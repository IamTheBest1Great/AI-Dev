# Module 6 — Model Context Protocol (MCP): Detailed Notes

## Table of Contents

- [6.1 MCP Architecture](#61-mcp-architecture)
  - [Hosts, clients, servers](#hosts-clients-servers)
  - [Tools, resources, prompts](#tools-resources-prompts)
  - [🆕 Currency note: the protocol is actively expanding](#-currency-note-the-protocol-is-actively-expanding)
  - [📋 Interview Questions — 6.1](#-interview-questions--61)
- [6.2 Building MCP Servers](#62-building-mcp-servers)
  - [Node.js implementation](#nodejs-implementation)
  - [Tool schema definition](#tool-schema-definition)
  - [Error handling](#error-handling)
  - [📋 Interview Questions — 6.2](#-interview-questions--62)
- [6.3 Publishing MCP Servers](#63-publishing-mcp-servers)
  - [npm publishing](#npm-publishing)
  - [Registry submission](#registry-submission)
  - [Claude Desktop / client integration](#claude-desktop--client-integration)
  - [📋 Interview Questions — 6.3](#-interview-questions--63)
- [6.4 MCP vs Traditional APIs 🆕](#64-mcp-vs-traditional-apis-)
  - [How MCP differs from plain function calling/REST integration](#how-mcp-differs-from-plain-function-callingrest-integration)
  - [When MCP is worth the overhead vs a direct API call](#when-mcp-is-worth-the-overhead-vs-a-direct-api-call)
  - [📋 Interview Questions — 6.4](#-interview-questions--64)
- [6.5 Ecosystem Strategy](#65-ecosystem-strategy)
  - [Early mover advantage and ecosystem formation timing](#early-mover-advantage-and-ecosystem-formation-timing)
  - [Visibility within the Claude ecosystem](#visibility-within-the-claude-ecosystem)
  - [📋 Interview Questions — 6.5](#-interview-questions--65)

> MCP itself is mid-transition as of mid-2026: a major spec revision (2026-07-28) and a new v2 TypeScript SDK are landing, while the registry and Claude's own connector surfaces have matured a lot since MCP's 2024 launch. These notes use the current production-recommended SDK and flag what's actively changing.

---

## 6.1 MCP Architecture

### Hosts, clients, servers
MCP's architecture has three distinct roles, and conflating them is the most common source of confusion when first learning the protocol:
- **Host** — the actual application the person uses (Claude Desktop, Claude Code, an IDE, Cowork). The host creates and manages MCP clients, enforces security policy, handles user consent/authorization prompts, and coordinates context across however many servers are connected.
- **Client** — lives *inside* the host and maintains a **1:1 stateful connection** to exactly one MCP server, implementing the protocol's client side (capability negotiation, message framing). A host with five connected servers is running five clients internally, even though the person only sees one application.
- **Server** — a separate process or remote service that exposes tools, resources, and prompts over the protocol. Critically, **a server has no direct relationship with the LLM at all** — it just exposes capabilities; the host and client are what bridge those capabilities into the model's context and tool-calling loop.

Two transport types connect a client to a server: **stdio** (the host spawns the server as a local child process and communicates over stdin/stdout — used for local integrations like filesystem access) and **Streamable HTTP** (the modern, recommended transport for remote servers, supporting both stateless and stateful/session-based modes). An older **HTTP+SSE** transport still exists but is now positioned as backwards-compatibility-only, not the recommended path for new servers.

### Tools, resources, prompts
MCP standardizes three distinct primitives, and the key differentiator between them is **who decides to invoke each one**:

| Primitive | Who invokes it | Analogy |
|---|---|---|
| **Tools** | The **model** decides to call it, based on the conversation | Function calling (Module 2), protocol-standardized |
| **Resources** | The **application/host** (or the user, via @-mention) attaches it | Read-only context, like attaching a file |
| **Prompts** | The **user** explicitly selects it | A reusable template, like a slash command |

- **Tools** let the LLM take actions — computation, side effects, network calls — exactly the function-calling concept from Module 2, just standardized at the protocol level so any MCP-compatible host can discover and call them the same way.
- **Resources** expose read-only data the client can surface to the user or model — e.g., a database table, a config file, a document — without the model needing to decide to "call" anything; the host/user controls when a resource gets pulled into context.
- **Prompts** are reusable, parameterized templates that help a user start a well-structured interaction with predefined arguments, rather than being something the model invokes mid-reasoning.

### 🆕 Currency note: the protocol is actively expanding
MCP's 2026-07-28 specification (the largest revision since launch) adds capability beyond the original three primitives, worth knowing about even at a conceptual level:
- **Tasks** — a formal extension for long-running, asynchronous operations, so a tool call doesn't have to block the whole interaction waiting for a slow operation to finish.
- **MCP Apps** — server-rendered UI as a protocol extension, directly connected to the interactive connector experiences now available in Claude's own interfaces (rich, in-conversation UI rather than plain text tool results).
- A formal **extensions framework** (reverse-DNS-identified, independently versioned) and a **deprecation policy**, meaning the protocol can now evolve new capabilities without breaking existing implementations — a maturity signal for a protocol that was still in rapid flux through 2025.
- The revision also moves toward a **stateless core that runs on ordinary HTTP infrastructure** (plain load balancers, no sticky sessions required), and tightens **OAuth/OIDC alignment** for remote server authorization.

### 📋 Interview Questions — 6.1
1. **Explain the difference between an MCP host and an MCP client, using a concrete example.**
   *Look for: the host (e.g., Claude Desktop) is the application the user sees and that manages everything; the client is the internal connection-handling component, one per connected server — a host with 5 connected servers runs 5 clients internally.*
2. **Why does a tool get invoked by the model, while a resource gets invoked by the application or user — what does that design choice actually buy you?**
   *Look for: tools are for actions the model should decide to take based on reasoning (needs LLM judgment); resources are context the host/user has already decided is relevant — putting that decision in the right place avoids the model needing to "guess" it should pull in static context it could just be given directly.*
3. **Why is "the server has no direct relationship with the LLM" an important architectural fact, not just a technicality?**
   *Look for: it clarifies that servers are reusable across any host/model combination — a server doesn't need to know or care which model is using it, which is exactly what makes one server usable by many different MCP clients.*
4. **When would you choose stdio transport over Streamable HTTP for an MCP server you're building?**
   *Look for: stdio for local-only integrations where the host spawns the server as a subprocess (e.g., filesystem access); Streamable HTTP for anything that needs to run remotely and be reachable over the network.*
5. **What problem does the Tasks extension in the 2026-07-28 spec solve that the original three primitives didn't address?**
   *Look for: long-running operations previously had no standardized way to avoid blocking the whole interaction; Tasks formalizes asynchronous, non-blocking tool execution.*

---

## 6.2 Building MCP Servers

### Node.js implementation
The official TypeScript SDK (`@modelcontextprotocol/sdk`, currently v1.x and the production-recommended version) provides a high-level `McpServer` class that handles protocol plumbing so you focus on defining capabilities:

```ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "weather-server", version: "1.0.0" });

server.registerTool(
  "get_weather",
  {
    title: "Get Weather",
    description: "Get the current weather for a location",
    inputSchema: { location: z.string().describe("City name, e.g. 'Mumbai'") },
  },
  async ({ location }) => {
    const data = await fetchWeather(location); // your own implementation
    return { content: [{ type: "text", text: `${data.tempC}°C, ${data.conditions}` }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

The same `registerResource` and `registerPrompt` methods exist for the other two primitives, following the same pattern: register a name, metadata, and a handler. For remote servers, swap `StdioServerTransport` for `StreamableHTTPServerTransport`.

**🆕 Currency note**: a separate, redesigned **v2 SDK** (a new package, `@modelcontextprotocol/server`, distinct from today's `@modelcontextprotocol/sdk`) is being built alongside the 2026-07-28 spec revision and is expected to reach stable release around Q3 2026. Until then, **v1.x remains the version to use for production**, and will keep receiving security/bug fixes for at least six months after v2 ships — worth knowing so you don't accidentally build against pre-release v2 APIs for a production server today.

### Tool schema definition
A tool's `inputSchema` is typically defined with Zod (the SDK has a required peer dependency on it, though it also supports other Standard-Schema-compatible libraries like ArkType and Valibot) — and every principle from Module 2.1 applies directly here, because an MCP tool schema *is* a JSON-Schema-like tool definition under the hood: narrow types over free-text strings, required fields marked deliberately, and clear per-field descriptions.

Beyond the input side, you can also declare an **`outputSchema`** and return **`structuredContent`** alongside the usual text content — giving callers a type-safe, validated structured result instead of having to parse it back out of a text blob, directly applying Module 2.3's structured-output principles to tool *results*, not just tool *inputs*:
```ts
server.registerTool(
  "calculate-bmi",
  {
    description: "Calculate Body Mass Index",
    inputSchema: { weightKg: z.number(), heightM: z.number() },
    outputSchema: { bmi: z.number() },
  },
  async ({ weightKg, heightM }) => {
    const output = { bmi: weightKg / (heightM * heightM) };
    return { content: [{ type: "text", text: JSON.stringify(output) }], structuredContent: output };
  }
);
```
Tools also support **annotations** (e.g., hints like whether a tool is read-only or destructive) — metadata the *host* can use to decide UI treatment or whether to require explicit confirmation before calling it, directly connecting to the human-in-the-loop and security patterns from Modules 1.6 and 9.3.

### Error handling
MCP distinguishes two different kinds of failure, and conflating them produces confusing agent behavior:
- **Protocol-level errors** — a malformed request, an unknown tool name — these are hard failures the client/transport layer surfaces as exceptions.
- **Tool-execution errors** — the tool ran, but the underlying operation failed (an API call 404'd, a file wasn't found). The well-behaved pattern is to return this *as a normal tool result* with an error flag set (`isError: true` in the content), rather than throwing — because that lets the **model see the failure as an observation it can reason about and recover from** (try a different argument, ask the user for clarification), exactly the Module 1.3 ReAct loop in action, instead of the whole interaction hard-crashing.

Beyond per-call error handling, two further disciplines matter for production servers:
- **Logging** — a server can declare a `logging` capability and send leveled messages back to the client (`debug` through `emergency`) for visibility during execution, directly supporting the observability practices from Module 9.2.
- **Output size limits** — clients commonly cap how much a single tool result can return into context (e.g., a default ceiling around 25,000 tokens, configurable, with a warning threshold well below that) specifically to prevent one verbose tool call from blowing out the conversation's context window — a direct, concrete mitigation for Module 1.5's context-overflow failure mode, enforced at the client/host level rather than left to each server author's discipline alone.

### 📋 Interview Questions — 6.2
1. **Why does the SDK encourage returning a tool execution failure as a normal result with `isError: true` instead of throwing an exception?**
   *Look for: returning it as a result lets the model see and reason about the failure as an observation (and potentially recover), while throwing surfaces it as a hard protocol-level failure the agent loop can't gracefully incorporate.*
2. **What's the practical benefit of declaring an `outputSchema` and returning `structuredContent`, beyond just returning text that happens to contain JSON?**
   *Look for: type-safe, validated structured output the calling code can trust programmatically, rather than needing to parse and hope the text actually contains valid JSON in the expected shape — the same Module 2.3 principle applied to tool outputs.*
3. **Why would a host enforce a maximum token limit on MCP tool output, and what failure mode does that protect against?**
   *Look for: prevents a single verbose tool call from consuming a large fraction of the context window — a direct, systemic mitigation for the context-overflow failure mode from Module 1.5.*
4. **A tool's `inputSchema` accepts any string for a `date` field. What would you change, and why does it matter more for an MCP tool than it might seem?**
   *Look for: narrowing to a validated date format/type — the same schema-design discipline from Module 2.1; it matters especially here because the tool may be consumed by many different hosts/clients you don't control, so looser validation propagates inconsistency more broadly than a single bespoke integration would.*
5. **If you're starting a new production MCP server today, would you build it against the current SDK or the in-development v2 SDK? Why?**
   *Look for: the current v1.x SDK — it's the explicitly recommended version for production, with v2 still pre-stable and tied to a spec revision that hasn't shipped yet; building against pre-release APIs for production work is a real currency/stability risk.*

---

## 6.3 Publishing MCP Servers

### npm publishing
A Node.js-based MCP server is packaged and distributed like any other npm package — typically with a `bin` entry so it can be invoked directly via `npx your-package-name`, which is exactly how most reference servers and community servers are designed to be run (no separate install step required by the end user; `npx` fetches and runs it on demand). Publishing itself is the ordinary `npm publish` flow — npm is simply the underlying code-hosting layer here, not an MCP-specific concept.

### Registry submission
**🆕 The official MCP Registry** (`registry.modelcontextprotocol.io`) is a **metaregistry**: it stores only *metadata* describing where to find a server, not the server's actual code or binary. The real package still lives on npm, PyPI, Docker Hub, NuGet, or as a GitHub release — the registry's job is giving every MCP client a single, canonical place to discover and resolve that metadata, rather than every client needing its own hand-maintained server list.

Publishing involves:
1. Authoring a `server.json` manifest (name, description, version, and the `packages` array pointing at the actual npm/PyPI/etc. package).
2. Publishing it via the official `mcp-publisher` CLI.
3. **Namespace authentication** — server names follow a reverse-DNS-style format (`io.github.<username>/<server-name>` verified via GitHub OAuth, or `<your-domain>/<name>` verified via DNS/HTTP challenge) — this ties every published name to a verified identity, specifically to prevent impersonation and namespace squatting in an open, permissionless publishing system.

The registry is explicitly **not** for private servers — if a server is only reachable on an internal network or a private package registry, the guidance is to run your own private sub-registry rather than publish it to the public one. The official registry is also designed to be consumed primarily by **downstream aggregators and marketplaces** (Anthropic's own connector Directory, PulseMCP, and others), who layer curation, ratings, and additional security scanning on top — the base registry itself focuses narrowly on namespace authentication and spam prevention, not deep code-level vetting.

### Claude Desktop / client integration
There are now multiple distinct paths to get an MCP server connected to a Claude client, and they serve different audiences:
- **Manual config file** (`claude_desktop_config.json`) — the original method, still supported, most relevant for local stdio servers you're developing or running entirely on your own machine.
- **Custom Connectors (UI)** — Claude Desktop's modern, recommended way to add a *remote* MCP server, with a guided flow and built-in OAuth support, requiring no manual JSON editing. This is also how remote MCP servers connect across claude.ai, Cowork, and the mobile apps generally — note that since the connection is brokered through Anthropic's cloud infrastructure for remote connectors, your server needs to be reachable over the public internet (allowlisting Anthropic's IP ranges), not hidden behind a corporate VPN or firewall.
- **Desktop Extensions (`.mcpb` files)** — one-click installable packages (an open specification, evolved from the earlier `.dxt` format) bundling a `manifest.json` with the server itself, supporting Node.js, Python, or compiled binary servers. Sensitive configuration fields (API keys, tokens) marked `"sensitive": true` in the manifest are automatically encrypted using the OS's native secure storage (Keychain on macOS, Credential Manager on Windows) rather than sitting in plain text.
- **Anthropic's reviewed Directory** — a curated set of connectors browsable directly inside Claude's "Connectors" UI; submitting a desktop extension or connector for review is a separate, additional step from simply publishing to the open MCP Registry, and grants meaningfully higher visibility and trust within Claude's own surfaces specifically.

For **Claude Code**, integration happens via CLI: `claude mcp add --transport http <name> <url> --header "Authorization: Bearer <token>"` for remote servers (HTTP is the recommended, most widely supported transport), or `claude mcp add --transport stdio <name> -- <command> [args...]` for local servers. Claude Code also keeps MCP's context footprint low via **tool search** — only tool names and server instructions load into context at session start, with full tool definitions deferred until actually needed, so connecting many MCP servers doesn't automatically bloat every conversation's context window.

For **developers building their own applications** on the Claude API directly, the **MCP connector** lets you pass `mcp_servers` straight into a Messages API call without running a separate MCP client process yourself — the API handles the connection and tool-calling loop server-side. As of the current beta (`mcp-client-2025-11-20`; the earlier `mcp-client-2025-04-04` version is deprecated), this is available on the Claude API, Claude Platform on AWS, and Microsoft Foundry, but not currently on Amazon Bedrock or Vertex AI.

### 📋 Interview Questions — 6.3
1. **Why is the official MCP Registry described as a "metaregistry," and what does that imply about where you'd actually go to download a server's code?**
   *Look for: the registry only stores metadata/pointers; the real package still lives on npm/PyPI/Docker Hub/etc. — the registry is a discovery layer, not a hosting layer.*
2. **What problem does namespace authentication (`io.github.<username>/<name>`) solve in an open, permissionless publishing system?**
   *Look for: prevents impersonation and namespace squatting by tying every published server name to a verified GitHub or domain identity, rather than allowing arbitrary unverified names.*
3. **What's the difference between publishing to the open MCP Registry and getting accepted into Anthropic's connector Directory, and why might a server author want to pursue both?**
   *Look for: the open registry maximizes discoverability across every MCP client (Cursor, Windsurf, custom agents, etc.); the Directory is a separate, curated, higher-trust surface specifically inside Claude's own interfaces — different audiences, different vetting bars, complementary rather than redundant.*
4. **Why does a remote MCP server need to be reachable from the public internet to work as a Custom Connector in Claude Desktop, even though Claude Desktop itself runs locally?**
   *Look for: remote connectors are brokered through Anthropic's cloud infrastructure even when the host app is local, so the connection to your server originates from Anthropic's IP ranges, not the user's own machine — meaning a server behind a corporate VPN/firewall won't be reachable unless explicitly allowlisted.*
5. **How does Claude Code's "tool search" behavior change the calculus of connecting many MCP servers to one agent, compared to a naive implementation that loads every tool's full schema upfront?**
   *Look for: deferring full tool definitions until actually needed (loading only names/instructions at session start) means adding more connected servers doesn't proportionally bloat the context window — directly mitigating the Module 2.2 "more tools dilutes selection accuracy / costs more tokens" problem at scale.*

---

## 6.4 MCP vs Traditional APIs 🆕

### How MCP differs from plain function calling/REST integration
With **plain function calling** (Module 2), you hand-write a tool schema for every capability you want *your specific agent* to have, wired directly into your own code and agent loop. With a **raw REST API**, the calling code (or the model, if given raw access) has to understand that particular service's own auth scheme, response shapes, and pagination conventions — none of which are LLM-native or standardized across services.

MCP's actual contribution is **standardizing the interface, not the underlying capability**: once a service exposes an MCP server, *any* MCP-compatible host or client — Claude Desktop, Claude Code, Cursor, a custom-built agent — can discover and use it through the same standardized operations (`tools/list`, `tools/call`, and the equivalent for resources/prompts), with zero bespoke integration code per consuming application. This decouples **"someone built an integration for service X"** from **"which specific agent gets to use it"**: without MCP, supporting N services across M different agent applications means up to N×M custom integrations; with MCP, it's N servers that every M-th client can use for free, once they speak the protocol.

MCP also offers primitives plain function calling has no standardized analog for: **resources** (read-only, application/user-attached context, not just model-invoked actions) and **prompts** (reusable, user-selectable interaction templates) — giving a server a way to expose more than just callable functions.

### When MCP is worth the overhead vs a direct API call
MCP isn't free: it means running and hosting an extra server process, handling transport/protocol plumbing, and capability negotiation — real overhead a single, simple, bespoke function-calling tool definition doesn't carry. For one agent calling one straightforward API a handful of times, plain function calling (Module 2) is usually simpler and has fewer moving parts.

MCP earns its overhead when at least one of these is true:
- **Multiple consumers** — the same integration needs to be usable by more than one agent/application/team, not just the one you're currently building.
- **External or non-engineering consumers** — you want other developers, or non-engineers using a no-code/agent platform, to be able to plug into your service without reading custom API documentation.
- **Richer interaction needs** — the integration genuinely benefits from primitives beyond a single function call: long-running async tasks, server-initiated sampling, reusable prompt templates, or dynamically discoverable resources.
- **Ecosystem distribution intent** — you actually want to publish and have the integration discovered broadly (via the registry, a Directory listing) rather than keep it private to one application.

A reasonable rule of thumb: **building one agent for one internal use case → plain function calling is usually simpler and faster to ship. Building a reusable integration meant to be consumed by many different AI applications (including ones you don't control) → MCP is the better investment.**

### 📋 Interview Questions — 6.4
1. **What does MCP actually standardize — the capability itself, or the interface to the capability? Why does that distinction matter?**
   *Look for: it standardizes the *interface* (discovery and invocation pattern), not the underlying capability — a GitHub MCP server still has to implement GitHub-specific logic; what MCP buys you is that any compliant client can talk to it the same way.*
2. **Explain the N×M integration problem MCP is designed to collapse, with a concrete example.**
   *Look for: without a standard, N services × M consuming applications can require up to N×M bespoke integrations; with MCP, each of the N services needs one server, and each of the M applications needs to speak MCP once — not once per service.*
3. **A team is building a single internal agent that calls exactly one internal API a few times. Would you recommend they build an MCP server for it? Why or why not?**
   *Look for: probably not — plain function calling is simpler for a single-consumer, single-service case; MCP's overhead (server process, protocol plumbing) isn't justified without multiple consumers or richer interaction needs.*
4. **What can an MCP server expose that a plain function-calling tool definition structurally cannot?**
   *Look for: resources (application/user-attached read-only context) and prompts (reusable, user-selectable templates) — primitives beyond a single callable action, which plain function calling has no standardized equivalent for.*
5. **If a company wants their product to be usable by both their own internal Claude-based agent and external developers building on completely different AI platforms, what does that imply about whether to invest in an MCP server?**
   *Look for: strongly favors MCP — the "external, multiple, unknown consumers" condition is exactly the scenario where the protocol-standardization investment pays off, versus building separate bespoke integrations per consuming platform.*

---

## 6.5 Ecosystem Strategy

### Early mover advantage and ecosystem formation timing
MCP is still a genuinely young, actively-evolving ecosystem as of mid-2026 — the registry only reached a preview in September 2025, hit an API freeze (v0.1) in October 2025, and the protocol itself is undergoing its largest revision yet (2026-07-28), alongside a new v2 SDK generation. This matters strategically in two opposing ways:
- **Being early and well-built in a category captures outsized, durable advantage.** Namespace authentication prevents literal name-squatting, but being the first genuinely solid, well-reviewed server in an important category (e.g., "the" GitHub server, "the" Slack server) means later entrants compete against an incumbent with accumulated usage history, reviews, and often reference-implementation status — the official `modelcontextprotocol/servers` repository deliberately houses only a small number of steering-group-maintained reference servers, leaving the rest of the ecosystem to compete on quality and adoption.
- **Protocol churn is itself a live strategic risk**, cutting the other way. With a major breaking spec revision landing and a new SDK generation in development, early movers who keep pace with spec changes (SDK tier-support expectations exist specifically to hold implementers to a timeline) retain their advantage; early movers who *don't* keep up risk being stuck on a deprecating transport (HTTP+SSE is now explicitly backwards-compatibility-only) while newer, spec-current entrants leapfrog them on capability and reliability.

The practical takeaway: "early mover advantage" in MCP isn't a one-time bet you place and then ignore — it requires continued maintenance investment to stay current as the protocol itself matures, or the early-mover advantage erodes.

### Visibility within the Claude ecosystem
Within Claude's own surfaces specifically, there are two distinct visibility channels worth telling apart:
- **The open MCP Registry** — maximizes discoverability across *every* MCP-compatible client broadly (Claude, Cursor, Windsurf, custom-built agents, and anything else that speaks the protocol). This is the right target if the goal is protocol-wide reach.
- **Anthropic's reviewed connector Directory** (and Desktop Extensions review) — a separate, curated, higher-trust surface specifically inside Claude Desktop, claude.ai, and Cowork's "Connectors" UI. Getting listed here is a materially different (and generally higher-bar) outcome than simply publishing to the open registry — it's specifically optimized for visibility and trust *within Claude's own surfaces*, at the cost of needing to go through Anthropic's review process rather than the registry's lighter namespace-authentication-only bar.

These two channels serve different audiences and aren't mutually exclusive — a server aiming for maximum reach typically pursues both: open registry publication for protocol-wide discoverability, and Directory/extension review submission specifically to capture the additional trust and visibility that comes from being surfaced directly inside Claude's own connector UI.

### 📋 Interview Questions — 6.5
1. **Why is MCP's "early mover advantage" not simply a one-time strategic bet, given the state of the protocol in 2026?**
   *Look for: ongoing spec revisions and SDK generations mean early movers have to keep investing to stay current, or risk being left on deprecated transports/APIs while newer, spec-current entrants catch up — the advantage requires maintenance, not just a head start.*
2. **What's the practical difference between publishing to the open MCP Registry and getting accepted into Anthropic's connector Directory, in terms of strategy?**
   *Look for: the open registry targets protocol-wide reach across any MCP client; the Directory specifically targets visibility and trust within Claude's own surfaces, with a higher review bar — different goals, often pursued together rather than as alternatives.*
3. **Why does the official `modelcontextprotocol/servers` repository deliberately house only a small number of reference servers rather than trying to be the canonical home for every server?**
   *Look for: it's explicitly meant as educational reference implementations maintained by the steering group, not a comprehensive catalog — the broader ecosystem (registry, community servers) is where breadth and competition actually happen.*
4. **A team built one of the very first MCP servers for a popular SaaS tool in 2024, but hasn't updated it since. How would you expect their competitive position to have changed by mid-2026, and why?**
   *Look for: recognizing that without keeping pace with spec revisions (deprecated transports, new capabilities like Tasks/MCP Apps), an unmaintained early server can lose its advantage to newer, spec-current competitors, despite the head start.*
5. **If you were advising a company building an MCP server for their product, would you recommend targeting the open registry, Anthropic's Directory, or both — and why?**
   *Look for: generally both, since they serve genuinely different audiences (protocol-wide reach vs. Claude-surface-specific trust/visibility) rather than being substitutes for each other.*

---

*End of Module 6 detailed notes — 25 interview questions total across 5 sections. MCP's spec, SDKs, and registry are all actively evolving as of mid-2026 — re-verify against current docs (modelcontextprotocol.io, docs.claude.com) before building production integrations.*
