# Module 3 — Frameworks: Detailed Notes

## Table of Contents

- [3.1 LangChain 🆕](#31-langchain-)
  - [Core abstractions (chains, runnables, agent executors)](#core-abstractions-chains-runnables-agent-executors)
  - [When LangChain is the right layer vs raw API calls](#when-langchain-is-the-right-layer-vs-raw-api-calls)
  - [📋 Interview Questions — 3.1](#-interview-questions--31)
- [3.2 LangGraph](#32-langgraph)
  - [Stateful workflows](#stateful-workflows)
  - [Conditional branching](#conditional-branching)
  - [Cycles and loops](#cycles-and-loops)
  - [Checkpoints and resumability](#checkpoints-and-resumability)
  - [Human-in-the-loop nodes](#human-in-the-loop-nodes)
  - [📋 Interview Questions — 3.2](#-interview-questions--32)
- [3.3 OpenAI Assistants/Agents API](#33-openai-assistantsagents-api)
  - [Threads and messages](#threads-and-messages)
  - [File search](#file-search)
  - [Code interpreter](#code-interpreter)
  - [Function calling](#function-calling)
  - [🆕 Note: OpenAI's agent-facing APIs have shifted rapidly — current status](#-note-openais-agent-facing-apis-have-shifted-rapidly--current-status)
  - [📋 Interview Questions — 3.3](#-interview-questions--33)
- [3.4 CrewAI / CrewAI.js](#34-crewai--crewaijs)
  - [Multi-agent with roles](#multi-agent-with-roles)
  - [Goal and delegation](#goal-and-delegation)
  - [Task dependencies](#task-dependencies)
  - [Agent communication](#agent-communication)
  - [🆕 A note on "CrewAI.js"](#-a-note-on-crewaijs)
  - [📋 Interview Questions — 3.4](#-interview-questions--34)
- [3.5 AutoGen / AG2 🆕](#35-autogen--ag2-)
  - [Conversable agents](#conversable-agents)
  - [Group chat orchestration](#group-chat-orchestration)
  - [Code execution agents](#code-execution-agents)
  - [🆕 Current status — important to know before recommending this stack](#-current-status--important-to-know-before-recommending-this-stack)
  - [📋 Interview Questions — 3.5](#-interview-questions--35)
- [3.6 LlamaIndex Agents 🆕](#36-llamaindex-agents-)
  - [Data agents over indexed corpora](#data-agents-over-indexed-corpora)
  - [Query engines as tools](#query-engines-as-tools)
  - [📋 Interview Questions — 3.6](#-interview-questions--36)
- [3.7 Other Notable Frameworks 🆕](#37-other-notable-frameworks-)
  - [Semantic Kernel (Microsoft)](#semantic-kernel-microsoft)
  - [Vercel AI SDK agent primitives](#vercel-ai-sdk-agent-primitives)
  - [Google Agent Development Kit (ADK)](#google-agent-development-kit-adk)
  - [📋 Interview Questions — 3.7](#-interview-questions--37)
- [3.8 Framework Selection 🆕](#38-framework-selection-)
  - [Build-vs-buy: raw API + custom loop vs framework](#build-vs-buy-raw-api--custom-loop-vs-framework)
  - [Comparison criteria: control, debuggability, lock-in, community size](#comparison-criteria-control-debuggability-lock-in-community-size)
  - [📋 Interview Questions — 3.8](#-interview-questions--38)

> Frameworks in this space move fast — several of the frameworks below changed meaningfully between when most tutorials were written and today. Where it matters, these notes call out the *current* (mid-2026) state explicitly rather than the historically popular pattern, since interviewers increasingly probe for whether a candidate's knowledge is current.

---

## 3.1 LangChain 🆕

### Core abstractions (chains, runnables, agent executors)
LangChain's composition layer is built on the **Runnable protocol** — a standard interface (`invoke`, `batch`, `stream`, and their async equivalents) that every component implements, so models, prompts, parsers, and retrievers can all be composed the same way using `.pipe()` / `RunnableSequence` / `RunnableParallel` (this composition style is often called LCEL). A **chain** is simply a fixed composition of Runnables — a predetermined sequence with no model-driven branching, matching the "fixed workflow" point on the agency spectrum from Module 1.1.

**Important currency note**: the original `AgentExecutor` (and `initialize_agent`) — the class that historically ran the dynamic agent loop — is now legacy. As of LangChain's 1.0 release, the legacy agent APIs moved into a `langchain-classic` package, and **`create_agent`** is the current recommended entry point for building an agent: it takes a model, tools, and optional middleware, and internally compiles to a LangGraph-based runtime rather than the old hidden-scratchpad loop. In other words, "agent executor" as a concept survives, but the implementation underneath it has moved to the graph-based runtime described in 3.2.

### When LangChain is the right layer vs raw API calls
LangChain earns its abstraction overhead when you need:
- **Multi-provider portability** — swapping model providers or vector stores without rewriting integration code.
- **A large, linear pipeline** — RAG retrieval → prompt assembly → generation → parsing, where the path is fixed ("dumb pipes," in practitioner shorthand).
- **A standardized way to plug into the broader ecosystem** — tracing (LangSmith), prebuilt retrievers, document loaders, output parsers.

Raw API calls are usually the better choice when:
- The task is simple enough that the abstraction adds more indirection than value (a single tool-calling loop doesn't need a framework).
- Debuggability matters more than speed of initial development — extra abstraction layers make it harder to see exactly what's being sent to the model.
- You want to avoid the dependency surface and version-churn risk that comes with a fast-moving framework (LangChain's own API surface changed substantially across its pre-1.0 releases).

The current practitioner consensus (mid-2026) is **not** "LangChain vs LangGraph" as a binary choice — it's "LangChain for the linear/integration parts, LangGraph for the cyclic/agentic execution," often used together in the same application.

### 📋 Interview Questions — 3.1
1. **What is a Runnable, and why does standardizing on that interface matter for composability?**
   *Look for: a single interface (invoke/batch/stream) that every component shares, so anything can be piped into anything else without bespoke glue code.*
2. **Why is `AgentExecutor` no longer the recommended way to build a LangChain agent, and what replaced it?**
   *Look for: awareness that it's now legacy/moved to `langchain-classic`; `create_agent` is current, and it's built on the LangGraph runtime under the hood — not a separate, simpler loop.*
3. **A team wants to build a simple, fixed RAG pipeline (retrieve → prompt → generate). Would you reach for LangChain or raw API calls, and why?**
   *Look for: recognizing this is a "dumb pipe" — LangChain's integration ecosystem (retrievers, parsers) earns its keep here, since there's no agentic branching to justify avoiding the abstraction.*
4. **What's the actual relationship between LangChain and LangGraph in 2026 — are they competitors?**
   *Look for: they're layers of the same stack; LangChain's high-level `create_agent` runs on LangGraph's runtime, not a competing one.*
5. **What's a concrete risk of building production infrastructure on a framework with LangChain's history of API churn (AgentExecutor → create_react_agent → create_agent)?**
   *Look for: migration cost, stale tutorials/training data leading to deprecated patterns, and the discipline of checking current docs rather than memory or old blog posts.*

---

## 3.2 LangGraph

### Stateful workflows
LangGraph models an agent (or any multi-step LLM application) as an explicit **graph**: nodes are functions that read and update a shared **state object** (typically a typed schema — `TypedDict` or a Pydantic model), and edges define how control passes between nodes. Unlike LangChain's linear Runnable composition, state here is explicit and persists across every node, rather than implicitly flowing through a pipe.

### Conditional branching
Edges can be **conditional** — a routing function inspects the current state and decides which node executes next. This is how classification-then-route patterns (Module 2.2's "two-stage routing") and "is the task done or not" decisions are implemented structurally, rather than being buried inside a single large prompt.

### Cycles and loops
The defining capability that separates LangGraph from a plain DAG-based pipeline tool: **edges can form cycles**. A node can route back to an earlier node, which is exactly what a ReAct-style think→act→observe→repeat loop requires (Module 1.3). This is *why* `create_agent` is built on LangGraph rather than LCEL — LCEL's Runnable composition has no native way to express "go back and do this again."

### Checkpoints and resumability
A **checkpointer** persists the graph's state at each step to durable storage (Postgres, Redis, SQLite, etc.). This buys two distinct things:
- **Resumability** — execution can pause and resume later, even after a process restart, which is essential for the asynchronous approval pattern from Module 1.6.
- **"Time-travel" debugging** — because every intermediate state is saved, you can replay a run from any checkpoint or fork a new execution path from a past state to test an alternative without re-running everything from scratch.

### Human-in-the-loop nodes
LangGraph supports **interrupts** placed before or after specific nodes (e.g., a node that sends an email or executes code) — when execution reaches that point, the graph pauses, the state is checkpointed, a human is notified, and execution resumes exactly where it left off once approved. This is a direct, structural implementation of the checkpoints/approval-workflow concepts from Module 1.6, rather than something bolted on with custom code.

### 📋 Interview Questions — 3.2
1. **Why can't a purely linear Runnable/LCEL pipeline implement a ReAct loop, but a LangGraph graph can?**
   *Look for: LCEL composition is acyclic by construction; LangGraph explicitly supports cycles, which a repeat-until-done loop requires.*
2. **What two distinct benefits does a checkpointer provide, beyond "the agent can pause"?**
   *Look for: resumability across process restarts/long waits, and time-travel debugging (replay/fork from any saved state).*
3. **How would you implement a "confirm before this agent sends an email" requirement in LangGraph, structurally rather than with ad hoc code?**
   *Look for: placing an interrupt before the email-sending node, relying on the checkpointer to persist state during the pause.*
4. **What's the difference between a conditional edge and a node in LangGraph's mental model?**
   *Look for: nodes do work and update state; conditional edges are pure routing logic that decide which node runs next based on the current state — they shouldn't themselves perform side effects.*
5. **A junior engineer says "we don't need LangGraph, we can just write a while loop." When would they be right, and when would a graph-based approach clearly win?**
   *Look for: a simple, single-path loop with no branching/checkpointing/HITL needs is fine as a while loop; LangGraph earns its complexity once you need durable state, conditional routing, or pause/resume semantics.*

---

## 3.3 OpenAI Assistants/Agents API

### Threads and messages
In the original Assistants API architecture, a **thread** was a server-managed conversation container holding a sequence of **messages**; OpenAI handled storing and (when needed) truncating conversation history, removing the burden of managing context windows manually.

### File search
A built-in retrieval tool: you uploaded files, OpenAI created and managed the underlying vector store, and the assistant could search across them as a tool — handling chunking/embedding/retrieval without you building that pipeline yourself.

### Code interpreter
A hosted, sandboxed Python execution tool — the assistant could write and run code (e.g., for data analysis or calculations) in an OpenAI-managed sandbox, rather than you provisioning your own execution environment.

### Function calling
Standard custom tool definitions, the same conceptual mechanism covered in Module 2 — the assistant could call your own application's functions in addition to the built-in tools above.

### 🆕 Note: OpenAI's agent-facing APIs have shifted rapidly — current status
This is no longer just a caution to "check current docs" — there is now a concrete, dated change worth knowing:
- OpenAI announced the **deprecation of the Assistants API in August 2025**, with a confirmed **sunset/shutdown date of August 26, 2026**.
- The replacement is the **Responses API** (paired with a separate **Conversations API** for server-side history management). Conceptually: Assistants → Prompts (now dashboard-managed, not API-created), Threads → Conversations, Runs → Responses.
- The Responses API is described as **"agentic by default"** — a single API call can let the model invoke multiple built-in tools (web search, file search, code interpreter, computer use, remote MCP servers) *and* your custom functions within one request, rather than requiring you to orchestrate the polling loop the Assistants API needed (`requires_action` states, manual run polling).
- OpenAI has stated there is **no automated migration tool** for Threads → Conversations — teams need to plan a manual migration before the shutdown date.
- As recently as June 2026, OpenAI also deprecated its separate "Agent Builder" product, pointing developers toward the **Agents SDK** (built on the Responses API) instead.

**Practical takeaway for anyone building today**: don't start a new integration on the Assistants API — build directly on the Responses API / Agents SDK, and if you inherit an existing Assistants API integration, treat the August 2026 deadline as a hard constraint, not a someday-migration.

### 📋 Interview Questions — 3.3
1. **If you inherited a production app built on the Assistants API today, what would your first action be?**
   *Look for: checking the confirmed sunset date and migration guide, not assuming there's open-ended time — the deadline is fixed and there's no automated Thread→Conversation migration tool.*
2. **What does "agentic by default" mean about the Responses API compared to the old Assistants API run-polling model?**
   *Look for: a single request can drive multiple tool calls (built-in and custom) in one agentic loop server-side, vs. the old pattern of manually polling run status and handling `requires_action` states client-side.*
3. **Why is it risky to learn "how to build an OpenAI agent" purely from a tutorial without checking the publish date?**
   *Look for: a concrete, recent example of API churn in this very space (Assistants API deprecation) — old tutorials teach a sunsetting pattern.*
4. **What's conceptually similar between OpenAI's Code Interpreter and a custom function-calling tool, and what's different?**
   *Look for: both are "tools" the model can invoke; Code Interpreter is hosted/sandboxed and managed by the provider, vs. a custom function which you implement and host yourself.*
5. **A teammate argues "threads are just like LangGraph checkpoints." Is that a fair comparison?**
   *Look for: partial — both persist state across turns/steps, but threads are a server-managed conversation history object, while LangGraph checkpoints persist an entire graph execution state (which can include far more than message history, like intermediate variables and pending interrupts).*

---

## 3.4 CrewAI / CrewAI.js

### Multi-agent with roles
CrewAI's central abstraction is the **Agent**, defined by a `role`, `goal`, and `backstory` — a short natural-language persona that shapes how that agent reasons and which tasks it's suited for (e.g., "Senior Data Analyst" vs. "Marketing Copywriter"). A more specific role tends to produce more focused, reliable behavior than a vague one, similar to the "tool descriptions as prompts" principle from Module 2.4 — role descriptions are prompts too.

CrewAI has two distinct orchestration modes, which is worth knowing explicitly:
- **Crews** — optimized for autonomy and collaborative, role-based reasoning between agents (closer to the "autonomous agent" end of the agency spectrum).
- **Flows** — a deterministic, event-driven orchestration layer with explicit state management, positioned for production cases that need precise, auditable control (closer to a conditional workflow, and structurally similar in spirit to LangGraph's graph model).

### Goal and delegation
Each agent has a `goal` it's working toward, and CrewAI supports a **hierarchical process** where a manager agent can delegate sub-tasks to other agents based on their roles, in addition to a **sequential process** where tasks run in a fixed order.

### Task dependencies
A `Task` can declare a `context` of other tasks whose output it depends on — CrewAI ensures those dependency tasks complete first and passes their output into the dependent task's context automatically, rather than you wiring that data flow by hand.

### Agent communication
Within a crew, agents share context through the crew's task pipeline and can be given delegation tools that let one agent explicitly hand off a sub-task to another. For communication *across* systems or with externally-hosted agents, CrewAI has added support for open protocols: **MCP** (Module 6) for tool access, and **A2A** (Agent-to-Agent) for cross-agent discovery and collaboration beyond a single crew.

### 🆕 A note on "CrewAI.js"
Worth flagging explicitly: CrewAI's officially maintained framework is **Python-only**. "CrewAI.js" / `crewai-js`-style packages that exist are **unofficial, community-built ports**, not maintained by CrewAI Inc. — at least one such repository's own maintainer has stated they've stopped maintaining it. If a JS/TS-native multi-agent framework is a hard requirement, this is a real constraint worth surfacing early rather than discovering after committing to the Python ecosystem's API surface in a JS port that may lag or go stale.

### 📋 Interview Questions — 3.4
1. **What's the practical difference between CrewAI's "Crews" and "Flows," and when would you pick one over the other?**
   *Look for: Crews = autonomous, role-based collaboration; Flows = deterministic, event-driven, explicit state — pick Flows when you need auditability/predictability, Crews when the task benefits from open-ended agent collaboration.*
2. **Why does writing a vague agent role like "Assistant" tend to produce worse results than "Senior Financial Analyst specializing in SEC filings"?**
   *Look for: role/goal/backstory function as prompts shaping reasoning — same principle as tool description quality from Module 2.4, applied to agent personas instead of tools.*
3. **How does CrewAI handle passing one task's output into a dependent task's context, and why does that matter for reliability?**
   *Look for: the `context` parameter on `Task` declares the dependency explicitly, and CrewAI sequences execution and injects the output automatically — avoiding manual, error-prone data wiring.*
4. **A candidate claims they're shipping a production Node.js app using "CrewAI.js." What would you want to verify before trusting that claim?**
   *Look for: recognizing CrewAI's official framework is Python-only, and that JS ports are unofficial/community-maintained — worth checking maintenance status and feature parity before relying on it in production.*
5. **What problem does A2A support solve that MCP support doesn't, in a multi-agent system?**
   *Look for: MCP standardizes how an agent accesses *tools and data*; A2A standardizes how separate *agents* discover and communicate with each other — different layers of the stack.*

---

## 3.5 AutoGen / AG2 🆕

### Conversable agents
AutoGen's foundational abstraction is the **conversable agent** — an agent defined primarily by how it participates in a message exchange: it can be configured to auto-reply, request human input at certain points, or terminate the conversation under specific conditions. Agents communicate by exchanging messages with each other, much like a multi-party chat, rather than through an explicit graph of function calls.

### Group chat orchestration
AutoGen popularized the **GroupChat** pattern: multiple conversable agents participate in a shared conversation, and a **manager** (itself often LLM-driven) dynamically decides which agent should "speak" (act) next based on the conversation so far. This is an *implicit*, dynamically-decided coordination model — contrast this with LangGraph's *explicit* graph edges, where the routing logic is defined in code ahead of time rather than decided turn-by-turn by a manager agent.

### Code execution agents
AutoGen includes agents specifically designed to write and execute code, typically inside a sandboxed environment (e.g., a Docker-based code executor) — useful for tasks like data analysis or verification-by-running-tests, conceptually similar to OpenAI's hosted Code Interpreter (3.3) but self-hosted and configurable.

### 🆕 Current status — important to know before recommending this stack
As of 2026, the AutoGen project landscape has split into three distinct paths, and this is a common interview/design-review question precisely because it trips up anyone relying on older material:
1. **Microsoft Agent Framework (MAF)** — the official, production-grade successor. It **merges AutoGen's multi-agent orchestration concepts with Semantic Kernel's enterprise stability** into one unified framework (see 3.7), reaching Release Candidate status for both .NET and Python in February 2026, with built-in checkpointing, multi-provider model support (including Claude), and support for A2A, AG-UI, and MCP.
2. **AutoGen itself** is now in **maintenance mode** — it still runs and is community-managed, but receives no new features. Microsoft's explicit guidance is that new users should start with MAF instead.
3. **AG2** (`ag2ai/ag2`) is a **community-led fork** that continues developing the legacy v0.2-style GroupChat API for teams who specifically want to keep that implicit-conversation pattern alive outside Microsoft's roadmap.

The broader industry direction mirrors what's happening elsewhere in this module: a shift from AutoGen's original **implicit, dynamically-managed GroupChat** model toward **explicit, graph-based Workflows** with typed state and defined edges — the same shift LangChain made toward LangGraph, and the same shift CrewAI made toward Flows.

### 📋 Interview Questions — 3.5
1. **What's the core difference between AutoGen's GroupChat coordination model and LangGraph's graph-based routing?**
   *Look for: GroupChat = implicit, a manager agent decides the next speaker dynamically at runtime; LangGraph = explicit, routing logic is defined as code/edges ahead of time.*
2. **If a team is starting a new multi-agent project on Microsoft's stack today, would you recommend AutoGen or Microsoft Agent Framework, and why?**
   *Look for: MAF — AutoGen is in maintenance mode with no new features; Microsoft's own guidance points new users to MAF.*
3. **What's the difference between AG2 and Microsoft Agent Framework, and why would a team choose the former despite MAF being the "official" path?**
   *Look for: AG2 is a community fork preserving the legacy v0.2 GroupChat API/pattern; a team might prefer it if they specifically value that implicit conversational model and don't want MAF's more structured, heavier workflow approach.*
4. **What risk does "implicit" coordination (a manager LLM deciding who speaks next) introduce that explicit graph-based routing avoids?**
   *Look for: less predictable/auditable execution paths, harder to test exhaustively, similar tradeoffs to the agent-vs-workflow discussion in Module 1.1 — flexibility traded for predictability.*
5. **Why might "AutoGen is actively maintained, I checked GitHub activity last week" not fully answer whether to build new production work on it?**
   *Look for: distinguishing between a project still receiving commits/community activity vs. being the vendor's recommended forward path — maintenance-mode status and "no new features" matters more for production decisions than raw commit activity.*

---

## 3.6 LlamaIndex Agents 🆕

### Data agents over indexed corpora
LlamaIndex's core differentiator versus the other frameworks in this module is its origin and strength in **data ingestion, indexing, and retrieval** — connectors for hundreds of data sources, multiple index types (vector, keyword, tree/hierarchical, knowledge-graph), and retrieval strategies (sub-question decomposition, recursive retrieval, hybrid search) that go well beyond plain vector similarity search. An "agent" in this context is most often a **reasoning loop wrapped around one or more of these retrieval/data capabilities**, making LlamaIndex a natural fit when the agent's core job is reasoning over a large, heterogeneous body of private data rather than primarily taking actions in the world.

As of 2026, LlamaIndex's recommended way to compose any non-trivial application (agentic or not) is its **Workflows** abstraction — an event-driven model where steps are triggered by typed events and themselves emit further events, supporting loops, branches, and parallel execution without needing to hand-encode that control flow into graph edges the way LangGraph does.

### Query engines as tools
A **query engine** wraps an index (e.g., a vector index over a document corpus) behind a simple `query()` interface that handles retrieval + synthesis. The key agentic pattern: wrap a query engine as a `QueryEngineTool` — give it a `name` and `description` (exactly the tool-description-as-prompt principle from Module 2.4) — and hand it to an agent alongside other tools. This lets an agent reason about *which* corpus/index is relevant to a given sub-question and route to it, rather than always querying a single flat index.

LlamaIndex's current agent classes (`FunctionAgent` for models with native function-calling, `ReActAgent` for any model) are built on top of Workflows, and `AgentWorkflow` extends this to **multi-agent handoff** — multiple named agents, each with a description, where the active agent can hand off control to another based on the task, similar in spirit to AutoGen's GroupChat but implemented on LlamaIndex's typed event/Workflow model rather than an implicit chat manager.

### 📋 Interview Questions — 3.6
1. **What kind of agentic task is LlamaIndex specifically well-suited for, compared to a more general framework like LangGraph or CrewAI?**
   *Look for: tasks centered on reasoning over large/heterogeneous private data corpora — its retrieval depth (sub-question decomposition, recursive retrieval, hybrid search, multiple index types) is its differentiator.*
2. **What is a QueryEngineTool, and why does its `description` field matter as much as it does for any other tool (per Module 2)?**
   *Look for: it wraps an index's query() interface as a callable tool; the description is what lets the agent decide which corpus is relevant to a given question — same tool-description-as-prompt principle applies.*
3. **How does LlamaIndex's event-driven Workflows model differ from LangGraph's explicit node/edge graph model, conceptually?**
   *Look for: Workflows are triggered by typed events that steps emit and consume (can be more implicit/flexible about control flow); LangGraph requires explicitly defined nodes and edges up front — different ergonomics for similar underlying capability (loops, branches).*
4. **When would you choose multiple QueryEngineTools over a single QueryEngineTool with a larger combined index?**
   *Look for: when sources are meaningfully different in domain/structure and routing benefits from the agent explicitly choosing the right one — same tool-namespacing logic from Module 2.2, applied to retrieval sources.*
5. **A team wants an agent that both browses the web and answers questions over an internal document set. Would you build this purely in LlamaIndex, or combine frameworks — and why?**
   *Look for: a reasonable answer either way as long as it's justified — e.g., LlamaIndex's retrieval depth for the document side, possibly combined with a more general orchestration layer for broader action-taking; there's no single "correct" framework, only fit-for-purpose reasoning.*

---

## 3.7 Other Notable Frameworks 🆕

### Semantic Kernel (Microsoft)
Semantic Kernel was Microsoft's earlier SDK for combining LLM "skills"/plugins with planners that could sequence them — strong on enterprise integration and a multi-language story (.NET, Python, Java). **As of 2026, Semantic Kernel and AutoGen are being consolidated**: Microsoft Agent Framework (MAF) is explicitly positioned as the **unified successor to both**, combining AutoGen's multi-agent orchestration patterns with Semantic Kernel's enterprise stability and plugin model, under one multi-language (.NET + Python) programming model. Teams currently on Semantic Kernel are being actively guided toward migrating to MAF rather than continuing net-new development on Semantic Kernel alone.

### Vercel AI SDK agent primitives
A TypeScript-first SDK, popular in the Next.js ecosystem, offering lightweight primitives for streaming model output and tool calling (functions exposed to the model with typed parameters, similar in spirit to Module 2's schema design) plus simple loop-control primitives for letting an agent take multiple steps. Its appeal is being **lighter-weight** than graph-based frameworks like LangGraph — well-suited to teams already building in TypeScript/Next.js who want agent behavior without adopting a separate, heavier orchestration framework, at the cost of less built-in support for complex branching/checkpointing/HITL patterns than LangGraph or MAF provide natively.

### Google Agent Development Kit (ADK)
An open-source, **code-first** framework, available across Python, Go, Java, and TypeScript/Kotlin, designed to make agent development feel close to ordinary software development. Notable characteristics:
- Supports both **workflow agents** (deterministic, graph-based orchestration with explicit execution paths) and **agent-coordinated dynamic routing** (more adaptive, LLM-driven routing) — covering both ends of the agency spectrum from Module 1.1 within one framework.
- Native **multi-agent** support via parent-child agent composition, plus support for the **A2A protocol** for cross-agent/cross-system communication.
- A built-in **tool confirmation flow** — a structural mechanism for human-in-the-loop approval, conceptually parallel to LangGraph's interrupts.
- Optimized for Gemini/Vertex AI but explicitly designed to be **model-agnostic** and deployable outside Google Cloud.

### 📋 Interview Questions — 3.7
1. **Why is it now slightly misleading to describe Semantic Kernel and AutoGen as two separate, competing Microsoft frameworks?**
   *Look for: both are being consolidated into Microsoft Agent Framework as their unified successor — treating them as still-separate, actively-developed competitors reflects stale information.*
2. **When would the Vercel AI SDK be a better fit than LangGraph for a given team, independent of raw capability?**
   *Look for: a TypeScript/Next.js-native team wanting lightweight agent behavior without adopting a heavier, separate orchestration framework — fit-for-stack reasoning, not "which is more powerful."*
3. **What does it mean that Google ADK supports both "workflow agents" and "agent-coordinated dynamic routing" in one framework?**
   *Look for: it spans both ends of the agency spectrum (Module 1.1) — deterministic graph-based execution and LLM-driven dynamic routing — rather than forcing a choice between frameworks for different agency levels.*
4. **How does ADK's "tool confirmation flow" relate to concepts you've already learned in Module 1.6 and Module 3.2?**
   *Look for: it's ADK's structural implementation of the human-in-the-loop approval checkpoint pattern — same concept as LangGraph's interrupts, different framework's terminology.*
5. **A team building exclusively on Google Cloud/Gemini asks if ADK locks them into that ecosystem. How would you answer?**
   *Look for: ADK is explicitly designed to be model-agnostic and deployable outside Google Cloud, even though it's optimized for and most naturally paired with Gemini/Vertex AI — partial lock-in risk, not absolute.*

---

## 3.8 Framework Selection 🆕

### Build-vs-buy: raw API + custom loop vs framework
This is fundamentally a tradeoff between **control/transparency** and **development speed/ecosystem leverage**:

| | Raw API + custom loop | Framework |
|---|---|---|
| Control | Full — you see and control every prompt, every retry, every state transition | Partial — behavior is mediated by the framework's abstractions |
| Speed to first working version | Slower — you build the loop, retries, and state management yourself | Faster — these are provided |
| Debuggability | High — nothing hidden | Varies — some frameworks (e.g., the old AgentExecutor) historically hid reasoning in opaque internals; current frameworks have generally improved this with explicit state/tracing |
| Dependency/version risk | None | Real — frameworks in this space have shown significant API churn (see 3.1, 3.5) |
| Ecosystem leverage | None — you build every integration | High — prebuilt integrations, observability, community patterns |

A useful heuristic: **start with the raw API for a single, well-understood agent loop to deeply understand the mechanics (Module 1–2 concepts) before adopting a framework** — it makes framework abstractions legible rather than magical, and makes it much easier to debug when (not if) the framework's abstraction leaks.

### Comparison criteria: control, debuggability, lock-in, community size
When evaluating a framework for a specific project, weigh:
- **Control** — how much of the execution path can you inspect/override vs. how much is opaque framework internals?
- **Debuggability** — does the framework give you structured tracing/observability (state objects, checkpoints, step-by-step replay), or do you have to reverse-engineer behavior from logs?
- **Lock-in** — is the framework single-provider (tightly coupled to one model vendor or cloud) or provider-agnostic? How costly would migrating off it be later?
- **Community size & momentum** — a large community means more examples, faster bug fixes, and more third-party integrations, but also means more outdated tutorials in search results (a real cost, as seen repeatedly in this module) — verify against current official docs rather than the most popular search result.
- **Production-readiness out of the box** — does it provide durable execution, checkpointing, and human-in-the-loop natively (LangGraph, MAF, ADK, CrewAI Flows), or are those your responsibility to build (a thinner SDK like the Vercel AI SDK)?

**Summary comparison across this module's frameworks:**

| Framework | Primary language(s) | Orchestration model | Best fit |
|---|---|---|---|
| LangChain (`create_agent`) | Python, TS | High-level entry point over LangGraph | Linear pipelines + simple agents, multi-provider integrations |
| LangGraph | Python, TS | Explicit stateful graph (cycles, checkpoints) | Complex control flow, durable execution, HITL |
| OpenAI Responses API / Agents SDK | Any (HTTP API) | Server-managed agentic loop | OpenAI-model-centric apps wanting built-in hosted tools |
| CrewAI | Python (official); JS unofficial only | Role-based Crews + deterministic Flows | Role-based multi-agent collaboration, with a deterministic production mode |
| Microsoft Agent Framework | .NET, Python | Graph-based Workflows (successor to AutoGen + Semantic Kernel) | Enterprise multi-agent systems on Microsoft stack, multi-provider |
| LlamaIndex (Workflows/AgentWorkflow) | Python | Event-driven Workflows | Data/retrieval-heavy agents over large private corpora |
| Vercel AI SDK | TypeScript | Lightweight tool-calling + loop primitives | TS/Next.js-native teams wanting minimal-overhead agents |
| Google ADK | Python, Go, Java, TS/Kotlin | Workflow agents + dynamic routing, multi-agent | Cross-language enterprise builds, especially Gemini-centric |

### 📋 Interview Questions — 3.8
1. **Why would you recommend a team build their first agent with raw API calls before adopting a framework, even if they know they'll eventually want one?**
   *Look for: understanding the underlying loop/tool-call mechanics makes framework abstractions legible rather than magical, and makes debugging framework behavior much easier later.*
2. **What's the difference between "lock-in" and "dependency risk," and can a framework have one without the other?**
   *Look for: lock-in = tied to a specific vendor/provider (e.g., a single-model-vendor SDK); dependency risk = exposure to the framework's own API churn, independent of vendor — e.g., LangGraph is provider-agnostic (low lock-in) but has had real API evolution (real dependency risk).*
3. **A framework has a huge GitHub star count and tons of tutorials. Why might that be a double-edged sword rather than a pure positive signal?**
   *Look for: popularity also means a large volume of outdated tutorials/training-data patterns circulating, as seen concretely with LangChain's AgentExecutor and OpenAI's Assistants API in this module — popularity doesn't guarantee currency.*
4. **How would you evaluate "production-readiness out of the box" when comparing two frameworks for a project that needs human approval gates?**
   *Look for: checking whether checkpointing/durable state and HITL interrupts are native, first-class features (LangGraph, MAF, ADK, CrewAI Flows) vs. something you'd have to hand-build on a thinner SDK.*
5. **Walk through how you'd actually decide between two reasonable framework options for a new project — what's your process, not just your final pick?**
   *Look for: a structured process — map task type to agency-spectrum position (Module 1.1), weigh the control/debuggability/lock-in/community criteria against project constraints (team's existing language stack, need for durable execution, vendor neutrality requirements), rather than picking based on hype or familiarity alone.*

---

*End of Module 3 detailed notes — 40 interview questions total across 8 sections. Given how quickly this space moves, treat the framework-status notes (3.1, 3.3, 3.5, 3.7) as a mid-2026 snapshot and re-verify against current docs before making architecture decisions.*
