# Module 2 — Tool Use & Function Calling

## Learning Objective

Teach agents how to:
- Interact with external systems
- Select the correct tools
- Pass structured inputs
- Handle responses
- Recover from failures
- Safely execute actions

> This module fills the gap between the foundational agent concepts of Module 1 and advanced topics like memory systems and multi-agent coordination. **Function calling is the substrate** almost everything else in agent design sits on top of — memory retrieval, browser automation, MCP, and multi-agent communication are all, under the hood, tool calls.

***

## Table of Contents

- [2.1 Function Calling Fundamentals](#21-function-calling-fundamentals)
  - [What is function calling?](#what-is-function-calling)
  - [Why function calling exists](#why-function-calling-exists)
  - [Anatomy of a function call](#anatomy-of-a-function-call)
  - [Function calling vs ReAct](#function-calling-vs-react)
  - [Real-world examples](#real-world-examples)
  - [What is a tool schema?](#what-is-a-tool-schema)
  - [Anatomy of a tool schema](#anatomy-of-a-tool-schema)
  - [JSON schema design for tool parameters](#json-schema-design-for-tool-parameters)
  - [Typed inputs/outputs and why strict typing reduces hallucinated calls](#typed-inputsoutputs-and-why-strict-typing-reduces-hallucinated-calls)
  - [Multi-tool execution](#multi-tool-execution)
  - [Parallel tool execution](#parallel-tool-execution)
  - [Single-call vs multi-call (parallel) tool invocation](#single-call-vs-multi-call-parallel-tool-invocation)
  - [📋 Interview Questions — 2.1](#-interview-questions--21)

- [2.2 Tool Selection & Routing](#22-tool-selection--routing)
  - [How agents read tool schemas](#how-agents-read-tool-schemas)
  - [Tool selection logic](#tool-selection-logic)
  - [How agents choose among many available tools](#how-agents-choose-among-many-available-tools)
  - [Tool namespacing and grouping at scale (10s–100s of tools)](#tool-namespacing-and-grouping-at-scale-10s100s-of-tools)
  - [Dynamic tool loading (only expose relevant tools per task)](#dynamic-tool-loading-only-expose-relevant-tools-per-task)
  - [📋 Interview Questions — 2.2](#-interview-questions--22)

- [2.3 Structured Output & Validation](#23-structured-output--validation)
  - [Tool lifecycle overview](#tool-lifecycle-overview)
  - [Tool execution lifecycle](#tool-execution-lifecycle)
  - [Tool input validation](#tool-input-validation)
  - [Schema validation libraries (Pydantic, Zod, instructor)](#schema-validation-libraries-pydantic-zod-instructor)
  - [Tool responses & observations](#tool-responses--observations)
  - [Error handling & recovery](#error-handling--recovery)
  - [Repair strategies for malformed tool calls](#repair-strategies-for-malformed-tool-calls)
  - [Retry-with-feedback loops on validation failure](#retry-with-feedback-loops-on-validation-failure)
  - [📋 Interview Questions — 2.3](#-interview-questions--23)

- [2.4 Tool Descriptions as Prompts](#24-tool-descriptions-as-prompts)
  - [Good vs bad tool descriptions](#good-vs-bad-tool-descriptions)
  - [Schema design failure modes](#schema-design-failure-modes)
  - [Description quality's impact on tool selection accuracy](#description-qualitys-impact-on-tool-selection-accuracy)
  - [Example-driven descriptions (few-shot inside the schema)](#example-driven-descriptions-few-shot-inside-the-schema)
  - [Common anti-patterns](#common-anti-patterns)
  - [Principles of good tool design](#principles-of-good-tool-design)
  - [Designing agent-friendly tools](#designing-agent-friendly-tools)
  - [Input design best practices](#input-design-best-practices)
  - [Output design best practices](#output-design-best-practices)
  - [Tool observability](#tool-observability)
  - [Tool security & permissions](#tool-security--permissions)
  - [Production tool architecture](#production-tool-architecture)
  - [📋 Interview Questions — 2.4](#-interview-questions--24)

***

## 2.1 Function Calling Fundamentals

### What is function calling?

**Purpose:** Introduce function calling as the foundation of tool use.

**Concepts:**
- LLMs calling tools
- Structured inputs
- Structured outputs
- External actions

**Function calling flow:**

```text
User
  ↓
LLM
  ↓
Function Call
  ↓
Tool
  ↓
Result
  ↓
Answer
```

Function calling is the mechanism that allows an LLM to request execution of external code (a "tool") with structured arguments, receive the result, and use that result to inform its next decision.

### Why function calling exists

**Purpose:** Show limitations of pure LLMs.

**Comparison:**

| Without function calling | With function calling |
|--------------------------|----------------------|
| `LLM → Guess` | `LLM → Tool → Real Data` |

**Examples:** weather APIs, stock prices, databases, external APIs

Pure LLMs are limited to their training knowledge and the prompt context. They cannot:
- Access real-time data
- Query databases
- Call external APIs
- Perform actions in the world

Function calling bridges this gap by giving the model a way to request external capabilities.

### Anatomy of a function call

**Purpose:** Break down every component.

**Components:**
```text
Function Name
Parameters
Schema
Execution
Response
```

**Example:**

```json
{
  "name": "get_weather",
  "city": "Mumbai"
}
```

Every function call consists of:
1. **Function name** — which tool to invoke
2. **Parameters** — the structured arguments
3. **Schema** — the contract defining valid parameters
4. **Execution** — the tool runs with those parameters
5. **Response** — the tool's result is returned

### Function calling vs ReAct

**Purpose:** Show the relationship.

| Function Calling | ReAct |
|------------------|-------|
| One Tool → One Result | Reason → Tool → Observe → Reason → ... |
| Single-step tool use | Multi-step reasoning loop with tools |

**Function calling** is the mechanism for invoking a single tool. **ReAct** is a reasoning pattern that uses function calling as part of a larger loop (think → call tool → observe → think again → ...).

Function calling is *how* the model invokes a tool; ReAct is *when and why* it chooses to do so in a multi-step process.

### Real-world examples

**Purpose:** Show practical use cases.

**Examples:**
- Email agent — call `send_email`, `search_emails`, `list_contacts`
- Calendar agent — call `create_event`, `list_events`, `cancel_event`
- CRM agent — call `create_customer`, `update_record`, `search_pipeline`
- Database agent — call `query_db`, `insert_row`, `update_record`
- Search agent — call `web_search`, `get_page`, `summarize_results`

Every agent that interacts with external systems (apis, databases, files) uses function calling under the hood.

### What is a tool schema?

**Purpose:** Introduce schemas.

```text
Tool
  ↓
Description
  ↓
Inputs
  ↓
Validation
```

A **tool schema** is the contract between the model and the tool — it defines:
- The tool's name
- What the tool does (description)
- What parameters it accepts
- What types those parameters must have
- Which parameters are required

The model reads the schema to understand *what* tool to call and *how* to format the arguments.

### Anatomy of a tool schema

**Purpose:** Explain schema structure.

**Components:**
- **Name** — identifier for the tool
- **Description** — what the tool does and when to use it
- **Parameters** — the object defining accepted arguments
- **Types** — data types for each parameter (string, integer, boolean, array, etc.)
- **Required Fields** — which parameters must be provided

### JSON schema design for tool parameters

A tool definition consists of three parts the model actually reads: a **name**, a **description**, and a **parameters schema** (almost always JSON Schema). The schema constrains the *shape* of the model's output — each field's type, whether it is required, and any value constraints.

```json
{
  "name": "create_calendar_event",
  "description": "Creates a calendar event for the current user. Use only after confirming date and time with the user — do not guess.",
  "parameters": {
    "type": "object",
    "properties": {
      "title": { "type": "string", "description": "Short event title, e.g. 'Dentist appointment'" },
      "start_time": { "type": "string", "format": "date-time", "description": "ISO 8601, e.g. 2026-06-25T14:00:00+05:30" },
      "duration_minutes": { "type": "integer", "minimum": 5, "maximum": 480 },
      "attendees": { "type": "array", "items": { "type": "string", "format": "email" } }
    },
    "required": ["title", "start_time", "duration_minutes"]
  }
}
```

**Principles of good schema design:**

| Principle | Why it matters |
|-----------|---------------|
| **Narrow the type as much as the real world allows** | Use `enum` over free-text `string` when valid values are known; use `integer` with `minimum`/`maximum` over unconstrained numbers |
| **Prefer flat structures over deeply nested objects** | Deep nesting is harder for the model to populate correctly |
| **Mark `required` deliberately** | Every optional field is one the model might omit or fabricate a default for; make something optional only when "absent" is a meaningful, distinct case |
| **Field-level descriptions matter** | `"start_time"` alone is ambiguous; `"ISO 8601, e.g. 2026-06-25T14:00:00+05:30"` eliminates formatting errors |

### Typed inputs/outputs and why strict typing reduces hallucinated calls

Loosely typed schemas (everything as `string`) give the model maximum freedom — which is exactly the problem. A free-text `string` field invites inconsistent date formats, fabricated IDs, or made-up values that look plausible but aren't grounded in anything real. Strict typing narrows the space of what the model can emit *before* it ever reaches your code:

- An `enum` field makes an invalid category a **schema violation**, not a silent bad value.
- A `format: "date-time"` field catches malformed dates at **validation time**.
- Many tool-calling APIs use **constrained/grammar-based decoding** under the hood — the model is literally prevented from generating tokens that would produce invalid JSON or an out-of-enum value, rather than just being asked nicely to follow the schema.

**General principle:** push correctness constraints into the schema wherever possible, rather than relying on the model "reading" instructions in the description and complying. A constraint enforced by the type system cannot be hallucinated past; a constraint only stated in prose sometimes will be.

### Multi-tool execution

**Purpose:** Show chaining.

**Example:**
```text
Search → Database → Email → CRM
```

Agents often need to call multiple tools in sequence, where each tool's output feeds into the next:
1. Search for information
2. Store results in a database
3. Email a summary to a user
4. Update the CRM with the interaction

This is multi-step tool use — each execution depends on the previous result.

### Parallel tool execution

**Purpose:** Performance optimization.

```text
Weather
      ↘
Stocks  → Merge Results
      ↗
News
```

When multiple tools are **independent** (no tool needs another's output), they can be called in parallel:
1. Request weather, stocks, and news simultaneously
2. Wait for all results
3. Merge and process

This reduces latency from N sequential round-trips to 1 parallel round-trip.

### Single-call vs multi-call (parallel) tool invocation

Modern tool-calling APIs let the model request **multiple tool calls in a single turn** when the calls are independent of each other — for example, checking the weather in three cities at once, rather than three separate think→act→observe round trips.

| Aspect | Single-call (sequential) | Multi-call (parallel) |
|--------|-------------------------|----------------------|
| **When to use** | Calls depend on each other's output | Calls are independent |
| **Latency** | One round trip per call | One round trip for all independent calls |
| **Risk** | Lower — order is naturally enforced | Must guard against accidental side-effect collisions |
| **Example** | "Get the user's order ID, then look up that order" | "Get weather for Mumbai, Delhi, and Bangalore" |

**The key judgment call is dependency detection.** Parallelizing calls where one's output is actually needed as another's input produces calls made with stale or placeholder arguments. Also, side-effecting tools (writes, sends, payments) called in parallel must be safe to run concurrently/idempotently — never parallelize two calls that both modify the same resource.

### 📋 Interview Questions — 2.1

**1. Why does making a parameter `enum` instead of free-text `string` reduce hallucination, beyond just "it's stricter"?**  
*Look for: many APIs use constrained decoding that makes an out-of-enum value structurally impossible to generate, not just discouraged — it is a hard guarantee, not a soft hint.*

**2. A tool has 12 parameters, 6 of them nested two levels deep. What problems would you expect, and how would you redesign it?**  
*Look for: deep nesting increases the chance of malformed/incomplete population; flattening or splitting into multiple simpler tools.*

**3. When would parallel tool calls actually make an agent worse, not just slower if avoided?**  
*Look for: when calls have a hidden dependency (one needs the other's output) or when concurrent side-effecting calls can race/conflict on the same resource.*

**4. What's the difference between validating a tool call after generation vs constraining it during generation, and why does that distinction matter?**  
*Look for: post-hoc validation catches errors after the fact (requires a retry loop); constrained decoding prevents many invalid outputs from being generated at all.*

**5. Why might making every parameter optional seem "flexible" but actually hurt reliability?**  
*Look for: every optional field is one the model can omit, infer, or fill with a fabricated default — required fields force the model (or orchestrator) to actually resolve ambiguity rather than guessing.*

***

## 2.2 Tool Selection & Routing

### How agents read tool schemas

**Purpose:** Show internal decision making.

```text
User Request
  ↓
Read Tool Descriptions
  ↓
Choose Best Tool
  ↓
Execute
```

At inference time, tool selection is effectively the model performing **classification over its available tools**, conditioned on:
- The task description
- Each tool's name
- Each tool's description

The model scans all available tool schemas and selects the one that best matches the current request.

### Tool selection logic

**Purpose:** How agents choose tools.

```text
Request
  ↓
Intent Detection
  ↓
Tool Ranking
  ↓
Selection
```

**Process:**
1. Extract intent from the user request
2. Rank all available tools by relevance to that intent
3. Select the top-ranked tool
4. Execute with generated parameters

**Examples:** Search Tool, Database Tool, Email Tool

### How agents choose among many available tools

At inference time, tool selection is effectively the model performing classification over its available tools, conditioned on the task description and each tool's name/description. This means **tool selection accuracy degrades the same way any classification task does**: with more, more similar, or more poorly distinguished classes (tools).

With 5 clearly distinct tools, selection is close to trivial. With 80 tools, several of them overlapping, it becomes a real bottleneck on agent reliability — independent of how good the underlying model is.

### Tool namespacing and grouping at scale (10s–100s of tools)

As the tool count grows, flat lists stop working. Common mitigations:

| Technique | How it works | Benefit |
|-----------|-------------|---------|
| **Namespacing** | Group related tools under a prefix (`calendar.create_event`, `calendar.list_events`, `email.send`, `email.search`) so the model can reason about *category* before *specific tool* | Near-duplicate names across domains don't collide (`search` meaning different things in `email.search` vs `docs.search`) |
| **Two-stage routing** | Lightweight first step picks the *category/domain* (e.g., "this is a calendar task"), then only that category's specific tools are exposed to a second reasoning step | Shrinks the effective choice set at the point where precision matters most |
| **Sub-agent delegation** | At very large scale, route to a specialized sub-agent that only knows about one domain's tools | Keeps agents fluent in manageable tool sets rather than one agent trying to know all of them |

### Dynamic tool loading (only expose relevant tools per task)

Rather than always injecting every available tool's schema into context (expensive in tokens, and it dilutes selection accuracy), retrieve only the tools likely to be relevant to the current task and load just those:

- Embed each tool's name/description once, embed the current task/query, and retrieve the top-K most similar tools — the same retrieval pattern as RAG, applied to a "tool corpus" instead of a document corpus.
- This both **reduces token cost** (fewer schemas in context) and **improves selection accuracy** (fewer, more relevant candidates to choose between).
- Tradeoff: if retrieval misses a tool the task actually needed, the agent simply cannot see it — so retrieval recall matters as much as tool-call precision.

### 📋 Interview Questions — 2.2

**1. Why does adding more tools to an agent sometimes make it *worse* at using the tools it already had, even though nothing about those original tools changed?**  
*Look for: tool selection is a classification problem over the full set; more (especially more similar) options dilute selection accuracy across the board.*

**2. How would you design tool organization for an agent with 150 available tools?**  
*Look for: namespacing/grouping, two-stage routing, possibly dynamic retrieval or sub-agent delegation rather than one flat list of 150.*

**3. What's the main risk introduced by dynamic tool loading (retrieval-based tool selection), and how would you mitigate it?**  
*Look for: recall failure — the right tool isn't retrieved into context at all, so the agent can't even attempt to use it; mitigate with retrieval evals, hybrid keyword+embedding retrieval, or always including a small "core" tool set.*

**4. Two tools, `search_docs` and `lookup_docs`, do almost the same thing. What problem does this cause, and how would you fix it?**  
*Look for: ambiguous boundary causes inconsistent selection; fix by merging them, or sharply differentiating their descriptions/use cases (or namespacing them under different domains if they really are different).*

**5. When would you choose sub-agent delegation over dynamic tool loading to handle a large tool set?**  
*Look for: delegation makes sense when domains require genuinely different reasoning/context, not just a smaller tool list — e.g., a "finance" sub-agent vs a "scheduling" sub-agent with different judgment calls, not just different APIs.*

***

## 2.3 Structured Output & Validation

### Tool lifecycle overview

**Purpose:** Introduce the full lifecycle.

```text
Need
  ↓
Select
  ↓
Execute
  ↓
Observe
  ↓
Continue
```

The tool lifecycle is the complete flow from recognizing a need for external capability to using that capability and continuing:
1. Recognize a need (e.g., "I need weather data")
2. Select the right tool (e.g., `get_weather`)
3. Execute with parameters
4. Observe the result
5. Continue reasoning with that result

### Tool execution lifecycle

**Purpose:** Detailed breakdown.

```text
User Request
  ↓
Reasoning
  ↓
Tool Selection
  ↓
Parameter Generation
  ↓
Execution
  ↓
Result
  ↓
Next Decision
```

**Step-by-step:**
1. **User Request** — task arrives
2. **Reasoning** — model analyzes what's needed
3. **Tool Selection** — chooses which tool to call
4. **Parameter Generation** — constructs arguments matching the schema
5. **Execution** — tool runs with those parameters
6. **Result** — tool returns output
7. **Next Decision** — model uses result to decide what to do next

### Tool input validation

**Purpose:** Prevent bad requests.

**Examples of validation failures:**
- Missing email field
- Negative amount (when only positive allowed)
- Invalid date format

**Validation layer:** Always validate tool inputs before execution — never trust raw model output. Schema validation catches structural errors (wrong types, missing required fields), but you may also need semantic validation (e.g., "is this email address actually valid?").

### Schema validation libraries (Pydantic, Zod, instructor)

Even with a well-designed schema and a capable model, tool calls should be validated programmatically before execution — never trust the raw output:

| Library | Ecosystem | What it adds |
|---------|-----------|--------------|
| **Pydantic** | Python | Typed models + validation errors |
| **Zod** | TypeScript/JS | Same, for the JS ecosystem |
| **instructor** | Python/JS (wraps the above) | Auto-retry loop with validation-error feedback built in |

- **Pydantic** (Python): define the expected shape as a typed model; parsing raw model output into it raises a clear validation error on mismatch.
- **Zod** (TypeScript): the equivalent pattern for the TS/JS ecosystem — schema-as-code with runtime validation.
- **instructor**: a thin layer over LLM calls that pairs the API call with a Pydantic/Zod-style schema and automatically retries the call (feeding back the validation error) until it gets a valid structured response or hits a retry limit.

### Tool responses & observations

**Purpose:** Understand what comes back.

```text
Tool
  ↓
Response
  ↓
Observation
  ↓
Reasoning
```

**Types of responses:**
- **Success** — tool completed and returned expected data
- **Partial Success** — tool completed but with warnings or incomplete data
- **Failure** — tool threw an error (timeout, auth error, invalid input, etc.)

The tool's response becomes the next **observation** in the ReAct loop, which the model uses to inform its next decision.

### Error handling & recovery

**Purpose:** Production-grade tool usage.

**Examples:**
```text
API Timeout   → Retry
Rate Limit    → Wait
Tool Failure  → Fallback
```

**Common patterns:**
- **Timeout** — retry with the same parameters
- **Rate limit** — wait, then retry
- **Tool failure** — try a fallback tool or escalate to human
- **Validation error** — feed error back to model and retry with corrected parameters

### Repair strategies for malformed tool calls

When a tool call fails validation, you have three broad options, roughly in order of preference:

1. **Programmatic coercion** — fix what's safely fixable in code (trim unexpected extra fields, coerce `"5"` to `5`, normalize a date format) without going back to the model. Fast, cheap, but only safe for unambiguous fixes.
2. **Feed the error back to the model and ask it to retry** — the most common and most general approach.
3. **Fall back to a default or fail gracefully** — when retries are exhausted, either use a safe default (if one exists) or surface the failure rather than guessing.

### Retry-with-feedback loops on validation failure

The standard pattern: catch the validation exception, format it as a clear error message, and feed it back into the agent's context as if it were a tool observation (e.g., *"Your last call failed: `start_time` must be ISO 8601 format, you provided `'tomorrow at 2'`. Please retry with a valid value."*) — then let the model attempt the call again. This is far more effective than silently retrying the identical call, because the model gets a chance to actually correct the specific mistake rather than repeating it.

- Always bound this with a **max retry count** — an unbounded retry loop is just a slower version of the infinite-loop failure mode from Module 1.5.
- The quality of the error message matters: "invalid input" gives the model nothing to act on; a specific, field-level error message gives it something to fix.

### 📋 Interview Questions — 2.3

**1. Why is it important to feed the *specific* validation error back to the model rather than just saying "that didn't work, try again"?**  
*Look for: specific, actionable error messages let the model correct the actual mistake; vague feedback often produces the same wrong call repeated, just like an unguided retry loop.*

**2. What's the difference between what Pydantic/Zod do and what `instructor` adds on top?**  
*Look for: Pydantic/Zod validate; instructor wraps the LLM call itself with an automatic validate→retry loop, so you don't hand-build that loop yourself.*

**3. When is it safe to programmatically coerce a malformed tool call instead of asking the model to retry, and when is it not?**  
*Look for: safe for unambiguous, low-risk fixes (type coercion, trimming extras); unsafe when the "fix" requires guessing intent (e.g., silently picking a default date when none was given).*

**4. Why must a retry-with-feedback loop have a hard cap, even though retrying seems strictly safer than not retrying?**  
*Look for: an unbounded retry loop is functionally the same failure mode as an infinite loop — cost, latency, and the possibility the model never converges on a valid call.*

**5. A tool call passes schema validation but is still semantically wrong (e.g., a real, valid email address that's the wrong recipient). How does this connect back to the "tool misuse" failure mode from Module 1.5?**  
*Look for: schema validation only catches structural errors, not semantic ones — this is the "right tool, wrong parameters" pattern, and needs a different kind of check (e.g., confirmation step) than schema validation alone.*

***

## 2.4 Tool Descriptions as Prompts

### Good vs bad tool descriptions

**Purpose:** One of the most important distinctions in tool design.

| Bad description | Good description |
|---------------|-----------------|
| `"Get data"` | `"Retrieve customer records using customer ID."` |

**Key differences:**
- **Ambiguity** vs **precision**
- **Missing context** vs **explicit context**
- **Generic action** vs **specific use case**

A tool's `description` field isn't documentation for a human — it's a prompt the model reads, every time, to decide *whether* to call this tool and *how* to fill in its arguments.

### Schema design failure modes

**Purpose:** Common mistakes to avoid.

**Common failure modes:**
- **Missing parameters** — required fields not declared
- **Ambiguous descriptions** — doesn't say when to use the tool
- **Wrong types** — using `string` instead of `integer`, `date-time`, etc.
- **Poor validation** — no constraints on values (minimum, maximum, enum, format)

Each of these directly increases the chance of malformed tool calls or wrong tool selection.

### Description quality's impact on tool selection accuracy

A tool's `description` field isn't documentation for a human — it's a prompt the model reads, every time, to decide *whether* to call this tool and *how* to fill in its arguments. Treat it with the same care as a system prompt: **ambiguous, sparse, or jargon-heavy descriptions directly cause wrong tool selection and malformed arguments**, even when the underlying schema is technically correct.

**Weak description:**
```json
"name": "search",
"description": "Searches for things."
```

**Strong description:**
```json
"name": "search_internal_docs",
"description": "Searches the company's internal documentation (wikis, design docs, runbooks). Use this for questions about internal processes, past decisions, or how something is implemented internally. Do NOT use this for questions about public information — use web_search for that instead."
```

The strong version does three things the weak one doesn't:
1. States **what** is being searched
2. States **when** to use it
3. Explicitly disambiguates it from a similarly-named tool

### Example-driven descriptions (few-shot inside the schema)

Just as a few examples in a prompt improve task performance, a short example call (or example input/output pair) embedded in a tool's description improves argument-filling accuracy — especially for tools with non-obvious formatting requirements:

```json
"description": "Schedules a meeting. Example: to schedule a 30-minute sync tomorrow at 2pm IST, call with start_time='2026-06-22T14:00:00+05:30', duration_minutes=30."
```

This is most valuable for fields where the **correct format** isn't obvious from the type alone (dates, IDs with a specific structure, units that could be ambiguous).

### Common anti-patterns

| Anti-pattern | Problem | Fix |
|--------------|---------|-----|
| **Vague names** (`do_thing`, `handle_request`, `process`) | Model has no signal about when the tool applies | Names should describe the action and object as specifically as possible (`create_calendar_event`, not `calendar_action`) |
| **Overlapping tools with no clear boundary** | Two tools that both plausibly handle the same request force the model to guess | Merge them, or sharply differentiate their descriptions/use cases |
| **Missing constraints** | Not specifying expected date format, valid ID pattern, or units leaves the model to infer inconsistently | Add `format`, `enum`, `minimum`, `maximum`, etc. |
| **Tool sprawl without grouping** | Many individually fine tools, with no namespacing or organization, recreate the selection-accuracy problem | Use namespacing, two-stage routing, or sub-agent delegation |
| **Descriptions written for a human reader, not the model** | Marketing-style language ("Our powerful search tool!") wastes tokens and signal | Focus on disambiguating use cases, not praise |

### Principles of good tool design

**Purpose:** Foundational best practices.

**Principles:**
- **Simple** — narrow scope, clear intent, few parameters
- **Explicit** — descriptions say exactly when to use and what format arguments need
- **Reliable** — schema constrains as much as possible; validation catches errors before execution
- **Observable** — every call is logged, with inputs, outputs, and timing

### Designing agent-friendly tools

**Purpose:** How agents think about tools.

| Human Tool | Agent Tool |
|------------|------------|
| Generic, flexible, guessable | Specific, constrained, unambiguous |
| Written for a human to read | Written for a model to parse |
| "Figure it out" attitude | "Every case specified" attitude |

**Agent-friendly tools:**
- Have narrow, specific scopes
- Use strict typing and constraints
- Provide explicit, unambiguous descriptions
- Include examples for non-obvious formats
- Return structured, self-explanatory outputs

### Input design best practices

**Purpose:** Create predictable inputs.

**Topics:**
- **Required fields** — mark fields required when absence is not meaningful
- **Defaults** — avoid implicit defaults; if a default exists, declare it
- **Validation** — validate at the schema level before the model generates
- **Constraints** — use `minimum`, `maximum`, `enum`, `format` wherever possible

### Output design best practices

**Purpose:** Make responses easy to reason about.

| Bad output | Good output |
|------------|-------------|
| `{"status": "ok"}` | `{"status": "success", "emailSent": true, "messageId": "123"}` |

**Good outputs:**
- Use structured, typed fields
- Include specific identifiers (messageId, recordId)
- Indicate success/failure explicitly
- Avoid ambiguous status strings

### Tool observability

**Purpose:** Monitor everything.

```text
Request
  ↓
Tool
  ↓
Logs
  ↓
Metrics
  ↓
Tracing
```

**What to observe:**
- **Request** — what triggered the tool call
- **Tool** — which tool was called, with what parameters
- **Logs** — detailed execution logs
- **Metrics** — latency, success rate, error rate per tool
- **Tracing** — end-to-end trace of the agent's trajectory

### Tool security & permissions

**Purpose:** Safe execution.

**Concepts:**
- **RBAC** (role-based access control) — who can call which tools
- **Permissions** — what each tool can do (read-only, write, admin)
- **Allow Lists** — whitelist of tools the agent is allowed to call
- **Credential Vaults** — secure storage for API keys, tokens, passwords

### Production tool architecture

**Purpose:** Complete production design.

```text
Agent
  ↓
Tool Registry
  ↓
Validation
  ↓
Execution Layer
  ↓
Monitoring
  ↓
Audit Logs
```

**Production architecture:**
1. **Agent** — initiates tool calls
2. **Tool Registry** — maps tool names to implementations
3. **Validation** — schema validation + semantic checks
4. **Execution Layer** — runs the tool with error handling
5. **Monitoring** — metrics, tracing, alerting
6. **Audit Logs** — persistent record of all tool calls

### 📋 Interview Questions — 2.4

**1. Why should a tool description be thought of as a prompt rather than documentation?**  
*Look for: it is read by the model at every decision point and directly shapes behavior, unlike documentation which a human reads once, out of band.*

**2. You inherit an agent with a tool called `search`. What questions would you ask before trusting its description is sufficient?**  
*Look for: what does it search? When should/shouldn't it be used? Does it overlap with other tools? Are argument formats specified?*

**3. When does adding an example call to a tool description help most, and when is it unnecessary overhead?**  
*Look for: most valuable when the correct argument format isn't obvious from the type/field name alone; unnecessary for simple, self-evident parameters.*

**4. Two tools overlap in what they can technically do. What are your two main options to resolve the ambiguity, and what's the tradeoff between them?**  
*Look for: merge them into one tool (simpler, less flexible) vs sharply differentiate their descriptions/scope (keeps both, requires more careful prompt-engineering and ongoing maintenance).*

**5. How would you test whether a tool description is "good enough" before shipping an agent that uses it?**  
*Look for: empirical testing — running varied task phrasings against the tool set and measuring selection accuracy/argument correctness, not just eyeballing the description.*
