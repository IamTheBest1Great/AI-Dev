# 📚 Table of Contents

* [6. Layer 4 — LLM Application Fundamentals](#6-layer-4--llm-application-fundamentals)
  * [6.1 Provider Integration](#61-provider-integration)
    * [Provider Landscape (OpenAI / Anthropic / Google / Others)](#provider-landscape-openai--anthropic--google--others)
    * [API Key / Project Management](#api-key--project-management)
    * [Usage Limits](#usage-limits)
    * [Rate Limits](#rate-limits)
    * [Error Handling](#error-handling)
    * [Provider-Specific Capabilities](#provider-specific-capabilities)
  * [6.2 Prompt Engineering](#62-prompt-engineering)
    * [Zero-Shot / One-Shot / Few-Shot](#zero-shot--one-shot--few-shot)
    * [Instruction Design & Role Definition](#instruction-design--role-definition)
    * [Delimiters](#delimiters)
    * [Output Constraints](#output-constraints)
    * [Prompt Decomposition vs Task Decomposition](#prompt-decomposition-vs-task-decomposition)
    * [Prompt Versioning & Testing](#prompt-versioning--testing)
  * [6.3 Structured Outputs](#63-structured-outputs)
    * [JSON Schema & Pydantic Validation](#json-schema--pydantic-validation)
    * [Structured Output APIs](#structured-output-apis)
    * [Function / Tool Schemas](#function--tool-schemas)
    * [Repair Loops & Retry Strategies](#repair-loops--retry-strategies)
    * [Schema Evolution](#schema-evolution)
  * [6.4 Streaming](#64-streaming)
    * [SSE vs WebSockets](#sse-vs-websockets)
    * [Streaming Text & Structured Events](#streaming-text--structured-events)
    * [Backpressure](#backpressure)
    * [Disconnect & Reconnect Handling](#disconnect--reconnect-handling)
  * [6.5 Cost Optimization](#65-cost-optimization)
    * [Token Budgeting & Output Limits](#token-budgeting--output-limits)
    * [Prompt Caching](#prompt-caching)
    * [Response Caching vs Semantic Caching](#response-caching-vs-semantic-caching)
    * [Batch Processing](#batch-processing)
    * [Model Routing](#model-routing)
    * [Context Reduction & Request Deduplication](#context-reduction--request-deduplication)
* [💡 Key Insights](#-key-insights)
* [⚠️ Common Mistakes](#️-common-mistakes)
* [🔍 Common Confusions](#-common-confusions)
* [🛠️ Practical Applications](#️-practical-applications)
* [📌 Important Terms](#-important-terms)
* [⚡ Quick Revision](#-quick-revision)
* [🎯 Interview Preparation](#-interview-preparation)
  * [Level 1 — Fundamentals](#level-1--fundamentals)
  * [Level 2 — Conceptual Understanding](#level-2--conceptual-understanding)
  * [Level 3 — Practical / Engineering](#level-3--practical--engineering)
  * [Level 4 — Advanced / Deep Understanding](#level-4--advanced--deep-understanding)
  * [Level 5 — Scenario-Based Questions](#level-5--scenario-based-questions)
  * [Common Confusion Questions](#common-confusion-questions)
  * [⚠️ Deep / Trick Questions](#️-deep--trick-questions)
* [⭐ Top Questions You MUST Know](#-top-questions-you-must-know)
* [🎯 Interview Readiness Checklist](#-interview-readiness-checklist)
* [🧠 What You Should Be Able to Explain](#-what-you-should-be-able-to-explain)

---

# 6. Layer 4 — LLM Application Fundamentals

## 6.1 Provider Integration

> **One-line understanding:** Provider integration is about connecting your app to LLM APIs (OpenAI, Anthropic, Google, etc.) through a common abstraction, so limits, errors, and provider differences are handled without locking your app to one vendor.

### 📌 Quick Info

| Field | Answer |
|---|---|
| **What?** | The practices/infrastructure for calling LLM provider APIs reliably |
| **Why?** | Providers differ in API shape, pricing, limits, failure modes |
| **How?** | Common internal interface + per-provider adapters |
| **When?** | Any production app calling one or more LLM providers |
| **When NOT?** | N/A — always needed once you call an external LLM API |
| **Benefits** | Portability, resilience, easier multi-model strategy |
| **Trade-offs** | Abstraction layer adds engineering overhead upfront |

### Provider Landscape (OpenAI / Anthropic / Google / Others)

/notes

**🧠 Simple Explanation:** Different companies sell access to their models through different APIs — same idea (send messages, get text back), different syntax and extra features.

**🔬 Technical Explanation**

| Provider | API Style | Notable Strengths |
|---|---|---|
| **OpenAI** | Chat-completions / Responses API | Broad model lineup, tool calling, structured outputs |
| **Anthropic** | Messages API | Strong tool use, long context, separate system prompt |
| **Google** | Gemini API | Native multimodality, very large context windows |
| **Others** | Azure OpenAI, AWS Bedrock, Vertex AI, open-weight hosts | Enterprise wrapping (VPC, compliance), self-hosted models |

⭐ **Key Point:** Don't memorize exact request/response JSON shapes — they change. Learn the shared concepts: messages, roles, tool calls, streaming, limits.

/handwritten

```text
Core idea: Multiple vendors, same core concepts, different syntax
Why: Avoid lock-in, handle outages/limits gracefully
How: Build one internal interface, adapt per provider
Remember: OpenAI=Chat/Responses, Anthropic=Messages, Google=Gemini
Interview point: "learn abstraction, not syntax"
```

### API Key / Project Management

> **One-line understanding:** API keys authenticate requests; projects/workspaces scope billing and permissions across teams and environments.

⚠️ **Important:** Never place API keys in frontend code — always proxy calls through a backend.

* Use separate keys per environment (dev/staging/prod) for clean usage attribution and incident isolation.

### Usage Limits

> **One-line understanding:** Usage limits cap **total** consumption (spend/tokens/requests) over a billing period.

🎯 **Interview Tip:** Be ready to distinguish this immediately from rate limits (see next) — it's a very common confusion question.

### Rate Limits

### 📌 Quick Info

| Field | Answer |
|---|---|
| **What?** | Caps on how *fast* requests/tokens can be sent (e.g., per minute) |
| **Why?** | Protects provider infra, ensures fair access |
| **How?** | Provider returns `429` when exceeded |
| **When?** | Always active on every API account |
| **Example** | 500 requests/min, 200K tokens/min |
| **Trade-offs** | Requires backoff/retry logic; can throttle bursty apps |

🛠️ **Practical Use:** Implement exponential backoff **with jitter** on `429`; track both request-count and token-count limits separately — high-token requests can hit token limits before request-count limits.

### Error Handling

> **One-line understanding:** Not every failed API call should be retried — only transient errors should be.

| Error Type | Cause | Strategy |
|---|---|---|
| `429` Rate limited | Too fast | Exponential backoff + retry |
| `500`/`503` Server error | Provider-side issue | Retry with backoff; circuit breaker if persistent |
| `400` Bad request | Malformed payload | Fix client-side; **do not retry blindly** |
| Timeout | Slow gen / network | Retry with longer timeout or fallback model |
| Content filtered | Safety trigger | Handle in UX; don't retry identically |

⚠️ **Common Mistake:** Retrying a `400` endlessly — it will fail identically every time since it reflects a real client-side bug.

### Provider-Specific Capabilities

> **One-line understanding:** Some features (native prompt caching, extended reasoning tokens, batch APIs) only exist on certain providers, so a multi-provider app needs conditional handling.

🛠️ **Practical Use:** Build an internal interface like `generate(messages, tools, schema)` with provider adapters underneath — isolates the rest of the app from provider churn.

---

## 6.2 Prompt Engineering

> **One-line understanding:** Prompt engineering is designing the input given to an LLM to reliably get the output you want — foundational, but not where most production quality comes from at scale.

⚠️ **Important — De-emphasis Note:** Beyond core principles, further prompt-wording tweaks have diminishing returns. Production quality increasingly depends on **context quality, tool design, retrieval, and evaluation** — not prompt tricks.

### Zero-Shot / One-Shot / Few-Shot

### 📌 Quick Info

| Field | Answer |
|---|---|
| **What?** | Number of examples given in the prompt before the real task |
| **Why?** | More examples → more consistent format/behavior |
| **When?** | Zero-shot for simple/well-known tasks; few-shot for format-sensitive/edge-case-heavy tasks |
| **When NOT?** | Few-shot when tokens/cost are tight and the task is simple |
| **Trade-offs** | More examples = more tokens = more cost/latency |

**🧠 Analogy**
**Think of it like:** Showing someone a few solved practice problems before a new one — they infer the pattern instead of being told the rule explicitly.
**In the actual concept:** Few-shot examples in the prompt let the model infer format/pattern from demonstrations, not instructions.

### Example

**Scenario:** Classify ticket urgency using few-shot prompting.

**Input:**
```text
Ticket: "App crashes on login, all users affected." → high
Ticket: "Minor typo on settings page." → low
Ticket: "Checkout fails intermittently for some users." → 
```

**Process:**
1. Model sees the pattern from two labeled examples.
2. Infers severity-language-to-label mapping.
3. Applies it to the new ticket.

**Result:** `medium`

/handwritten

```text
Core idea: zero=no examples, one=1 example, few=multiple examples
Why: examples teach format/pattern faster than instructions alone
When: few-shot when format/edge cases matter
Remember: more examples = more tokens = more cost
```

### Instruction Design & Role Definition

> **One-line understanding:** Clear, explicit instructions plus a persistent system role produce more consistent model behavior than vague prompts.

* State the task explicitly; specify output format directly.
* Put critical instructions near the beginning/end (models weight context edges more reliably).
* Use a system message to set tone/persona/constraints for the whole conversation.

```text
System: "You are a precise technical documentation assistant.
Always answer in bullet points. Never speculate beyond provided context."
```

### Delimiters

> **One-line understanding:** Delimiters (tags, quotes, fences) separate instructions from data so the model doesn't confuse the two.

```text
Summarize the text between the tags.
<document>{{user_provided_text}}</document>
```

⚠️ **Common Mistake:** Pasting untrusted user content directly into an instruction without delimiters — raises **prompt injection** risk (embedded text interpreted as new instructions).

### Output Constraints

> **One-line understanding:** Explicit rules that narrow the shape of the response (length, format, allowed values).

```text
"Respond with exactly one word: 'positive', 'negative', or 'neutral'. No explanation."
```

### Prompt Decomposition vs Task Decomposition

### 🔍 Common Confusions

| Concept A | Concept B | Key Difference |
|---|---|---|
| Prompt decomposition | Task decomposition | Prompt decomposition organizes a **single prompt's** structure (role+context+task+format); task decomposition splits a **task** into multiple LLM calls/steps |

```text
Large task: "Read this contract and produce a risk report."
      ↓ decompose
Step 1: Extract key clauses
Step 2: Classify each clause by risk category
Step 3: Summarize findings into a report
```

💡 **Key Insight:** Smaller, focused sub-tasks are individually more reliable and independently validatable — a single large multi-part prompt has no intermediate checkpoint to catch a partial failure.

### Prompt Versioning & Testing

> **One-line understanding:** Treat prompts like code — version them, and test changes against a regression set before deploying.

🛠️ **Practical Use:** Store prompts in source control; maintain a regression test set of representative inputs with expected criteria; run it automatically on every prompt change; roll back on regression.

---

## 6.3 Structured Outputs

> **One-line understanding:** Structured outputs get an LLM to produce machine-parseable output (usually JSON) reliably enough for code to consume it directly.

### 📌 Quick Info

| Field | Answer |
|---|---|
| **What?** | Techniques to constrain/validate model output into a defined schema |
| **Why?** | Free-form text requires fragile parsing; apps need reliable structure |
| **How?** | Schema definition + validation + (optionally) constrained decoding |
| **When?** | Any time output feeds directly into code (DB writes, API calls, UI) |
| **Trade-offs** | Guarantees shape, not factual correctness |

### JSON Schema & Pydantic Validation

**🔬 Technical Explanation**

```json
{
  "type": "object",
  "properties": {
    "sentiment": { "type": "string", "enum": ["positive", "negative", "neutral"] },
    "confidence": { "type": "number" }
  },
  "required": ["sentiment", "confidence"]
}
```

* **JSON Schema** formally specifies expected shape/types/required fields.
* **Pydantic** (Python) defines the same structure as classes and validates/parses the response — turns "hope it's valid JSON" into "get a typed object or an explicit error."

### Structured Output APIs

> **One-line understanding:** Provider-native features that constrain generation itself so output structurally matches a schema, rather than just hoping the prompt worked.

**🔬 Technical Explanation:** Implemented via **constrained decoding** — restricting which tokens can be sampled at each step so only schema-valid tokens are possible.

⚠️ **Important:** Structural validity ≠ content correctness. The model can produce a perfectly-formed JSON object with hallucinated/wrong values.

### Function / Tool Schemas

> **One-line understanding:** Schemas describing available functions the model can "call" by producing structured arguments — the mechanism underlying agentic tool use.

```text
Tool: get_weather(location: string, unit: "celsius" | "fahrenheit")
```

### Repair Loops & Retry Strategies

/notes

**What?** If output fails validation, send the error back to the model and ask it to fix the output instead of failing outright.

```text
Model output → Validate → ❌ Fails →
  "Your JSON was invalid: missing 'confidence' field. Please fix." →
  Model retries → Validate → ✅ Passes
```

| Retry Strategy | When to Use |
|---|---|
| Immediate retry (same prompt) | Transient/random failures |
| Retry with repair instructions | Structural validation failures |
| Retry with lower temperature | Reduce randomness-driven errors |
| Fallback to a different model | Persistent failures with primary model |
| Give up after N attempts | Avoid infinite loops/cost blowup |

/handwritten

```text
Core idea: validate → if fail, tell model the error → retry (capped)
Why: structured-output guarantees aren't perfect
Remember: always cap retries to avoid cost blowup
```

### Schema Evolution

> **One-line understanding:** Managing schema changes over time (adding optional fields, deprecating old ones, versioning) without breaking existing consumers — same discipline as traditional API versioning.

---

## 6.4 Streaming

> **One-line understanding:** Streaming delivers model output incrementally as it's generated, dramatically improving *perceived* latency (not actual generation time).

### 📌 Quick Info

| Field | Answer |
|---|---|
| **What?** | Incremental delivery of generated output |
| **Why?** | Users see first tokens almost immediately instead of waiting for the full response |
| **How?** | SSE (most common) or WebSockets |
| **When?** | User-facing chat/generation UIs |
| **When NOT?** | Background/batch jobs with no live viewer |

### SSE vs WebSockets

### 🔍 Common Confusions

| SSE | WebSockets |
|---|---|
| One-directional (server → client) | Full-duplex (bidirectional) |
| Simpler, works over plain HTTP, native browser `EventSource` | Needed only if client must send data mid-stream too |
| Most common for LLM text streaming | Used for interactive voice, live collaborative editing |

### Streaming Text & Structured Events

```text
Client connects → Server streams: "The" → " quick" → " brown" → " fox" → [done]
```

* **Streaming structured events**: typed events like `{"type": "tool_call_start"}`, `{"type": "text_delta"}`, `{"type": "message_stop"}` — common in agentic/tool-use APIs.
* **Progress events**: explicit stage signals ("retrieving documents," "running tool") for long-running agent UX.

### Backpressure

> **One-line understanding:** Backpressure is what happens when a client/network can't consume streamed data as fast as the server produces it — needs buffering/flow-control to avoid unbounded memory growth.

### Disconnect & Reconnect Handling

🛠️ **Practical Use:** Detect disconnects server-side and **cancel the in-flight generation** immediately — avoids paying for tokens nobody will see.

* **Reconnect strategy:** simple = restart request; advanced = track a stream/response ID and resume from last chunk (needs provider/infra support).

---

## 6.5 Cost Optimization

> **One-line understanding:** Cost optimization combines caching, routing, and context reduction to cut LLM spend at scale without meaningfully hurting quality.

### Token Budgeting & Output Limits

🛠️ **Practical Use:** Track token usage per prompt component (system prompt, history, retrieved context); set hard caps (e.g., "retrieved context ≤ 2,000 tokens"); cap `max_tokens` on generation to prevent runaway-length responses.

### Prompt Caching

### 📌 Quick Info

| Field | Answer |
|---|---|
| **What?** | Provider caches the processing of a repeated prompt **prefix** |
| **Why?** | Reprocessing a large static context on every request is wasteful |
| **How?** | Requires a **stable, identical prefix** across requests |
| **When?** | Long static system prompt / reference doc reused across many queries |
| **When NOT?** | Fully dynamic prompts (e.g., unique RAG context every request) |

💡 **Key Insight:** Structure prompts with static content **first**, dynamic content (user query) **last**, to maximize cache reuse.

### Response Caching vs Semantic Caching

| Response Caching | Semantic Caching |
|---|---|
| Matches **exact** request text | Matches by **meaning/embedding similarity** |
| Zero mismatch risk | Risk of serving wrong answer if threshold too loose |
| Lower hit rate | Higher hit rate |
| Use when phrasing is consistent | Use when users phrase things many ways |

```text
Cached: "What's your refund policy?"
New query: "How do refunds work?" → similarity above threshold → serve cached answer
```

### Batch Processing

> **One-line understanding:** Grouping non-urgent requests into a provider's batch API trades higher latency (often hours) for significantly lower per-token cost.

🎯 **Interview Tip:** Good answer for any "large offline job" scenario (bulk classification, dataset labeling) — mention it explicitly.

### Model Routing

```text
Incoming request
      ↓
Classify complexity
      ↓
 Simple → small/cheap model
 Complex → larger/reasoning model
```

⚠️ **Common Mistake:** Using one (usually the most powerful) model for every task regardless of complexity — wastes budget on tasks a cheaper model handles fine.

### Context Reduction & Request Deduplication

* **Context reduction:** summarize conversation history, retrieve only relevant chunks (not whole documents), trim verbose system prompts.
* **Request deduplication:** collapse duplicate in-flight/near-simultaneous identical requests (double-clicks, concurrent identical jobs) into one model call.

---

# 💡 Key Insights

* Provider APIs churn constantly — the durable skill is the **shared abstraction** (messages, roles, tools, streaming, limits), not memorized syntax.
* Not all errors deserve a retry — distinguishing **transient** from **permanent** failures is core to production reliability.
* Prompting matters, but **context, tools, retrieval, and evaluation** drive more production quality at scale than prompt-wording tweaks.
* Structured output guarantees **shape**, never **truth** — validation and repair loops handle format, not factual correctness.
* Streaming improves **perceived** latency only — total generation time is unchanged.
* Prompt caching rewards **stable prefixes**; RAG's dynamic context inherently limits cache effectiveness.
* Cost optimization is rarely one technique — production systems combine caching + routing + context reduction + batching.

---

# ⚠️ Common Mistakes

| Mistake | Correct Understanding |
|---|---|
| ❌ Retrying every failed call, including `400`s | Only retry transient errors (429/500/503/timeout) |
| ❌ API keys in frontend code | Always proxy through a backend |
| ❌ Treating valid JSON as proof of correctness | Schema validity ≠ content accuracy |
| ❌ Over-investing in prompt wording | Context/retrieval/tools matter more at scale |
| ❌ Not canceling generation on disconnect | Wastes cost on tokens nobody sees |
| ❌ One model for every task | Wastes budget vs. routing by complexity |
| ❌ Deploying prompt changes untested | Silent regressions go unnoticed without a test set |

---

# 🔍 Common Confusions

| Concept A | Concept B | Key Difference |
|---|---|---|
| Usage limits | Rate limits | Usage = total consumption cap; Rate = speed-of-consumption cap |
| Response caching | Semantic caching | Exact match vs. meaning-similarity match |
| Structured output API | Prompting for JSON | API guarantees structure via constrained decoding; prompting only increases likelihood |
| SSE | WebSockets | One-directional vs. full-duplex |
| Prompt caching | Response caching | Speeds up reprocessing a prefix vs. skips the model call entirely |
| Task decomposition | Prompt decomposition | Splits a task into steps vs. organizes one prompt's internal structure |

---

# 🛠️ Practical Applications

| Question | Answer |
|---|---|
| **Where used?** | Chat products, coding assistants, extraction pipelines, support bots |
| **Problem solved?** | Reliable, affordable, responsive LLM-powered features in production |
| **Typical use case?** | RAG chatbots, structured data extraction, streaming chat UIs |
| **Engineering consideration?** | Error classification, caching strategy, routing, schema validation |

---

# 📌 Important Terms

| Term | Simple Meaning | Why It Matters |
|---|---|---|
| Rate limit | Cap on request/token speed | Drives retry/backoff design |
| Usage limit | Cap on total consumption | Drives budget/capacity planning |
| Exponential backoff | Increasing retry delays | Standard resilience pattern |
| Few-shot | Multiple examples in prompt | Improves format consistency |
| Delimiters | Markers separating instructions/data | Reduces prompt injection risk |
| Constrained decoding | Restricts tokens to schema-valid ones | Powers structured output APIs |
| Repair loop | Re-prompt model to fix invalid output | Recovers from validation failures |
| SSE | One-way streaming protocol | Standard for LLM text streaming |
| Backpressure | Consumer slower than producer | Needed for streaming flow control |
| Prompt caching | Reuses processed prompt prefix | Major cost/latency lever |
| Semantic caching | Caches by meaning similarity | Higher hit rate, some risk |
| Model routing | Picks model by task complexity | Core cost-control technique |

---

# ⚡ Quick Revision

| # | Key Point |
|---|---|
| 1 | Learn the shared provider abstraction, not exact API syntax |
| 2 | Retry only transient errors (429/500/503/timeout), never blind-retry a 400 |
| 3 | Few-shot > zero-shot for format-sensitive tasks; don't over-invest in prompt tricks |
| 4 | Structured output = guaranteed shape, never guaranteed correctness |
| 5 | Streaming improves perceived latency only; cancel generation on disconnect |
| 6 | Combine prompt/response/semantic caching + routing + context reduction for cost control |

---

# 🎯 Interview Preparation

## Level 1 — Fundamentals

**Q1. What is the difference between usage limits and rate limits?**
Usage limits cap total consumption (tokens/spend) over a billing period; rate limits cap how fast requests/tokens can be sent within a short window.

**Q2. What is few-shot prompting?**
Providing multiple example input/output pairs so the model infers the desired pattern/format before handling the real task.

**Q3. What is a JSON Schema used for?**
Formally specifying the expected structure, types, and required fields of model output so it can be validated and consumed by code.

**Q4. What is SSE and why is it common for LLM streaming?**
Server-Sent Events — a simple, one-directional HTTP streaming protocol; used because LLM output only needs to flow server→client.

**Q5. What is prompt caching?**
A provider feature that caches processing of a repeated prompt prefix so future requests reusing it are cheaper/faster.

**Q6. What is model routing?**
Dynamically choosing which model handles a request based on task complexity.

**Q7. Why should API keys never be in frontend code?**
Frontend code is extractable by any user; an exposed key can be stolen and abused — proxy calls through a backend instead.

**Q8. What is a repair loop?**
Sending validation errors back to the model and asking it to fix its output, instead of failing the request outright.

### 🧠 Knowledge Check

**If you can explain these in your own words, you understand Level 1:**
* The distinction between rate and usage limits
* Why examples in a prompt change model behavior
* Why structured output needs a schema, not just hope

---

## Level 2 — Conceptual Understanding

**Q1. Why shouldn't every API error be retried automatically?**
Transient errors (429/500/503/timeout) are likely to succeed on retry; a `400` reflects a real client-side bug that will fail identically — retrying it wastes time and can mask the actual issue.

**Q2. Why is structural validity not the same as output correctness?**
Schema validation confirms shape/types/required fields, not that the values are factually accurate — the model can still hallucinate a schema-valid but wrong answer.

**Q3. Why does streaming improve perceived latency without reducing total generation time?**
The model still takes the same total time to generate, but delivering tokens as produced lets users see output almost immediately instead of waiting for completion.

**Q4. Why does prompt caching require a stable prefix?**
Caching reuses the processed state of an unchanged prefix; any change invalidates it — so static content should come first, dynamic content last.

**Q5. Why is semantic caching riskier than exact-match caching?**
It relies on embedding similarity as a proxy for "same question," which is approximate — a loose threshold can serve a wrong answer for a meaningfully different query.

**Q6. Why is prompt engineering de-emphasized relative to context/retrieval/tools?**
Gains from prompt wording plateau after core principles are applied; production issues more often stem from poor context or tool design, which have larger marginal impact.

**Possible Follow-ups (Q6):**
1. What would you check first if quality is poor? → Retrieved context relevance before prompt wording.
2. Does that mean prompting doesn't matter? → No — baseline clarity/format still matters; it's about where the *next* unit of effort pays off most.

---

## Level 3 — Practical / Engineering

**Q1. How would you design error handling for a production LLM integration?**
Classify errors as retryable vs. non-retryable; apply exponential backoff with jitter and a max attempt count for retryable ones; log/alert on persistent failures; consider a fallback model for degraded primary-provider scenarios.

**Q2. How would you reduce cost for a high-traffic FAQ chatbot?**
Combine response/semantic caching for repeated questions, route simple queries to a smaller model, cap output length, and use prompt caching for a reused static knowledge excerpt.

**Q3. How would you handle a client disconnecting mid-stream?**
Detect the disconnect server-side and cancel the in-flight generation immediately to avoid paying for undelivered tokens.

**Q4. How would you validate and recover from malformed structured output?**
Parse against the schema (e.g., Pydantic); on failure, feed the specific error back in a repair loop with a capped retry count before failing gracefully.

**Q5. How would you version and test prompt changes safely?**
Store prompts in source control; maintain a regression test set; run it automatically on every change; roll back on regression.

---

## Level 4 — Advanced / Deep Understanding

**Q1. What hidden trade-offs exist in aggressive model routing?**
The complexity classifier can misroute — sending hard tasks to a weak model (quality risk) or easy tasks to an expensive model (cost risk); the classifier itself adds latency/cost that must be smaller than the savings it produces.

**Q2. Why can constrained decoding still fail to produce useful results despite guaranteeing valid JSON?**
It restricts *which tokens* can be chosen to stay schema-valid, but doesn't constrain *what the model believes* — it can fill required fields with confidently wrong values while remaining fully schema-compliant.

**Q3. What failure mode does semantic caching introduce that exact-match caching structurally cannot?**
Serving a plausible-but-wrong cached answer for a subtly different query (different scope/time/entity) — exact-match can only ever return answers to truly identical prior requests.

**Q4. How would you handle backpressure in a streaming app at scale?**
Bounded per-connection queues, pausing/dropping delivery if the client can't keep up, and monitoring persistently slow connections as a signal of client/network issues.

---

## Level 5 — Scenario-Based Questions

### Scenario

Your LLM costs grew 5x in three months; leadership wants a 50% cut without a major quality drop.

**Question:** What would you do and why?

### Model Answer
1. **Recommended approach:** Audit token usage by component first, then layer in model routing, prompt caching, response/semantic caching, and output length caps.
2. **Reasoning:** Cost growth is usually concentrated in a few high-volume patterns — measure before optimizing.
3. **Alternatives:** Blanket switch to a cheaper model (rejected — risks regressions on tasks needing the stronger model).
4. **Trade-offs:** Caching risks staleness/correctness if unscoped; routing needs a reliable classifier.
5. **Failure cases:** Semantic cache serving wrong answers for near-duplicate queries — mitigate with conservative thresholds.
6. **Production considerations:** Roll out incrementally with A/B monitoring on both cost **and** quality.

---

## Common Confusion Questions

### Q. Rate limits vs. usage limits?

| Rate Limits | Usage Limits |
|---|---|
| Caps speed (requests/tokens per minute) | Caps total (spend/tokens per period) |
| Triggers `429`, handled via backoff | Hard stop once cap reached |

**When would you choose to fix which?** A `429` calls for retry logic; a usage-limit hit calls for raising the limit or reducing consumption — retrying won't help.

---

## ⚠️ Deep / Trick Questions

### Does a structured output API guarantee correct data?

**Correct Understanding:**
* No — it guarantees shape/type conformance via constrained decoding, not factual correctness.
* Application-level content validation is still required.

### Is more prompt engineering always the right lever for a struggling app?

**Correct Understanding:**
* No — beyond core principles, returns diminish.
* Struggling apps more often need better context/retrieval or tool design.

---

# ⭐ Top Questions You MUST Know

1. Rate limits vs. usage limits — and how to handle each.
2. Which API errors are retryable, and why.
3. Few-shot vs. zero-shot — when does it help?
4. Why structured output guarantees shape, not correctness.
5. What is a repair loop, and when to use one.
6. Why SSE is the default for LLM streaming.
7. Why streaming improves perceived, not actual, latency.
8. Why prompt caching needs a stable prefix.
9. Response caching vs. semantic caching trade-offs.
10. Model routing — benefit and risk.
11. Why disconnect handling matters for cost.
12. Why prompt engineering is de-emphasized at scale.
13. Designing error handling for production.
14. Why API keys never belong in frontend code.
15. How to cut LLM costs 50% without hurting quality.

---

# 🎯 Interview Readiness Checklist

| Skill | Can I explain it? |
|---|---|
| Rate vs. usage limits | ☐ |
| Retryable vs. non-retryable errors | ☐ |
| Zero/one/few-shot prompting | ☐ |
| Structured output: shape vs. correctness | ☐ |
| Repair loops & retry strategies | ☐ |
| SSE vs. WebSockets | ☐ |
| Why streaming = perceived latency | ☐ |
| Prompt/response/semantic caching | ☐ |
| Model routing rationale & risk | ☐ |
| Backpressure & disconnect handling | ☐ |
| A real cost-optimization scenario | ☐ |

---

# 🧠 What You Should Be Able to Explain

1. Why a common provider abstraction matters more than any one vendor's API syntax.
2. How to distinguish retryable from non-retryable errors and design handling for each.
3. Core prompting techniques and why they matter less at scale than context/tools/retrieval.
4. Why structured outputs guarantee format, not correctness, and how repair loops help.
5. Why streaming improves perceived (not actual) latency, and how to handle disconnects/backpressure.
6. The trade-offs between prompt, response, and semantic caching.
7. How routing, batching, and context reduction combine to control cost in production.
