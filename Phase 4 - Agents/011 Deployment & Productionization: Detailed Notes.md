# Module 11 — Deployment & Productionization: Detailed Notes 🆕

## Table of Contents

- [11.1 Sync vs Async Execution Models](#111-sync-vs-async-execution-models)
  - [The fundamental distinction](#the-fundamental-distinction)
  - [Sync execution — when it fits](#sync-execution--when-it-fits)
  - [Async execution and job queues](#async-execution-and-job-queues)
  - [Job queue architecture for agents](#job-queue-architecture-for-agents)
  - [📋 Interview Questions — 11.1](#-interview-questions--111)
- [11.2 Durable Execution and State Persistence](#112-durable-execution-and-state-persistence)
  - [Why in-memory state is a production anti-pattern for agents](#why-in-memory-state-is-a-production-anti-pattern-for-agents)
  - [Durable execution — the core concept](#durable-execution--the-core-concept)
  - [Temporal — the production-validated implementation](#temporal--the-production-validated-implementation)
  - [The workflow/activity split — why it maps cleanly onto agents](#the-workflowactivity-split--why-it-maps-cleanly-onto-agents)
  - [DB-backed checkpoints — the lighter alternative](#db-backed-checkpoints--the-lighter-alternative)
  - [📋 Interview Questions — 11.2](#-interview-questions--112)
- [11.3 Scaling Considerations](#113-scaling-considerations)
  - [Concurrency limits and worker pools](#concurrency-limits-and-worker-pools)
  - [Provider rate limits and multi-provider routing](#provider-rate-limits-and-multi-provider-routing)
  - [Multi-model routing at the architecture level](#multi-model-routing-at-the-architecture-level)
  - [📋 Interview Questions — 11.3](#-interview-questions--113)
- [11.4 Versioning Agents, Prompts, and Tool Schemas](#114-versioning-agents-prompts-and-tool-schemas)
  - [Why versioning is harder for agents than for software](#why-versioning-is-harder-for-agents-than-for-software)
  - [Versioning prompts](#versioning-prompts)
  - [Versioning tool schemas](#versioning-tool-schemas)
  - [Versioning whole agents](#versioning-whole-agents)
  - [📋 Interview Questions — 11.4](#-interview-questions--114)
- [11.5 Rollout Strategies](#115-rollout-strategies)
  - [Why standard software rollout strategies apply — with extra care](#why-standard-software-rollout-strategies-apply--with-extra-care)
  - [Canary releases for agents](#canary-releases-for-agents)
  - [Feature flags for agent behavior changes](#feature-flags-for-agent-behavior-changes)
  - [Rollback — what it means for agents specifically](#rollback--what-it-means-for-agents-specifically)
  - [📋 Interview Questions — 11.5](#-interview-questions--115)

> Deployment is where every architectural decision made in Modules 1–10 meets reality. The patterns here directly operationalize the theoretical reliability and security concepts — durable execution for Module 1.6's async approvals, rollout strategies for Module 10.4's A/B testing, versioning for Module 10.3's regression testing. These aren't optional polish; they're what separates a working demo from a production system.

> Deployment is where every architectural decision made in Modules 1–10 meets reality. The patterns here directly operationalize the theoretical reliability and security concepts — durable execution for Module 1.6's async approvals, rollout strategies for Module 10.4's A/B testing, versioning for Module 10.3's regression testing. These aren't optional polish; they're what separates a working demo from a production system.

---

## 11.1 Sync vs Async Execution Models

### The fundamental distinction
**Synchronous execution** holds the calling connection open for the entire duration of an agent's run — the client sends a request and waits, blocked, until the agent finishes and returns a result. This is the natural default when building an agent for the first time (it's the HTTP request-response model most developers reach for instinctively) and is entirely appropriate for short, fast tasks. Its ceiling is straightforward: anything that takes longer than a few seconds starts to run into real problems — HTTP connection timeouts (typically 30–120 seconds by default), user-facing latency degradation, and an inability to do any other work while waiting.

**Asynchronous execution** decouples the moment of request submission from the moment of result delivery. The client submits a task, receives an immediate acknowledgment (a job ID, a status URL), and the agent runs detached from that original connection — the client polls for completion or receives a callback/webhook when done. This is the model required for any agent task that takes longer than a handful of seconds.

### Sync execution — when it fits
Synchronous execution is the right default when:
- **Latency is short** — the agent's full run (all steps, all tool calls) reliably completes in under roughly 10–30 seconds.
- **The user is actively waiting** — a chat-style interface where the user expects a quick response; streaming output from each step is often the practical enhancement that makes sync tolerable even at slightly longer durations.
- **Failure is cheap** — if the request fails or times out, retrying from scratch is acceptable and not costly.

The largest production case for synchronous agents in practice: streaming token-by-token output (using SSE or WebSockets) that shows the user progressive results rather than a blank screen while waiting for the full agent run to finish. This doesn't change the underlying sync model but dramatically changes the user experience within it.

### Async execution and job queues
Async execution requires a **queue layer** between the request submission and the execution worker — the queue decouples them so neither blocks the other. The standard pattern:
1. Client submits task → API server validates and enqueues a job, returns job ID immediately.
2. A worker pool pulls jobs from the queue and executes them (each worker runs one agent task at a time, or concurrently with a defined limit).
3. Results are written to durable storage when complete.
4. Client polls a status endpoint or receives a webhook/callback with the result.

This is the same job-queue architecture from Module 5.5 (queues and retries for long-running pipeline jobs), applied specifically to agent tasks. The same discipline — exponential backoff with jitter, idempotency, dead-letter queues for exhausted retries — applies unchanged. The n8n queue-mode architecture from Module 5.2 is a specific implementation of this same pattern at the workflow-automation layer.

### Job queue architecture for agents
Three components the queue layer adds that an in-process model doesn't have:
- **Backpressure** — when more jobs are submitted than workers can currently handle, the queue absorbs the burst rather than either dropping requests or overwhelming the worker pool. The queue depth at any point in time is a first-class operational metric that needs to be monitored and alerted on.
- **Horizontal scaling** — adding more workers is the mechanism for increasing throughput. Because workers pull from a shared queue rather than each having a fixed partition, scaling workers doesn't require re-partitioning work — it's operationally simpler than most stateful scaling approaches.
- **Work deduplication and idempotency** — a job submitted twice (e.g., due to a client retry after a network timeout) should result in one execution, not two. This requires an idempotency key on the job submission — the same discipline from Module 5.5, now at the agent-job level rather than the pipeline-task level.

### 📋 Interview Questions — 11.1
1. **What's the practical ceiling of synchronous execution for agents, and what usually breaks first when you exceed it?**
   *Look for: HTTP timeout limits (30–120s default), blocking the calling thread for the entire duration, user-facing latency; what breaks first in practice is the HTTP connection timing out before the agent finishes, causing the client to receive an error even though the agent may have completed successfully.*
2. **A streaming response (token by token) uses synchronous HTTP underneath. What problem does streaming solve, and what problem does it leave unsolved?**
   *Look for: streaming improves perceived latency and UX by showing progressive output, but the underlying connection is still open and synchronous — it doesn't solve connection timeouts for very long tasks or allow the user to disconnect and reconnect without losing state.*
3. **Why does a job queue's idempotency key matter specifically for agent tasks, and what goes wrong without it?**
   *Look for: network timeouts or client retries can submit the same task twice; without an idempotency key, both submissions execute independently — potentially causing duplicate side effects (two emails sent, two records created, double spend) rather than recognizing the second submission as a duplicate of the first.*
4. **How does queue depth function as an operational signal, and what would you do if you observed it growing continuously?**
   *Look for: continuously growing queue depth means work is arriving faster than workers can process it — the first response is to scale out workers (horizontal scaling); if that's already at its limit, the root cause may be worker efficiency (slow agent runs) or an unexpected traffic spike needing upstream rate limiting.*
5. **When would you choose to keep an agent synchronous (with streaming) rather than moving to async job queues, even for a task that takes 45 seconds?**
   *Look for: when the user genuinely needs to watch the agent's reasoning live and can't usefully do anything else while waiting; when the product UX is built around real-time interaction; when the added operational complexity of a queue layer isn't justified by the marginal improvement over a long streaming response with an appropriately generous timeout.*

---

## 11.2 Durable Execution and State Persistence

### Why in-memory state is a production anti-pattern for agents
Most agent frameworks, by default, keep the agent's execution state — the conversation history, the current step in the plan, the results of completed tool calls — entirely in memory. This is fine for a demo. In production, it means:
- A process restart (deployment, crash, OOM kill, planned maintenance) **loses everything** — the agent's full state evaporates and the task has to restart from scratch.
- A task running across hours or days **holds memory for that entire duration** — a resource leak, not a feature.
- A paused human-approval gate (Module 1.6) **blocks a process** or requires the task to be abandoned if the process needs to restart before approval arrives.

The failure mode this produces is exactly the silent-error-compounding problem from Module 1.5, now at the infrastructure level: an agent that appeared to be making progress crashes at step 47 of 50, restarts, and re-executes the first 47 steps from scratch — paying the full token and tool-call cost again, and potentially taking side-effecting actions (sending messages, writing records) a second time if those steps weren't idempotent.

### Durable execution — the core concept
Durable execution is the infrastructure guarantee that an agent's execution state survives any single-point failure — process restarts, machine failures, deliberate pauses — and can be resumed exactly where it left off, not from the beginning. The core mechanism: **every completed step's result is persisted to durable storage before the next step begins**, so recovery means replaying the workflow's coordination logic using already-recorded step results, not re-executing the steps themselves.

Workflow replay does not re-execute activities — it only re-runs the workflow's coordination code using the recorded outputs from the event history. API calls, database writes, and LLM calls are not repeated during replay. This is the core guarantee that makes durable execution safe and cost-efficient. This distinction is critical: replay re-runs the deterministic coordination code (which path to take next, which step to execute), but uses the *recorded results* from the durable log rather than re-executing the non-deterministic operations (LLM calls, tool calls) again. This means neither the token cost nor the side effects of already-completed steps are doubled on recovery.

### Temporal — the production-validated implementation
Temporal provides an open-source workflow orchestration engine designed for complex, high-scale distributed systems. As of 2026 it has become the most widely cited infrastructure layer for production-grade long-running agents — a $300M raise in February 2026 and the GA of the OpenAI Agents SDK integration signal that the industry has reached consensus on this pattern. Temporal now has native integrations with Google ADK and the OpenAI Agents SDK, announced at Replay 2026, alongside new capabilities including Serverless Workers, Standalone Activities, and Workflow Streams.

The core programming model in Python:

```python
@workflow.defn
class AgentWorkflow:
    @workflow.run
    async def run(self, goal: str) -> str:
        messages = [{"role": "user", "content": goal}]
        while True:
            # LLM calls automatically retry on failure
            response = await workflow.execute_activity(
                call_llm,
                args=[messages, tools],
                start_to_close_timeout=timedelta(seconds=60)
            )
            if not response.tool_calls:
                return response.content  # Agent is done
            # Tool execution is durable - survives crashes
            result = await workflow.execute_activity(
                run_tool,
                args=[response.tool_call],
                start_to_close_timeout=timedelta(minutes=5)
            )
            messages.append(response.message)
            messages.append({"role": "tool", "content": result})
```

Every `execute_activity` call is durably recorded. If the process crashes mid-loop, Temporal replays the workflow code from the beginning but injects the recorded results for already-completed activities rather than re-executing them — the agent resumes exactly at the step that was interrupted, with zero re-execution of completed steps.

### The workflow/activity split — why it maps cleanly onto agents
When using Temporal to build an AI application, the predetermined flow (simple or complex), or the agentic loop, is implemented as a Temporal Workflow. LLMs, Actions, and UX are components coordinated by the workflow.

The design constraint that makes this work: **workflows must be deterministic** — given the same event history, replaying the workflow must always produce the same sequence of activity calls. This means no randomness, no `datetime.now()`, no direct I/O inside workflow code. Everything non-deterministic goes into an **activity**: LLM calls, tool calls, API requests, database reads and writes, current-time lookups. The workflow coordinates; the activities do.

This maps cleanly onto the agent architecture from Module 1.2: the orchestration/runtime layer (deterministic coordination logic) is the Temporal workflow; every LLM call and tool call is a Temporal activity. The split that good agent architecture already recommends is precisely the split Temporal requires — which is why the integration feels natural rather than forced.

### DB-backed checkpoints — the lighter alternative
For teams not ready for Temporal's operational complexity, a simpler form of durability: persist the agent's full state (conversation history, step index, intermediate results) to a database after each completed step, and implement restart logic that checks for an existing checkpoint before beginning execution.

LangGraph's checkpointer (Module 3.2) is the most common implementation of this pattern within an agent framework — a Postgres, Redis, or SQLite backend that persists the graph's state at each node. The tradeoff vs. Temporal: simpler to operate, no new infrastructure to run, but lacks Temporal's event-sourced replay guarantee, its built-in retry policy engine, and its visibility tooling (Temporal's Web UI shows every workflow's full execution history across all workers in real time).

The right choice depends on task horizon and operational maturity: DB-backed checkpoints are entirely sufficient for tasks that pause for human approval and resume (hours to a day or two); Temporal becomes clearly worthwhile for tasks running days to weeks, tasks that must survive arbitrary infrastructure failures, and tasks where the audit trail of every step's inputs and outputs has compliance or debugging value beyond just "can it resume."

### 📋 Interview Questions — 11.2
1. **What exactly happens during a Temporal workflow replay, and why doesn't it re-execute LLM or tool calls?**
   *Look for: replay re-runs the deterministic workflow coordination code but injects recorded results from the durable event history for every already-completed activity — no LLM calls, API calls, or tool calls are re-executed, so neither their cost nor their side effects are repeated.*
2. **Why must Temporal workflows be deterministic, and what does "deterministic" specifically prohibit?**
   *Look for: deterministic means the workflow code produces the same sequence of activity calls given the same event history — this prohibits randomness, datetime.now() calls, direct I/O, or any non-deterministic logic inside workflow code; those operations belong in activities.*
3. **How does the Temporal workflow/activity split map onto the agent anatomy from Module 1.2?**
   *Look for: the orchestration/runtime layer (coordination logic, loop control) = Temporal workflow (deterministic); every LLM call, tool call, and API request = Temporal activity (non-deterministic, durable) — the split that good agent architecture already recommends is exactly what Temporal requires.*
4. **When would DB-backed checkpoints (e.g., LangGraph's checkpointer) be sufficient, versus when does Temporal's full durable-execution model become clearly worth the operational overhead?**
   *Look for: DB checkpoints are sufficient for tasks that pause for human approval over hours or a day or two; Temporal earns its overhead for multi-day tasks, tasks that must survive arbitrary infrastructure failures reliably, or tasks where the full step-by-step audit trail has compliance/debugging value.*
5. **A production agent at step 47 of 50 crashes due to an OOM kill. What happens next under: (a) in-memory state only, (b) DB-backed checkpoints, (c) Temporal?**
   *Look for: (a) task restarts from scratch, re-paying all costs and re-executing all side effects; (b) task resumes from the last checkpoint, re-executing only from that point — but if the checkpoint was at step 40, steps 40–47 re-execute; (c) task resumes exactly at step 47 using recorded activity results — no step is re-executed, no costs are doubled.*

---

## 11.3 Scaling Considerations

### Concurrency limits and worker pools
At low volume, every agent task can run immediately. At real production volume, three distinct concurrency constraints require active management rather than assuming the infrastructure will handle them:

- **Worker concurrency** — how many agent tasks can run simultaneously across your worker pool. Exceeding this means tasks queue up rather than starting immediately. The right number balances throughput against per-task resource consumption (memory per running agent, number of simultaneous open database/API connections) — it's a resource sizing problem, not just "more is better."
- **Per-task concurrency within an agent** — the parallel sub-agent and tool-call patterns from Modules 7.4 and 2.1 mean a single task can itself fan out into many simultaneous operations. This internal concurrency multiplies the worker's resource footprint and the rate at which it hits external rate limits — a task that "uses 1 unit" of concurrency at the worker level may use 10 units of concurrency against a downstream API.
- **Semaphore-based admission control** — a critical production pattern for sub-agent spawning (Module 7.4's runaway-spawn mitigation): a semaphore with a defined max-concurrent count prevents any individual task from spawning an unbounded number of sub-agents, even if the agent's own reasoning suggests it should. The semaphore is the structural circuit breaker from Module 9.1 applied at the concurrency level rather than the step-count level.

### Provider rate limits and multi-provider routing
LLM providers impose rate limits on two distinct dimensions, and both must be managed separately:
- **Requests per minute (RPM)** — how many API calls your account can make per minute. Exceeded by many small, fast requests (e.g., high-volume routing/classification steps in the module 9.4 model-routing pattern).
- **Tokens per minute (TPM)** — how many tokens in total can flow through API calls per minute. Exceeded by fewer but larger requests (long contexts, verbose tool outputs, many accumulated steps in a single agent's context).

Single-provider deployments hit hard ceilings at scale — and rate limit errors are silent-failure risks if not handled correctly (an uncaught 429 that gets retried without backoff can produce a thundering herd that makes the rate limit worse, the same problem from Module 5.5's retry design). Multi-provider routing provides both redundancy and headroom:

- **Fallback routing** — a primary provider is used by default; a secondary provider receives traffic when the primary is rate-limited or degraded. Requires the application layer to abstract provider differences (different SDK shapes, response formats, capability availability) so the fallback is operationally transparent.
- **Load-balanced routing** — traffic is distributed across multiple provider accounts (e.g., multiple Anthropic API keys if permitted under ToS, or different providers) from the first request, spreading rate-limit exposure rather than concentrating it on one account until it saturates.
- **AI gateway services** (e.g., LiteLLM, Portkey) — a thin proxy layer that abstracts multi-provider routing, provides unified observability across providers, and handles retry/fallback logic centrally rather than requiring each application to re-implement it independently.

### Multi-model routing at the architecture level
Module 9.4 covered per-step model routing for cost optimization. At the deployment/architecture level, routing is a broader concern: which model for which task type, enforced systematically across all agent runs rather than left to per-agent configuration. Concretely:
- Define routing rules based on task classification (a simple request → small fast model; a complex multi-step research task → frontier model), and enforce them at the infrastructure layer (the AI gateway or a custom router), not inside each individual agent's prompt.
- Measure and track cost-per-task-type separately by model to make routing decisions evidence-based rather than assumption-based.
- Use **A/B testing across model versions** (Module 10.4) before routing changes go live, using the same regression-suite discipline — never upgrade a model in the routing table without testing it first.

### 📋 Interview Questions — 11.3
1. **What's the difference between worker-level concurrency and per-task internal concurrency (parallel tool calls / sub-agents), and why does conflating them lead to bad capacity planning?**
   *Look for: worker concurrency = how many tasks run simultaneously across the pool; per-task internal concurrency = how many operations a single running task fans out to simultaneously — a single worker slot can consume 10x or 100x its apparent resource footprint if the task itself parallelizes aggressively, which a capacity plan based only on worker count will underestimate.*
2. **Why must RPM and TPM be managed as separate rate-limit dimensions, and what different architectural patterns cause each to be the binding constraint?**
   *Look for: RPM is exhausted by many small, fast requests (high-frequency routing/classification); TPM is exhausted by fewer but larger requests (long contexts, verbose tool outputs) — a system can be well within RPM while hitting TPM limits (or vice versa), requiring different mitigations.*
3. **How does a semaphore-based admission control pattern address the runaway sub-agent spawning risk from Module 7.4?**
   *Look for: a semaphore with a defined max-concurrent count is a structural ceiling on how many sub-agents can run simultaneously regardless of what the agent's reasoning decides — it's the circuit-breaker pattern (Module 9.1) applied to concurrency rather than step count.*
4. **What's the difference between fallback routing and load-balanced routing, and which is appropriate for: (a) a team that's close to their rate limit at peak, (b) a team that wants redundancy but rarely hits rate limits?**
   *Look for: (a) load-balanced from the start — they need headroom across providers, not just a fallback they'll hit constantly; (b) fallback — they want resilience against occasional degradation without the complexity of distributing all traffic across providers.*
5. **Why should multi-model routing rules be enforced at the infrastructure layer (a gateway or router) rather than inside each individual agent's prompt or configuration?**
   *Look for: agent-level configuration means each agent can diverge from the policy independently, making routing behavior hard to audit, test, or change consistently; infrastructure-layer enforcement means routing is a single-point policy change that applies uniformly without touching individual agent implementations.*

---

## 11.4 Versioning Agents, Prompts, and Tool Schemas

### Why versioning is harder for agents than for software
Software versioning is well-understood: the artifact (a binary, a library, a container image) has a precise, reproducible definition. Agent "behavior" is defined by the combination of model weights, system prompt, tool schemas, and orchestration logic — and any of these can change independently, with the interaction effects between them being non-obvious and often surprising. A prompt change that's safe with model version A may produce a regression with model version B. A tool schema change may be backward-compatible in schema terms but change the agent's selection behavior in ways the schema alone doesn't predict.

This means agent versioning has to be **compositional** — tracking the version of every component independently while also testing the specific combinations in use, not just individual components in isolation.

### Versioning prompts
The core discipline (established in Module 10.4 but expanded here into deployment practice):
- **Every system prompt is a versioned artifact** with a unique, human-readable identifier (not just a hash): `customer-support-v2.3-2026-06` is debuggable and traceable; `8f3a2c1` is not.
- **Prompts live in version control** (the same repository as the code that uses them, or a dedicated prompt registry), not in a database or dashboard field that can be edited without review.
- **The prompt version in use is emitted as a structured log field** on every agent run, so when a production incident is investigated, you can precisely identify which prompt version was running at the time without guessing.
- A **prompt registry** (a lightweight service or database table that maps prompt identifiers to prompt content and metadata) enables runtime prompt loading rather than hard-coding prompts in deployment artifacts — making it possible to roll back a prompt change without a full redeployment.

### Versioning tool schemas
Tool schemas (Module 2) are contracts between the agent and the tool — and changes to them are breaking changes in a very specific sense: even a "backward-compatible" schema change (adding an optional field) may change the agent's behavior because the model reads and is influenced by the full schema at inference time. The disciplines:
- Treat tool schema changes with the same regression-testing discipline as prompt changes: run the full evaluation suite on the new schema before deploying it.
- Maintain a changelog of tool schema versions, tied to the agent versions that use each schema version.
- For tools called externally (MCP servers, Module 6), confirm which schema version the server is currently serving before routing production traffic to it — schema version mismatches between the agent's expectations and the server's actual behavior are a common, hard-to-diagnose category of production bug.

### Versioning whole agents
A "whole agent version" is the specific combination of: model version + system prompt version + tool schema versions + orchestration logic version (the code, including framework version). This combination determines what behavior a user will experience, and all four components need to be captured together to reproduce any historical behavior.

Practically this means:
- **A deployment manifest** (or equivalent) that specifies all four components explicitly, not just the code version. A container image version alone is insufficient if the prompt is loaded at runtime from a registry (and it should be).
- **A canonical agent ID** that maps to a specific set of component versions, so observability tooling, experiments, and incident reports all reference the same granularity of version identifier.
- **Immutable version identifiers** — once a version is in production, its components don't change. Any change — even a prompt typo fix — creates a new version number.

### 📋 Interview Questions — 11.4
1. **Why is tracking the code version alone insufficient for reproducible agent behavior, when it's usually sufficient for traditional software?**
   *Look for: agent behavior is determined by the combination of model weights, system prompt, tool schemas, and orchestration code — any of the non-code components can change independently of the code version, and the interaction effects between them are non-obvious; all four must be captured together.*
2. **What's the specific risk of treating a "backward-compatible" tool schema change (adding an optional field) as safely non-breaking for an agent?**
   *Look for: the model reads the full schema at inference time and is influenced by it — even an optional new field can change tool-selection behavior or argument patterns in ways the schema-compatibility definition doesn't capture; requires empirical testing, not just schema analysis.*
3. **Why should the prompt version be emitted as a structured log field on every agent run, and how does this change incident investigation?**
   *Look for: structured logging of prompt version means you can precisely identify which prompt was running during a production incident without guessing or reconstructing from deployment history; "find all runs where customer-support-v2.3 was active" becomes a simple log query rather than an archaeological investigation.*
4. **What problem does a prompt registry (loading prompts at runtime from a service) solve that hard-coding prompts in deployment artifacts doesn't?**
   *Look for: runtime loading enables rollback of a prompt change without a full code redeployment — a critical operational difference when a prompt regression needs to be reverted immediately and a full deployment pipeline takes 20–40 minutes.*
5. **A team uses a semantic version (1.4.2) for their agent, but it only tracks changes to the orchestration code. What's missing, and how would you redesign the versioning scheme?**
   *Look for: a version that only tracks code is missing prompt version, tool schema versions, and model version — redesign as a composite manifest (e.g., `agent-v1.4.2 / prompt-customer-support-v2.3 / tools-v3.1 / model-claude-opus-4.6`) that captures all four components explicitly.*

---

## 11.5 Rollout Strategies

### Why standard software rollout strategies apply — with extra care
The standard software deployment playbook — canary releases, feature flags, gradual rollout, rollback triggers — applies directly to agent deployments. The extra care: agent behavior changes are **more opaque** than software behavior changes. A code change has a diff; you can read what changed. A prompt or model change may have a clear human-readable diff but an unclear behavioral diff — two prompts that look almost identical can produce meaningfully different agent behavior on specific input categories that weren't anticipated when the change was written. This opacity means rollout strategies need to be more conservative and more instrumented for agents than for equivalent software changes.

### Canary releases for agents
A canary release routes a small fraction of real production traffic (typically 1–5%) to the new agent version while the remaining traffic continues on the current version. The canary cohort must be large enough to produce statistically meaningful quality metrics within a defined observation window (typically 24–72 hours for agents, given that evaluation metrics need multiple runs per task to stabilize). If quality metrics for the canary cohort match or exceed the control cohort's metrics within the observation window, rollout proceeds to the next stage; if they degrade, rollout is halted and the canary is rolled back.

Canary releases for agents have a specific additional concern: **distribution skew**. If the 1–5% canary cohort receives a non-representative sample of task types (e.g., only easy tasks, or only tasks from one geographic region or user segment), the canary metrics won't generalize. Randomize traffic routing at the task level, not at the session or user level, and stratify the canary cohort to match the known distribution of task types rather than relying on random sampling to produce a representative distribution in a small cohort.

### Feature flags for agent behavior changes
Feature flags allow behavior changes to be deployed in code but activated (or deactivated) at runtime without a redeployment. For agent behavior specifically, feature flags are most useful for:
- **Enabling a new capability for a subset of users** (e.g., a new tool set, a new agent mode) while keeping the existing behavior for everyone else.
- **Targeting specific user segments** for behavioral experiments that aren't appropriate for a simple random A/B split.
- **Immediate kill-switch capability** — if a new behavior causes problems, a flag can deactivate it in seconds without waiting for a deployment pipeline.

The operational discipline: feature flags that control agent behavior changes are **not housekeeping items to be cleaned up eventually** — they're active safety controls while a change is in rollout. Define a flag's expected active lifetime and a mandatory cleanup deadline when the flag is created, so the codebase doesn't accumulate indefinitely-live flags that nobody knows are safe to remove.

### Rollback — what it means for agents specifically
Rolling back a traditional software deployment means pointing production traffic back at the previous deployment artifact. Rolling back an agent version means:
1. Routing traffic back to the previous prompt version (from the prompt registry — this is why runtime loading matters, Module 11.4).
2. Routing traffic back to the previous model version (updating the routing table).
3. Routing traffic back to the previous tool schema version.
4. If using durable execution (Module 11.2), considering whether in-flight tasks should be allowed to complete on the new version or should be migrated to the old version — and recognizing that migrating in-flight Temporal workflows to a different workflow definition version requires Temporal's versioning APIs (patch/get_version), which must be planned for, not retrofitted.

The last point is the most frequently overlooked: a rollback policy that's only thought about *after* an incident occurs, at 2am, is not a rollback policy — it's a wish. The rollback procedure needs to be documented, tested, and practiced before any production deployment, the same as any other disaster-recovery procedure.

### 📋 Interview Questions — 11.5
1. **Why do agent behavior changes require more conservative rollout strategies than equivalent software changes, even when both use a canary release?**
   *Look for: the behavioral diff of a prompt or model change is opaque in a way a code diff isn't — two nearly-identical prompts can produce meaningfully different behavior on unanticipated input categories, which isn't visible from reading the change itself and requires empirical measurement in production to detect.*
2. **Why must canary traffic for agents be randomized at the task level (not the session or user level), and why does stratification matter for a small canary cohort?**
   *Look for: session/user-level randomization can produce a skewed task distribution in a small cohort; task-level randomization with stratification ensures the canary sees a representative mix of task types, so its quality metrics will generalize to the full population.*
3. **What makes a feature flag for an agent behavior change different from a housekeeping item, and what operational discipline does that imply?**
   *Look for: an active agent-behavior feature flag is a live safety control during rollout — defining an expected lifetime and mandatory cleanup deadline at flag creation time prevents indefinitely-live flags that become invisible technical debt and unknown risk.*
4. **A rollback to the previous agent version is triggered at 2am after a production incident. What are the four components that need to be rolled back, and which one is most likely to have been overlooked in the rollback plan?**
   *Look for: prompt version (via registry), model version (via routing table), tool schema version, and in-flight durable workflow handling — in-flight workflow migration is almost always the overlooked one, since it requires Temporal's versioning APIs planned in advance.*
5. **Why is testing your rollback procedure before an incident the most important part of having a rollback procedure at all?**
   *Look for: an untested rollback procedure fails exactly when it's needed most — under pressure, at 2am, with an incident in progress — because the steps weren't validated when there was time to discover gaps; rollback must be treated as a rehearsed procedure, not a theoretical capability.*

---

*End of Module 11 detailed notes — 25 interview questions total across 5 sections. The Temporal ecosystem is moving fast — GA integrations with Google ADK and OpenAI Agents SDK were announced at Replay 2026, and Serverless Workers / Workflow Streams are new capabilities — re-verify current integration support before building against any specific SDK helper.*
