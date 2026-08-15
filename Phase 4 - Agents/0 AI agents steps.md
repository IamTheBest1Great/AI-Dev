Realistic answer: it depends heavily on scope, but here's a grounded breakdown so you can estimate your own case.

## Quick answer by agent complexity

| Type | Solo developer | Small team (2-3 people) |
|---|---|---|
| Simple single-tool agent (few tools, one workflow) | 3–7 days | 2–4 days |
| Moderate agent (multiple tools, some business logic, human review) | 2–4 weeks | 1–2 weeks |
| Complex/multi-agent, high-stakes, compliance needs | 2–4 months | 3–6 weeks |

These numbers assume you already know your stack and aren't learning APIs/tools from scratch as you go — add 30–50% if you are.

## Time breakdown per step (for a "moderate" agent — the most common real case)

| Step | Time | Notes |
|---|---|---|
| **1. Scope the use case** | 0.5–1 day | Fast if you know the domain; longer if requirements are fuzzy or need stakeholder alignment |
| **2. Build the eval set** | 1–2 days | This is usually underestimated. Writing 20-50 good test cases with expected behavior takes real thought |
| **3. Build the basic loop** | 1–3 days | The actual "hello world" agent loop is fast — a few hours to a day. Wiring up real tools/APIs (auth, error handling per tool) takes longer |
| **4. Narrow decision surface / prompt design** | 2–4 days (spread across iteration) | Not a discrete step — you'll revisit this constantly during Step 5 |
| **5. Iterate against eval set** | 3–7 days | This is usually the biggest time sink. Expect several rounds of "fix, rerun, still failing on X, fix again" |
| **6. Security guardrails** | 1–3 days | Fast if tools already have proper access scoping; slower if you're retrofitting permissions onto existing systems |
| **7. Human review workflow** | 1–2 days | Simple if it's just "flag to a Slack channel"; longer if you need a proper review UI |
| **8. Observability/logging** | 1–2 days | Basic structured logging is quick; a dashboard takes longer |
| **9. Scalability work** | 0 days (usually) at this stage | Skip until you have real traffic — don't budget for this upfront |
| **10. Gradual rollout setup** | 0.5–1 day | Mostly config/infra work if your deployment pipeline already exists |
| **11. Ongoing maintenance** | Ongoing, not a one-time cost | Budget a few hours a week after launch, indefinitely |

**Rough total: 2–4 weeks solo, 1–2 weeks with a small team**, for a genuinely production-worthy (not toy) agent.

## What actually eats the most time in practice

1. **Step 5 (iteration) almost always takes longer than planned.** Getting an agent from "works on the happy path" to "reliably handles edge cases" is where most of the real time goes — often 2-3x the time it took to write the initial loop.
2. **Tool integration, not the LLM logic.** Writing the actual agent loop is fast. Handling auth, rate limits, weird API error formats, and retries for each real tool you connect to is the slow part.
3. **Getting the eval set genuinely good.** A rushed eval set gives false confidence; a good one takes real thought about edge cases you haven't seen yet.

## What can shrink the timeline

- Using an existing framework (LangGraph, OpenAI Agents SDK, etc.) instead of building tool-calling plumbing yourself — saves days on the wiring, though you trade off some debugging transparency
- Reusing MCP servers instead of writing custom integrations for common tools (Slack, GitHub, databases) — can save a full day per integration
- Starting with a genuinely narrow scope (see earlier in our conversation) — the single biggest lever on timeline is resisting scope creep in Step 1

## What can blow up the timeline

- Discovering mid-build that the "narrow use case" wasn't actually narrow (very common)
- Tool/API access issues (getting credentials, permissions, sandbox environments set up) — often underestimated, especially in larger orgs
- Skipping the eval set, then spending 3x longer later debugging production issues you have no systematic way to catch

If you tell me the actual thing you're building (what tools it needs, roughly how many, and whether it's for personal use, internal team use, or customer-facing), I can give you a much tighter time estimate specific to your case.

Good instinct — the 11-step version I gave you is the "core spine," but there are real steps that production teams add depending on maturity and stakes. Here's what's commonly missing, organized by where it fits in the timeline.

## Steps to add BEFORE building (pre-build)

### A. Model selection & benchmarking
- **What:** Actually test 2-3 candidate models against your eval set before committing, rather than defaulting to whichever you used last time.
- **When:** Right after Step 2 (eval set), before Step 3.
- **Why:** Different models have different strengths for tool use, instruction-following, and cost/latency tradeoffs. The "best" model is often task-dependent.
- **How:** Run your eval set against 2-3 models, compare accuracy/cost/latency, pick based on your actual constraints.

### B. Legal/compliance review (for regulated domains)
- **What:** Check if your agent's domain has specific requirements — data residency, disclosure rules (e.g., "you are talking to an AI"), industry regulations (HIPAA, GDPR, financial compliance).
- **When:** During scoping (Step 1), before any real data touches the system.
- **Why:** Retrofitting compliance after launch is far more expensive than designing for it upfront. This is often the actual bottleneck in enterprise agent projects, not the tech.
- **How:** Loop in legal/compliance early — even a 30-minute conversation at Step 1 saves weeks later.

## Steps to add DURING build (often skipped)

### C. Structured output validation / schema enforcement
- **What:** Force tool calls and outputs to match a strict schema (JSON schema, Pydantic models) rather than trusting free-text parsing.
- **When:** Alongside Step 3 (building the loop).
- **Why:** Unvalidated outputs are a silent source of production bugs — the agent "mostly" returns valid JSON until it doesn't, and that failure mode is hard to catch without validation in place.
- **How:** Use function-calling/tool-use schemas natively rather than parsing free text; reject and retry on schema mismatch.

### D. Fallback & degraded-mode handling
- **What:** Explicit behavior for when a tool fails, the model times out, or a dependency is down — not just "the agent breaks."
- **When:** Alongside Step 3–5.
- **Why:** In a demo, failures don't matter. In production, "the payment API is down" happens weekly, and the agent needs a defined fallback (retry, queue, escalate to human, apologize gracefully) instead of an ugly error.
- **How:** For every tool, explicitly define: what happens on timeout, on error, on empty result.

### E. Adversarial / red-team testing
- **What:** Deliberately try to break the agent — prompt injection, jailbreak attempts, edge-case abuse, trying to get it to take unauthorized actions.
- **When:** After Step 5 (iteration), before Step 6 (security) is finalized.
- **Why:** Your eval set tests "does it work correctly." Red-teaming tests "can someone make it misbehave" — a genuinely different question that regular evals don't cover.
- **How:** Have someone (ideally not the builder) actively try to manipulate the agent for a few hours; add findings to your eval/guardrail set.

### F. Cost monitoring & budget alerts
- **What:** Track $ spent per request/session, with alerts if costs spike unexpectedly.
- **When:** Alongside Step 8 (observability).
- **Why:** A buggy loop (e.g., an agent stuck retrying) can burn through API budget fast and silently — this is one of the most common "surprise" production incidents with agents specifically.
- **How:** Log token usage per call, set a per-session cost ceiling, alert on anomalies.

## Steps to add BEFORE launch (pre-launch)

### G. Load/latency testing
- **What:** Test how the agent performs under realistic concurrent traffic, not just one request at a time.
- **When:** Just before Step 10 (gradual rollout).
- **Why:** Agents that work fine for one user can hit rate limits, timeouts, or resource contention under real load — this is invisible until you actually test it.
- **How:** Simulate concurrent requests against a staging environment; watch for rate-limit errors, timeout cascades.

### H. User transparency / disclosure
- **What:** Deciding how (and whether) users are told they're interacting with an AI agent, and what recourse they have (e.g., "talk to a human").
- **When:** Before any customer-facing launch.
- **Why:** Increasingly a legal requirement in several jurisdictions, and independent of that, it's good trust practice — users who feel misled about talking to AI churn faster.
- **How:** Simple UI disclosure + a clear escalation path to a human.

### I. Documentation & runbook for whoever's on call
- **What:** A short doc: what the agent does, common failure modes, how to disable it, who to contact.
- **When:** Right before launch.
- **Why:** When something goes wrong at 2am, whoever's on call needs to know how to kill the agent or roll it back — this is often forgotten until the first incident.
- **How:** One page: kill switch location, common errors and fixes, escalation contacts.

## Steps to add AFTER launch (ongoing, beyond what I covered)

### J. A/B testing framework for changes
- **What:** Formal comparison between agent versions (not just eval-set pass/fail) using real user outcomes.
- **When:** Once you're making iterative improvements post-launch.
- **Why:** Eval sets catch known failure modes; A/B tests catch things you didn't think to test for, using real usage patterns.
- **How:** Route a % of traffic to each version, compare business metrics (satisfaction, task completion, escalation rate).

### K. Model deprecation / version drift handling
- **What:** A plan for when your underlying model gets deprecated or silently updated by the provider.
- **When:** Ongoing, revisited whenever a model provider announces changes.
- **Why:** Agents can subtly change behavior when a provider updates a model version — production teams get caught off guard by this regularly.
- **How:** Pin model versions explicitly where possible; re-run your eval set whenever a model version changes.

---

## The honest priority order if you can't do everything

If you only add a few of these to the core 11 steps, in order of actual impact:
1. **Fallback/degraded-mode handling (D)** — the #1 cause of embarrassing production failures
2. **Structured output validation (C)** — cheap to add, prevents a whole category of silent bugs
3. **Adversarial testing (E)** — especially if the agent is customer-facing or handles anything sensitive
4. **Cost monitoring (F)** — cheap insurance against a very common and painful surprise
5. **Documentation/runbook (I)** — costs an hour, saves a 2am scramble

The rest (compliance, load testing, A/B testing) scale in importance with the stakes and maturity of what you're building — worth knowing about, not necessarily worth doing on day one of a small internal tool.
