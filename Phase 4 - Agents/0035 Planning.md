# Planning Module — Task Decomposition, Planner-Executor & Reflection: Detailed Notes

## Table of Contents

- [P.1 Task Decomposition](#p1-task-decomposition)
  - [Why decomposition matters](#why-decomposition-matters)
  - [Types of decomposition](#types-of-decomposition)
  - [Decomposition strategies](#decomposition-strategies)
  - [Granularity — how small is small enough?](#granularity--how-small-is-small-enough)
  - [Dependency mapping](#dependency-mapping)
  - [Failure modes specific to decomposition](#failure-modes-specific-to-decomposition)
  - [📋 Interview Questions — P.1](#-interview-questions--p1)
- [P.2 Planner-Executor](#p2-planner-executor)
  - [The core separation of concerns](#the-core-separation-of-concerns)
  - [How the planner works](#how-the-planner-works)
  - [How the executor works](#how-the-executor-works)
  - [Plan revision and replanning](#plan-revision-and-replanning)
  - [Human-in-the-loop at the planning layer](#human-in-the-loop-at-the-planning-layer)
  - [When Planner-Executor outperforms plain ReAct](#when-planner-executor-outperforms-plain-react)
  - [📋 Interview Questions — P.2](#-interview-questions--p2)
- [P.3 Reflection](#p3-reflection)
  - [What reflection means in agent systems](#what-reflection-means-in-agent-systems)
  - [Reflexion — the formal pattern](#reflexion--the-formal-pattern)
  - [Self-critique vs external critique](#self-critique-vs-external-critique)
  - [Meta-cognitive reflection vs step-level reflection](#meta-cognitive-reflection-vs-step-level-reflection)
  - [Reflection in multi-agent systems](#reflection-in-multi-agent-systems)
  - [Failure modes of reflection itself](#failure-modes-of-reflection-itself)
  - [📋 Interview Questions — P.3](#-interview-questions--p3)

> This module expands the planning concepts briefly introduced in Module 1.3 (ReAct, Plan-and-Execute, Reflexion) into full depth. Cross-references to other modules are explicit throughout — planning doesn't exist in isolation, it's the connective tissue between all agent capabilities.

---

## P.1 Task Decomposition

### Why decomposition matters
The single most common reason a capable model fails at a complex task is not a reasoning failure — it's a **working-memory/context problem**: the full task plus all its intermediate steps and relevant context eventually exceeds what can be effectively held in one coherent attention span, whether that's a human's short-term memory or an LLM's effective context utilization. A model that's excellent at 10-step problems often degrades noticeably on 50-step problems not because it became less capable, but because the problem outgrew the cognitive span it's effective within.

Decomposition solves this by dividing the problem into chunks that each fit comfortably within effective reasoning range, letting the model be excellent at each sub-task rather than mediocre across the whole undivided problem. This directly parallels how professional knowledge work actually happens — a software architect doesn't try to hold an entire codebase in working memory while writing a single function; a legal analyst doesn't try to hold an entire case file while drafting a single clause.

There's a secondary benefit worth naming explicitly: **sub-tasks have checkable intermediate outputs**. An undivided long task has one output at the very end, which either succeeded or failed. A decomposed task produces intermediate artifacts (a sub-plan, a research summary, a draft section) that can each be inspected and corrected early, rather than discovering a fundamental error only after the model has built 40 further steps on top of it — the same "validate close to the source" principle from Module 5.1's validation pipelines, applied to task execution rather than data cleaning.

### Types of decomposition
Not all tasks decompose the same way, and the right decomposition strategy depends on the task's structure:

- **Sequential decomposition** — sub-tasks that must run in a fixed order, each depending on the previous one's output. The simplest and most common pattern. `research → outline → draft → edit → finalize` is a sequential decomposition of "write a report."
- **Parallel decomposition** — genuinely independent sub-tasks that can run concurrently (Module 7.4). `research competitor A, research competitor B, research competitor C` — none depends on the others. Use when sub-tasks don't share dependencies and latency matters.
- **Hierarchical decomposition** — sub-tasks that are themselves complex enough to need their own further decomposition: a tree of tasks rather than a flat list. Matches the hierarchical multi-agent pattern from Module 7.1.
- **Conditional decomposition** — the specific sub-tasks to execute depend on an intermediate decision point: "if the research finds a patent conflict, the next sub-tasks are legal review steps; if it doesn't, the next sub-tasks are a standard commercialization plan." The plan branches rather than being fully enumerable upfront.
- **Iterative decomposition** — a task that loops: solve a draft → evaluate → improve → evaluate → improve until a quality bar is met. This is the decomposition pattern underlying Reflexion (P.3) and test-driven coding agents (Module 8.2).

| Type | Sub-task relationship | Best for |
|---|---|---|
| Sequential | Ordered, each depends on prior | Linear workflows, most writing tasks |
| Parallel | Independent, can run concurrently | Research across multiple sources, multi-domain analysis |
| Hierarchical | Tree of tasks within tasks | Large-scale, multi-layer complex problems |
| Conditional | Branch on intermediate results | Decision-dependent workflows with unknowable paths upfront |
| Iterative | Loop with quality gates | Refinement tasks: writing, code, design |

### Decomposition strategies
Several distinct prompting and architectural approaches for generating a decomposition:

**Least-to-Most Prompting** — explicitly ask the model to first identify and solve easier sub-problems, then use those answers as scaffolding for the harder ones. Introduced as a prompting technique specifically to improve compositional generalization: a model that would get a complex problem wrong in a single shot often gets it right when explicitly given an ordered sequence of easier stepping stones to the same answer.

**Top-Down Decomposition** — start from the high-level goal and progressively break it into sub-goals, then sub-sub-goals, until individual tasks are small enough to execute directly. Generates the full tree structure upfront, which makes it easy to visualize and approve the full scope before any execution begins — but assumes the tree can be fully specified upfront, which isn't always true for genuinely open-ended tasks.

**Bottom-Up Decomposition** — identify concrete, immediately-executable tasks first, then group them into higher-level phases. Works well when the implementer has deep domain knowledge of what concrete steps are needed but finds it harder to reason top-down about the abstract goal structure. Common in engineering contexts.

**Goal-Condition Decomposition** — for each sub-task, explicitly state what the world should look like when it's successfully complete (a "postcondition"), rather than just naming the task. This grounds each sub-task in a checkable outcome rather than an activity — directly enabling the verification loops from Module 8.2's test-driven coding agents.

### Granularity — how small is small enough?
Over-granular decomposition (too many tiny steps) creates as much problem as under-granular decomposition (sub-tasks still too large to execute reliably). The right granularity is a sub-task that:
- Can be executed in **one agent turn** with the information available at that point.
- Has an output that is **verifiable** — the executor can tell if it succeeded or failed without ambiguity.
- Is **atomic with respect to failure** — if it fails, the failure is contained to that one sub-task and doesn't silently corrupt 10 subsequent ones built on top of it.

A sub-task that fails the "verifiable" test is almost always the wrong unit — split it until success/failure is unambiguous, or define an explicit postcondition that makes it testable.

### Dependency mapping
After decomposition, the dependencies between sub-tasks must be explicitly mapped — this is not optional or implicit. Two fundamental dependency types:
- **Data dependency** — Sub-task B needs Sub-task A's output as its own input. A can't start until B is done.
- **Resource dependency** — Sub-tasks C and D both need the same limited resource (a shared tool with a rate limit, an exclusive lock on a file). They must be serialized or one will fail even though neither has a data dependency on the other.

Concretely mapping these dependencies drives the execution order (sequential, parallel, or conditional) and directly shapes which multi-agent topology from Module 7.1 is the right match — a fully sequential dependency chain suggests a supervisor pattern; genuinely independent sub-tasks with no cross-dependencies are the right candidates for parallel execution (Module 7.4).

### Failure modes specific to decomposition
- **Over-decomposing stable, well-understood tasks** — adding complexity and orchestration overhead to tasks that a single agent in one pass handles reliably; the same "premature multi-agent" anti-pattern from Module 7.6, applied at the decomposition level.
- **Under-decomposing genuinely complex tasks** — leaving sub-tasks that are themselves too large to execute reliably, just renamed rather than actually broken down; cosmetic decomposition that doesn't actually solve the working-memory problem.
- **Missing a cross-sub-task dependency** — starting a sub-task before a dependency it actually has is resolved, causing it to execute on stale or missing input. This is the multi-task version of Module 1.5's "error compounding" failure mode — an early, undetected missing dependency contaminates everything that follows it.
- **Front-loading all planning before any execution** — generating a full, fixed decomposition at the start and never revisiting it, even when early execution results reveal the decomposition was wrong. This is the failure mode Plan-and-Execute patterns specifically guard against with replanning capabilities (P.2).

### 📋 Interview Questions — P.1
1. **Why does decomposition help a capable model succeed on complex tasks it would otherwise fail, even without making the model itself more capable?**
   *Look for: decomposition puts each sub-task within the model's effective reasoning range — the failure on complex undivided tasks is primarily a working-memory/context-span problem, not a raw capability problem.*
2. **What's the difference between parallel and sequential decomposition, and how would you decide which applies to a given task?**
   *Look for: sequential = ordered with data/resource dependencies between sub-tasks; parallel = genuinely independent sub-tasks that can run concurrently — the decision turns entirely on whether dependencies exist, not on preference.*
3. **A candidate says "I decomposed the task into 15 sub-tasks before execution started." What's the risk of fully upfront decomposition without any replanning capability?**
   *Look for: early execution results may reveal the decomposition itself was wrong — without replanning, the system continues executing on a flawed plan rather than adapting; front-loading all planning assumes a predictability the task may not actually have.*
4. **What makes a sub-task "atomic with respect to failure," and why does that property matter for a multi-step agent?**
   *Look for: failure is contained to that one sub-task and doesn't silently corrupt subsequent steps built on top of it — without this property, a single early failure can cascade through the entire plan before anyone catches it.*
5. **Why is explicitly mapping data and resource dependencies between sub-tasks not optional, even when the dependency seems obvious?**
   *Look for: an unmapped dependency can cause a sub-task to start before it's actually ready (running on missing/stale input), which is Module 1.5's error-compounding failure mode at the plan level — the "obvious" dependency that nobody documented is exactly the one that gets missed under deadline pressure.*
6. **How does goal-condition decomposition differ from simply listing tasks, and what specifically does it enable that a task-list doesn't?**
   *Look for: postconditions make sub-task success explicitly verifiable — you can check whether the world matches the stated condition, rather than just whether the activity was attempted; this is what enables test-driven verification loops like Module 8.2's coding agents.*

---

## P.2 Planner-Executor

### The core separation of concerns
The Planner-Executor pattern applies a fundamental software-engineering principle — **separation of concerns** — to the agent architecture: one component reasons at the *strategic* level (what needs to happen, in what order, and why), while a separate component operates at the *tactical* level (actually doing each step, using tools, observing results, handling the noise of real-world execution). Neither component tries to do both simultaneously.

This separation is distinct from just writing a plan before acting. In a plain ReAct loop (Module 1.3), the model simultaneously maintains the high-level goal, the current execution state, the list of tools available, and the specific action to take — all in one attention pass. In a Planner-Executor split, the Planner forms a clear strategy without the cognitive overhead of also managing tool-call arguments and execution errors; the Executor manages the messy details of step-by-step execution without re-deriving the high-level strategy on every single iteration.

The analogy that makes this concrete: **an architect (Planner) produces a blueprint for what to build; a construction team (Executor) actually builds it, handling the thousand practical complications of turning a blueprint into a real structure.** The architect doesn't re-derive the building's purpose every time the construction team hits an unexpected underground pipe; the construction team doesn't redesign the building's layout when they need to reroute that pipe.

### How the planner works
The Planner's job is to produce an explicit, structured plan from the task goal — a sequence of steps, with enough detail that an Executor can carry out each one without needing to re-infer the overall intent from scratch. Good planning output explicitly captures:
- **What** needs to happen at each step (the action/sub-task).
- **Why** this step is needed (how it contributes to the overall goal) — this is often skipped but matters enormously for replanning, when the Executor's results reveal that the "why" of a step has been satisfied some other way.
- **Success criteria** for each step (what a good result looks like), enabling the verification loops from Module 8.2 and P.1's goal-condition decomposition.
- **Dependency ordering** between steps (which steps must precede others and why).

The Planner typically operates at a **higher abstraction level** than the Executor — it doesn't need to know the exact API endpoint or parameter format for a tool call; it just needs to know that "retrieve the latest sales data from the CRM" is the right next step. The Executor resolves those implementation details.

### How the executor works
The Executor takes one step from the plan at a time, carries it out using available tools (exactly the Module 2 function-calling loop), observes the result, and reports back to the Planner with either a success confirmation or a failure description (including *why* it failed and what was actually returned). It applies all the low-level reliability patterns from Module 9 — circuit breakers, validation, retry-with-feedback — within each individual step, without needing to involve the Planner in those implementation details unless a step genuinely cannot be completed by any available means.

Critically, the Executor **should not itself make strategic decisions** — deciding whether a step is still needed, whether the plan should change, or whether to skip a step because it seems unnecessary. Those decisions belong to the Planner. This sounds obvious but is a common implementation failure: an Executor that silently decides to skip or modify a step is reintroducing the mixed-concerns problem the pattern was designed to solve, in a way that makes the system harder to debug because the plan-vs-execution divergence is now invisible.

### Plan revision and replanning
A rigid, fixed plan that never updates on feedback is the classic failure mode of purely top-down decomposition — execution results frequently reveal that the plan was based on incorrect assumptions, and a system with no replanning capability has to either proceed on a wrong plan or stop entirely. Replanning is what makes the Planner-Executor pattern adaptive rather than just structured:

After each step (or batch of steps), the Executor feeds a summary of actual results back to the Planner. The Planner evaluates whether the plan's remaining steps are still valid given what was actually learned. When a significant discrepancy is found — the data the plan assumed would be available isn't, an assumption was wrong, or a step produced a result that satisfies the next three planned steps' goals simultaneously — the Planner revises the remaining plan rather than continuing to execute it literally.

This creates a genuine **feedback loop between strategy and execution**, the same Reason-Act-Observe structure as the base ReAct loop (Module 1.3), but operating at the *planning level* rather than the *step level* — meta-level adaptation overlaid on top of step-level adaptation.

```
Goal
 │
 ▼
┌──────────┐   full plan (or revised plan)   ┌──────────────┐
│  PLANNER │ ──────────────────────────────▶ │   EXECUTOR   │
│          │ ◀────────────────────────────── │              │
└──────────┘   step result / failure report  └──────────────┘
                                                     │
                                               tools / world
```

### Human-in-the-loop at the planning layer
The Planner-Executor split creates a natural, clean human review point that a pure ReAct loop doesn't: **pause after the Planner produces a plan, before the Executor does anything**. This is where a human can review the full intended sequence, spot a strategic mistake before it's acted on, and approve, reject, or modify the plan — a significantly cheaper intervention than catching and recovering from a mistake after the Executor has already taken real-world actions. This directly implements Module 1.6's approval-workflow pattern at the *plan* level rather than the *individual action* level: reviewing and approving a plan of 10 steps is dramatically more efficient than individually approving each of the 10 steps as they execute, while providing most of the same safety assurance if the plan is actually well-specified.

### When Planner-Executor outperforms plain ReAct
Plain ReAct is the simpler default — don't add architecture complexity without a reason. The Planner-Executor pattern earns its overhead when:
- **The task has many steps** — the single-pass attention overhead of keeping the whole strategy plus current execution state in mind degrades significantly as step count grows; separating planning from execution avoids this degradation.
- **The plan benefits from human review before execution** — when the stakes of a wrong strategy are high, pausing after the Planner step and before the Executor step is a valuable safety gate without needing to review every individual tool call.
- **Replanning is expected** — if execution results are likely to invalidate assumptions the plan was built on (research-dependent tasks, tasks in uncertain environments), the structured Planner-Executor loop manages that feedback more cleanly than an ad hoc "sometimes step back and rethink" policy inside a plain ReAct loop.
- **Step-level efficiency matters** — the Executor can use a lighter model than the Planner (Module 9.4's model routing) precisely because individual steps are narrower, well-scoped execution tasks rather than both high-level reasoning and low-level tool use at the same time.

### 📋 Interview Questions — P.2
1. **What's the core problem the Planner-Executor separation solves that plain ReAct doesn't fully address?**
   *Look for: the mixed cognitive load of maintaining strategic context plus detailed execution state in a single attention pass — separating them lets each component be optimized for its own concern, and both become more reliable.*
2. **Why is "the Executor silently decides to skip a step that seems unnecessary" a serious architecture failure, even if the skip turns out to be correct?**
   *Look for: it reintroduces the mixed-concerns problem invisibly — the plan-vs-execution divergence is now hidden, making it dramatically harder to debug when something later goes wrong because the recorded plan no longer matches what actually ran.*
3. **How does pausing for human review between the Planner and Executor steps differ from, and compare to, approving each individual tool call in a pure ReAct loop?**
   *Look for: plan-level review is far more efficient (review 10 steps in one pass vs. reviewing each individually) and catches strategic errors earlier, while individual-action approval provides finer-grained control but creates much more friction and doesn't necessarily catch strategic errors until they've already partially executed.*
4. **What triggers a replanning event, and what specifically changes when replanning happens?**
   *Look for: significant discrepancy between what execution actually returned and what the plan assumed — not small surprises, but results that genuinely invalidate a remaining planned step's premise; replanning revises the Planner's remaining plan, not just the current step's retry.*
5. **Why can the Executor often use a cheaper, smaller model than the Planner, specifically because of the Planner-Executor split?**
   *Look for: individual Executor steps are narrow, well-scoped execution tasks (call this tool with these arguments) — they don't require the high-level strategic reasoning that justifies a frontier model; the split enables per-role model routing (Module 9.4) that a mixed-role agent can't cleanly apply.*
6. **A team is building a 5-step agent for a stable, well-understood workflow that runs the same way every time. Would you recommend Planner-Executor over plain ReAct?**
   *Look for: probably not — the overhead of a Planner-Executor split isn't justified for a short, stable, fully-known workflow; the pattern earns its complexity for longer, more uncertain, or human-reviewable tasks.*

---

## P.3 Reflection

### What reflection means in agent systems
In agent systems, "reflection" refers to a model explicitly **reasoning about its own outputs, reasoning, or trajectory** — treating its own prior output as an object to examine and critique, rather than blindly accepting it and moving forward. This is in deliberate contrast to a standard generator that produces output and stops. The motivating insight: the same model that generated a poor first attempt often has enough latent capability to *identify* that the attempt was poor and *articulate specifically why* — if given a structured opportunity to do so, rather than immediately being pushed to the next turn.

Reflection operates at multiple timescales and levels of granularity:
- **Step-level reflection** — pausing after each individual action and asking "was that a good action, and did it get me closer to the goal?"
- **Trajectory-level reflection** — after a full attempt (or a multi-step segment), reviewing the entire sequence: "looking at what I tried, what went wrong, and what should I do differently on the next attempt?"
- **Meta-cognitive reflection** — evaluating the reasoning *process* rather than specific outputs: "am I approaching this problem the right way, or should I reframe the task itself?"

### Reflexion — the formal pattern
Reflexion (Shinn et al., 2023) is the canonical, formalized version of trajectory-level reflection. The structure:

```
Attempt 1
   │
   ▼
Evaluation (did it succeed? external signal: test pass/fail, task success, human rating)
   │
   ▼
Verbal Reflection (LLM produces a natural-language critique: "I failed because X; next time I should Y")
   │
   ▼
Episodic Memory (the critique is stored explicitly — not just in the conversation history but as a named, retrievable "lesson")
   │
   ▼
Attempt 2 (conditioned on the stored lesson from the critique)
   │
   ▼
... repeat until success or budget exhausted
```

Three components distinguish Reflexion from a simple retry loop:
1. **The critique is explicit and verbal** — the model articulates *specifically* what went wrong, not just "it failed." This is what conditions the next attempt on something meaningfully different.
2. **The critique is stored as episodic memory** (Module 1.4) — explicitly persisted and retrieved into the next attempt's context, so it isn't just a vague "previous attempt summary" but a specific, named lesson.
3. **The evaluation signal drives the critique** — Reflexion works best when there's a concrete, preferably automated success signal (tests pass or fail, a task is completed or not) rather than pure self-assessment, because a model evaluating its own output without external ground truth tends to overrate it.

This directly generalizes Module 8.2's test-driven coding agent loop — Reflexion is essentially "test-driven development as a cognitive pattern applied to any agent task with a checkable outcome," not just code.

### Self-critique vs external critique
Reflection sources differ meaningfully in reliability:

**Self-critique** (the model critiques its own output) has a fundamental limitation: the same model that produced a flawed output is often subject to the same biases and blind spots when evaluating it. Self-critique is most useful for catching *surface-level* errors (missing information, structural problems, factual inconsistencies visible from the text itself) and for tasks where the model has strong priors about what "good" looks like. It's less reliable for catching *reasoning* errors (a flawed inference chain that reaches a confident but wrong conclusion) and for tasks outside the model's domain of strong calibration.

**External critique** — feedback from a separate, different model (a "critic" LLM), automated tests, a human reviewer, or an external validation service — provides a genuinely independent perspective that isn't subject to the generator's own blind spots. Where it's feasible, external critique produces more reliable correction signals than self-critique. The cost is the additional inference call (a separate critic model), or the latency of an automated test suite, or the throughput bottleneck of a human reviewer — each justified at different levels of task stakes and error-tolerance.

The practical pattern many production systems converge on: **self-critique as the fast, cheap first pass** (the model flags obvious issues without an additional inference call), **external tests/validators as the ground-truth check** for any outcome that has a checkable correct answer, and **human review** reserved for high-stakes, ambiguous, or non-computable quality judgments.

### Meta-cognitive reflection vs step-level reflection
These operate at different levels and serve different functions:

**Step-level reflection** ("was that tool call the right one, given what I know now?") is the embedded self-check after each individual action — essentially the "Observe" phase of the ReAct loop (Module 1.3), but with an explicit evaluative judgment rather than just passively receiving the tool output. It catches local, recoverable errors before they compound into the next step.

**Meta-cognitive reflection** ("is my overall approach to this problem the right one, or am I solving the wrong version of the task?") operates at a higher level of abstraction and is the more powerful but rarer form. A model can execute a plan perfectly at the step level and still fail catastrophically if the plan itself was a mis-framing of the problem. Meta-cognitive reflection creates the opportunity to catch that mis-framing before the execution is complete. Practically, it's most valuable:
- At explicit "checkpoint" moments in a long task (between major phases, not after every step).
- When execution results consistently don't match expectations (a repeated pattern of wrong or surprising results often signals a meta-level framing problem, not just a series of unrelated execution failures).
- As a pre-execution step for high-stakes or novel tasks: "before I start executing this plan, let me reason about whether this framing of the task is correct."

### Reflection in multi-agent systems
Reflection doesn't only apply to a single agent reflecting on its own output — it can be **architected as a dedicated role** in a multi-agent system. A "Critic" agent (Module 3.4's CrewAI naming for this role, though the pattern appears across frameworks) specializes specifically in evaluating other agents' outputs, applying focused critical attention that a generalist agent spreading attention across generation, planning, and evaluation simultaneously rarely achieves.

A common production pattern: a **Generator agent** produces a draft, a **Critic agent** evaluates it against defined criteria and produces a structured critique, and that critique is fed back to the Generator (or a separate Reviser agent) for the next pass — repeating until the Critic's evaluation exceeds a quality threshold or a budget is exhausted. This is Reflexion's episodic-memory-and-retry loop implemented as a multi-agent interaction rather than a single agent's internal loop, which distributes the cognitive load and avoids the self-evaluation blind spot problem.

### Failure modes of reflection itself
Reflection is not free of its own failure modes — naively applying it introduces new problems:
- **Reflection inflation / sycophantic critique** — a self-critiquing model tends to rate its own outputs better than they deserve, especially under pressure to "move forward." The fix: external ground truth (test results, a separate critic model) rather than relying on the generator's self-assessment as the primary evaluation signal.
- **Infinite reflection loops** — adding reflection without a convergence condition or budget creates a loop that never terminates ("this could always be slightly better"). The fix is the same as for any agent loop: an explicit stopping criterion, whether that's a quality threshold, a max-iteration count, or a time budget.
- **Superficial critique that doesn't change the underlying approach** — a reflection that says "I should be more thorough" without identifying a specific, actionable change produces a next attempt that's essentially the same as the first with slightly different phrasing. The fix: require the critique to be specific (what exactly was wrong, what exactly should change) and check whether the next attempt actually demonstrates the stated change.
- **Hallucinated self-improvement** — a model claiming in its critique that the next attempt will be better, without actually changing anything substantive, then producing a next attempt that's nearly identical. The fix: compare the critique's stated intended changes against the diff between the first and second attempts, rather than accepting the critique's self-report at face value.

### 📋 Interview Questions — P.3
1. **What distinguishes Reflexion from a plain "retry on failure" loop, and why does that distinction determine whether the retry is actually likely to help?**
   *Look for: Reflexion produces an explicit, stored verbal critique of specifically what went wrong; a plain retry repeats the same attempt with the same priors — without a specific diagnosis of what to change, there's no reason to expect a different outcome.*
2. **Why does Reflexion work best with an external evaluation signal rather than pure self-assessment?**
   *Look for: the same model that generated a flawed output is often subject to the same blind spots when evaluating it — external signals (test pass/fail, a separate critic model, human rating) provide a genuinely independent perspective the generator can't fully provide for itself.*
3. **What's the difference between step-level and meta-cognitive reflection, and when would you specifically want to trigger the latter?**
   *Look for: step-level = local check on individual actions; meta-cognitive = evaluating whether the overall approach/framing is correct — trigger meta-cognitive at major phase boundaries, when results consistently surprise, or before executing a high-stakes plan.*
4. **A candidate says "we added reflection to the agent and now it's always critiquing itself before moving forward." What failure mode should you immediately probe for?**
   *Look for: infinite reflection loops without a convergence condition or budget, and superficial critique that doesn't actually change the subsequent attempt — both are common when reflection is added without explicit stopping criteria.*
5. **How does architecting reflection as a dedicated "Critic" agent role in a multi-agent system address the self-evaluation blind spot problem?**
   *Look for: a separate Critic agent applies focused critical attention that isn't subject to the Generator's own biases/blind spots — the generator isn't evaluating its own output; a genuinely independent agent with a different context and focus is.*
6. **How would you detect "hallucinated self-improvement" — a model claiming it will improve without actually changing anything — in a Reflexion loop?**
   *Look for: compare the critique's stated specific intended changes against the concrete diff between Attempt N and Attempt N+1, rather than accepting the model's self-report that it improved; if the diff doesn't reflect the stated changes, the critique was cosmetic.*

---

*End of Planning Module detailed notes — 18 interview questions total across 3 sections. These topics were briefly introduced in Module 1.3 of the main curriculum — these notes provide the full depth those introductory mentions pointed toward.*
