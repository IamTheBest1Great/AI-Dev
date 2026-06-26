# Module 12 — Agent UX & Design Patterns: Detailed Notes 🆕

## Table of Contents

- [12.1 Streaming vs Batching Output](#121-streaming-vs-batching-output)
  - [The core decision](#the-core-decision)
  - [Streaming patterns and their tradeoffs](#streaming-patterns-and-their-tradeoffs)
  - [Batching — when it's the right call](#batching--when-its-the-right-call)
  - [The hybrid: progressive disclosure](#the-hybrid-progressive-disclosure)
  - [📋 Interview Questions — 12.1](#-interview-questions--121)
- [12.2 Showing vs Hiding Intermediate Reasoning](#122-showing-vs-hiding-intermediate-reasoning)
  - [The scratchpad problem](#the-scratchpad-problem)
  - [The case for showing steps](#the-case-for-showing-steps)
  - [The case for hiding steps](#the-case-for-hiding-steps)
  - [Planning visibility — the emerging consensus](#planning-visibility--the-emerging-consensus)
  - [Explainability on demand](#explainability-on-demand)
  - [📋 Interview Questions — 12.2](#-interview-questions--122)
- [12.3 Interruptibility and Steerability Mid-Task](#123-interruptibility-and-steerability-mid-task)
  - [Why interruptibility is a design problem, not just an engineering one](#why-interruptibility-is-a-design-problem-not-just-an-engineering-one)
  - [Variable autonomy levels](#variable-autonomy-levels)
  - [Mid-task steering mechanisms](#mid-task-steering-mechanisms)
  - [Progressive delegation](#progressive-delegation)
  - [📋 Interview Questions — 12.3](#-interview-questions--123)
- [12.4 Designing for Trust](#124-designing-for-trust)
  - [Why trust is the primary design constraint](#why-trust-is-the-primary-design-constraint)
  - [Transparency signals](#transparency-signals)
  - [Confidence signals](#confidence-signals)
  - [Citations and source attribution](#citations-and-source-attribution)
  - [Memory surfacing](#memory-surfacing)
  - [📋 Interview Questions — 12.4](#-interview-questions--124)
- [12.5 Error Recovery UX](#125-error-recovery-ux)
  - [Why error recovery UX is structurally different for agents](#why-error-recovery-ux-is-structurally-different-for-agents)
  - [Graceful degradation](#graceful-degradation)
  - [Fast correction loops](#fast-correction-loops)
  - [Clear handoff to a human](#clear-handoff-to-a-human)
  - [The seven-point trust checklist](#the-seven-point-trust-checklist)
  - [📋 Interview Questions — 12.5](#-interview-questions--125)

> Most AI feature failures in 2026 are not model failures — they are design failures. The models are good enough. What they need is a user interface that signals confidence honestly, handles failure gracefully, gives the user control, and respects their time and data. This module is where everything built in Modules 1–11 either earns user trust or loses it. The models are good enough. What they need is a user interface that signals confidence honestly, handles failure gracefully, gives the user control, and respects their time and data. This module is where everything built in Modules 1–11 either earns user trust or loses it.

---

## 12.1 Streaming vs Batching Output

### The core decision
How an agent delivers its output is the first and most visible UX decision in the entire stack — it shapes the user's perception of speed, effort, and reliability before they've evaluated a single word of the actual result. Two fundamental models:

**Streaming** delivers output progressively as it's generated — tokens appear on screen as the model produces them, intermediate steps surface as they complete, and the user can begin reading and reacting before the full task is done. **Batching** withholds all output until the task is entirely complete, then presents the final result as a single, polished unit.

Neither is universally correct. The choice depends on task duration, output type, the user's role (active collaborator vs. passive recipient), and whether intermediate states are meaningful or noisy to the specific person receiving them.

### Streaming patterns and their tradeoffs
**Token-level streaming** (the classic chat-interface pattern: words appear one by one as the model generates them) is effective for text-generation tasks because it makes long responses feel faster than their actual latency — the user starts reading while generation continues. Its ceiling: it's only useful when the output is text that makes sense to read sequentially. A streaming JSON object that's half-formed, or a streaming code block that doesn't compile until complete, gives the user something to look at but nothing they can act on.

**Step-level streaming** (showing each completed tool call or agent step as it finishes, rather than individual tokens within a step) is the right pattern for multi-step agent tasks. The user sees: "Searching the web... ✓  Analyzing results... ✓  Drafting the report..." rather than a blank screen for 90 seconds. This is the pattern NNGroup identified in their State of UX 2026 report as "multi-step workflow tracking" — one of five interface decisions that every enterprise agent needs to get right. It surfaces the agent's progress trajectory without showing the raw model-level scratchpad detail (see 12.2).

**Technical considerations**: streaming requires either SSE (server-sent events) or WebSocket connections rather than standard HTTP request-response; streaming architectures are harder to cache at the CDN layer; and streaming LLM output specifically must handle the challenge that many output formats (JSON, structured tables, code blocks) aren't parseable until complete, requiring the client to buffer and render incrementally rather than waiting for a parse-complete signal.

### Batching — when it's the right call
Batching is the right choice when:
- **The intermediate state is meaningless or confusing.** A half-generated data table with missing columns isn't helpful — it requires the user to hold a "this isn't done yet" mental model for a result format that expects them to evaluate columns in parallel. Better to batch the whole table and present it complete.
- **The task runs in the background.** An asynchronous job (Module 11.1) that the user kicks off and walks away from has no audience for streaming output — deliver the final result when it's ready.
- **The output has high editing cost if interrupted.** Some deliverables (a formal document, a production code PR) are better received as complete, reviewed artifacts rather than as works-in-progress the user might react to prematurely — the same logic as reviewing a completed plan before execution begins (Module P.2's human-in-the-loop at the planning layer).

### The hybrid: progressive disclosure
The most effective production pattern for complex, multi-step agents is neither pure streaming nor pure batching — it's **progressive disclosure**: stream high-level status (which step is running, when each completes, how long the task is taking overall) while batching the detailed output of each step until that step is fully complete before surfacing it to the user.

The user gets the perceptual benefit of streaming (a sense of active progress, something meaningful to watch) without the cognitive cost of managing half-formed intermediate outputs. This is the same principle from Module 12.2's "planning visibility" concept — the user sees what the agent is doing at the right level of abstraction, not at the raw implementation detail level.

### 📋 Interview Questions — 12.1
1. **Why is token-level streaming (characters appearing one at a time) not always the right default for a multi-step agent, even though it feels fast?**
   *Look for: individual tokens from an agent's reasoning or a half-formed tool result aren't meaningful to act on — step-level streaming (each completed step surfaced as a unit) is more useful for multi-step tasks where the meaningful unit is a completed step, not an individual token.*
2. **What specific technical challenges does streaming output introduce that batched responses don't have?**
   *Look for: requires SSE or WebSocket (not standard HTTP request-response), CDN caching is harder, and many structured output formats (JSON, tables, code) aren't parseable until complete — requiring the client to handle incremental rendering or buffering explicitly.*
3. **A user is running a long-horizon research agent that takes 8 minutes. Should the output stream token by token, step by step, or batch entirely? Defend your answer.**
   *Look for: step-level streaming is almost certainly right — 8 minutes of blank screen destroys the user experience, but raw token streaming of reasoning notes creates noise; showing completed steps (search done, analysis done, draft section done) maintains engagement at the right granularity.*
4. **What's the user-experience argument for batching a generated data table rather than streaming its rows as they're produced?**
   *Look for: a half-formed table with missing columns requires the user to suppress their natural instinct to evaluate what they're seeing — it imposes a "don't read this yet" cognitive tax that batching eliminates; tables are evaluated as complete structures, not as sequential streams.*
5. **Describe the "progressive disclosure" hybrid pattern and explain what problem it solves that neither pure streaming nor pure batching solves alone.**
   *Look for: streams high-level status (which step is active) while batching each step's detailed output until it's complete — gives the perceptual benefit of streaming (sense of active progress) without the cognitive cost of managing half-formed intermediate results.*

---

## 12.2 Showing vs Hiding Intermediate Reasoning

### The scratchpad problem
The model's "scratchpad" — the chain-of-thought, the tool-call parameters it considered, the intermediate reasoning it worked through — is an engineering artifact, not a user feature. Dumping raw scratchpad content onto a user interface is a design failure that disguises itself as transparency: it gives users access to information they can't act on, in a format they weren't trained to interpret, at a cognitive cost that trades against actually using the final output. This is the reasoning equivalent of showing a user a compiler's intermediate representation instead of the program output.

The design question is not "should we show reasoning?" but "**which level of reasoning, abstracted to what degree, to which users, at what moment in the task?**" — four dimensions that require explicit design decisions, not a single binary toggle.

### The case for showing steps
Planning visibility means the user sees the agent's intended action sequence before execution begins. This earns two distinct benefits: it gives the user a chance to catch a misunderstood goal *before* the agent acts on it, and it builds the kind of observable, predictable behavior that trust requires over time. When an agent acts — especially without being asked — users immediately want to know why. If they have to go looking for the answer, trust erodes fast. The fix is simple: show the reasoning, briefly, right next to the action.

"Briefly, right next to the action" is the operative phrase: the well-designed pattern shows **a summary of the reasoning** (one or two sentences: "I'm searching for recent pricing data because your question referenced 2026 numbers") co-located with the action it motivated, not the full internal chain-of-thought. The reasoning is **anchored to the decision it explains**, not floating in a separate reasoning pane the user has to correlate manually with what actually happened.

Intermediate steps belong on screen when: the user is a **power user or developer** evaluating agent behavior; the task is **high-stakes** (an action with real-world consequences that the user might want to stop); or the user is **actively collaborating** on a multi-step task where their feedback at intermediate stages would improve the outcome.

### The case for hiding steps
For end users in consumer or enterprise-productivity products, raw intermediate reasoning is almost always more hindrance than help:
- It **creates false evaluation burden** — the user feels they should review each step, even when they lack the context to evaluate whether it was correct.
- It **inflates perceived complexity** — an agent that shows 15 intermediate steps before answering a straightforward question feels unreliable rather than thorough.
- It **front-loads cognitive load** at the worst moment — the user is waiting for the answer, not seeking a reasoning tutorial.

Users can request explainability on demand — this is the design pattern that resolves the tension: hide intermediate reasoning by default, but surface it immediately and completely when the user asks "how did you get this?" or "why did you do that?" This is "explainability on demand" — transparency available when wanted, invisible when not.

### Planning visibility — the emerging consensus
Agent UX patterns are the interface decisions that turn raw agent architecture into something a user can supervise. Five patterns apply to every enterprise agent regardless of model or framework: planning visibility, tool-use disclosure, memory surfacing, multi-step workflow tracking, and recovery routing.

"Planning visibility" specifically means showing the agent's intended *action plan* (what it plans to do, in what order, toward what goal) before execution — not the raw reasoning behind the plan. This is exactly the human-in-the-loop checkpoint from Module P.2 and Module 1.6, expressed as a UX pattern: the user reviews and can modify the plan before the agent executes it, at a level of abstraction that makes the plan evaluable (a list of intended actions) rather than opaque (raw chain-of-thought) or overwhelming (every tool-call argument).

### Explainability on demand
The concrete implementation:
- **Default state**: the agent shows only the completed output, with a minimal status trail ("Searched 4 sources, analyzed 12 results, summarized findings").
- **On-demand detail**: an expandable reasoning section, a "how did you get this?" button, or a step-by-step trace view that opens when the user requests it — surfaces the full reasoning trail without imposing it on users who didn't ask.
- **For flagged outputs**: when the agent has low confidence or made an assumption the user might disagree with, proactively surfacing the relevant part of the reasoning right at that specific point in the output (not in a separate pane) — targeted rather than wholesale transparency.

### 📋 Interview Questions — 12.2
1. **Why is dumping raw chain-of-thought onto the UI a design failure even though it feels like transparency?**
   *Look for: raw scratchpad is information users can't act on, in a format they weren't trained to interpret, at a cognitive cost that trades against actually using the final output — it's the engineering artifact, not a user feature; transparency at the wrong level of abstraction doesn't serve the user.*
2. **What's the difference between planning visibility and showing raw reasoning, and why does that distinction matter for user trust?**
   *Look for: planning visibility shows the intended action sequence at a level the user can evaluate and approve (a list of steps); raw reasoning shows the model's internal derivation — the former is evaluable by the user, the latter usually isn't; only the former builds trust through oversight.*
3. **How does "explainability on demand" resolve the tension between hiding intermediate reasoning and being transparent about it?**
   *Look for: hides reasoning by default (reducing cognitive load for most users) while surfacing it immediately and completely when explicitly requested — transparency becomes available when wanted rather than always imposed; users who want oversight can get it, users who don't aren't burdened by it.*
4. **For which specific user types and task types would you show intermediate steps by default rather than on demand?**
   *Look for: power users/developers evaluating agent behavior, high-stakes tasks with real-world consequences where users might want to stop mid-execution, and active collaborative tasks where user feedback at intermediate stages would improve the outcome.*
5. **A product manager proposes showing all 15 intermediate steps for every agent response to "make the agent feel more thorough." What would you push back on?**
   *Look for: this creates false evaluation burden (users feel they should review each step), inflates perceived complexity (15 steps before a simple answer feels unreliable, not thorough), and front-loads cognitive load exactly when the user is waiting for a result — thoroughness should be available on demand, not imposed by default.*

---

## 12.3 Interruptibility and Steerability Mid-Task

### Why interruptibility is a design problem, not just an engineering one
An agent that can be technically interrupted (a stop button exists in the API) but that leaves the user uncertain about what state the world is in after interruption has failed at the UX level regardless of the engineering success. When an agent is interrupted mid-task, the user immediately needs to know: what actions were already taken? What state did those leave the world in? Is it safe to retry from the beginning, or would retrying duplicate a side effect? What needs to be manually cleaned up?

Interruptibility done well means these questions are answered proactively, at the moment of interruption, not left for the user to investigate. A stop button without a "here's what had already happened before you stopped" disclosure is half an implementation of interruptibility.

### Variable autonomy levels
Autonomy shouldn't be binary. In 2026, users expect variable control levels when designing for AI agents. The spectrum:

- **Full supervision** — the agent proposes every action before taking it, requiring explicit approval for each step. Maximum control, minimum throughput; right for high-stakes or unfamiliar task types.
- **Supervised autonomy** — the agent executes routine actions independently but pauses for approval at defined checkpoints (irreversible actions, budget thresholds, scope boundaries). The Module 1.6 pattern expressed as a user-configurable preference.
- **Full autonomy** — the agent completes the entire task without interruption, surfacing only the final result. Minimum friction, maximum throughput; right for high-volume routine tasks where the user has established confidence through repeated supervised runs.

The design principle: **autonomy level should be configurable by the user**, not fixed by the product. A user who's run the same task 50 times successfully wants full autonomy; a user running it for the first time wants full supervision. Designing a single fixed autonomy level for all users of a task is designing for the wrong user on one end or the other.

### Mid-task steering mechanisms
Beyond a simple stop button, genuinely steerable agents offer:
- **Redirect** — "keep going, but focus on X instead of Y" without stopping and restarting from scratch. Requires the agent to update its remaining plan (Module P.2's replanning) rather than either continuing unchanged or abandoning all progress.
- **Scope adjustment** — "stop after this section" or "include the financial analysis too" — expanding or contracting the task scope mid-run without canceling.
- **Priority change** — "do the legal review first before the financial analysis" — reordering remaining steps without discarding the work already done.

Each of these is a different kind of mid-task intervention, and each requires different underlying support in the agent's architecture (primarily the Planner-Executor pattern from Module P.2 — you can redirect, scope, and reprioritize because there's an explicit, revisable plan, not because the agent is replanning from scratch from a raw goal).

### Progressive delegation
Progressive delegation directly addresses [user trust] by letting the user's own approval history set the pace of autonomy expansion. The system earns permission through demonstrated reliability rather than demanding it at launch.

This is the UX pattern that connects autonomy level to observed track record rather than to a fixed product decision:
- A new user starts at supervised autonomy.
- Each successfully completed task with no corrections needed is recorded.
- After N consecutive successful completions (configurable threshold), the system suggests raising the autonomy level for that specific task type — "You've approved 10 consecutive email drafts without changes. Would you like me to send future drafts automatically?"
- Trust earned in one task type doesn't automatically transfer to a different task type — a user who's given full autonomy on email drafting hasn't thereby given it on calendar changes.

This is the agent-UX expression of the broader principle from Module 9.3's least-privilege design: start with minimum necessary permissions and earn additional scope through demonstrated reliability, not by requesting it all upfront.

### 📋 Interview Questions — 12.3
1. **Why does a technically working stop button still fail at the UX level if it doesn't answer "what state is the world in now"?**
   *Look for: interruption leaves the user needing to know what already executed, what state those actions left things in, whether retrying from the beginning is safe or would duplicate side effects, and what needs manual cleanup — an unsupported stop button outsources all of this investigation to the user.*
2. **What's the principle behind "variable autonomy levels," and why does designing a single fixed autonomy level for all users fail?**
   *Look for: different users have different established trust and different task familiarity — a first-time user needs full supervision; a veteran of 50 successful runs wants full autonomy; a single fixed level serves the wrong user on one end or the other.*
3. **How does the Planner-Executor pattern (Module P.2) enable redirect and scope-adjustment mid-task in a way a pure ReAct loop doesn't?**
   *Look for: a pure ReAct loop has no explicit, revisable plan to adjust — interrupting it means either continuing unchanged or restarting from scratch; a Planner-Executor split has an explicit remaining-plan that can be revised in place (redirect, reorder, scope-adjust) without discarding completed steps.*
4. **What's "progressive delegation" and how does it differ from simply offering an autonomy-level setting the user can toggle?**
   *Look for: progressive delegation ties autonomy expansion to the user's own observed approval history — the system earns permission through demonstrated reliability on specific task types rather than requiring a user to proactively configure a setting; it's trust earned from the system's behavior, not trust assumed and offered as a toggle.*
5. **A user is running a 20-step research agent and at step 12 says "actually, focus on the European market only." What does the agent need to do, and why can't it just continue from step 13 unchanged?**
   *Look for: the agent needs to replan the remaining steps (13–20) against the new scope, not just continue executing a plan designed for a broader scope — continuing unchanged would produce steps that research markets outside Europe despite the redirect, wasting effort and producing the wrong output; replanning updates the remaining plan in place.*

---

## 12.4 Designing for Trust

### Why trust is the primary design constraint
NNGroup's State of UX 2026 report identifies trust as a major design challenge for AI experiences, noting that users burned by premature AI features resist adopting new ones. This has a concrete behavioral implication: **one bad experience with an AI feature is disproportionately destructive to trust** compared to one good experience being constructive of it — the same asymmetry that applies to brand trust in general applies here at product scale. Designing an agent that occasionally does something unexpected and gives the user no explanation, no recourse, and no understanding of what happened is not a minor UX debt — it's a trust-destroying event that can make users refuse to re-engage with the feature permanently.

Trust in agentic AI is built over time through observable outcomes and responsive learning. UX should provide feedback that shows how the AI is performing, what it has learned, and how user input influences future behavior.

### Transparency signals
Transparency signals communicate *what the agent is doing* — not just that it's doing something:
- **Tool-use disclosure** — surfacing which external tools, APIs, or data sources the agent accessed for a given response, close to the point in the output where that information was used. "Based on a search of your company's Confluence space from this morning" is more trustworthy than "Based on my knowledge."
- **Scope communication** — what the agent did and didn't look at. "I reviewed your last 90 days of email but not your calendar" sets accurate expectations and prevents users from drawing false conclusions from an answer that silently had limited input scope.
- **Action disclosure before execution** (planning visibility from 12.2) — for irreversible or consequential actions, showing the specific action and its parameters before executing it, not just the category of action.

### Confidence signals
Confidence signals communicate *how certain the agent is* — and, crucially, they must be **calibrated**, not decorative:
- An uncalibrated confidence signal (one that says "high confidence" regardless of actual uncertainty) is worse than no confidence signal, because it trains the user to ignore it. The first time a "high confidence" result is demonstrably wrong, the signal loses all credibility.
- Useful confidence signals distinguish between different sources of uncertainty: uncertain because the evidence is contradictory; uncertain because the data is recent and may not reflect the latest state; uncertain because this is outside the agent's reliable domain; uncertain because the user's question was ambiguous.
- A colour transition from amber to green as a confidence score updates communicates improving certainty — micro-animations (100–300ms, purposeful, used sparingly) are one practical way to communicate confidence state changes without requiring the user to re-read a confidence label.

### Citations and source attribution
Citations serve a dual purpose in agent UX: they're a quality signal (the agent retrieved real, specific sources rather than generating from parametric memory) and a trust mechanism (the user can verify the claim independently, rather than having to trust the agent on faith). Design principles for citations that actually serve these purposes:
- **Inline rather than footnoted** — a citation at the end of a 500-word response is not connected to the specific claim it evidences in the user's reading experience; a citation immediately after the specific claim is far more legible as a trust signal.
- **Linked and accessible** — a citation that's a bare title or a URL that resolves to a paywall doesn't actually enable verification; the link needs to work, and for paywalled content, a quote or excerpt from the specific supporting text should accompany it.
- **Specific, not sweeping** — "based on three sources" is not a citation; "per the Q3 2026 earnings release (link), revenue grew 14%" is.

### Memory surfacing
Memory surfacing — making it visible to the user what the agent "remembers" about them or their preferences, and giving them control over those memories — is an emerging trust and control pattern specific to persistent-memory agents (Module 1.4's long-term memory types). Users who don't know what the agent has stored about them, and have no way to review or correct it, experience the persistent-memory feature as uncanny or invasive rather than helpful. Concrete implementation: a "what you know about me" view that surfaces stored memories, lets users edit or delete individual memories, and shows which past interaction produced each memory.

### 📋 Interview Questions — 12.4
1. **Why is a single bad AI feature experience disproportionately trust-destroying compared to a single good experience being trust-building?**
   *Look for: the same asymmetry as brand trust generally — negative experiences are encoded more strongly and durably than positive ones; one unexplained wrong action with no recourse can make a user permanently unwilling to re-engage, while one correct answer just meets expectations.*
2. **What's the difference between a tool-use disclosure and a citation, and why do both matter as distinct trust mechanisms?**
   *Look for: tool-use disclosure reveals what the agent accessed (which tools, data sources) and when — a process signal; a citation shows specifically which information from a source supports a specific claim — an evidence signal; both matter because users need to understand both how the agent worked and what specifically it's basing claims on.*
3. **Why is an uncalibrated confidence signal worse than no confidence signal at all?**
   *Look for: an uncalibrated signal ("high confidence" regardless of actual uncertainty) trains users to ignore it; the first demonstrably wrong "high confidence" result destroys the signal's credibility permanently, leaving the user worse off than if they'd never been given a confidence signal to calibrate against.*
4. **Why should citations be inline (immediately after the specific claim) rather than footnoted at the end of a response?**
   *Look for: inline placement connects the citation to the specific claim it evidences in the user's reading flow — a footnote requires the user to hold the claim in memory, find the footnote, and mentally reconnect them; inline citation is legible as a trust signal, footnoted citation is a technicality that most users won't trace.*
5. **Why does memory surfacing matter specifically for long-term-memory agents, and what happens to user trust without it?**
   *Look for: users who don't know what the agent has stored about them and can't review or correct it experience persistent memory as uncanny or invasive rather than helpful — they feel surveilled, not served; memory surfacing turns an unsettling feature into a trustworthy one by giving the user visibility and control.*

---

## 12.5 Error Recovery UX

### Why error recovery UX is structurally different for agents
In traditional software, error recovery is straightforward: something failed, here's an error message, try again or contact support. For agents, error recovery is structurally more complex for three reasons:
- **The error may have already had side effects.** An agent that got halfway through a task before failing may have sent a message, booked a resource, or modified a record. The "undo" question is real and urgent, not rhetorical.
- **The user may not know exactly what failed.** A multi-step agent run has many potential failure points; "something went wrong" is unacceptable as an error message when the user needs to know whether to retry from the beginning, from partway through, or not at all.
- **The same failure may require different responses at different task stages.** A failure at step 2 of 20 is very different from a failure at step 19 of 20 — the former invites retrying; the latter demands a specific partial-result handoff rather than "sorry, start over."

### Graceful degradation
Even the best AI agents make mistakes — which is why agentic systems require fast correction loops. Graceful degradation means the agent fails without catastrophe — providing partial value and a clear state report rather than either pretending to succeed or producing an unhelpful generic error:
- **Partial result delivery** — if the agent completed 8 of 10 research sources before a rate limit failure, delivering the 8 completed analyses with a clear "2 sources could not be retrieved" notice is more useful than withholding all 8 while the user waits for a retry.
- **Scope-limited fallback** — if the agent can't access live data, fall back to the most recent cached version with a clear timestamp and "live data was unavailable" notice, rather than failing entirely.
- **Honest capability communication** — if the task genuinely exceeds the agent's current capability (not a transient error but a structural limitation), say so specifically ("I can draft the analysis but I can't access your company's CRM data from this session") rather than producing a generic failure.

### Fast correction loops
Agentic systems require fast correction loops: users can request corrections, agents acknowledge and implement them, and the result is updated accordingly. The design requirements for a correction loop that actually builds trust rather than eroding it:
- **Targeted correction** — the user should be able to correct a specific section or claim ("the revenue figure in paragraph 3 is wrong") without invalidating the entire response; the agent re-executes only what needs re-execution, not the whole task.
- **Correction acknowledgment** — the agent explicitly confirms what it understood the correction to be and what it changed, rather than silently producing a new version and hoping the user can tell what changed.
- **Correction audit trail** — in high-stakes contexts (financial analysis, legal documents, medical information), showing the before/after delta of a correction, not just the updated output, gives users a way to verify the correction actually addressed their concern rather than introducing a new one.

### Clear handoff to a human
When the agent reaches a state where it genuinely cannot proceed — the error is not transient, no fallback strategy works, or the decision genuinely requires human judgment that the agent shouldn't be making autonomously — the **handoff experience must be actively designed**, not treated as the absence of an agent experience. The handoff should:
- **Identify specifically what the agent could and couldn't do** — "I completed steps 1–4 and got stuck on step 5 because X" — not "an error occurred."
- **Transfer context, not just control** — the human who picks this up needs to know exactly where the agent got to, what it tried, and what the specific blocker is. A handoff without context transfer forces the human to start from scratch.
- **Provide a clear next action** — "Please approve the data access request in your IT portal (link) so I can retry" is a handoff with a path forward; "I was unable to complete this task" is a dead end.

The broader principle connecting all five subsections of this module: the difference between AI features users trust and ones they turn off usually comes down to six design decisions: showing the plan before acting, letting users set their own autonomy level, explaining reasoning, adapting the interface to the task, making recovery effortless, and knowing when to escalate. Get those right and the agent feels like a partner. Get them wrong and users disable it after the first mistake.

### The seven-point trust checklist
A feature that passes all seven feels trustworthy. A feature that fails on two or more will have trouble with adoption, no matter how good the underlying model is. These seven are the synthesized set from current practitioner research:

1. **Signals confidence honestly** — calibrated, not decorative.
2. **Handles failure gracefully** — partial value + clear state, not blank failure.
3. **Gives the user control** — configurable autonomy level, interruptible, steerable.
4. **Respects the user's time** — doesn't impose intermediate reasoning on users who didn't ask for it.
5. **Is transparent about what it accessed** — tool-use disclosure, scope communication.
6. **Attributes claims to specific sources** — inline citations, not sweeping references.
7. **Makes recovery effortless** — fast correction loops, clear human handoff with context transfer.

### 📋 Interview Questions — 12.5
1. **Why does a multi-step agent failure require a more complex response than a traditional software error, structurally?**
   *Look for: the agent may have already taken side-effecting actions before failing (what state is the world in now?), the user may not know which of many steps failed, and the appropriate recovery differs based on *which* step failed — none of these complexities exist in a stateless request-response failure.*
2. **What's the difference between graceful degradation and a generic error message, and what specific elements make a degraded response still useful?**
   *Look for: graceful degradation delivers partial value with a clear state report (what completed, what didn't, what the current state is) rather than withholding everything or producing an uninformative failure; the user can act on partial results while the agent or a human resolves the blocker.*
3. **Why should a targeted correction ("the revenue figure in paragraph 3 is wrong") not trigger a full task re-execution?**
   *Look for: re-executing the entire task is wasteful (re-pays all costs for completed steps), potentially re-introducing different errors in completed sections, and slower; targeted correction re-executes only the specific affected component, preserving completed work — this requires the agent to have component-level state, not just a single atomic output.*
4. **What does "transferring context, not just control" mean in a human handoff, and what happens when context isn't transferred?**
   *Look for: transferring context means the human receiving the handoff gets the full picture of what was completed, what was tried, and what the specific blocker is — without this, they restart from scratch rather than picking up from the agent's stopping point, which defeats the purpose of having the agent work on the task first.*
5. **Why is designing the human handoff experience as carefully as the success experience a mark of a production-grade agent product, rather than an edge-case afterthought?**
   *Look for: agents fail regularly enough in production that the handoff path is a common user journey, not a rare exception; a poorly designed handoff (generic error, no context, no clear next step) is experienced as often as a successful run for tasks in difficult domains — treating it as an afterthought means a significant fraction of users always have a bad experience.*

---

*End of Module 12 detailed notes — 25 interview questions total across 5 sections. The NNGroup State of UX 2026 report and the practitioner research cited here represent the current (mid-2026) state of agent UX practice — this field is still being actively formalized, and the patterns here represent the leading edge of what's working in production rather than an established, stable canon.*
