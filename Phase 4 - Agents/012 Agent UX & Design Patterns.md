I’ve compared the interactive version with your original detailed notes. Almost everything is preserved, but **one paragraph** from the end of Section 12.5 was accidentally dropped. Here it is:

> *“The broader principle connecting all five subsections of this module: the difference between AI features users trust and ones they turn off usually comes down to six design decisions: showing the plan before acting, letting users set their own autonomy level, explaining reasoning, adapting the interface to the task, making recovery effortless, and knowing when to escalate. Get those right and the agent feels like a partner. Get them wrong and users disable it after the first mistake.”*

Everything else – all subsections, tables, examples, the Seven-Point Trust Checklist, and all 25 interview questions – is intact. Below is the **fully corrected and complete interactive module** with that paragraph restored.

---

# Module 12 — Agent UX & Design Patterns: Detailed Notes (Interactive Format)

> **“The models are good enough. What they need is a user interface that signals confidence honestly, handles failure gracefully, gives the user control, and respects their time and data.”**

---

## Table of Contents (full – with subsections)

- [12.1 Streaming vs Batching Output](#121-streaming-vs-batching-output)
  - [The core decision](#the-core-decision)
  - [Streaming patterns & trade-offs](#streaming-patterns--trade-offs)
  - [Batching – when it's the right call](#batching--when-its-the-right-call)
  - [The hybrid: progressive disclosure](#the-hybrid-progressive-disclosure)
  - [📋 Interview Questions – 12.1](#-interview-questions--121)
- [12.2 Showing vs Hiding Intermediate Reasoning](#122-showing-vs-hiding-intermediate-reasoning)
  - [The scratchpad problem](#the-scratchpad-problem)
  - [The case for showing steps](#the-case-for-showing-steps)
  - [The case for hiding steps](#the-case-for-hiding-steps)
  - [Planning visibility – the emerging consensus](#planning-visibility--the-emerging-consensus)
  - [Explainability on demand](#explainability-on-demand)
  - [📋 Interview Questions – 12.2](#-interview-questions--122)
- [12.3 Interruptibility and Steerability Mid-Task](#123-interruptibility-and-steerability-mid-task)
  - [Why interruptibility is a design problem, not just an engineering one](#why-interruptibility-is-a-design-problem-not-just-an-engineering-one)
  - [Variable autonomy levels](#variable-autonomy-levels)
  - [Mid‑task steering mechanisms](#mid-task-steering-mechanisms)
  - [Progressive delegation](#progressive-delegation)
  - [📋 Interview Questions – 12.3](#-interview-questions--123)
- [12.4 Designing for Trust](#124-designing-for-trust)
  - [Why trust is the primary design constraint](#why-trust-is-the-primary-design-constraint)
  - [Transparency signals](#transparency-signals)
  - [Confidence signals](#confidence-signals)
  - [Citations and source attribution](#citations-and-source-attribution)
  - [Memory surfacing](#memory-surfacing)
  - [📋 Interview Questions – 12.4](#-interview-questions--124)
- [12.5 Error Recovery UX](#125-error-recovery-ux)
  - [Why error recovery UX is structurally different for agents](#why-error-recovery-ux-is-structurally-different-for-agents)
  - [Graceful degradation](#graceful-degradation)
  - [Fast correction loops](#fast-correction-loops)
  - [Clear handoff to a human](#clear-handoff-to-a-human)
  - [The broader principle](#the-broader-principle)
  - [The Seven‑Point Trust Checklist](#the-seven-point-trust-checklist)
  - [📋 Interview Questions – 12.5](#-interview-questions--125)

---

## 12.1 Streaming vs Batching Output

### The core decision
> **Output delivery shapes perceived speed, effort, and reliability before the user reads a single word.**

| Streaming | Batching |
|-----------|----------|
| Output appears progressively as generated | All output held back, shown only when complete |
| User can start reading/reacting early | User sees a single polished unit |
| Good for collaboration, long tasks | Good for background jobs, structured data |

The right choice depends on: **task duration**, **output type**, **user role**, and whether **intermediate states are meaningful to that user**.

---

### Streaming patterns & trade-offs

| Pattern | What it looks like | Best for | Watch out! |
|---------|-------------------|----------|------------|
| **Token‑level** | Words appear one‑by‑one (classic chat) | Text‑generation (emails, explanations) | Useless for half‑formed JSON, tables, code |
| **Step‑level** | Each completed tool call / agent step shown | Multi‑step agent tasks | Shows progress without raw scratchpad noise |

**Step‑level example**:  
*“Searching the web… ✓ Analysing results… ✓ Drafting the report…”*  
→ This is the **“multi‑step workflow tracking”** pattern (NNGroup, State of UX 2026).

**Technical headaches of streaming**:
- Requires **SSE** or **WebSocket** – not plain HTTP request‑response
- Harder to **cache** (CDN)
- Structured formats (JSON, tables, code) **can’t be parsed until complete** → client must buffer / incrementally render

---

### Batching – when it's the right call

| Scenario | Reason |
|----------|--------|
| Intermediate state is **meaningless or confusing** | e.g. a half‑generated table with missing columns – forces user to hold “not done yet” model |
| Task runs **asynchronously / in background** | Nobody watching; deliver final result when ready |
| Output has **high editing cost if interrupted** | Formal document, production code PR → better received as a complete, reviewed artifact |

---

### The hybrid: progressive disclosure

**Progressive disclosure** = stream **high‑level status** + batch **detailed step output**.

- User sees *what step is running*, *when it completes*, *how long the task is taking*
- But each step’s detailed result appears only **once that step is fully complete**

✅ **Perceptual benefit of streaming** (sense of progress)  
✅ **No cognitive cost** of managing half‑formed intermediate outputs  

---

### 📋 Interview Questions – 12.1

**1. Why is token‑level streaming not always the right default for a multi‑step agent, even though it feels fast?**  
*Answer guide:* Individual tokens from reasoning or half‑formed tool results aren’t meaningful to act on. Step‑level streaming (each completed step as a unit) is more useful because the meaningful unit is a finished step, not a token.

**2. What specific technical challenges does streaming output introduce that batched responses don’t have?**  
*Answer guide:* Requires SSE/WebSocket (not plain HTTP), CDN caching is harder, and structured formats (JSON, tables, code) aren’t parseable until complete – the client must handle incremental rendering/buffering.

**3. A user is running a long‑horizon research agent that takes 8 minutes. Should the output stream token‑by‑token, step‑by‑step, or batch entirely? Defend your answer.**  
*Answer guide:* Step‑level streaming is almost certainly right – 8 minutes of blank screen destroys experience. Raw token streaming of reasoning notes creates noise. Showing completed steps (search done, analysis done, draft section done) maintains engagement at the right granularity.

**4. What’s the user‑experience argument for batching a generated data table rather than streaming its rows as they’re produced?**  
*Answer guide:* A half‑formed table with missing columns forces the user to suppress evaluation – “don’t read this yet” cognitive tax. Tables are evaluated as complete structures, not as sequential streams. Batching eliminates that burden.

**5. Describe the “progressive disclosure” hybrid pattern and explain what problem it solves that neither pure streaming nor pure batching solves alone.**  
*Answer guide:* Streams high‑level status (which step is active) while batching each step’s detailed output until complete. Gives the perceptual benefit of streaming (active progress) without the cognitive cost of managing half‑formed intermediate results.

---

## 12.2 Showing vs Hiding Intermediate Reasoning

### The scratchpad problem

> **The model’s chain‑of‑thought is an engineering artifact, not a user feature.**

Dumping raw scratchpad onto the UI **masquerades as transparency** but actually:
- Gives users **information they can’t act on**
- In a **format they weren’t trained to interpret**
- At a **cognitive cost** that trades against using the final output

**The right question is not “Should we show reasoning?” but:**  
*“Which level of reasoning, abstracted to what degree, to which users, at what moment?”*

---

### The case for showing steps

**Planning visibility** = showing the **intended action sequence before execution**.

| Benefit | Why it matters |
|---------|----------------|
| **Catch misunderstood goals** before action | Prevents irreversible mistakes |
| **Builds trust** through observable, predictable behaviour | Users see *why* the agent acted, not just that it acted |

**Good pattern**:  
A **summary of the reasoning** (1–2 sentences) placed **right next to the action** it explains.

❌ *Raw chain‑of‑thought in a separate pane*  
✅ *“I’m searching for recent pricing data because your question referenced 2026 numbers.”* anchored to the search action

**Show intermediate steps by default when**:
- User is a **power user / developer** evaluating behaviour
- Task is **high‑stakes** (real‑world consequences, user might want to stop)
- User is **actively collaborating** and feedback at intermediate stages improves the outcome

---

### The case for hiding steps

For **end users** in consumer/productivity products, raw reasoning often:

- **Creates false evaluation burden** – user feels they *should* review each step, even without context to judge it
- **Inflates perceived complexity** – 15 intermediate steps before a simple answer feels unreliable, not thorough
- **Front‑loads cognitive load** – user is waiting for an answer, not a reasoning tutorial

---

### Planning visibility – the emerging consensus

Five interface decisions every enterprise agent needs (NNGroup, State of UX 2026):
1. Planning visibility
2. Tool‑use disclosure
3. Memory surfacing
4. Multi‑step workflow tracking
5. Recovery routing

**Planning visibility** specifically means:  
Show the *action plan* (what it will do, in what order, toward what goal) **before execution** – at a level the user can evaluate and approve.  
→ Exactly the human‑in‑the‑loop checkpoint from Modules P.2 & 1.6.

---

### Explainability on demand

**Default state**: Minimal status trail  
*“Searched 4 sources, analysed 12 results, summarised findings.”*

**On‑demand detail**:
- Expandable reasoning section
- “How did you get this?” button
- Step‑by‑step trace view

**Proactive for flagged outputs**:
- Low confidence, assumptions user might disagree with
- Surface the relevant part of the reasoning **at that specific point in the output** (targeted, not wholesale)

---

### 📋 Interview Questions – 12.2

**1. Why is dumping raw chain‑of‑thought onto the UI a design failure even though it feels like transparency?**  
*Answer guide:* Raw scratchpad is information users can’t act on, in a format they aren’t trained to interpret, at a cognitive cost that trades against using the final output. It’s an engineering artifact, not a user feature. Transparency at the wrong abstraction level doesn’t serve the user.

**2. What’s the difference between planning visibility and showing raw reasoning, and why does that distinction matter for user trust?**  
*Answer guide:* Planning visibility shows the intended action sequence at a level the user can evaluate and approve (a list of steps). Raw reasoning shows the model’s internal derivation. Only the former is evaluable and builds trust through oversight.

**3. How does “explainability on demand” resolve the tension between hiding intermediate reasoning and being transparent about it?**  
*Answer guide:* Hides reasoning by default (reducing cognitive load) while surfacing it immediately and completely when explicitly requested. Transparency is available when wanted, not always imposed. Users who want oversight get it; others aren’t burdened.

**4. For which specific user types and task types would you show intermediate steps by default rather than on demand?**  
*Answer guide:* Power users/developers evaluating behaviour, high‑stakes tasks with real‑world consequences (user might want to stop mid‑execution), and active collaborative tasks where user feedback at intermediate stages would improve the outcome.

**5. A product manager proposes showing all 15 intermediate steps for every agent response to “make the agent feel more thorough.” What would you push back on?**  
*Answer guide:* This creates false evaluation burden (users feel they should review each step), inflates perceived complexity (15 steps before a simple answer feels unreliable, not thorough), and front‑loads cognitive load when the user is waiting for a result. Thoroughness should be available on demand, not imposed.

---

## 12.3 Interruptibility and Steerability Mid‑Task

### Why interruptibility is a design problem, not just an engineering one

> **A stop button without a “here’s what already happened” disclosure is half an implementation.**

When an agent is interrupted mid‑task, the user immediately needs to know:
- What actions were **already taken**?
- What **state** did those actions leave the world in?
- Is it **safe to retry** from the beginning, or will that **duplicate a side‑effect**?
- What needs **manual cleanup**?

✅ Good interruptibility answers these **proactively, at the moment of interruption**.

---

### Variable autonomy levels

Autonomy should not be binary. Users expect a **spectrum of control**.

| Autonomy level | Description | Best for |
|----------------|-------------|----------|
| **Full supervision** | Agent proposes every action; user must approve each. Maximum control. | High‑stakes, unfamiliar tasks |
| **Supervised autonomy** | Agent executes routine actions independently, pauses at defined checkpoints (irreversible actions, budget thresholds, scope boundaries). | Recurring tasks with occasional oversight |
| **Full autonomy** | Agent completes the entire task, surfaces only the final result. Minimum friction. | High‑volume routine tasks after many successful supervised runs |

**Design principle**: Autonomy level should be **configurable by the user**, not fixed by the product. A first‑time user needs full supervision; a veteran of 50 runs wants full autonomy.

---

### Mid‑task steering mechanisms

Beyond a simple stop button:

| Mechanism | What the user says | What the agent does |
|-----------|-------------------|---------------------|
| **Redirect** | “Keep going, but focus on X instead of Y” | Replans remaining steps – doesn’t stop or restart from scratch |
| **Scope adjustment** | “Stop after this section” / “Include the financial analysis too” | Expands or contracts task scope mid‑run without cancelling |
| **Priority change** | “Do the legal review first before the financial analysis” | Reorders remaining steps without discarding completed work |

🔑 These rely on an **explicit, revisable plan** (Planner‑Executor pattern, Module P.2). A pure ReAct loop has no plan to adjust – you can only continue unchanged or restart from scratch.

---

### Progressive delegation

> **The system earns permission through demonstrated reliability, rather than demanding it at launch.**

**How it works**:
1. New user starts at **supervised autonomy**.
2. Each successfully completed task (no corrections needed) is recorded.
3. After **N consecutive successes** (configurable), the system suggests raising autonomy for that specific task type:  
   *“You’ve approved 10 consecutive email drafts without changes. Would you like me to send future drafts automatically?”*
4. Trust earned in one task type **does not automatically transfer** to another.

→ Agent‑UX expression of **least‑privilege** (Module 9.3): start with minimum permissions, earn more scope through demonstrated reliability.

---

### 📋 Interview Questions – 12.3

**1. Why does a technically working stop button still fail at the UX level if it doesn’t answer “what state is the world in now”?**  
*Answer guide:* Interruption leaves the user needing to know what already executed, what state those actions left, whether retrying is safe or would duplicate side‑effects, and what needs manual cleanup. An unsupported stop button outsources all of this investigation to the user.

**2. What’s the principle behind “variable autonomy levels,” and why does designing a single fixed autonomy level for all users fail?**  
*Answer guide:* Different users have different established trust and task familiarity – a first‑timer needs full supervision, a veteran of 50 successful runs wants full autonomy. A single fixed level serves the wrong user on one end.

**3. How does the Planner‑Executor pattern (Module P.2) enable redirect and scope‑adjustment mid‑task in a way a pure ReAct loop doesn’t?**  
*Answer guide:* A pure ReAct loop has no explicit, revisable plan – interrupting means either continuing unchanged or restarting. Planner‑Executor has an explicit remaining‑plan that can be revised in place (redirect, reorder, scope‑adjust) without discarding completed steps.

**4. What’s “progressive delegation” and how does it differ from simply offering an autonomy‑level setting the user can toggle?**  
*Answer guide:* Progressive delegation ties autonomy expansion to the user’s own observed approval history – the system earns permission through demonstrated reliability on specific task types, rather than requiring the user to proactively configure a setting. It’s trust earned, not trust assumed.

**5. A user is running a 20‑step research agent and at step 12 says “actually, focus on the European market only.” What does the agent need to do, and why can’t it just continue from step 13 unchanged?**  
*Answer guide:* The agent must replan steps 13–20 against the new scope. Continuing unchanged would execute steps designed for a broader scope, wasting effort and producing wrong output. Replanning updates the remaining plan in place.

---

## 12.4 Designing for Trust

### Why trust is the primary design constraint

> **One bad AI experience is disproportionately destructive to trust.**  
> (Same asymmetry as brand trust: negative experiences encode more strongly and durably than positive ones.)

NNGroup’s State of UX 2026: “Trust is a major design challenge for AI experiences.” Users burned by premature AI features resist adopting new ones.

Trust in agentic AI is built over time through **observable outcomes** and **responsive learning**. UX should show:
- How the AI is performing
- What it has learned
- How user input influences future behaviour

---

### Transparency signals

| Signal | What it communicates | Example |
|--------|----------------------|---------|
| **Tool‑use disclosure** | Which external tools, APIs, data sources were accessed | *“Based on a search of your Confluence space from this morning…”* |
| **Scope communication** | What the agent did and didn’t look at | *“I reviewed your last 90 days of email but not your calendar.”* |
| **Action disclosure before execution** | For irreversible actions, the specific action and parameters | Planning visibility, Module 12.2 |

---

### Confidence signals

**Calibrated, not decorative.**

- An **uncalibrated** confidence signal (always “high confidence”) is **worse than none** – the first wrong “high confidence” result destroys all credibility.
- Useful confidence signals **distinguish sources of uncertainty**:
  - Contradictory evidence
  - Data freshness / recency
  - Outside the agent’s reliable domain
  - Ambiguous user question

**Micro‑animations** (100–300 ms) can communicate confidence state changes – e.g., colour transition from amber to green as certainty improves.

---

### Citations and source attribution

Citations serve two purposes:
1. **Quality signal** – agent retrieved real, specific sources, not parametric memory
2. **Trust mechanism** – user can verify the claim independently

| Good practice | Why |
|---------------|-----|
| **Inline** (immediately after the claim) | Connects citation to the specific claim in reading flow; footnotes force mental reconnection |
| **Linked & accessible** | A paywalled bare title doesn’t enable verification; include a quote or excerpt for paywalled content |
| **Specific, not sweeping** | *“Per the Q3 2026 earnings release (link), revenue grew 14%”* not *“based on three sources”* |

---

### Memory surfacing

For agents with **persistent memory** (Module 1.4):

- Without visibility → users feel **surveilled, not served** (uncanny/invasive).
- With **“what you know about me” view** → user can review, edit, delete stored memories, see which past interaction produced each.

---

### 📋 Interview Questions – 12.4

**1. Why is a single bad AI feature experience disproportionately trust‑destroying compared to a single good experience being trust‑building?**  
*Answer guide:* Negative experiences are encoded more strongly and durably than positive ones (brand‑trust asymmetry). One unexplained wrong action with no recourse can make a user permanently unwilling to re‑engage; a correct answer just meets expectations.

**2. What’s the difference between a tool‑use disclosure and a citation, and why do both matter as distinct trust mechanisms?**  
*Answer guide:* Tool‑use disclosure reveals what the agent accessed and when (a process signal). A citation shows which specific information from a source supports a specific claim (an evidence signal). Both matter because users need to understand how the agent worked and what it’s basing claims on.

**3. Why is an uncalibrated confidence signal worse than no confidence signal at all?**  
*Answer guide:* An uncalibrated signal (“high confidence” regardless of actual uncertainty) trains users to ignore it. The first demonstrably wrong “high confidence” result destroys the signal’s credibility permanently, leaving the user worse off than if they’d never been given a signal to calibrate against.

**4. Why should citations be inline (immediately after the specific claim) rather than footnoted at the end of a response?**  
*Answer guide:* Inline placement connects the citation to the claim it evidences in the user’s reading flow. A footnote requires the user to hold the claim in memory, find the footnote, and mentally reconnect them – inline is legible as a trust signal, footnoted is a technicality most users won’t trace.

**5. Why does memory surfacing matter specifically for long‑term‑memory agents, and what happens to user trust without it?**  
*Answer guide:* Without visibility and control over stored memories, users experience persistent memory as uncanny or invasive – they feel surveilled, not served. Memory surfacing turns an unsettling feature into a trustworthy one by giving the user visibility and control.

---

## 12.5 Error Recovery UX

### Why error recovery UX is structurally different for agents

Unlike traditional software errors, agent errors:

| Challenge | Why it’s harder |
|-----------|-----------------|
| **Side effects already occurred** | Agent may have sent a message, booked a resource, modified a record. “Undo” is real and urgent. |
| **User doesn’t know what failed** | Multi‑step run has many failure points; “something went wrong” is unacceptable. |
| **Recovery depends on which step failed** | Failure at step 2 of 20 → retry; failure at step 19 → specific partial‑result handoff, not “start over”. |

---

### Graceful degradation

> **Fail without catastrophe** – provide partial value + clear state report.

| Strategy | Example |
|----------|---------|
| **Partial result delivery** | 8 of 10 analyses completed; deliver them with a clear “2 sources could not be retrieved” notice |
| **Scope‑limited fallback** | Live data unavailable → fall back to most recent cached version with a timestamp and notice |
| **Honest capability communication** | *“I can draft the analysis but I can’t access your company’s CRM from this session.”* |

---

### Fast correction loops

Agentic systems require **fast correction loops**:
- User requests correction
- Agent acknowledges and implements it
- Result updated accordingly

**Design requirements**:

| Requirement | Why |
|-------------|-----|
| **Targeted correction** | Correct a specific section/claim (“revenue in paragraph 3 is wrong”) without re‑executing the whole task |
| **Correction acknowledgment** | Agent explicitly confirms what it understood and what it changed, not silently producing a new version |
| **Correction audit trail** | In high‑stakes contexts, show the before/after delta – users can verify the correction didn’t introduce new errors |

---

### Clear handoff to a human

When the agent genuinely cannot proceed, the **handoff experience must be actively designed**, not treated as an afterthought.

| Element | Good handoff | Bad handoff |
|---------|-------------|-------------|
| **State summary** | *“I completed steps 1–4 and got stuck on step 5 because X”* | *“An error occurred.”* |
| **Context transfer** | Human receives full picture of what was done, tried, and the blocker | Human starts from scratch (defeats the purpose of the agent) |
| **Clear next action** | *“Please approve the data access request in your IT portal (link) so I can retry.”* | *“I was unable to complete this task.”* |

---

### The broader principle

> The difference between AI features users trust and ones they turn off usually comes down to six design decisions: **showing the plan before acting, letting users set their own autonomy level, explaining reasoning, adapting the interface to the task, making recovery effortless, and knowing when to escalate**. Get those right and the agent feels like a partner. Get them wrong and users disable it after the first mistake.

---

### The Seven‑Point Trust Checklist

A feature that passes all seven **feels trustworthy**. Fail on two or more → adoption trouble, no matter how good the model.

| # | Principle | Description |
|---|-----------|-------------|
| 1 | **Signals confidence honestly** | Calibrated, not decorative |
| 2 | **Handles failure gracefully** | Partial value + clear state, not blank failure |
| 3 | **Gives the user control** | Configurable autonomy, interruptible, steerable |
| 4 | **Respects the user’s time** | Doesn’t impose intermediate reasoning on users who didn’t ask |
| 5 | **Transparent about what it accessed** | Tool‑use disclosure, scope communication |
| 6 | **Attributes claims to specific sources** | Inline citations, not sweeping references |
| 7 | **Makes recovery effortless** | Fast correction loops, clear human handoff with context transfer |

---

### 📋 Interview Questions – 12.5

**1. Why does a multi‑step agent failure require a more complex response than a traditional software error, structurally?**  
*Answer guide:* The agent may have already taken side‑effecting actions (what state is the world in now?), the user may not know which step failed, and the appropriate recovery differs based on *which* step failed. None of these complexities exist in a stateless request‑response failure.

**2. What’s the difference between graceful degradation and a generic error message, and what specific elements make a degraded response still useful?**  
*Answer guide:* Graceful degradation delivers partial value with a clear state report (what completed, what didn’t, current state), rather than withholding everything. The user can act on partial results while the agent or a human resolves the blocker.

**3. Why should a targeted correction (“the revenue figure in paragraph 3 is wrong”) not trigger a full task re‑execution?**  
*Answer guide:* Re‑executing the whole task wastes resources, risks introducing different errors in completed sections, and is slower. Targeted correction re‑executes only the affected component – preserving completed work. This requires component‑level state, not just a single atomic output.

**4. What does “transferring context, not just control” mean in a human handoff, and what happens when context isn’t transferred?**  
*Answer guide:* Transferring context means the human receives the full picture of what was completed, what was tried, and the specific blocker. Without it, they restart from scratch, defeating the purpose of having the agent work on the task first.

**5. Why is designing the human handoff experience as carefully as the success experience a mark of a production‑grade agent product, rather than an edge‑case afterthought?**  
*Answer guide:* Agents fail regularly enough in production that the handoff path is a *common user journey*, not a rare exception. A poorly designed handoff (generic error, no context, no next step) is experienced as often as a successful run in difficult domains – treating it as an afterthought means a significant fraction of users always have a bad experience.

---

*End of Module 12 – 25 interview questions across 5 sections. The patterns here represent the leading edge of what’s working in production as of mid‑2026.*
