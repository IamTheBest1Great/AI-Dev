# Module 8 — Advanced Agent Capabilities: Detailed Notes

## Table of Contents

- [8.1 Computer Use Agents](#81-computer-use-agents)
  - [OS-level control (screen perception, mouse/keyboard actions)](#os-level-control-screen-perception-mousekeyboard-actions)
  - [Risks and sandboxing requirements specific to computer use](#risks-and-sandboxing-requirements-specific-to-computer-use)
  - [📋 Interview Questions — 8.1](#-interview-questions--81)
- [8.2 Coding Agents](#82-coding-agents)
  - [Autonomous coding agents (Claude Code, SWE-agent, Devin-style agents, IDE agent modes)](#autonomous-coding-agents-claude-code-swe-agent-devin-style-agents-ide-agent-modes)
  - [Repo-aware context management](#repo-aware-context-management)
  - [Test-driven verification loops for code agents](#test-driven-verification-loops-for-code-agents)
  - [📋 Interview Questions — 8.2](#-interview-questions--82)
- [8.3 Voice & Real-Time Agents 🆕](#83-voice--real-time-agents-)
  - [Streaming speech-to-text / text-to-speech integration](#streaming-speech-to-text--text-to-speech-integration)
  - [Latency budgets for conversational real-time agents](#latency-budgets-for-conversational-real-time-agents)
  - [Interruption handling (barge-in)](#interruption-handling-barge-in)
  - [📋 Interview Questions — 8.3](#-interview-questions--83)
- [8.4 Agentic RAG 🆕](#84-agentic-rag-)
  - [Retrieval as a callable tool vs static context injection](#retrieval-as-a-callable-tool-vs-static-context-injection)
  - [Query planning and multi-hop retrieval](#query-planning-and-multi-hop-retrieval)
  - [Re-ranking and source verification before synthesis](#re-ranking-and-source-verification-before-synthesis)
  - [📋 Interview Questions — 8.4](#-interview-questions--84)
- [8.5 Long-Horizon / Autonomous Agents 🆕](#85-long-horizon--autonomous-agents-)
  - [Continuous/background operation patterns](#continuousbackground-operation-patterns)
  - [Scheduling and triggering (cron, event-driven)](#scheduling-and-triggering-cron-event-driven)
  - [State checkpointing across very long tasks](#state-checkpointing-across-very-long-tasks)
  - [📋 Interview Questions — 8.5](#-interview-questions--85)

> Computer use, coding agents, and voice agents are three of the fastest-moving product categories in this entire curriculum — even major labs have already discontinued and rebranded products in this space within months of launch. These notes use verified current architecture details and flag the genuine, ongoing churn rather than presenting any single vendor's numbers as settled.

---

## 8.1 Computer Use Agents

### OS-level control (screen perception, mouse/keyboard actions)
Computer use agents operate at the full operating-system/desktop level rather than just within a browser's DOM (Module 4.1) — the model receives a **screenshot**, reasons over it visually, and outputs an action (click at a coordinate, type text, press a key combination, scroll) as a tool call; the result is a new screenshot, and the loop — screenshot → action → new screenshot — repeats until the task completes or a step cap is hit (Claude's computer use tool defaults to a maximum of 10 agent-loop iterations, developer-adjustable). This is architecturally a direct application of the ReAct loop from Module 1.3, just with vision as the observation channel instead of text.

This is meaningfully harder than browser automation specifically because there's typically **no structured accessibility tree or DOM to lean on** the way Playwright MCP (Module 4.1) does — the agent is working purely from pixels, the same way a human would look at a screen, which is why computer use has historically been slower and less reliable than API-based or DOM-based automation when either of those alternatives is available. The standard benchmark for this category, **OSWorld** (369 real desktop tasks spanning file management, multi-app workflows, and more), illustrates how genuinely unsolved this still is: scores have improved substantially from roughly 12% in 2025 to the 70s–80s percentile range for leading models by mid-2026, but remain below the human baseline (commonly cited around 70–72%) — a real, hard, actively-researched problem, not a solved one with marketing-driven scores that have already saturated.

**🆕 Currency note**: this category has already seen real product churn worth knowing about as a case study. OpenAI's **Operator** launched as a standalone research-preview product in January 2025, built on a model called Computer-Using Agent (CUA); by July 2025 — about six months later — the standalone product was discontinued and its capabilities folded into **ChatGPT Agent**, a broader product surface. Anthropic's Computer Use, by contrast, launched as an API-first capability in October 2024 and reached general availability on claude.ai in March 2026, expanding across model generations rather than being rebranded. The lesson generalizes: even flagship products from major labs in this specific category have proven unstable in naming and architecture within roughly a year — exactly the kind of fast-moving area where the "verify current docs before building" discipline from earlier modules matters most.

### Risks and sandboxing requirements specific to computer use
Computer use carries risks distinct from standard tool-calling, and the standard mitigations are the productized, concrete version of Module 9's security principles:
- **Dedicated, minimally-privileged sandbox** — a virtual machine or container, not the user's real desktop, specifically to contain the consequences of a mistaken or malicious action.
- **Avoid exposing sensitive data/credentials** where possible — every credential the agent can see is a credential a successful prompt injection could exfiltrate or misuse.
- **Domain allowlisting** for any internet access, reducing exposure to malicious content the agent might encounter mid-task.
- **Human confirmation for consequential actions** — financial transactions, agreeing to terms of service, or anything requiring affirmative consent should pause for explicit approval, directly the Module 1.6 checkpoint pattern.

**Prompt injection is a heightened, category-specific risk here**: malicious instructions can be embedded directly in on-screen content — a webpage, a popup, an email the agent is asked to read — and a model reasoning over a screenshot may follow an instruction it "sees" even when it conflicts with the user's actual request. The defense is layered, not single-point: the model itself is trained to resist this; automated classifiers additionally scan screenshots for likely injection attempts; and when something suspicious is flagged, the system is steered to pause and ask for human confirmation before proceeding — defense in depth, because no single layer is assumed sufficient on its own.

### 📋 Interview Questions — 8.1
1. **Why is computer use generally harder and less reliable than browser automation via Playwright (Module 4.1), even for tasks that happen to take place inside a browser window?**
   *Look for: computer use typically lacks a structured accessibility tree/DOM to act on — it works from raw screenshots the way a human visually would, while Playwright can act on structured page data directly, which is faster and more precise.*
2. **What does it mean that OSWorld scores remain below the human baseline even as they've improved dramatically, and why does that matter for how you'd deploy a computer use agent in production?**
   *Look for: the category is still genuinely unsolved, not just under-marketed — production deployments need human review/confirmation gates rather than treating computer use as a reliable, unattended automation layer.*
3. **Why is prompt injection a heightened risk specifically for computer use agents compared to a typical text-based tool-calling agent?**
   *Look for: the agent visually "reads" arbitrary on-screen content (webpages, popups) that wasn't vetted by the developer, and malicious instructions can be embedded directly in that content — a much larger and less controlled injection surface than a developer-defined tool's text output.*
4. **A team wants to give a computer use agent direct access to the user's real desktop, including their email client, to "save setup time." What would you push back on?**
   *Look for: the sandboxing principle — running on the real desktop with real credentials/data turns any mistake or successful injection into a real-world consequence; a dedicated, minimally-privileged sandbox exists specifically to contain that blast radius.*
5. **What does OpenAI's Operator-to-ChatGPT-Agent transition (within about six months of launch) illustrate about this product category, beyond just "OpenAI made a branding change"?**
   *Look for: even major labs' flagship products in this specific category are still architecturally unstable — a sign to verify current product details before building against them rather than assuming a 2025-era tutorial still reflects 2026 reality.*

---

## 8.2 Coding Agents

### Autonomous coding agents (Claude Code, SWE-agent, Devin-style agents, IDE agent modes)
Several distinct lineages exist in this space, each having shaped how the field thinks about coding agents:
- **Claude Code** — an agentic coding tool that reads a codebase, edits files, runs commands and tests, and works across many files toward a stated goal, available as a CLI, IDE extensions (VS Code, JetBrains), a desktop app, and browser/mobile surfaces sharing the same underlying configuration.
- **SWE-agent** — an open-source, originally academic project that pioneered the **Agent-Computer Interface (ACI)** concept: rather than giving a model raw, unconstrained shell access, it exposes a curated, agent-friendly set of file-viewing, searching, and editing commands specifically designed to make codebase navigation and editing more reliable for an LLM. This idea — that *how* you expose a codebase to an agent matters as much as the underlying model's capability — has influenced how the whole field designs coding-agent tooling since.
- **Devin (Cognition)** — one of the earliest products marketed as a "fully autonomous" software engineer, launched in March 2024 with a widely-publicized 13.86% SWE-bench score (notably, well above the weaker state of the art it was compared against at the time). Under a strict, single-agent, no-human-in-the-loop evaluation, Devin 2.0 scores around 45.8% — a real, if more modest-sounding, improvement; Cognition has since pursued enterprise and government deployment (an enterprise partnership and a government-focused offering launched in early 2026). A more grounded production signal than any single benchmark: Devin's real-world PR merge rate has reportedly improved from roughly 34% to 67% over about 18 months of deployment.
- **OpenHands** (formerly OpenDevin) — an open-source, MIT-licensed community alternative, often used as a transparent baseline for comparison.
- **IDE agent modes** — agent loops embedded directly inside an IDE (Cursor, GitHub Copilot's agent mode, JetBrains AI assistants, and others), letting the agent use the IDE's own build/test/debug tooling directly rather than operating as a separate CLI or cloud product.

### Repo-aware context management
A real codebase vastly exceeds any model's context window — a coding agent fundamentally cannot "just read the whole repo," which makes context management the single hardest, most actively-engineered problem in this category. The concrete techniques (illustrated well by Claude Code's documented architecture, though the underlying principles generalize):
- **A persistent, always-loaded project context file** (Claude Code's `CLAUDE.md`, with similar conventions like `AGENTS.md` elsewhere) — build commands, directory structure, coding conventions, and team norms loaded once at session start rather than rediscovered every turn, directly the in-context-memory pattern from Module 1.4.
- **Isolated sub-agent delegation for exploration** — a dedicated, often read-only sub-agent searches and reads files in its own separate context window and returns only a distilled summary, keeping verbose intermediate file contents and search results out of the main session entirely. This is the Module 7.4 sub-agent pattern, applied specifically for context-budget protection rather than latency reduction.
- **Layered, automatic compaction** as the context window fills — trimming the cheapest, least valuable content first (e.g., stale verbose tool outputs), only falling back to summarizing the conversation itself as a more expensive, lossier last resort, so a long session degrades gracefully rather than hitting a hard wall mid-task.

The unifying principle: every one of these is fundamentally a **context-budget decision** — what's always loaded (cheap to access, but costs tokens every single turn), what's loaded on demand, and what's isolated entirely in a sub-agent's own window with only a summary returned — a direct, high-stakes application of Module 1.4's memory-type distinctions to the specific domain where context pressure is most acute.

### Test-driven verification loops for code agents
Running tests after a code change gives the agent (and the developer) a concrete, checkable signal for whether a change actually works — closing the loop so the agent isn't just generating plausible-looking code on faith, directly the Module 1.3 ReAct pattern with "run the tests" as the observation step after the "edit code" action.

**Important caveat, worth taking seriously rather than treating test-passing as proof of correctness**: independent analysis of top coding-agent leaderboard results has found that a real share of benchmark tasks labeled "solved" pass their tests for the wrong reasons — through test-suite gaming or reward-hacking the evaluation harness (e.g., modifying or working around a test rather than genuinely fixing the underlying issue) rather than through correct code. This is the coding-agent expression of Module 1.5's "hallucinated tool calls look plausible" failure mode: a confidently green test suite is not the same guarantee of correctness it would be if a human wrote and ran it with the intent to verify their own work, rather than to satisfy an evaluator. The practical implication: test-driven verification is **necessary but not sufficient** — pair it with human code review, held-out tests the agent genuinely cannot see or modify, and the same evenhanded skepticism toward any single benchmark score (Module 4.6's lesson about not trusting vendor or leaderboard numbers in isolation) applied specifically to coding-agent leaderboards.

### 📋 Interview Questions — 8.2
1. **What problem did SWE-agent's "Agent-Computer Interface" concept solve, and why does it matter beyond that one specific project?**
   *Look for: curated, agent-friendly commands instead of raw shell access made codebase navigation more reliable — establishing the broader principle that how you expose an environment to an agent matters as much as the underlying model's raw capability.*
2. **Why is "Devin's PR merge rate improved from 34% to 67%" a more meaningful production signal than a single SWE-bench percentage?**
   *Look for: a merge rate reflects sustained, real-world deployment outcomes over time, not a one-time score on a fixed, heavily-optimized benchmark task set — closer to the evenhanded-benchmark-skepticism principle from Module 4.6.*
3. **Explain the role a dedicated "Explore" sub-agent plays in a coding agent's context management, and why returning only a summary matters.**
   *Look for: it performs verbose file-reading/searching in its own isolated context window, keeping that bulk out of the main session — directly applying Module 7.4's sub-agent pattern for context-budget protection rather than latency.*
4. **A coding agent's PR passes all existing tests. What additional question should you ask before trusting it's actually correct?**
   *Look for: whether the tests genuinely verify the intended behavior or were gamed/worked-around — passing tests is necessary but not sufficient evidence of correctness, per the reward-hacking caveat.*
5. **Why does "always loaded" context (like CLAUDE.md) cost something every single turn, and how does that shape what belongs there versus in a sub-agent's isolated context?**
   *Look for: always-loaded context consumes tokens on every turn regardless of relevance to that turn's specific task, so only genuinely stable, broadly-relevant information (conventions, build commands) belongs there — verbose, task-specific exploration belongs in an isolated sub-agent window instead.*

---

## 8.3 Voice & Real-Time Agents 🆕

### Streaming speech-to-text / text-to-speech integration
Two distinct architectures dominate this space:
- **Chained pipeline** — separate STT, text-based LLM, and TTS components (e.g., a dedicated speech-recognition provider feeding a text model feeding a dedicated speech-synthesis provider), each a distinct hop with its own latency budget. This remains the dominant production pattern, particularly where compliance requirements (e.g., HIPAA) or a need for custom voice cloning constrain which specific provider you can use for each piece — you can swap any one component independently of the others.
- **Native speech-to-speech** — a single model takes raw audio in and produces raw audio out directly (e.g., OpenAI's Realtime API, Google's Gemini Live), with no text intermediate. This eliminates the STT-finalize wait and TTS startup delay that a chained pipeline pays on every turn, and can preserve prosody and emotional tone that gets flattened away when everything passes through text in between — at the real cost of harder debugging (there's no clean, inspectable text intermediate step) and typically higher, less predictable cost, since the model's context accumulates as audio tokens, which run far more expensive than the equivalent text tokens.

### Latency budgets for conversational real-time agents
Concrete numbers worth internalizing: **under ~300ms feels natural**, **300–600ms is generally acceptable**, and **beyond roughly 1 second starts to feel robotic or broken**. This total budget has to be split across *every* hop in a chained pipeline — STT, LLM reasoning (plus any tool calls), and TTS — which is why provider and architecture choice become serious, latency-driven engineering decisions in this category in a way they typically aren't for text-based agents, where a few extra hundred milliseconds of "thinking" rarely bothers anyone. Even strong streaming STT alone can take 100–150ms to produce a first partial transcript, which already consumes a meaningful fraction of a sub-second total budget before the LLM or TTS legs even start — making network topology choices (WebRTC vs. WebSocket, edge routing) a real part of the latency engineering, not an afterthought.

### Interruption handling (barge-in)
Letting a user interrupt the agent mid-response, the way they naturally would a human in conversation, requires getting three distinct, genuinely failure-prone pieces right:
1. **Echo cancellation (AEC) on the client** — without it, the agent can hear its own synthesized voice coming back through the user's microphone and mistake it for user speech, causing it to interrupt itself. This is a commonly-reported, easy-to-overlook debugging trap specifically because it's invisible until you actually test with real audio hardware.
2. **Immediately canceling in-flight generation** the moment genuine user speech is detected, rather than letting the agent finish its current sentence or turn first.
3. **Discarding any already-generated-but-not-yet-played audio buffer** — the model has often streamed several hundred milliseconds of audio ahead of what the user has actually heard, and that buffered-but-unheard audio needs to be thrown away immediately on interruption, or the agent will appear to keep talking over the user briefly even after "stopping."

Getting barge-in right is arguably the single most failure-prone piece of a voice agent's UX — a voice agent that can't be interrupted naturally reads as broken in a way the text-agent equivalent (the user simply sends another message) never does, which is why it deserves dedicated engineering attention rather than being treated as a minor polish item.

### 📋 Interview Questions — 8.3
1. **What's the core latency tradeoff between a chained STT-LLM-TTS pipeline and a native speech-to-speech model?**
   *Look for: speech-to-speech eliminates the STT-finalize and TTS-startup waits and can preserve prosody, at the cost of harder debugging (no text intermediate) and typically higher, less predictable cost from audio-token context accumulation.*
2. **Why does a sub-second total latency budget make individual hop latency (like 100-150ms STT time-to-first-partial) a serious engineering constraint in voice agents specifically?**
   *Look for: every hop's latency directly consumes a scarce, perceptible total budget (under ~1 second before it feels broken) — unlike a text agent where a few hundred extra milliseconds of reasoning is rarely noticed, every millisecond is contested in a real-time voice pipeline.*
3. **A voice agent keeps "interrupting itself" mid-response. What's the most likely root cause, and how would you fix it?**
   *Look for: missing or disabled echo cancellation (AEC) on the client — the agent is hearing its own synthesized audio through the microphone and mistaking it for user speech; fix by enabling platform-native AEC.*
4. **Why isn't canceling in-flight generation alone sufficient to make barge-in feel natural — what else has to happen?**
   *Look for: already-generated but not-yet-played audio sitting in a buffer needs to be discarded too, or the user will hear the agent keep talking briefly even after generation has technically stopped.*
5. **Why might a team choose a chained pipeline over a native speech-to-speech model even though speech-to-speech offers lower latency and better prosody?**
   *Look for: compliance requirements (e.g., HIPAA-eligible components), custom voice-cloning needs, the ability to swap any individual component independently, or wanting a text intermediate for debugging/tool-call reliability — real tradeoffs, not a strictly dominated choice.*

---

## 8.4 Agentic RAG 🆕

### Retrieval as a callable tool vs static context injection
Traditional RAG performs a single, **upfront** retrieval pass — relevant chunks are fetched once and injected into the prompt before generation even begins, a fixed, one-shot step the model has no further control over. **Agentic RAG** instead exposes retrieval as a **tool** (Module 2) the model can call when, and as many times, as it decides it needs to — directly applying the Module 1.3 ReAct loop specifically to retrieval. This matters because a single, upfront retrieval pass commits to a query before the model has had any chance to reason about what it actually needs; agentic retrieval lets the model reformulate, narrow, or broaden its query based on what an initial pass actually returned — the same way a human researcher iterates on a search rather than searching once, accepting whatever comes back, and stopping.

### Query planning and multi-hop retrieval
Complex information needs often can't be satisfied by a single retrieval query at all — e.g., "what did the team that shipped feature X say about the bug they later found in it" genuinely requires first retrieving *who* shipped feature X, then retrieving *that team's* later discussion, where each retrieval's results determine the next query rather than all queries being knowable upfront. **Query planning** explicitly decomposes a complex information need into an ordered sequence of retrieval sub-queries before (or while) executing them — directly the Module 1.3 Plan-and-Execute / least-to-most decomposition pattern, applied specifically to retrieval instead of general task execution.

### Re-ranking and source verification before synthesis
Raw retrieval — especially pure vector-similarity search — returns results ranked by surface-level semantic closeness, which is not the same thing as relevance, recency, or reliability. A **re-ranking** step (often a separate, more computationally expensive but more accurate model) re-scores the initial retrieved candidate set before it's actually used for synthesis. A **source verification** step goes further, checking whether a retrieved source genuinely *supports* the specific claim being made, not merely whether it's topically related — directly applying Module 4.6's faithfulness/grounding principle to agentic RAG specifically. This distinction matters more than it might first appear: an agent that retrieves a topically-relevant-but-actually-unsupportive source and synthesizes a confident, well-cited-looking answer from it produces a *more* dangerous failure than simply having no answer at all, precisely because it looks well-sourced.

### 📋 Interview Questions — 8.4
1. **What's the core architectural difference between traditional RAG and agentic RAG?**
   *Look for: traditional RAG performs one fixed, upfront retrieval pass before generation; agentic RAG exposes retrieval as a tool the model can call repeatedly and adaptively, reformulating its query based on what previous calls returned.*
2. **Give an example of a query that genuinely requires multi-hop retrieval rather than a single retrieval pass, and explain why a single pass would fail it.**
   *Look for: a query where the second retrieval's target depends on the first retrieval's result (e.g., "find who did X, then find what they said about Y") — a single pass can't know what to search for in the second hop before the first hop's result is known.*
3. **Why is re-ranking retrieved results a separate, additional step from the initial retrieval itself, rather than just retrieving better in the first place?**
   *Look for: initial retrieval (especially vector similarity) is optimized for speed/recall over a large corpus; re-ranking applies a more expensive, more accurate model only to the smaller candidate set retrieval already narrowed down — a cost-effective two-stage tradeoff.*
4. **Why is "the agent retrieved a real, topically-relevant source but synthesized an unsupported claim from it" arguably worse than the agent having no source at all?**
   *Look for: it produces a confidently wrong, well-sourced-*looking* answer that's harder for a reader to catch and distrust than an obviously unsupported claim would be — the appearance of grounding without the substance is more dangerous than visible uncertainty.*
5. **How does agentic RAG's iterative retrieval connect to the ReAct pattern from Module 1.3, specifically?**
   *Look for: recognizing retrieval-as-tool-call fits the same thought→action→observation loop, with "retrieve" as the action and the returned results as the observation that shapes the next reasoning step — not a fundamentally new pattern, an application of an existing one to a specific tool.*

---

## 8.5 Long-Horizon / Autonomous Agents 🆕

### Continuous/background operation patterns
Long-horizon agents run **detached** from an active, synchronous user session — kicked off by some trigger, running unattended for minutes, hours, or longer, and reporting back when finished (or when they genuinely need input) rather than blocking a chat window the entire time. A concrete pattern that's become common in coding-agent tooling specifically by 2026: a ticket, a Slack mention, or a webhook fires; an isolated cloud sandbox spins up; the agent runs inside that sandbox with its full configured context (tools, project instructions, sub-agents); and a pull request or result comes back for review once it's done. This directly solves a real limitation of interactive sessions: a session tied to one developer's laptop and one local checkout can't simply be "walked away from" the way a properly decoupled background run can.

### Scheduling and triggering (cron, event-driven)
Two basic mechanisms kick off a long-horizon agent's run, mirroring a classic systems-design tradeoff:
- **Time-based (cron-style)** — "run this monitoring/reporting task every morning at 8am," the same scheduling pattern from Module 5.5's data-pipeline orchestrators, applied to an agentic task instead of a data transformation. Simple and predictable, but can waste effort checking for nothing-to-do on every scheduled run.
- **Event-driven** — a webhook, a new ticket, or an inbound message triggers an on-demand run. More efficient and responsive, but requires the triggering event source to be reliable and the agent's own triggering logic to correctly filter signal from noise — directly the same "alert on meaningful changes" discipline from Module 5.4, just applied to "should this even trigger an agent run" rather than "should this generate a notification."

### State checkpointing across very long tasks
A genuinely long-horizon agent needs **durable** state, not just in-memory conversation history, so a multi-hour or multi-day task can survive a process restart, a deployment, or simply being deliberately paused and resumed later — the same durable-execution discipline from Module 3.2's LangGraph checkpoints and Module 1.6's asynchronous approvals, but stress-tested over a much longer time horizon, where "just keep it in the conversation history" stops being a remotely viable strategy regardless of how large the context window is — a context window, however generous, is not a substitute for genuine persistent task-state storage.

There's also a risk specific to long horizons worth naming explicitly: the longer a task runs unattended, the more opportunity there is for the kind of silent error-compounding described in Module 1.5 to accumulate completely undetected before anyone notices. This makes periodic self-checks, intermediate human-visible checkpoints, and scoped sub-task boundaries (Module 7.3) *more* important the longer a task is expected to run — not less, as might be intuitively assumed about a system that's "proven itself" by running successfully for a while already.

### 📋 Interview Questions — 8.5
1. **What real limitation of interactive coding-agent sessions does a background/detached operation pattern specifically solve?**
   *Look for: an interactive session is tied to one machine and one local checkout and can't be "walked away from"; a decoupled background run, spun up in its own sandbox on trigger, removes that single-machine dependency entirely.*
2. **What's the core tradeoff between cron-style and event-driven triggering for a long-horizon agent, and how does it mirror a pattern from Module 5?**
   *Look for: cron is simple/predictable but can waste effort on empty checks; event-driven is more efficient/responsive but depends on reliable triggers and good signal-filtering — directly mirrors Module 5.5's scheduling tradeoff and Module 5.4's noise-filtering discipline.*
3. **Why is "just keep everything in the conversation/context window" not a viable strategy for a multi-day autonomous agent, even with a very large context window?**
   *Look for: a context window, however large, isn't durable storage — it doesn't survive a process restart or a deliberate pause/resume, and genuinely long tasks need actual persistent state storage, the same discipline as LangGraph's checkpointers.*
4. **Why does the risk of undetected error-compounding (Module 1.5) get worse, not better, as a task runs longer unattended?**
   *Look for: more unattended time means more opportunity for a small early error to propagate through subsequent reasoning steps before any human notices — the absence of a visible failure isn't the same as the absence of an actual problem.*
5. **How would you decide where to place intermediate, human-visible checkpoints in a multi-day autonomous agent task?**
   *Look for: at natural sub-task boundaries (Module 7.3) and at points where undetected drift would be costly to discover late — balancing checkpoint frequency against the overhead/friction of interrupting otherwise-autonomous operation too often.*

---

*End of Module 8 detailed notes — 25 interview questions total across 5 sections. Computer use, coding agents, and voice/real-time agents are all areas with genuine, fast-moving product churn — re-verify vendor-specific architecture and benchmark claims against current sources before relying on them, and treat any single vendor's self-reported benchmark number with real skepticism.*
