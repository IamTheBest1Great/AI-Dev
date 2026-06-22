# Module 7 — Multi-Agent Systems: Detailed Notes 🆕

## Table of Contents

- [7.1 Orchestration Patterns](#71-orchestration-patterns)
  - [Supervisor / orchestrator-worker](#supervisor--orchestrator-worker)
  - [Hierarchical](#hierarchical)
  - [Peer-to-peer / swarm](#peer-to-peer--swarm)
  - [Blackboard](#blackboard)
  - [📋 Interview Questions — 7.1](#-interview-questions--71)
- [7.2 Agent-to-Agent Communication Protocols and Message Passing](#72-agent-to-agent-communication-protocols-and-message-passing)
  - [The core problem](#the-core-problem)
  - [🆕 A2A (Agent2Agent) protocol — the emerging standard](#-a2a-agent2agent-protocol--the-emerging-standard)
  - [When the protocol overhead isn't worth it](#when-the-protocol-overhead-isnt-worth-it)
  - [📋 Interview Questions — 7.2](#-interview-questions--72)
- [7.3 Task Decomposition and Delegation Strategies](#73-task-decomposition-and-delegation-strategies)
  - [Breaking down the goal](#breaking-down-the-goal)
  - [Static vs dynamic decomposition](#static-vs-dynamic-decomposition)
  - [Delegation criteria](#delegation-criteria)
  - [Failure handling in delegation](#failure-handling-in-delegation)
  - [📋 Interview Questions — 7.3](#-interview-questions--73)
- [7.4 Parallel Agents / Sub-Agent Spawning](#74-parallel-agents--sub-agent-spawning)
  - [Running sub-tasks concurrently](#running-sub-tasks-concurrently)
  - [Costs and risks](#costs-and-risks)
  - [Dynamic spawning and its own runaway-loop risk](#dynamic-spawning-and-its-own-runaway-loop-risk)
  - [📋 Interview Questions — 7.4](#-interview-questions--74)
- [7.5 Shared State and Coordination Problems](#75-shared-state-and-coordination-problems)
  - [Race conditions](#race-conditions)
  - [Conflicting actions](#conflicting-actions)
  - [Deadlock](#deadlock)
  - [Mitigations](#mitigations)
  - [The broader principle](#the-broader-principle)
  - [📋 Interview Questions — 7.5](#-interview-questions--75)
- [7.6 Cost/Latency Tradeoffs of Multi-Agent vs Single-Agent-with-Tools](#76-costlatency-tradeoffs-of-multi-agent-vs-single-agent-with-tools)
  - [The core question](#the-core-question)
  - [How multi-agent systems multiply cost in non-obvious ways](#how-multi-agent-systems-multiply-cost-in-non-obvious-ways)
  - [When multi-agent genuinely wins](#when-multi-agent-genuinely-wins)
  - [When a single agent with more tools usually wins](#when-a-single-agent-with-more-tools-usually-wins)
  - [Practical heuristic](#practical-heuristic)
  - [📋 Interview Questions — 7.6](#-interview-questions--76)

> Your original outline lists this module as six flat topics rather than numbered subsections — these notes group them as 7.1–7.6 so each gets the same depth and a dedicated interview-question set as every other module, while keeping your original topic order and wording intact.

---

## 7.1 Orchestration Patterns

Four recurring topologies show up across almost every multi-agent framework in Module 3, even when each framework uses its own terminology for them:

### Supervisor / orchestrator-worker
A single central agent (the supervisor) decomposes the overall task, delegates sub-tasks to worker agents, and collects/synthesizes their results. Workers don't talk to each other directly — every piece of communication routes through the supervisor, a **hub-and-spoke topology**. This is the most common and most controllable pattern, and it's what's underneath CrewAI's hierarchical process (Module 3.4), a typical LangGraph supervisor graph (Module 3.2), and Google ADK's parent-child agent composition (Module 3.7) — different frameworks, same underlying topology.

### Hierarchical
A generalization of the supervisor pattern across **multiple levels**: a top-level orchestrator delegates to mid-level supervisors, who themselves delegate further down to worker agents — forming a tree rather than a single flat hub-and-spoke layer. This earns its extra complexity when sub-tasks are themselves complex enough to need their own decomposition — e.g., a top-level "ship the feature" orchestrator delegating to a "backend" sub-supervisor and a "frontend" sub-supervisor, each independently managing their own workers.

### Peer-to-peer / swarm
No central coordinator at all — agents communicate directly with each other and hand off control or collaborate as equals based on the evolving conversation/task state. This is AutoGen's GroupChat model (Module 3.5) in its purest form, and was also the model OpenAI's early experimental "Swarm" framework pioneered before being folded into the now-current Agents SDK. It's more flexible and adaptive than a fixed hierarchy, but — exactly mirroring the agency-spectrum tradeoff from Module 1.1, just one level up — harder to predict, test exhaustively, or debug, because there's no single point where you can observe and constrain the full flow of control.

### Blackboard
A shared, structured workspace (the "blackboard") that every agent can read from and write to; agents act **opportunistically** whenever they notice something useful to contribute to the shared state, rather than being explicitly invoked by a coordinator or a peer's handoff. This pattern predates modern LLM agents entirely — it originates in classical AI systems research (e.g., the Hearsay-II speech recognition system from the 1970s) and is well suited to problems where the right contributor to a given sub-step **isn't known in advance** and only becomes apparent from the current state of the blackboard itself. The tradeoff: it requires careful conflict resolution, since multiple agents can simultaneously want to read or write the same region of shared state (this connects directly to 7.5's coordination problems).

| | Supervisor | Hierarchical | Peer-to-peer/swarm | Blackboard |
|---|---|---|---|---|
| Central control | Yes, one supervisor | Yes, multiple levels | No | No |
| Predictability | High | Moderate | Lower | Lower |
| Debuggability | Easiest — one place to inspect | Moderate — trace down the tree | Hardest — flow isn't centrally observable | Hard — contributions are opportunistic, not scripted |
| Best fit | Most production systems; clear task decomposition | Large tasks with naturally nested sub-problems | Open-ended collaboration, unclear upfront structure | Problems where the right contributor emerges from state, not from a predetermined plan |

### 📋 Interview Questions — 7.1
1. **What's the structural difference between the supervisor pattern and the hierarchical pattern, and when does that difference actually matter?**
   *Look for: hierarchical is a multi-level generalization of supervisor — it matters once individual sub-tasks are themselves complex enough to need their own decomposition, not just simple delegation.*
2. **Why is the peer-to-peer/swarm pattern harder to debug than a supervisor pattern, structurally — not just "it's more complex"?**
   *Look for: there's no single point of control where the full flow can be observed and constrained; in a supervisor pattern, the supervisor's own context/decisions are that single observable point.*
3. **What problem does the blackboard pattern solve that a fixed supervisor-delegation pattern can't handle well?**
   *Look for: situations where the right contributor to a sub-step isn't knowable in advance and only emerges from the evolving shared state — a supervisor would need to know who to delegate to upfront, which blackboard doesn't require.*
4. **A team is building a multi-agent system and defaults to peer-to-peer because "it's more flexible." What would you want to know before agreeing that's the right call?**
   *Look for: whether the task genuinely needs that flexibility, or whether a supervisor pattern would provide the same outcome with far better predictability/debuggability — flexibility should be a deliberate tradeoff, not a default.*
5. **Why does the blackboard pattern specifically require more conflict-resolution machinery than the other three patterns?**
   *Look for: because any agent can write to shared state opportunistically and without coordination, conflicting writes are structurally more likely than in patterns where communication is routed through a single supervisor or explicit peer handoff.*

---

## 7.2 Agent-to-Agent Communication Protocols and Message Passing

### The core problem
Once agents need to exchange task requests, results, and status updates — especially across different codebases, teams, or vendors — you face the same standardization problem Module 6.4 covered for agent-to-*tool* communication, but one level up: agent-to-*agent*. Without a shared protocol, every pair of interoperating agents needs its own bespoke integration, the same N×M problem from Module 6.4, just with agents on both sides instead of an agent and a tool.

### 🆕 A2A (Agent2Agent) protocol — the emerging standard
A2A was created by Google (April 2025) and donated to the **Linux Foundation** two months later for neutral, vendor-agnostic governance under its Agentic AI Foundation. The adoption trajectory has been genuinely fast: **150+ supporting organizations**, deep production integration across Google, Microsoft, AWS, Salesforce, SAP, and ServiceNow platforms, and over 22,000 GitHub stars within roughly its first year — a meaningfully different adoption curve than most emerging protocols see.

Technically, A2A runs over **JSON-RPC** (with gRPC added as an alternative binding as of v1.0), and its core discovery mechanism is the **Agent Card** — a document an agent publishes describing its capabilities, conceptually similar to an OpenAPI spec but describing agent capabilities rather than REST endpoints. As of the v1.0/v1.2 releases (reached in 2026), A2A added:
- **Signed Agent Cards** — a cryptographic signature lets a receiving agent verify a card was actually issued by the domain it claims, which is the trust mechanism that makes *decentralized* agent discovery viable at all (without it, anyone could publish a forged card claiming to be a different, trusted agent).
- **Multi-tenancy** — a single endpoint can host multiple distinct agents, letting a SaaS provider serve different agents per customer/tenant from one deployment.
- **Agent Payments Protocol (AP2)** — an extension specifically for agent-initiated payments, reflecting how far "agents acting autonomously in the world" has progressed as a real production concern, not just a research idea.

**The mental model worth memorizing**: **MCP is the vertical layer (agent ↔ tool/data), A2A is the horizontal layer (agent ↔ agent)** — these are explicitly designed as complementary, not competing, standards. A real production multi-agent system commonly uses both simultaneously: each agent uses MCP to reach its own tools and data sources, and A2A to delegate to or coordinate with other agents, possibly built on entirely different frameworks or owned by different teams/vendors.

### When the protocol overhead isn't worth it
When all the agents in a system run in the same codebase/process and there's no need for cross-vendor, cross-framework interoperability, plain in-process message passing — direct function calls, a shared object, or an internal message queue — is simpler and has none of A2A's protocol overhead (capability discovery, card verification, network round-trips). A2A earns its overhead specifically when agents *might* live in different codebases, teams, or vendor platforms — the same "is this reusable across multiple consumers" calculus from Module 6.4, applied to agent-to-agent instead of agent-to-tool.

### 📋 Interview Questions — 7.2
1. **What's the precise distinction between what MCP standardizes and what A2A standardizes?**
   *Look for: MCP = agent-to-tool/data (vertical); A2A = agent-to-agent (horizontal) — they're complementary layers a single system can use together, not competing alternatives.*
2. **Why does "Signed Agent Cards" matter specifically for *decentralized* agent discovery, more than it would in a tightly controlled internal system?**
   *Look for: without cryptographic verification, anyone could publish a forged card claiming to be a trusted agent/domain — signing is what makes open, decentralized discovery trustworthy rather than just convenient.*
3. **A startup is building three internal agents that will always run in the same Python process. Would you recommend they adopt A2A for communication between them? Why or why not?**
   *Look for: probably not — A2A's overhead (discovery, card verification, network protocol) isn't justified when there's no cross-codebase/cross-vendor interoperability need; plain in-process function calls or a message queue are simpler and sufficient.*
4. **What does A2A's Agent Payments Protocol (AP2) extension tell you about how far autonomous agent deployment has progressed in production?**
   *Look for: recognizing that agent-initiated payments being standardized at the protocol level reflects agents now taking genuinely consequential real-world actions in production, not just research demos — a sign of how mainstream autonomous-agent deployment has become.*
5. **Why might a company choose to support both MCP and A2A in the same product, rather than picking one?**
   *Look for: they solve different problems (tool access vs. agent coordination) and are explicitly designed to be used together — supporting only one would leave either the tool-access or the agent-coordination half of the system without a standard.*

---

## 7.3 Task Decomposition and Delegation Strategies

### Breaking down the goal
Decomposition means splitting a large goal into smaller, assignable sub-tasks — ideally structured so each sub-task maps naturally to whichever agent (or agent role) is best suited to it. This applies the same reasoning patterns from Module 1.3 (Plan-and-Execute, least-to-most decomposition), with one key difference: the "executor" for each sub-task here is a **distinct agent**, not the same agent looping back on itself.

### Static vs dynamic decomposition
- **Static decomposition** — the plan and the assignment of sub-tasks to specific agents is fixed upfront, closer to a predefined workflow (Module 1.1's agency spectrum). More predictable and testable, at the cost of flexibility if the task turns out to need a different breakdown than anticipated.
- **Dynamic decomposition** — a supervisor/orchestrator agent decides delegation on the fly, based on intermediate results as they come in. More adaptive to genuinely unpredictable tasks, but — the same tradeoff that recurs throughout this curriculum — harder to test exhaustively and less predictable in production.

### Delegation criteria
A good delegation strategy weighs several factors, not just "which agent is generically capable":
- **Role/specialization fit** — matching a sub-task to an agent whose role/persona is suited to it (the same role-as-prompt principle from CrewAI in Module 3.4).
- **Current load/availability** — avoiding overloading one worker agent with a queue of sub-tasks while others sit idle, especially relevant once you're running many agents concurrently (7.4).
- **Task dependency ordering** — a sub-task that depends on another's output can't be delegated or started until that dependency resolves, the same dependency-tracking discipline as CrewAI's `context` parameter (Module 3.4) or a data pipeline's task graph (Module 5.5).

### Failure handling in delegation
What happens when a delegated sub-task fails needs to be designed explicitly, not left implicit: does the supervisor retry the same sub-task with a different worker agent, re-decompose the overall task differently given the new information that this sub-task failed, or escalate to a human (Module 1.6)? This is the orchestration-layer version of Module 1.5's single-agent failure modes — the same discipline of having an explicit, designed response to failure (rather than blind retry) applies one level up.

### 📋 Interview Questions — 7.3
1. **What's the practical tradeoff between static and dynamic task decomposition, and how does it mirror a distinction from Module 1.1?**
   *Look for: static = more predictable/testable but less adaptive (workflow-like); dynamic = more adaptive but less predictable (agent-like) — the same agency-spectrum tradeoff, applied at the decomposition-strategy level rather than the single-action level.*
2. **Why does delegation based purely on "which agent is generically capable" miss something important compared to factoring in current load?**
   *Look for: even a capable agent can become a bottleneck if it's already handling a queue of other sub-tasks while other equally-capable agents sit idle — load-awareness matters for actual throughput, not just task-agent fit.*
3. **A delegated sub-task fails. What are the three broad response options a supervisor could take, and how would you decide between them?**
   *Look for: retry with a different worker, re-decompose the task differently, or escalate to a human — the choice depends on whether the failure suggests a worker-specific problem, a task-decomposition problem, or something requiring human judgment.*
4. **Why can't a sub-task with an unresolved dependency simply be delegated immediately alongside its independent siblings?**
   *Look for: it would either execute on stale/missing input or need to block/retry until the dependency resolves — dependency ordering has to be tracked explicitly, the same discipline as a data pipeline's task graph.*
5. **What's the risk of a supervisor that always responds to a delegated sub-task's failure by simply retrying it with the same worker, unchanged?**
   *Look for: this is the multi-agent version of Module 1.5's infinite-loop/blind-retry failure mode — without changing something (a different worker, a different decomposition, human input), a retry just repeats the same failure.*

---

## 7.4 Parallel Agents / Sub-Agent Spawning

### Running sub-tasks concurrently
When sub-tasks are genuinely independent (no shared dependency between them), an orchestrator can spawn multiple worker agents to run **concurrently** rather than sequentially — directly analogous to Module 2.1's parallel tool calls, except each "call" here is a full agent with its own reasoning loop, not a single function invocation. The benefit is real and significant: a task like "research these 5 competitors" can drop from 5x sequential latency to roughly 1x wall-clock latency by running 5 research sub-agents in parallel instead of one after another.

### Costs and risks
Parallelism doesn't make the work free, it just changes *where* the cost shows up:
- **Multiplied compute/token cost** — N concurrent agents cost roughly N times as much as one, even though wall-clock time drops; you're trading money for latency, not eliminating the underlying cost.
- **Resource contention** — sub-agents sharing an underlying constrained resource (the same rate-limited API, the same browser session pool from Module 4.2) can bottleneck or interfere with each other even while running "in parallel," undermining the latency benefit you were trying to capture.
- **Result-aggregation complexity** — someone has to merge N independent agents' outputs into one coherent final result. This is a genuinely non-trivial synthesis task in its own right (resolving overlaps, contradictions, differing confidence levels across sub-agent outputs), not just string concatenation — and it deserves the same design attention as any other step in the pipeline, typically falling on the supervisor from 7.1.

### Dynamic spawning and its own runaway-loop risk
An agent deciding **mid-task** to spawn additional sub-agents for newly discovered sub-problems (rather than a fixed N decided upfront) is more adaptive, but introduces its own version of Module 1.5's runaway-loop failure mode — except now the "loop" is uncontrolled growth in the *number of agents* rather than the number of steps one agent takes. The mitigation is structurally the same: explicit circuit breakers and a hard cap (max concurrent sub-agents, max spawn depth), not an unbounded "spawn whenever it seems useful" policy.

### 📋 Interview Questions — 7.4
1. **Why does spawning N parallel sub-agents reduce wall-clock latency without reducing total cost — and why is that distinction important to communicate to stakeholders?**
   *Look for: parallelism trades cost for latency, it doesn't eliminate the underlying compute cost — N agents still cost roughly N times as much, even though the task finishes faster; conflating "faster" with "cheaper" sets the wrong expectation.*
2. **Two sub-agents run in parallel against the same rate-limited third-party API. What's likely to happen, and how would you redesign around it?**
   *Look for: they'll contend for the same rate limit, likely causing throttling/failures that erode or eliminate the latency benefit — redesign by either serializing access to that specific shared resource or provisioning separate rate-limit allocations per sub-agent.*
3. **Why is "merge the sub-agents' outputs" not a trivial final step, and what design attention does it actually need?**
   *Look for: independent sub-agent outputs can overlap, contradict, or carry different confidence levels — synthesis requires real reasoning (resolving conflicts, deciding what to keep), not just concatenating text; this deserves the same care as any other pipeline step.*
4. **What's the multi-agent analog of Module 1.5's infinite-loop failure mode, and how would you guard against it?**
   *Look for: uncontrolled dynamic spawning of additional sub-agents mid-task — guard with explicit caps on max concurrent sub-agents and max spawn depth, the same circuit-breaker discipline applied to agent count instead of step count.*
5. **A task decomposes into 5 sub-problems, but only 2 are genuinely independent of each other. How would that change your approach to parallelizing it?**
   *Look for: only the genuinely independent sub-problems should run in parallel; the dependent ones need to respect their ordering (7.3) — naively parallelizing all 5 risks agents acting on stale or missing inputs from their dependencies.*

---

## 7.5 Shared State and Coordination Problems

### Race conditions
Two or more agents read a piece of shared state, each independently decides on an action based on that (now possibly stale) read, and both act — e.g., two agents both see "ticket is unassigned" and both assign themselves to it, or two agents both read an inventory count and both commit to fulfilling an order only one of them can actually complete. The core problem is the **gap between reading state and acting on it**, during which the state can change underneath an agent without it knowing.

### Conflicting actions
Even without a literal race condition on the same piece of state, two agents can each take an individually reasonable action that's mutually inconsistent with the other — e.g., one agent schedules a meeting at 2pm based on its view of a calendar, while another agent, working from slightly different or slightly stale context, cancels that same meeting moments later.

### Deadlock
Two or more agents each waiting on a resource or piece of information the other holds or controls, with neither able to proceed — directly analogous to classic OS-level deadlock, just caused by agentic decision loops (an agent waiting for another agent's output before it will act) rather than literal lock acquisition.

### Mitigations
These are genuinely the same category of problem traditional distributed/concurrent systems have dealt with for decades, just with LLM agents as the actors instead of threads or processes:
- **Locking/transactional patterns** — acquire an explicit lock on a shared resource before an agent acts on it, release it after, borrowed directly from traditional distributed-systems practice.
- **Blackboard-style conflict resolution** — a moderator/controller component that decides which agent's proposed write actually gets applied when conflicting writes are detected, rather than allowing silent last-write-wins corruption.
- **Favoring the supervisor/hierarchical pattern (7.1)** specifically *because* routing all coordination through one central point sidesteps most peer-to-peer race conditions by construction — at the real cost of less autonomy/flexibility, which is exactly why supervisor patterns dominate in production despite peer-to-peer's theoretical appeal.
- **Idempotency and optimistic-concurrency patterns** (the same idempotency discipline from Module 5.5) so that even if two agents do attempt the same action, the second attempt is a safe no-op rather than a duplicate or conflicting side effect.

### The broader principle
Multi-agent coordination problems are genuinely **a distributed-systems problem wearing an AI costume**. The same underlying theory — locking, consensus, optimistic concurrency, the tradeoffs distributed-systems engineers have formalized for decades — applies directly here. Teams without distributed-systems background tend to rediscover these problems the hard way (in production, as confusing intermittent bugs) rather than importing well-understood solutions from the start.

### 📋 Interview Questions — 7.5
1. **What's the precise difference between a race condition and a deadlock in a multi-agent system?**
   *Look for: a race condition is two agents acting on a stale/shared read without coordination (both proceed, conflicting outcome); deadlock is two agents each waiting on something the other holds, with neither able to proceed (neither acts at all).*
2. **Why does the supervisor pattern reduce coordination problems "by construction," and what's the cost of that benefit?**
   *Look for: routing all communication through one central point eliminates most peer-to-peer race conditions structurally, since agents never act on directly-shared, uncoordinated state — the cost is reduced autonomy/flexibility compared to a peer-to-peer design.*
3. **How does idempotency design (from Module 5.5) help with multi-agent race conditions specifically?**
   *Look for: even if two agents both attempt the same action due to a race, idempotent design means the second attempt is a safe no-op rather than a duplicated/conflicting side effect — it doesn't prevent the race, but it neutralizes its consequences.*
4. **Why would you describe multi-agent coordination as "a distributed-systems problem wearing an AI costume," and why does that framing matter practically?**
   *Look for: the underlying problems (races, conflicting writes, deadlock) and their solutions (locking, consensus, optimistic concurrency) are decades-old distributed-systems concepts — framing it this way means importing known solutions instead of re-deriving them from scratch through painful trial and error in production.*
5. **A blackboard-pattern system has two agents simultaneously trying to write different values to the same key. What's the risk of not having an explicit conflict-resolution mechanism, and what would one look like?**
   *Look for: risk is silent last-write-wins corruption with no record of the conflict; a mitigation is a moderator/controller component that explicitly arbitrates which write is applied (and ideally logs the conflict) rather than letting the underlying storage layer resolve it implicitly.*

---

## 7.6 Cost/Latency Tradeoffs of Multi-Agent vs Single-Agent-with-Tools

### The core question
Before reaching for a multi-agent architecture, the question to ask is genuinely simple: **does this task benefit from being split across multiple distinct agents, or would a single agent with the same overall tool access perform just as well at a fraction of the cost and complexity?** Multi-agent systems are not a strictly more powerful upgrade over a single agent with tools — they're a different tool with real, non-obvious costs.

### How multi-agent systems multiply cost in non-obvious ways
Every agent carries its own context window and reasoning overhead, and in a supervisor pattern (7.1) specifically, the supervisor's own context grows as it both delegates work *and* has to read every worker's result back in to synthesize a final answer. In measured comparisons, multi-agent setups have been shown to consume meaningfully more total tokens than an equivalent single agent for tasks that didn't actually require decomposition — without a corresponding improvement in output quality. The orchestration overhead is real, not just a theoretical concern.

### When multi-agent genuinely wins
- **Context-window pressure** — the task's necessary state/reasoning genuinely doesn't fit in one agent's context window, so splitting it is the only way to keep each agent's working context manageable.
- **Genuine role/expertise specialization** — tasks that benefit from distinctly different "expertise" framings (Module 3.4's role-as-prompt principle), where forcing one agent to context-switch between very different personas measurably degrades quality compared to dedicated, specialized agents.
- **Real parallelism opportunity** — tasks with genuinely independent sub-problems where wall-clock latency reduction (7.4) matters more than the multiplied compute cost it requires.
- **Permission/safety isolation** — giving a sub-agent narrow, scoped tool access for a specific risky sub-task, rather than one agent holding broad permissions for everything it might ever need — directly applying Module 9.3's least-privilege principle at the architecture level, not just the individual-tool level.

### When a single agent with more tools usually wins
Tasks that fit comfortably within one context window, don't need genuinely distinct "roles," and don't have real parallelism to exploit. In these cases, the entire orchestration problem space from 7.1–7.5 (topology design, communication protocol overhead, coordination/race-condition risk) is a cost being paid for no corresponding benefit.

### Practical heuristic
**Default to a single agent with tools.** Only move to a multi-agent architecture once you have a *concrete, specific* reason — context-window pressure, a measured role-specialization quality gain, real exploitable parallelism, or a genuine permission-isolation requirement — that a single agent genuinely can't satisfy. Multi-agent architecture should be a deliberate response to a demonstrated need, not a default starting design, however appealing the idea of a "team of AI agents" sounds.

### 📋 Interview Questions — 7.6
1. **Why can a multi-agent system end up costing more in total tokens than a single agent, even when it produces a comparable-quality result?**
   *Look for: every agent carries its own context/reasoning overhead, and a supervisor's context grows from both delegating work and re-reading every worker's output to synthesize a final answer — overhead that doesn't exist in a single-agent design.*
2. **A candidate proposes a 4-agent system for a task that clearly fits within one model's context window and has no parallelizable sub-problems. How would you push back?**
   *Look for: applying the practical heuristic — without context-window pressure, genuine role-specialization benefit, real parallelism, or permission isolation, the added orchestration complexity and cost has no offsetting benefit; recommend starting with a single agent with tools.*
3. **How does the "permission isolation" justification for multi-agent design connect to a principle from Module 9?**
   *Look for: it's the architecture-level application of Module 9.3's least-privilege principle — scoping a risky sub-task to a narrowly-permissioned sub-agent instead of giving one agent broad permissions for everything it might ever need.*
4. **What's the difference between a task that genuinely needs role-specialized agents versus one where a single agent could just be prompted to "think like X, then like Y"?**
   *Look for: genuine need exists when context-switching between personas measurably degrades a single agent's output quality; if a single well-prompted agent performs comparably, separate agents add cost/complexity without a real quality benefit — this should be tested, not assumed.*
5. **Why should "multi-agent architecture" be treated as a deliberate response to a demonstrated need rather than a default design choice, even though it's often the more exciting-sounding option?**
   *Look for: recognizing the real, multiplied costs (compute, coordination complexity, debuggability loss from 7.1/7.5) mean multi-agent should be justified by a specific limitation of single-agent design, not chosen by default because it sounds more sophisticated or impressive.*

---

*End of Module 7 detailed notes — 30 interview questions total across 6 sections. The A2A protocol details here are a mid-2026 snapshot of a fast-adopting but still-maturing standard — re-verify version/feature specifics against the Linux Foundation's A2A documentation before architecture decisions.*
