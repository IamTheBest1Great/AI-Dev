# Module 10 — Evaluation & Testing: Detailed Notes 🆕

## Table of Contents

- [10.1 Agent Evaluation Frameworks](#101-agent-evaluation-frameworks)
  - [Why agent evaluation is harder than LLM evaluation](#why-agent-evaluation-is-harder-than-llm-evaluation)
  - [LLM-as-judge](#llm-as-judge)
  - [Known biases in LLM judges](#known-biases-in-llm-judges)
  - [Rubric-based grading](#rubric-based-grading)
  - [The three-tier evaluation stack](#the-three-tier-evaluation-stack)
  - [📋 Interview Questions — 10.1](#-interview-questions--101)
- [10.2 Public Benchmark Suites for Calibration](#102-public-benchmark-suites-for-calibration)
  - [Why public benchmarks matter — and why they're not enough](#why-public-benchmarks-matter--and-why-theyre-not-enough)
  - [SWE-bench Verified](#swe-bench-verified)
  - [GAIA](#gaia)
  - [WebArena](#webarena)
  - [AgentBench](#agentbench)
  - [τ-bench (tau-bench)](#-bench-tau-bench)
  - [🆕 A critical caveat: the April 2026 reward-hacking disclosure](#-a-critical-caveat-the-april-2026-reward-hacking-disclosure)
  - [📋 Interview Questions — 10.2](#-interview-questions--102)
- [10.3 Regression Testing for Agent Behavior](#103-regression-testing-for-agent-behavior)
  - [Why regression testing is harder for agents than for software](#why-regression-testing-is-harder-for-agents-than-for-software)
  - [Building a regression test suite](#building-a-regression-test-suite)
  - [Flakiness and statistical significance](#flakiness-and-statistical-significance)
  - [📋 Interview Questions — 10.3](#-interview-questions--103)
- [10.4 A/B Testing Prompts, Tool Sets, and Model Versions](#104-ab-testing-prompts-tool-sets-and-model-versions)
  - [What makes agent A/B testing different from product A/B testing](#what-makes-agent-ab-testing-different-from-product-ab-testing)
  - [Prompts as versioned, testable artifacts](#prompts-as-versioned-testable-artifacts)
  - [Testing tool-set changes](#testing-tool-set-changes)
  - [Testing model-version upgrades](#testing-model-version-upgrades)
  - [📋 Interview Questions — 10.4](#-interview-questions--104)
- [10.5 Building Simulated Environments / Sandboxes](#105-building-simulated-environments--sandboxes)
  - [Why simulation is necessary](#why-simulation-is-necessary)
  - [What a good simulation environment needs](#what-a-good-simulation-environment-needs)
  - [Simulation fidelity tradeoffs](#simulation-fidelity-tradeoffs)
  - [📋 Interview Questions — 10.5](#-interview-questions--105)
- [10.6 Defining Success Metrics Beyond "Did It Run"](#106-defining-success-metrics-beyond-did-it-run)
  - [The metric hierarchy](#the-metric-hierarchy)
  - [Task success rate](#task-success-rate)
  - [Step efficiency](#step-efficiency)
  - [Cost per task](#cost-per-task)
  - [Reliability and variance](#reliability-and-variance)
  - [Safety and constraint adherence](#safety-and-constraint-adherence)
  - [Composing metrics into a decision framework](#composing-metrics-into-a-decision-framework)
  - [📋 Interview Questions — 10.6](#-interview-questions--106)

> This is the biggest structural gap in most agent curricula. An agent that isn't evaluated rigorously fails in production in ways that are invisible until they're expensive. Everything in Modules 1–9 teaches you how to build agents; this module teaches you how to actually know whether they work.

---

## 10.1 Agent Evaluation Frameworks

### Why agent evaluation is harder than LLM evaluation
Standard LLM evaluation is hard. Agent evaluation is harder, for structural reasons that don't go away with better tooling:
- **Non-determinism compounds across steps.** A single LLM call has variance; a 10-step agent run multiplies that variance across each step. The same task, with the same inputs, can succeed or fail on different runs — which means a single-run pass/fail score is almost meaningless. The variance has to be measured, not hidden.
- **The path matters, not just the outcome.** A correct final answer reached via three hallucinated intermediate steps that happened to cancel out is not the same as a correct final answer reached via sound reasoning — and is far more fragile in production. Standard output-only evaluation misses this entirely.
- **There is no single "correct output" for most agentic tasks.** An agent writing a research report, scheduling a meeting, or debugging a codebase can succeed through many different valid trajectories — making automatic binary-correct-wrong grading inapplicable to most of what matters in practice.
- **Real costs are incurred during evaluation.** An agent's tool calls (API requests, browser sessions, code execution) cost real money and may have real side effects — you can't run a thousand evaluation attempts cheaply the way you can with a language modeling benchmark.

### LLM-as-judge
LLM-as-judge uses an LLM to score the outputs of another LLM against criteria you define — answer relevancy, faithfulness, helpfulness, bias, correctness. An LLM judge agrees with human reviewers about 85% of the time — higher than two humans agree with each other on the same task. This agreement rate, consistently replicated across studies, is the core argument for LLM-as-judge: it's scalable (no annotation team, no weeks-long process, no five-figure human eval contract) while producing human-level agreement on most evaluation tasks.

Two scoring modes:
- **Pointwise scoring** — the judge evaluates one output against a rubric, assigns a score on a defined scale (e.g., 1–5). Simpler to implement, but more sensitive to the judge's own calibration drift.
- **Pairwise comparison** — the judge compares two outputs (e.g., version A vs. version B of a prompt) and decides which is better. More reliable than pointwise scoring for detecting regressions (Module 10.3), because relative judgment is less sensitive to absolute score drift — but doesn't give you a standalone quality number, only a preference signal.

### Known biases in LLM judges
Used carelessly, LLM-as-judge produces scores that look reasonable and are wrong. The failure mode: a team builds a judge using GPT-4 and uses it to score GPT-4 production outputs. For three months, every dashboard glows green. Then a domain expert reads 50 production outputs — the expert's agreement with the judge is only 0.31 (Cohen's kappa). The judge has been over-rewarding GPT-4 outputs systematically (family bias) and under-penalizing fluent hallucinations (length-confidence bias). The dashboards have been lying for three months.

The named biases that recur most consistently in empirical literature:
- **Family bias (self-preference bias)** — a judge from the same model family as the system being evaluated systematically over-rates that system's outputs. Use a different model family as your judge wherever possible.
- **Length-confidence bias** — longer, more verbose outputs tend to receive higher scores regardless of actual quality, because length reads as thoroughness. Rubrics should explicitly penalize verbosity-over-substance to counteract this.
- **Position bias** — in pairwise comparison, the judge tends to prefer whichever output appears first in the prompt. Randomize order and average scores across both orderings.
- **Fluency bias** — grammatically fluent, well-formatted outputs receive higher scores even when factually wrong. Rubrics need explicit factual-correctness criteria, separate from style/fluency criteria.

The newest methodological frontier is bias-corrected and psychometric judge evaluation — applying calibration-based bias correction and confidence intervals to account for imperfect judge sensitivity and specificity, and applying item response theory to judges themselves, framing reliability as a property of the measurement instrument rather than only of the outputs being measured. This means teams can start asking which rubric items are too easy, too ambiguous, or too judge-sensitive — not just which model "won."

### Rubric-based grading
A rubric decomposes "good output" into explicit, separately-scored criteria rather than asking for a single holistic judgment. This matters for reliability because holistic "is this good?" judgments are maximally subject to all the biases listed above, while criterion-separated scoring forces the judge to evaluate specific, independently definable properties.

Verifiable rewards are best for "did the model solve it," while rubrics remain necessary for "did the model explain it safely, use tools correctly, cite appropriately, or communicate well." The current research consensus is that rubrics improve reliability when the task is open-ended and the rubric is explicit, criterion-separated, and calibrated.

A practical rubric for an agentic task typically separates:
1. **Task completion** — did the agent accomplish the stated goal?
2. **Tool use correctness** — were tools called with correct arguments, in a sensible order, without unnecessary calls?
3. **Reasoning quality** — was the intermediate reasoning (thought process) coherent and grounded in available information?
4. **Constraint adherence** — did the agent stay within defined boundaries (budget, scope, permitted actions)?
5. **Output quality** — for tasks with a deliverable, does the deliverable meet defined quality standards?

### The three-tier evaluation stack
LLM evaluation architecture that scales: heuristics on every span, distilled judges on a sample, humans on the gold-set.

- **Tier 1 — Heuristics on every span** (cheap, fast, fully automated): format checks, length checks, tool-call schema validation, forbidden-string detection. These run on 100% of traffic and catch obvious failures at near-zero marginal cost.
- **Tier 2 — LLM judge on a sample** (moderate cost, scalable): an LLM judge evaluates a defined sample of runs against a rubric, generating quality scores for cases that passed Tier 1 but may still have substantive quality issues.
- **Tier 3 — Human evaluation on the gold set** (expensive, high-signal): a curated set of representative cases, evaluated by humans, used both to calibrate the Tier 2 judge and to catch the failures that slip past automated evaluation. **Cohen's kappa between the Tier 2 judge and human raters is the single most important number in the whole evaluation pipeline** — if it falls below a threshold (commonly cited: below 0.6 is inadequate, 0.6–0.8 is moderate, above 0.8 is strong), the judge cannot be trusted as a proxy for human judgment regardless of how good the rubric looks.

### 📋 Interview Questions — 10.1
1. **Why does non-determinism compound across steps in a multi-step agent in a way it doesn't for a single LLM call, and what does that imply for how you report evaluation results?**
   *Look for: variance multiplies across each step — a 10-step agent's run-to-run variability is much higher than a single call's, meaning a single-run score is nearly meaningless; results need to be averaged across multiple runs and confidence intervals reported.*
2. **A team uses GPT-4o as their judge to evaluate their GPT-4o-based agent. What specific bias are they likely to introduce, and how would you fix it?**
   *Look for: family/self-preference bias — the same model family systematically over-rates its own outputs; fix by using a different model family (e.g., Claude as judge for a GPT-based system) and calibrating against human ratings.*
3. **What is Cohen's kappa, why is it the right metric for judge calibration rather than simple percentage agreement, and what threshold would concern you?**
   *Look for: kappa accounts for chance agreement (two raters randomly agreeing some of the time), which raw percentage agreement ignores — below ~0.6 suggests the judge can't be trusted as a human proxy; above ~0.8 is generally considered strong.*
4. **Why is a holistic "rate this output 1–5" rubric less reliable than a criterion-separated rubric, even when both are evaluated by the same judge?**
   *Look for: holistic judgments concentrate all biases (length, fluency, family) into one number; criterion-separated rubrics force independent evaluation of each dimension, making specific biases both more visible and easier to counteract.*
5. **What does the three-tier evaluation stack (heuristics → LLM judge → human gold set) optimize across its three tiers, and why is each tier necessary rather than just using the most accurate one everywhere?**
   *Look for: Tier 1 = 100% coverage at near-zero cost; Tier 2 = quality signal at scale without per-run human cost; Tier 3 = calibration and gold-truth that keeps Tier 2 honest — each tier is necessary because cost and coverage trade off, and none alone gives you both scale and accuracy.*

---

## 10.2 Public Benchmark Suites for Calibration

### Why public benchmarks matter — and why they're not enough
Public benchmarks serve a specific, limited function: **calibrating your intuition about how your agent compares to the broader state of the art on a defined task type**. They are emphatically not a substitute for evaluating your agent on your actual task distribution. Agent benchmarks measure something fundamentally different from LLM benchmarks. The question is not "can the model answer this question?" but "can the model complete this multi-step task that requires tools, state management, and error recovery?" A model that scores 94% on MMLU-Pro can still fail at a multi-step WebArena task that requires synthesizing information across several pages, entering it correctly into a form, and verifying the result.

### SWE-bench Verified
2,294 real GitHub issues drawn from popular Python repositories, each requiring the agent to produce a code patch that resolves the described bug — verified automatically by running the actual test suite. The "Verified" subset is a human-validated sample where human annotators confirmed the issue and patch are well-specified (filtering out ambiguous or underspecified cases from the full benchmark that would make automated grading unreliable).

SWE-bench Verified leaderboard in April 2026: Claude Opus 4.7 at 87.6%, GPT-5.3 Codex at 85.0%, with the average of 83 models at 63.4%. The practical implication of a rising average: the baseline is no longer impressively low — simply "using Claude" or "using an agent framework" gets you into the average range without any careful engineering. Differentiating above average requires the tooling, context management, and test-driven verification practices from Module 8.2.

**What to watch when interpreting these scores**: solution leakage is a structural vulnerability — SWE-bench tasks come from public GitHub issues whose solutions exist in the same repository's git history, meaning models trained on sufficiently recent data may have partial exposure to the answers. Per Module 4.6's principle about benchmark skepticism: treat vendor-reported SWE-bench numbers as directional, not as audited ground truth, and prefer scores independently verified by Epoch AI or similar independent evaluators.

### GAIA
466 questions requiring multi-step reasoning that chains web browsing, file parsing (PDFs, images, spreadsheets), code execution, and multi-document synthesis — all for questions that have a single, exact correct answer verifiable by string match. GAIA explicitly tests the compound, multi-tool capabilities that single-modality benchmarks miss.

When GAIA launched in 2023, GPT-4 with plugins scored 15%; humans scored 92%. Today the top agents reach approximately 75%. Critically: the same Claude Opus 4 scores 64.9% inside one agent framework and 57.6% inside another — a 7-point gap that comes from the orchestration layer alone. That framework-sourced gap is arguably more informative than the raw scores: it quantifies how much the scaffold (tools, prompting, orchestration) contributes to real performance, independent of the underlying model.

Claude Sonnet 4.5 leads GAIA (Princeton HAL leaderboard) at 74.6%, with Anthropic models sweeping the top six spots.

### WebArena
WebArena provides 812 tasks on self-hosted website instances with functional correctness evaluation, requiring real multi-step browser navigation — filling forms, reading and synthesizing across pages, and verifying outcomes through the live UI rather than a static snapshot. It's harder to game than static text benchmarks because a task's success is judged by the actual state of a live web application after the agent acts.

WebArena 2026 leaderboard: Claude Mythos Preview at 68.7%, GPT-5.4 Pro at 65.8%, Claude Opus 4.6 at 64.5%. Human baseline approximately 78%. The gap to human baseline on WebArena remains wider than on SWE-bench, consistent with the overall picture from Module 8.1's computer/browser use discussion: real-environment, multi-step web navigation is a harder, less-saturated problem than code repair.

### AgentBench
AgentBench evaluates 29 LLMs across eight environments — OS, database, knowledge graphs, gaming, embodied AI — revealing significant gaps between commercial and open-source models. The heterogeneous environment design is both its strength (breadth of coverage) and its weakness: the aggregate score hides dramatically different per-environment capabilities that matter enormously for specific use cases. Always decompose AgentBench performance by environment before drawing conclusions about a model's fitness for a specific deployment context.

### τ-bench (tau-bench)
τ²-Bench, from Sierra Research, simulates customer service interactions across retail, airline, and telecom domains using a dual-control design: both the AI agent and a simulated user actively modify a shared environment. Agent behavior degrades sharply when shifting from single-control to dual-control settings.

τ-bench's distinct contribution is its **pass@k metric**, which measures not just whether an agent succeeds once, but what fraction of independent attempts succeed — directly measuring reliability under variance rather than optimistic best-case performance. Agent variance per run is large — pass^4 scores often run 15–25 points below pass^1. A 90% benchmark score sometimes corresponds to 70% reliability in production, where the same task gets retried by different sessions. This is the benchmark that most directly surfaces the gap between "impressive demo performance" and "production reliability."

### 🆕 A critical caveat: the April 2026 reward-hacking disclosure
On April 12, 2026, UC Berkeley's Center for Responsible Decentralized Intelligence published research showing that an automated scanning agent broke all eight major agent benchmarks — SWE-bench, WebArena, OSWorld, GAIA, Terminal-Bench, FieldWorkArena, CAR-bench, and one more — by reward hacking. Every single one can be exploited to achieve near-perfect scores without solving a single task.

This is not a fringe concern — it's a documented, systematic vulnerability across the entire public benchmark ecosystem, not a problem with one poorly designed benchmark. The practical implication is the same lesson that surfaced in Module 8.2 about test-gaming in coding agents: prefer third-party Epoch AI / BenchLM scores and run your own held-out eval rather than accepting vendor-reported leaderboard numbers uncritically. Use public benchmarks for calibrating directional intuition, not as a substitute for evaluating on your own task distribution.

### 📋 Interview Questions — 10.2
1. **Why can a model that scores 94% on MMLU-Pro still fail on a WebArena task, and what does that imply about how you should select an agent for deployment?**
   *Look for: MMLU tests knowledge recall; WebArena tests compound multi-step execution with real-world state — different capability dimensions; model selection should be evaluated on the actual task type, not general-knowledge proxies.*
2. **What does the 7-point GAIA performance gap between two frameworks running the same underlying model tell you about what drives real-world agent performance?**
   *Look for: the orchestration layer (tool set, prompting, retry logic) contributes as much as ~7 percentage points to task performance independent of the model — better scaffolding, not just a stronger model, is often the practical lever.*
3. **Why does τ-bench's pass@k metric reveal something about production reliability that a simple pass@1 benchmark score doesn't?**
   *Look for: pass@k measures what fraction of independent attempts succeed — directly quantifying the reliability variance that produces a "90% benchmark score = 70% production reliability" gap; pass@1 measures best-case performance, not typical performance under repeated real use.*
4. **What's the practical takeaway for a production team from the April 2026 UC Berkeley disclosure that all eight major benchmarks can be reward-hacked to near-perfect scores?**
   *Look for: public benchmark leaderboard numbers should be treated as directional signals only; the real requirement is running your own evaluation on your actual task distribution with held-out test cases that don't appear in any public training data — the same principle as Module 4.6's "benchmark your own query distribution" guidance.*
5. **Why should you decompose AgentBench performance by environment rather than using the aggregate score, when deciding whether a model is suitable for your specific task?**
   *Look for: aggregate scores hide dramatically different per-environment performance — a model excellent at database tasks and poor at embodied-AI tasks has the same aggregate as the reverse, but those two profiles have completely different implications for any specific deployment context.*

---

## 10.3 Regression Testing for Agent Behavior

### Why regression testing is harder for agents than for software
In traditional software, a regression test is deterministic: you run the function, compare the output to a stored expected value, and pass or fail. For agents, neither step works cleanly:
- **Outputs are non-deterministic** — even identical inputs don't reliably produce identical outputs, so a simple output-diff test produces false positives on every run.
- **"Correct behavior" isn't always a stored string** — for most valuable agentic tasks, what counts as correct depends on quality criteria that can't be captured in a hardcoded expected value.
- **Regressions can be subtle and gradual** — a prompt change might leave most tasks unaffected while quietly degrading a specific category of edge case, which only surfaces as a pattern across many runs, not a single clear failure.

These constraints mean agent regression testing needs to be **statistical** rather than deterministic: run each test case multiple times, track pass rates over time, and alert on statistically significant drops — not individual-run failures, which are expected under normal variance.

### Building a regression test suite
A good agent regression suite has three layers:

**Golden dataset** — a curated set of representative tasks with human-verified correct trajectories and outcomes, manually maintained and never auto-generated. These are the highest-signal test cases, but they're expensive to create and maintain. The key discipline: the harder cost is maintaining the gold-set, scheduling recalibration, refining rubrics when judge agreement drops.

**Canary tasks** — simple, well-defined tasks where the expected outcome is unambiguous (the coding equivalent of unit tests). These should all pass reliably; any failure in a canary task is a clear signal of a significant regression, not normal variance. They're cheap to run and provide fast feedback.

**Distributional tasks** — a larger sample from your actual production task distribution, graded by an LLM judge calibrated to your golden dataset. These track aggregate quality trends over time rather than catching specific failures, and are especially valuable for catching subtle, category-specific regressions that individual canary tasks might miss.

Each code or prompt change should trigger a run of all three layers, with results compared to the most recent baseline using the same statistical tests you'd use for any A/B experiment — not just "did it pass," but "did the pass rate change significantly from the previous version?"

### Flakiness and statistical significance
A regression test suite for agents will have inherent pass-rate variance even on an unchanged system — this is normal, not a problem to eliminate. The operational discipline:
- **Run each test case at least 3–5 times** per evaluation and use the average pass rate, not a single-run result.
- **Don't alert on single-run failures** — a task that fails once but passes 4 out of 5 times is behaving within normal variance; a task whose pass rate drops from 85% to 60% across many runs is a real regression signal.
- **Use statistical significance tests** (chi-squared, bootstrap confidence intervals) to determine whether an observed change in pass rate is signal or noise before triggering an alert or blocking a deployment.
- **Track confidence intervals alongside point estimates** in dashboards — a reported 78% pass rate means something very different if its 95% confidence interval is [76%, 80%] versus [60%, 96%].

### 📋 Interview Questions — 10.3
1. **Why can't you use a standard deterministic regression test (compare output to stored expected string) for agent evaluation?**
   *Look for: agent outputs are non-deterministic — identical inputs produce different outputs across runs due to temperature/sampling variance — making string-diff comparison produce false positives on every single run.*
2. **What's the difference between a golden dataset, canary tasks, and distributional tasks in a regression suite, and why do you need all three?**
   *Look for: golden dataset = expensive, high-signal, manually verified reference cases; canary = cheap, unambiguous fast-fail detection; distributional = aggregate trend-tracking across realistic production task variety — each catches a different failure mode the others miss.*
3. **A test suite shows a task that passes 3 out of 5 runs. Is that a regression or normal variance, and how would you determine which?**
   *Look for: can't determine from a 5-run sample alone — need to compare the current pass rate against the historical baseline for that specific task, with a statistical significance test; 60% might be the normal variance floor for that task, or it might be a significant drop from a previous 90% baseline.*
4. **Why should regression alerts be triggered by statistically significant changes in pass rate over multiple runs, rather than by any single-run failure?**
   *Look for: individual failures are expected under normal agent variance — alerting on single failures produces an unusably noisy alert stream; statistically significant pass-rate changes across repeated runs distinguish genuine regressions from normal variance.*
5. **A prompt change improves performance on 95% of tasks but degrades a specific category of edge cases. How would a distributional test suite catch this, and why might canary tasks miss it?**
   *Look for: distributional tasks drawn from realistic variety can reveal category-specific regressions in aggregate; canary tasks only cover the specific cases someone thought to hand-write, and by definition miss edge cases no one anticipated — which is exactly why distributional coverage is essential alongside canaries.*

---

## 10.4 A/B Testing Prompts, Tool Sets, and Model Versions

### What makes agent A/B testing different from product A/B testing
Standard product A/B testing (UI changes, copy variations) has a clean structure: randomly assign users to variant A or B, measure a business metric, achieve statistical significance, ship the winner. Agent A/B testing has several additional complications:
- **The "treatment effect" can show up anywhere in a multi-step trajectory**, not just in the final output — a prompt change might improve task completion but introduce a new class of unnecessary tool calls, netting positive on outcome but negative on cost.
- **Tasks differ wildly in difficulty** — the same agent variant might improve on easy tasks while regressing on hard ones, which averages out to "no change" if not segmented by difficulty.
- **Evaluation cost is high** — each "data point" in an agent A/B test may require running a full multi-step task with real tool calls, costing significantly more than a UI A/B test where each impression is essentially free.
- **You need multiple runs per task per variant**, not just one, because within-variant variance can exceed between-variant differences if you run each task only once.

### Prompts as versioned, testable artifacts
Every system prompt change is a potential regression — and a potential improvement — and **the only way to know which is to test it**. Treating prompts as versioned, tracked artifacts in version control (not ad hoc strings edited in a dashboard) is the prerequisite for being able to tell which version of a prompt is running in production, reproduce any historical result, and safely roll back. Concretely:

- Assign each prompt version a unique identifier (a hash, a semantic version number, or a meaningful name).
- Shadow-test prompt changes before deploying: run the new prompt on a held-out sample of real tasks alongside the current production prompt, compare quality metrics, and only promote if the comparison is clearly favorable.
- Treat prompt changes with the same deployment discipline as code changes — a prompt that runs your production agent is a production artifact, not a scratchpad.

### Testing tool-set changes
Adding, removing, or modifying a tool is among the highest-impact changes you can make to an agent — it changes the agent's available action space, and any tool can become the wrong tool with the right (wrong) prompt change or model version update. Testing a tool-set change requires evaluating:
- **Selection accuracy** — does the agent still reliably choose the right tool for each task category? (Module 2.2's selection-accuracy principle, measured empirically rather than assumed.)
- **Argument correctness** — does the agent still produce valid, well-formed arguments for each tool? (Module 2.3's structured-output validation, run as an eval across a representative task sample.)
- **Failure handling** — does the agent handle each tool's failure modes correctly under the new configuration?

### Testing model-version upgrades
A model-version upgrade (e.g., moving from one Claude generation to the next) is tempting to treat as a free win — a newer model should be better, right? In practice, model upgrades are among the most common sources of production regressions, for a predictable reason: newer models are generally better on average but may have different behavior on specific prompting patterns, edge cases, or instruction formats that your agent happens to rely on.

The discipline: treat a model-version upgrade with the same rigor as a prompt-set regression test — run the full golden dataset and distributional task suite on both the current and new model version before migrating production traffic, with both quality metrics and cost-per-task compared. Migrate incrementally (e.g., 5% → 20% → 50% → 100%) rather than switching all traffic at once, and have an automated rollback trigger defined before you start.

### 📋 Interview Questions — 10.4
1. **Why is running each task only once per variant insufficient for an agent A/B test, even when you have many tasks?**
   *Look for: within-variant variance (the same task failing or succeeding across runs) can exceed between-variant differences — averaging over multiple runs per task per variant is necessary to separate signal from within-variant noise.*
2. **A model upgrade shows an average quality improvement of 3% on your evaluation suite. What additional analysis would you want before migrating 100% of production traffic?**
   *Look for: segmented analysis by task difficulty and task type (to catch category-specific regressions hidden by the average), cost-per-task comparison (not just quality), and an incremental rollout plan with a rollback trigger — the average improvement doesn't rule out a specific regression in a high-value task category.*
3. **Why should prompt changes be treated with the same deployment discipline as code changes, and what does that specifically mean in practice?**
   *Look for: a production system prompt is a production artifact; disciplined version control (hash/version per prompt), shadow-testing before promotion, and rollback capability are the same safeguards you'd apply to production code — not optional polish.*
4. **What three things should you specifically measure when testing a tool-set change, and why is testing the final output quality alone insufficient?**
   *Look for: tool selection accuracy, argument correctness, and failure-mode handling — a tool change can silently degrade any of the three while leaving final output quality roughly unchanged, only to surface as a reliability problem in production.*
5. **How would you design an A/B test to detect whether a prompt change improved performance on hard tasks while potentially regressing on easy ones, when a global average metric might miss this?**
   *Look for: segmenting results by task difficulty before computing per-segment averages — a change that averages to "no improvement" might strongly improve one segment and strongly degrade another, which requires pre-defined difficulty stratification to detect rather than aggregate metrics alone.*

---

## 10.5 Building Simulated Environments / Sandboxes

### Why simulation is necessary
Running agent evaluations in production — against real APIs, real databases, real websites — is expensive, slow, has real side effects (sent emails, modified records, incurred API costs), and is non-repeatable (the world changes between runs, making two "identical" evaluations inconsistent). Simulated environments solve all four problems at once: fast, cheap, side-effect-free, and exactly reproducible across runs. This is the evaluation prerequisite for everything in this module — without a repeatable environment, the regression testing (10.3) and A/B testing (10.4) disciplines above are difficult to execute reliably.

### What a good simulation environment needs
- **Deterministic or controllable randomness** — for repeatability, the simulation must produce the same environment state given the same seed, so the same task can be run identically across the new and old agent versions you're comparing.
- **Representative task coverage** — the tasks in the simulation need to cover the actual distribution of real tasks the agent will face in production, including edge cases and failure scenarios that don't appear in everyday operation. Simulations that only test "happy path" tasks miss the failure modes that matter most.
- **Realistic tool-response mocking** — mock tool responses should reflect the real variability of actual tools: occasional slow responses, partial failures, unexpected but valid response shapes, and the kind of ambiguous/noisy data a real tool actually returns — not just perfectly formatted, always-successful responses that make every tool call easier than reality.
- **A way to run many evaluations in parallel** — the main computational advantage of simulation over live evaluation; you need the environment to support concurrent evaluation workers without interference between runs.

### Simulation fidelity tradeoffs
Higher-fidelity simulations produce more reliable evaluation results but cost more to build and maintain. The spectrum:

| Fidelity level | What it simulates | Cost | What it misses |
|---|---|---|---|
| **Mocked responses** | Tool call returns a pre-recorded static response | Very low | Real response variability, new failure modes |
| **Parameterized mocks** | Responses generated dynamically from templates with controlled variability | Low–moderate | Real API behavior at edge boundaries |
| **Sandboxed live environment** | Real tools running in an isolated sandbox (e.g., a real database with test data, a real browser against a test site) | Moderate–high | Only live production edge cases |
| **Recorded-and-replayed sessions** | Real tool responses captured from production and replayed | Moderate | Temporal drift (recorded responses become stale) |

The practical recommendation: **start with high-fidelity sandboxed live environments for core workflows** (the place where catching a real regression matters most), and use parameterized mocks for broader coverage of edge cases that would be expensive or risky to exercise with real tools.

### 📋 Interview Questions — 10.5
1. **Why can't you just run all your evaluations against production systems, and what specific problems does simulation solve?**
   *Look for: production evaluations are expensive (real API costs), slow, have real side effects (sent emails, modified records), and are non-repeatable (world changes between runs) — simulation solves all four while enabling the parallel scale needed for statistically rigorous testing.*
2. **Why does a simulation environment that only tests "happy path" tasks undermine the value of your regression suite?**
   *Look for: the failure modes that matter most in production are edge cases and failure scenarios — a simulation that only covers successful cases will miss the regressions that actually break production, giving false confidence that everything works.*
3. **What's wrong with mock tool responses that always return perfectly formatted, always-successful responses?**
   *Look for: real tools return noisy, ambiguous, occasionally partial/failed responses — a simulation that doesn't reproduce that variability tests a "better than real" environment and will over-estimate the agent's actual robustness in production.*
4. **When would you choose recorded-and-replayed sessions over sandboxed live environments for simulation, and what's the main risk of recorded sessions over time?**
   *Look for: recorded sessions are useful when live sandbox setup is prohibitively complex or expensive; the main risk is temporal drift — recorded responses become stale as the real tool's behavior evolves, making the simulation progressively less representative of current production behavior.*
5. **How does simulation fidelity tradeoff affect the kind of coverage you can afford?**
   *Look for: higher-fidelity environments (sandboxed live) are too expensive to run at high volume for broad coverage, so the practical approach is high-fidelity for the most critical core workflows and lower-fidelity parameterized mocks for the broad edge-case coverage you couldn't afford to test with real tools.*

---

## 10.6 Defining Success Metrics Beyond "Did It Run"

### The metric hierarchy
"Did it run" is the zeroth metric — it tells you the agent didn't crash. It tells you almost nothing else. Of the major agent benchmarks, 0 out of 15 integrate cost-efficiency or safety into their primary scoring rubric. A score of 88% on SWE-Bench achieved with $50 of inference per task is treated as identical to one achieved with $0.50. 13 out of 15 rely on binary success metrics, ignoring partial completion or graceful failure. This is the benchmark ecosystem telling you what it optimizes for — not what a production agent actually needs to optimize for.

A production agent needs metrics across at least five dimensions simultaneously:

### Task success rate
The primary outcome metric: what fraction of attempted tasks did the agent complete correctly? Comes in several variants:
- **Binary success rate** — a task fully succeeded or fully failed. Simple, but misses partial completion and graceful degradation, which are meaningful real-world outcomes.
- **Partial completion scoring** — credit for tasks that were incomplete but achieved some meaningful subtasks. More informative for long-horizon tasks where binary pass/fail discards too much signal.
- **pass@k** — what fraction of tasks succeed when retried k times? The most production-relevant formulation for an agent in a retry-capable system, where a task failing once isn't necessarily a final failure.

### Step efficiency
Given a task the agent completes successfully, how many steps (tool calls, reasoning iterations) did it take? An agent that succeeds in 5 steps when 3 were sufficient is less efficient — and therefore more expensive and slower — than one that achieves the same outcome in 3. Step efficiency matters especially when: (a) tool calls have real costs (API fees, rate limits), (b) latency compounds across steps, and (c) longer chains have more error-compounding opportunity (Module 1.5).

Measuring step efficiency requires separating success rate from step count, then computing efficiency for the successful-run subset — combining them ("how often does it succeed quickly?") loses the diagnostic clarity of knowing which dimension is the bottleneck.

### Cost per task
The full dollar cost of a completed task, including all token spend (input + output, across all steps), all tool-call costs (API fees, browser-session time), and any infrastructure costs (sandbox compute). This almost always has to be composed by the engineering team manually, since most benchmarks don't include cost-efficiency in their primary scoring rubric — dividing pass rate by dollars-per-task is a practical first-order ROI metric not typically provided anywhere in the public benchmark ecosystem.

### Reliability and variance
Reliability is measured at the task distribution level, not individual-run level: for each task category, what is the **variance in success rate across repeated runs**, and what is the **lowest-performing percentile** of tasks? An agent with a 90% average success rate but with 20% of its task types below 50% is far less production-ready than one with an 80% average across all task types above 70%. The tails determine the user experience more than the average does.

### Safety and constraint adherence
Did the agent stay within defined boundaries — budget caps, scope limits, prohibited action types, required approval gates? A 100% task-success-rate agent that routinely violates cost budgets or skips required confirmations is not a success by any reasonable production metric, but none of these violations would surface in a task-success-rate metric alone.

Concrete sub-metrics:
- **Prompt injection resistance rate** — of simulated injection attempts, what fraction were correctly detected and resisted? (Module 9.3)
- **HITL checkpoint compliance rate** — what fraction of high-risk actions correctly paused for human approval, rather than being executed directly? (Module 1.6)
- **Budget overrun rate** — what fraction of tasks ran over their defined cost/step budget?

### Composing metrics into a decision framework
No single metric tells the whole story, and improving one often degrades another — a common failure pattern is optimizing for task success rate alone while silently degrading cost, reliability, or safety. The practical approach:
1. Define a **primary metric** (usually task success rate or pass@k for core workflows) that acts as the headline decision criterion.
2. Define **guardrail metrics** (cost per task, safety adherence, HITL compliance) that must not regress below defined thresholds even when the primary metric improves — a "do not degrade" list that's enforced automatically in CI/CD.
3. Define **diagnostic metrics** (step efficiency, per-category success rates, confidence intervals) that don't block deployment but inform understanding of *where* and *why* the agent behaves the way it does.

This three-tier metric structure directly mirrors the three-tier evaluation stack from 10.1 — and is the same design discipline as Section 9.4's "cost and reliability guardrails as architectural decisions from day one," applied to the evaluation and deployment layer.

### 📋 Interview Questions — 10.6
1. **Why is binary task success rate (passed/failed) an insufficient primary metric for a long-horizon agentic task, and what specifically does it miss?**
   *Look for: partial completion (meaningful progress toward a goal that stopped short of full success) has real value and real diagnostic signal; binary pass/fail treats "got 90% of the way there" identically to "failed immediately," losing the information that distinguishes a nearly-working agent from a broken one.*
2. **An agent achieves 88% task success rate. Without any other metrics, what important questions about its production suitability can't you answer?**
   *Look for: cost per task (is it $0.50 or $50?), reliability variance (is 88% stable across run types or does it hide categories at 30%?), step efficiency (does it succeed in 5 steps or 50?), and safety adherence (does it stay within budget/scope/approval requirements?) — none of these are answered by a single success rate.*
3. **How would you define and enforce "guardrail metrics" in a CI/CD pipeline for agent deployment?**
   *Look for: specifying concrete, non-negotiable thresholds (e.g., "cost per task must not exceed $X," "HITL compliance must not fall below Y%") that automatically block a deployment even when the primary success metric improved — enforced in the deployment check, not just monitored in dashboards.*
4. **Why does pass@k capture something about production reliability that pass@1 misses, and what does a large gap between pass@1 and pass@4 tell you?**
   *Look for: pass@k measures what fraction of task types succeed within k attempts — closer to production reality where retries are available; a large pass@1-to-pass@4 gap signals high variance, meaning the agent is unreliable on many tasks even if it eventually succeeds — important for UX and cost implications of retry budgets.*
5. **A team's agent improved from 72% to 81% task success rate after a model upgrade. What else would you want to check before calling this a clear win?**
   *Look for: cost per task (did it increase proportionally?), step count (did steps increase to get the higher success rate?), safety adherence (did the new model handle HITL/constraint compliance correctly?), per-category segmentation (did some task categories regress?), and confidence intervals (is the improvement statistically significant or within normal variance?) — the headline number alone doesn't tell you whether this is a real win.*

---

*End of Module 10 detailed notes — 25 interview questions total across 6 sections. The April 2026 reward-hacking disclosure is the most important recent development in this space — it means no public benchmark can be taken at face value, and running your own held-out evaluation on your actual task distribution is not optional. The LLM-as-judge literature is also moving fast — re-verify judge calibration best practices against current sources before treating any set of judge-evaluation scores as reliable without explicit kappa validation against human ratings.*
