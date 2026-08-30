📚 Table of Contents

* [14. Layer 12 — Agent Runtime / Harness Engineering](#14-layer-12--agent-runtime--harness-engineering)

  * [14.1 Runtime Concepts](#141-runtime-concepts)

    * [14.1.1 Agent Loop](#1411-agent-loop)
    * [14.1.2 Execution Environment](#1412-execution-environment)
    * [14.1.3 Workspace](#1413-workspace)
    * [14.1.4 Filesystem](#1414-filesystem)
    * [14.1.5 Shell](#1415-shell)
    * [14.1.6 Network Access](#1416-network-access)
    * [14.1.7 Environment Variables](#1417-environment-variables)
    * [14.1.8 Secrets](#1418-secrets)
    * [14.1.9 Artifact Storage](#1419-artifact-storage)
    * [14.1.10 Process Management](#14110-process-management)
  * [14.2 Sandboxing](#142-sandboxing)

    * [14.2.1 Containers](#1421-containers)
    * [14.2.2 Process Isolation](#1422-process-isolation)
    * [14.2.3 Filesystem Isolation](#1423-filesystem-isolation)
    * [14.2.4 Network Restrictions](#1424-network-restrictions)
    * [14.2.5 CPU Limits](#1425-cpu-limits)
    * [14.2.6 Memory Limits](#1426-memory-limits)
    * [14.2.7 Time Limits](#1427-time-limits)
    * [14.2.8 Tool Allowlists](#1428-tool-allowlists)
    * [14.2.9 Domain Allowlists](#1429-domain-allowlists)
    * [14.2.10 Workspace Boundaries](#14210-workspace-boundaries)
    * [14.2.11 Sandboxing Defense-in-Depth](#14211-sandboxing-defense-in-depth)
  * [14.3 Runtime Lifecycle](#143-runtime-lifecycle)

    * [14.3.1 Create Task](#1431-create-task)
    * [14.3.2 Provision Environment](#1432-provision-environment)
    * [14.3.3 Load Context / Skills](#1433-load-context--skills)
    * [14.3.4 Execute Agent](#1434-execute-agent)
    * [14.3.5 Checkpoint](#1435-checkpoint)
    * [14.3.6 Continue / Pause / Approve](#1436-continue--pause--approve)
    * [14.3.7 Persist Artifacts](#1437-persist-artifacts)
    * [14.3.8 Complete / Fail / Cancel](#1438-complete--fail--cancel)
    * [14.3.9 Destroy or Retain Environment](#1439-destroy-or-retain-environment)
    * [14.3.10 Runtime Lifecycle Flow](#14310-runtime-lifecycle-flow)
  * [14.4 Runtime Control](#144-runtime-control)

    * [14.4.1 Cancellation](#1441-cancellation)
    * [14.4.2 Interruptibility](#1442-interruptibility)
    * [14.4.3 Timeouts](#1443-timeouts)
    * [14.4.4 Concurrency](#1444-concurrency)
    * [14.4.5 Quotas](#1445-quotas)
    * [14.4.6 Maximum Steps](#1446-maximum-steps)
    * [14.4.7 Maximum Tokens](#1447-maximum-tokens)
    * [14.4.8 Maximum Cost](#1448-maximum-cost)
    * [14.4.9 Emergency Stop](#1449-emergency-stop)
    * [14.4.10 Human Takeover](#14410-human-takeover)
    * [14.4.11 Runtime Control Matrix](#14411-runtime-control-matrix)
  * [14.5 Artifacts](#145-artifacts)

    * [14.5.1 Files](#1451-files)
    * [14.5.2 Reports](#1452-reports)
    * [14.5.3 Images](#1453-images)
    * [14.5.4 Generated Code](#1454-generated-code)
    * [14.5.5 Logs](#1455-logs)
    * [14.5.6 Datasets](#1456-datasets)
    * [14.5.7 Test Results](#1457-test-results)
    * [14.5.8 Build Outputs](#1458-build-outputs)
    * [14.5.9 Artifact Lifecycle](#1459-artifact-lifecycle)
  * [14.6 Runtime Architecture](#146-runtime-architecture)

    * [14.6.1 Control Plane](#1461-control-plane)
    * [14.6.2 Execution Plane](#1462-execution-plane)
    * [14.6.3 Workspace Layer](#1463-workspace-layer)
    * [14.6.4 Policy and Security Layer](#1464-policy-and-security-layer)
    * [14.6.5 State and Checkpoint Layer](#1465-state-and-checkpoint-layer)
    * [14.6.6 Artifact Layer](#1466-artifact-layer)
    * [14.6.7 Observability Layer](#1467-observability-layer)
  * [14.7 Sandboxed Agent Runtime Project](#147-sandboxed-agent-runtime-project)

    * [14.7.1 Project Goal](#1471-project-goal)
    * [14.7.2 Functional Requirements](#1472-functional-requirements)
    * [14.7.3 Project Architecture](#1473-project-architecture)
    * [14.7.4 Task Execution Workflow](#1474-task-execution-workflow)
    * [14.7.5 Workspace Design](#1475-workspace-design)
    * [14.7.6 Shell Execution](#1476-shell-execution)
    * [14.7.7 Resource Controls](#1477-resource-controls)
    * [14.7.8 Network Controls](#1478-network-controls)
    * [14.7.9 Permission Controls](#1479-permission-controls)
    * [14.7.10 Artifact Persistence](#14710-artifact-persistence)
    * [14.7.11 Cancellation and Recovery](#14711-cancellation-and-recovery)
    * [14.7.12 Runtime States](#14712-runtime-states)
    * [14.7.13 Example Execution](#14713-example-execution)
  * [14.8 Key Insights](#148-key-insights)
  * [14.9 Common Mistakes](#149-common-mistakes)
  * [14.10 Common Confusions](#1410-common-confusions)
  * [14.11 Practical Applications](#1411-practical-applications)
  * [14.12 Important Terms](#1412-important-terms)
  * [14.13 Quick Revision](#1413-quick-revision)
  * [14.14 Interview Preparation](#1414-interview-preparation)

    * [14.14.1 Level 1 — Fundamentals](#14141-level-1--fundamentals)
    * [14.14.2 Level 2 — Conceptual Understanding](#14142-level-2--conceptual-understanding)
    * [14.14.3 Level 3 — Practical / Engineering](#14143-level-3--practical--engineering)
    * [14.14.4 Level 4 — Advanced / Deep Understanding](#14144-level-4--advanced--deep-understanding)
    * [14.14.5 Level 5 — Scenario-Based Questions](#14145-level-5--scenario-based-questions)
    * [14.14.6 Knowledge Check](#14146-knowledge-check)
    * [14.14.7 Follow-up Questions](#14147-follow-up-questions)
    * [14.14.8 Common Confusion Questions](#14148-common-confusion-questions)
    * [14.14.9 Deep / Trick Questions](#14149-deep--trick-questions)
  * [14.15 Top Questions You MUST Know](#1415-top-questions-you-must-know)
  * [14.16 Interview Readiness Checklist](#1416-interview-readiness-checklist)
  * [14.17 What You Should Be Able to Explain](#1417-what-you-should-be-able-to-explain)

# 14. Layer 12 — Agent Runtime / Harness Engineering

🧠 **Simple Understanding:** Agent runtime / harness engineering is the engineering of the **execution environment around an AI agent**: where it runs, what it can access, how resources are limited, how actions are controlled, how state is persisted, and how the environment is cleaned up.

A useful mental model is:

```text
Agent
  ↓
Runtime / Harness
  ↓
┌─────────────────────────────────────────┐
│ Workspace                               │
│ Filesystem                              │
│ Shell                                   │
│ Network                                 │
│ Environment Variables                   │
│ Secrets                                 │
│ Processes                               │
│ Tools                                   │
│ Resource Limits                         │
│ Artifacts                               │
└─────────────────────────────────────────┘
  ↓
External Systems
```

The runtime is especially important for **modern long-running agents**, because autonomous systems need a controlled place to perform work rather than merely generate text.

⭐ **Core Principle:** The model decides and reasons, but the **harness controls execution**.

---

# 14.1 Runtime Concepts

## 14.1.1 Agent Loop

🧠 **Simple Understanding:** The agent loop repeatedly turns observations into actions until the task reaches a terminal condition.

```text
Goal
 ↓
Observe
 ↓
Reason
 ↓
Act
 ↓
Observe Result
 ↓
Update State
 ↓
Continue / Stop
```

The runtime executes and controls this loop.

### 📌 Quick Info

| Field         | Answer                               |
| ------------- | ------------------------------------ |
| **What?**     | Repeated agent execution cycle       |
| **Why?**      | Enables multi-step task completion   |
| **How?**      | Model + tools + state + observations |
| **When?**     | Stateful and action-oriented tasks   |
| **Main risk** | Loops, runaway cost, unsafe actions  |

---

## 14.1.2 Execution Environment

🧠 **Simple Understanding:** The execution environment is the controlled computing environment where the agent's actions actually happen.

It may include:

```text
CPU
Memory
Filesystem
Processes
Network
Shell
Environment variables
Secrets
Installed tools
```

Example:

```text
Agent
 ↓
Runtime
 ↓
Container
 ├── /workspace
 ├── shell
 ├── Python
 ├── Git
 └── restricted network
```

🔬 **Technical Explanation:** The execution environment separates **model reasoning** from **machine-side execution**. The model does not directly control the host system; it issues requests that the runtime decides whether and how to execute.

---

## 14.1.3 Workspace

🧠 **Simple Understanding:** A workspace is the working area assigned to a task.

Example:

```text
/workspaces/task-123/
├── input/
├── work/
├── output/
└── temp/
```

A workspace may contain:

* Uploaded files.
* Generated files.
* Intermediate outputs.
* Source code.
* Test results.

### Benefits

* Task isolation.
* Easy cleanup.
* Reproducibility.
* Artifact collection.

---

## 14.1.4 Filesystem

🧠 **Simple Understanding:** The filesystem provides persistent or temporary file access to the agent.

The runtime should define exactly what the agent can access.

Example:

```text
Allowed:
  /workspace/project/*

Blocked:
  /etc/*
  /home/other-user/*
  /system/*
```

⚠️ **Important:** Filesystem boundaries should be enforced by the runtime, not merely described in the agent's instructions.

---

## 14.1.5 Shell

🧠 **Simple Understanding:** A shell lets an agent execute operating-system commands.

Example:

```text
python app.py
pytest
git status
ls
```

Shell access is powerful because it can potentially:

* Read/write files.
* Start processes.
* Access networks.
* Consume CPU/memory.
* Modify the environment.

Therefore shell execution should generally occur inside a controlled sandbox when agents are untrusted or highly autonomous.

---

## 14.1.6 Network Access

🧠 **Simple Understanding:** Network access determines which external services the agent can reach.

Possible policies:

```text
No network
       ↓
Internal-only
       ↓
Allowlisted domains
       ↓
Restricted internet
       ↓
Broad internet
```

Network permissions should be treated as an explicit capability.

---

## 14.1.7 Environment Variables

🧠 **Simple Understanding:** Environment variables provide configuration to processes running in the workspace.

Examples:

```text
APP_ENV=production
LOG_LEVEL=info
API_ENDPOINT=...
```

Do not treat environment variables as inherently safe.

Sensitive values should be handled separately from ordinary configuration.

---

## 14.1.8 Secrets

🧠 **Simple Understanding:** Secrets are sensitive credentials such as API keys, tokens, passwords, or certificates.

Examples:

```text
OPENAI_API_KEY
DATABASE_PASSWORD
CLOUD_TOKEN
```

A runtime should minimize:

* Secret exposure.
* Secret persistence.
* Secret inheritance.
* Secret logging.

⭐ **Key Point:** An agent may need access to a capability without being given unrestricted access to the underlying secret.

---

## 14.1.9 Artifact Storage

🧠 **Simple Understanding:** Artifact storage holds outputs produced during agent execution.

Examples:

```text
report.pdf
generated_code/
chart.png
test_results.json
build.zip
```

Artifacts should normally be separated from ephemeral execution state.

```text
Runtime Workspace
      ↓
Completed Artifact
      ↓
Artifact Storage
```

---

## 14.1.10 Process Management

🧠 **Simple Understanding:** Process management controls programs started by the agent.

The runtime should know:

```text
Which process?
Who started it?
How long has it run?
How much CPU?
How much memory?
Can it be terminated?
```

Important operations:

* Start.
* Monitor.
* Signal.
* Kill.
* Reap/clean up.
* Detect orphaned processes.

---

# 14.2 Sandboxing

🧠 **Simple Understanding:** Sandboxing restricts what an agent can access and how much damage or resource consumption it can cause.

A sandbox can control:

```text
Execution
Files
Network
CPU
Memory
Time
Tools
Domains
Workspace
```

⭐ **Key Point:** Sandboxing should be treated as **defense in depth**, not a single security switch.

---

## 14.2.1 Containers

🧠 **Simple Understanding:** Containers provide isolated execution environments with controlled filesystem, process, and resource boundaries.

Conceptually:

```text
Host
├── Container A → Agent Task A
├── Container B → Agent Task B
└── Container C → Agent Task C
```

Containers can help isolate workloads, but their exact security properties depend on configuration and the underlying runtime.

---

## 14.2.2 Process Isolation

🧠 **Simple Understanding:** Process isolation prevents one task's processes from freely interacting with another task's processes or the host.

Goals:

```text
Task A processes
      ✕
Task B processes

Task
      ✕
Host processes
```

---

## 14.2.3 Filesystem Isolation

Filesystem isolation ensures the agent sees only the filesystem it should see.

Example:

```text
Sandbox
├── /workspace
├── /tmp
└── permitted runtime files
```

rather than:

```text
Entire host filesystem
```

---

## 14.2.4 Network Restrictions

🧠 **Simple Understanding:** Network controls determine what the sandbox can communicate with.

Possible controls:

```text
Outbound network disabled
        ↓
Specific domains
        ↓
Specific IPs / services
        ↓
Controlled proxy
```

Useful for:

* Reducing exfiltration risk.
* Preventing access to internal systems.
* Controlling external dependencies.
* Limiting unintended internet activity.

---

## 14.2.5 CPU Limits

🧠 **Simple Understanding:** CPU limits prevent a runaway process from consuming unlimited compute.

Example:

```text
Task budget:
2 CPU cores
```

This protects:

* Other workloads.
* Infrastructure.
* Cost.

---

## 14.2.6 Memory Limits

🧠 **Simple Understanding:** Memory limits cap how much RAM the execution environment can consume.

Example:

```text
Task memory:
4 GB
```

Useful against:

* Memory leaks.
* Large accidental allocations.
* Resource exhaustion.

---

## 14.2.7 Time Limits

🧠 **Simple Understanding:** Time limits bound how long a process or task can run.

```text
Start
 ↓
Work
 ↓
Timeout
 ↓
Terminate / Recover
```

Time limits should exist at multiple levels:

```text
Tool timeout
Process timeout
Node timeout
Task timeout
```

---

## 14.2.8 Tool Allowlists

🧠 **Simple Understanding:** A tool allowlist defines which capabilities an agent is allowed to use.

Example:

```text
Research task:
✓ web_search
✓ fetch_url
✓ save_file

✕ payment
✕ delete_database
✕ production_shell
```

This creates an explicit capability boundary.

---

## 14.2.9 Domain Allowlists

🧠 **Simple Understanding:** Domain allowlists restrict network access to specific domains.

Example:

```text
Allowed:
example.com
docs.example.com
api.example.com

Blocked:
everything else
```

Useful for:

* Research agents.
* Enterprise environments.
* Controlled browsing.
* Data-exfiltration prevention.

---

## 14.2.10 Workspace Boundaries

🧠 **Simple Understanding:** Workspace boundaries define which files and directories belong to the current task.

```text
Task A
└── /workspace/task-A/

Task B
└── /workspace/task-B/
```

Task A should not automatically access Task B's files.

---

## 14.2.11 Sandboxing Defense-in-Depth

A strong sandbox combines multiple controls:

```text
                 SANDBOX
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
  Process        Filesystem     Network
  Isolation      Isolation      Controls
      │             │             │
      └─────────────┼─────────────┘
                    ▼
              Resource Limits
                    │
           ┌────────┼────────┐
           ▼        ▼        ▼
          CPU      RAM      Time
                    │
                    ▼
             Capability Policy
                    │
              ┌─────┴─────┐
              ▼           ▼
          Tool List   Domain List
```

⭐ **Remember:** A single permission boundary failing should not automatically mean unrestricted compromise.

---

# 14.3 Runtime Lifecycle

The runtime lifecycle is:

```text
Create task
 ↓
Provision environment
 ↓
Load context / skills
 ↓
Execute agent
 ↓
Checkpoint
 ↓
Continue / pause / approve
 ↓
Persist artifacts
 ↓
Complete / fail / cancel
 ↓
Destroy or retain environment
```

---

## 14.3.1 Create Task

🧠 **Simple Understanding:** Create a durable task record before execution begins.

Example:

```json
{
  "task_id": "task-102",
  "status": "created",
  "user_id": "...",
  "runtime_profile": "research"
}
```

The task ID becomes the primary reference for:

* State.
* Logs.
* Artifacts.
* Traces.

---

## 14.3.2 Provision Environment

The runtime creates:

```text
Container
Workspace
Resource limits
Network policy
Tool permissions
Environment configuration
```

The environment should match the task's required capability profile.

---

## 14.3.3 Load Context / Skills

Before execution:

```text
Task
 ↓
Load:
├── Instructions
├── Task state
├── Relevant memory
├── Tools
├── Skills
└── Required artifacts
```

This connects runtime engineering to **context engineering**.

---

## 14.3.4 Execute Agent

The runtime starts the agent loop:

```text
Plan
 ↓
Action
 ↓
Observation
 ↓
State Update
 ↓
Next Action
```

It also monitors:

* CPU.
* Memory.
* Time.
* Tool calls.
* Processes.
* Costs.

---

## 14.3.5 Checkpoint

🧠 **Simple Understanding:** Save enough state to safely continue later.

Example:

```text
Checkpoint
├── Task state
├── Completed steps
├── Pending work
├── Tool results
├── Artifact references
└── Runtime metadata
```

Checkpointing is essential for long-running tasks.

---

## 14.3.6 Continue / Pause / Approve

The runtime may enter:

```text
RUNNING
   ↓
WAITING_FOR_APPROVAL
   ↓
APPROVED
   ↓
RESUME
```

or:

```text
RUNNING
   ↓
PAUSED
   ↓
RESUME
```

The environment may be retained or torn down depending on lifecycle policy.

---

## 14.3.7 Persist Artifacts

Before environment cleanup:

```text
Workspace
   ↓
Collect Outputs
   ↓
Validate
   ↓
Artifact Store
```

This allows useful outputs to survive environment destruction.

---

## 14.3.8 Complete / Fail / Cancel

Terminal task states should be explicit:

```text
COMPLETED
FAILED
CANCELLED
```

Other useful states include:

```text
PAUSED
WAITING_FOR_APPROVAL
TIMED_OUT
RECOVERING
```

---

## 14.3.9 Destroy or Retain Environment

🧠 **Simple Understanding:** After execution, the environment can be destroyed, preserved, or retained temporarily depending on task requirements.

### Destroy when:

* Task is complete.
* Workspace contains only disposable state.
* Security requires cleanup.

### Retain when:

* Human debugging is required.
* Task must resume later.
* Artifacts still need collection.
* Postmortem investigation requires the environment.

⭐ **Key Point:** Environment retention is both an operational and security decision.

---

## 14.3.10 Runtime Lifecycle Flow

```text
                    CREATE TASK
                        │
                        ▼
               PROVISION ENVIRONMENT
                        │
                        ▼
              LOAD CONTEXT / SKILLS
                        │
                        ▼
                  EXECUTE AGENT
                        │
                        ▼
                   CHECKPOINT
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
           Continue   Pause     Approval
              │         │         │
              │         ▼         ▼
              │       Resume    Resume
              │         │         │
              └─────────┼─────────┘
                        ▼
                 PERSIST ARTIFACTS
                        │
                        ▼
              COMPLETE / FAIL / CANCEL
                        │
                        ▼
                DESTROY / RETAIN
```

---

# 14.4 Runtime Control

## 14.4.1 Cancellation

🧠 **Simple Understanding:** Cancellation stops a task intentionally before normal completion.

A robust cancellation path should address:

```text
Agent loop
Processes
Tools
Network operations
Child processes
Queued work
```

Cancellation should be propagated rather than stopping only the top-level request.

---

## 14.4.2 Interruptibility

🧠 **Simple Understanding:** Interruptibility means execution can be safely paused at controlled boundaries.

Useful interruption points:

```text
Before tool execution
After tool execution
Before risky action
After checkpoint
Before expensive operation
```

Interruptibility is especially valuable for human approval and emergency intervention.

---

## 14.4.3 Timeouts

There should be bounded execution:

```text
Tool timeout
Node timeout
Process timeout
Task timeout
Environment lifetime
```

Example:

```text
Task starts
   ↓
Maximum runtime reached
   ↓
Cancel execution
   ↓
Persist partial state
   ↓
Mark TIMED_OUT
```

---

## 14.4.4 Concurrency

🧠 **Simple Understanding:** Concurrency controls how many agent tasks or processes can run simultaneously.

Examples:

```text
Global concurrency = 100
Per-user concurrency = 3
Per-tenant concurrency = 20
```

Concurrency controls prevent:

* Resource exhaustion.
* Noisy neighbors.
* Unexpected cost spikes.

---

## 14.4.5 Quotas

🧠 **Simple Understanding:** Quotas limit resource consumption over a defined period or scope.

Examples:

```text
100 tasks/day
10 GB storage
1,000 tool calls/hour
$50/day budget
```

Quotas can exist at:

* User level.
* Tenant level.
* Agent level.
* API level.
* Organization level.

---

## 14.4.6 Maximum Steps

🧠 **Simple Understanding:** A maximum step count limits the number of agent iterations.

```text
step 1
step 2
...
step 50
↓
Maximum reached
↓
Stop
```

Useful against:

* Infinite loops.
* Excessive exploration.
* Unbounded tool use.

---

## 14.4.7 Maximum Tokens

Limit model usage:

```text
Input tokens
+
Output tokens
=
Token budget
```

The runtime can stop or degrade execution when the budget is exceeded.

---

## 14.4.8 Maximum Cost

🧠 **Simple Understanding:** A cost budget prevents the agent from consuming unlimited paid resources.

Possible components:

```text
Model cost
+
Tool cost
+
Compute
+
Storage
+
Network
```

Example:

```text
Task budget = $2
```

Once the budget is exhausted:

```text
Stop
Fallback
Request approval
or
Escalate
```

---

## 14.4.9 Emergency Stop

🧠 **Simple Understanding:** An emergency stop immediately disables active execution when a severe problem is detected.

Potential triggers:

* Security incident.
* Runaway task.
* Dangerous tool behavior.
* Infrastructure emergency.
* Human operator intervention.

```text
EMERGENCY STOP
      ↓
Terminate active execution
      ↓
Persist state / logs
      ↓
Lock task
      ↓
Investigate
```

---

## 14.4.10 Human Takeover

🧠 **Simple Understanding:** Transfer control from the autonomous agent to a human operator.

```text
Agent
 ↓
Failure / uncertainty / risk
 ↓
Human takeover
 ↓
Human executes or repairs
```

The runtime should make the current state and environment understandable to the human.

---

## 14.4.11 Runtime Control Matrix

| Control           | Controls                | Typical Failure Prevented       |
| ----------------- | ----------------------- | ------------------------------- |
| Cancellation      | Intentional termination | Unwanted execution              |
| Interruptibility  | Safe pause points       | Unsafe continuation             |
| Timeout           | Duration                | Hanging tasks                   |
| Concurrency limit | Parallel workload       | Resource exhaustion             |
| Quota             | Aggregate resource use  | Abuse / cost spikes             |
| Max steps         | Agent iterations        | Infinite loops                  |
| Max tokens        | Model consumption       | Excessive inference             |
| Max cost          | Financial spend         | Runaway spending                |
| Emergency stop    | Immediate shutdown      | Severe incidents                |
| Human takeover    | Autonomous control      | High-risk / unrecoverable cases |

---

# 14.5 Artifacts

🧠 **Simple Understanding:** Artifacts are durable outputs produced by agent execution that users or downstream systems may need after the runtime finishes.

---

## 14.5.1 Files

Examples:

```text
CSV
JSON
TXT
PDF
DOCX
source files
configuration files
```

Files may be inputs, intermediate objects, or final outputs.

---

## 14.5.2 Reports

Examples:

```text
Research report
Audit report
Analysis report
Execution summary
```

Reports usually belong in persistent artifact storage rather than ephemeral runtime storage.

---

## 14.5.3 Images

Examples:

* Charts.
* Diagrams.
* Screenshots.
* Generated visual assets.

Metadata may include:

```text
artifact_id
task_id
format
size
created_at
source
```

---

## 14.5.4 Generated Code

Examples:

```text
Python project
TypeScript application
SQL scripts
Infrastructure configuration
Tests
```

Generated code should usually be isolated in the task workspace and persisted as an artifact when required.

---

## 14.5.5 Logs

Logs describe execution behavior:

```text
Task started
Tool executed
Process spawned
Error occurred
Artifact generated
Task completed
```

Logs are operational data, but can also become evidence for evaluation and debugging.

---

## 14.5.6 Datasets

Agents may generate:

```text
CSV datasets
JSONL files
Evaluation cases
Transformed datasets
Extracted records
```

Large datasets are generally better stored as artifacts with metadata rather than inserted entirely into runtime state.

---

## 14.5.7 Test Results

Examples:

```text
pytest results
benchmark results
evaluation scores
lint reports
security scan results
```

These can become inputs to downstream evaluation or deployment decisions.

---

## 14.5.8 Build Outputs

Examples:

```text
compiled binaries
packages
containers
deployment bundles
generated documentation
```

Build outputs should have traceable links to:

```text
Task
Source
Version
Execution
```

---

## 14.5.9 Artifact Lifecycle

```text
Create
  ↓
Validate
  ↓
Tag / Metadata
  ↓
Persist
  ↓
Access
  ↓
Version
  ↓
Retain
  ↓
Expire / Delete
```

Artifact storage should distinguish:

```text
Ephemeral
Temporary
Durable
Archived
```

---

# 14.6 Runtime Architecture

A production runtime can be understood as several layers.

```text
                         AGENT RUNTIME

                    ┌───────────────────┐
                    │   Control Plane   │
                    │ task • policy •   │
                    │ scheduling • stop │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Execution Plane   │
                    │ agent • tools •   │
                    │ processes         │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  Workspace             Security / Policy      State / Checkpoints
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                        Artifact Layer
                              │
                              ▼
                      Observability Layer
```

---

## 14.6.1 Control Plane

🧠 **Simple Understanding:** The control plane decides **what should run and under what constraints**.

Responsibilities:

* Task creation.
* Scheduling.
* Resource assignment.
* Permissions.
* Cancellation.
* Quotas.
* Lifecycle management.

---

## 14.6.2 Execution Plane

🧠 **Simple Understanding:** The execution plane is where the agent and its processes actually run.

Components may include:

```text
Agent process
Tool executor
Shell
Container
Subprocesses
```

---

## 14.6.3 Workspace Layer

Responsible for:

* Filesystem.
* Input files.
* Temporary files.
* Generated files.
* Workspace cleanup.

---

## 14.6.4 Policy and Security Layer

Responsible for:

```text
Authentication
Authorization
Sandbox policy
Network policy
Tool allowlists
Secret access
Resource limits
```

---

## 14.6.5 State and Checkpoint Layer

Stores:

* Agent state.
* Workflow progress.
* Checkpoints.
* Pending work.
* Runtime metadata.

This enables:

```text
Pause
Resume
Recover
Replay / inspect
```

---

## 14.6.6 Artifact Layer

Responsible for durable outputs:

```text
Files
Reports
Images
Code
Datasets
Test results
Build outputs
```

---

## 14.6.7 Observability Layer

Tracks:

```text
Task
 ↓
Runtime
 ↓
Process
 ↓
Agent
 ↓
Tool
 ↓
Artifact
 ↓
Outcome
```

Useful telemetry:

* Logs.
* Metrics.
* Traces.
* Resource usage.
* Tool execution.
* State transitions.
* Errors.
* Costs.

⭐ **Key Point:** For autonomous systems, **runtime observability is part of correctness**, because unexplained execution cannot be reliably debugged or evaluated.

---

# 14.7 Sandboxed Agent Runtime Project

## 14.7.1 Project Goal

🧠 **Simple Understanding:** Build a runtime that allows an agent to execute shell commands inside an isolated workspace while enforcing resource, network, and permission boundaries.

Core capability:

```text
Agent
 ↓
Request Shell Action
 ↓
Runtime Policy
 ↓
Sandbox
 ↓
Execute
 ↓
Capture Result
 ↓
Update State
 ↓
Persist Artifacts
```

---

## 14.7.2 Functional Requirements

| Capability       | Requirement                       |
| ---------------- | --------------------------------- |
| Task creation    | Create durable execution record   |
| Workspace        | Isolated per-task directory       |
| Shell execution  | Execute approved commands         |
| Process control  | Start/monitor/terminate processes |
| CPU limit        | Prevent excessive CPU use         |
| Memory limit     | Prevent excessive RAM use         |
| Time limit       | Prevent runaway execution         |
| Network policy   | Restrict outbound communication   |
| Tool allowlist   | Restrict capabilities             |
| Domain allowlist | Restrict network destinations     |
| Secrets          | Controlled credential injection   |
| Checkpoints      | Persist execution state           |
| Artifacts        | Persist outputs                   |
| Cancellation     | Stop active tasks                 |
| Audit            | Record significant runtime events |

---

## 14.7.3 Project Architecture

```text
                              USER
                               │
                               ▼
                         API / Task Service
                               │
                               ▼
                        Runtime Controller
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                 Policy     State      Scheduler
                 Engine      Store
                    │
                    ▼
             Sandbox Provisioner
                    │
                    ▼
              ┌───────────────┐
              │   SANDBOX     │
              │               │
              │ Workspace     │
              │ Shell         │
              │ Processes     │
              │ CPU Limit     │
              │ Memory Limit  │
              │ Time Limit    │
              │ Network Rule  │
              └───────┬───────┘
                      │
              ┌───────┼────────┐
              ▼       ▼        ▼
            Tools   Files    Network
              │       │        │
              └───────┼────────┘
                      ▼
                 Artifacts
                      │
                      ▼
                 Audit / Trace
```

---

## 14.7.4 Task Execution Workflow

```text
Create Task
    ↓
Validate Task Policy
    ↓
Provision Sandbox
    ↓
Create Workspace
    ↓
Inject Allowed Context / Tools
    ↓
Start Agent
    ↓
Agent Requests Action
    ↓
Policy Check
    ↓
Execute
    ↓
Capture stdout / stderr / exit code
    ↓
Update State
    ↓
Checkpoint
    ↓
Continue?
 ┌──┴───────┐
Yes         No
 │           │
 ▼           ▼
Continue   Finalize
             ↓
         Persist Artifacts
             ↓
        Complete / Fail
             ↓
      Destroy / Retain Sandbox
```

---

## 14.7.5 Workspace Design

A practical layout:

```text
/workspace/
├── input/
├── project/
├── output/
├── logs/
└── tmp/
```

Recommended separation:

| Directory  | Purpose                      |
| ---------- | ---------------------------- |
| `input/`   | User-provided inputs         |
| `project/` | Main working files           |
| `output/`  | Expected final artifacts     |
| `logs/`    | Task-specific logs           |
| `tmp/`     | Disposable intermediate data |

The workspace should be unique to the task unless explicit sharing is required.

---

## 14.7.6 Shell Execution

A shell request might conceptually contain:

```json
{
  "command": "pytest",
  "cwd": "/workspace/project",
  "timeout_seconds": 120
}
```

The runtime should validate:

```text
Command policy
Working directory
Timeout
Resource limits
Network permissions
Environment
```

Then return structured results:

```json
{
  "exit_code": 0,
  "stdout": "...",
  "stderr": "",
  "duration_ms": 12450
}
```

⭐ **Key Point:** The runtime should capture **structured execution results**, not only raw text.

---

## 14.7.7 Resource Controls

A task profile might conceptually specify:

```json
{
  "cpu": 2,
  "memory_mb": 4096,
  "timeout_seconds": 600,
  "max_steps": 50,
  "max_cost": 2.00
}
```

These are configuration examples, not universal defaults.

Controls should exist at multiple levels:

```text
Process
Node
Agent
Task
Tenant
Infrastructure
```

---

## 14.7.8 Network Controls

A runtime can implement:

```text
Network Policy
├── Disabled
├── Internal-only
├── Allowlisted domains
└── Restricted external access
```

Example:

```text
Research Agent

✓ wikipedia.org
✓ arxiv.org
✓ official-domain.example

✕ private internal network
✕ arbitrary IP ranges
```

The exact policy should follow the task's requirements.

---

## 14.7.9 Permission Controls

A tool/action can carry metadata:

```json
{
  "tool": "shell",
  "risk": "high",
  "requires_approval": false,
  "network": "restricted",
  "workspace": "task-only"
}
```

The runtime evaluates:

```text
Who?
 ↓
Which task?
 ↓
Which tool?
 ↓
Which resource?
 ↓
Which permissions?
 ↓
Allowed?
```

---

## 14.7.10 Artifact Persistence

At task completion:

```text
Workspace
   ↓
Identify artifacts
   ↓
Validate outputs
   ↓
Assign metadata
   ↓
Store externally
   ↓
Return artifact references
```

Example metadata:

```json
{
  "artifact_id": "artifact-77",
  "task_id": "task-102",
  "path": "/workspace/output/report.pdf",
  "type": "application/pdf",
  "size_bytes": 182340
}
```

---

## 14.7.11 Cancellation and Recovery

Cancellation:

```text
User cancels
 ↓
Runtime marks cancellation requested
 ↓
Interrupt agent
 ↓
Terminate child processes
 ↓
Persist state
 ↓
Collect useful artifacts/logs
 ↓
Finalize task as CANCELLED
```

Recovery after failure:

```text
Failure
 ↓
Checkpoint available?
 ├── Yes → Restore
 │          ↓
 │       Verify state
 │          ↓
 │       Resume / Recover
 │
 └── No  → Fail / Escalate
```

---

## 14.7.12 Runtime States

A useful state machine:

```text
CREATED
   ↓
PROVISIONING
   ↓
READY
   ↓
RUNNING
   ├──────────────► PAUSED
   │                   │
   │                   ▼
   │                 RESUMING
   │                   │
   │                   └──────► RUNNING
   │
   ├──────────────► WAITING_FOR_APPROVAL
   │                   │
   │                   └──────► RUNNING
   │
   ├──────────────► RECOVERING
   │                   │
   │                   ├────► RUNNING
   │                   └────► FAILED
   │
   ├──────────────► TIMED_OUT
   │
   ├──────────────► CANCELLED
   │
   └──────────────► COMPLETED
```

---

## 14.7.13 Example Execution

### Scenario

An agent receives:

> "Run the test suite and generate a report."

### Execution

```text
Task Created
    ↓
Sandbox Provisioned
    ↓
Repository Mounted / Prepared
    ↓
Agent loads task context
    ↓
Agent calls shell
    ↓
pytest
    ↓
Results captured
    ↓
Agent analyzes failures
    ↓
Agent generates report
    ↓
report.json
report.html
    ↓
Artifacts persisted
    ↓
Task completed
    ↓
Sandbox destroyed
```

### Result

```text
Task:
COMPLETED

Artifacts:
- report.html
- report.json

Trace:
- shell call
- test results
- report generation
```

This illustrates the difference between:

```text
Agent capability
```

and:

```text
Runtime execution infrastructure
```

---

# 14.8 Key Insights

💡 **Key Insights**

1. **The runtime is part of the agent system.** The model alone cannot safely perform arbitrary machine-side actions.

2. **The harness enforces boundaries.** Filesystem, network, CPU, memory, time, tools, and secrets should be controlled outside the model.

3. **Long-running agents require lifecycle management.** Creation, provisioning, checkpointing, pause/resume, completion, and cleanup must be explicit.

4. **Sandboxing is defense in depth.** Containers alone should not be treated as the entire security model; combine isolation, allowlists, resource limits, and policy enforcement.

5. **Runtime controls protect both safety and economics.** Max steps, tokens, cost, CPU, memory, and time prevent runaway execution.

6. **Artifacts should outlive ephemeral environments.** A sandbox can be destroyed while its useful outputs remain in durable artifact storage.

7. **State and environment are different layers.** The runtime must track the agent's execution state while also controlling and observing the actual environment.

---

# 14.9 Common Mistakes

⚠️ **Common Mistakes**

| Mistake                                                | Correct Understanding                                                                |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| "The prompt tells the agent what files it can access." | Filesystem access must be enforced by runtime controls.                              |
| "Containers automatically make execution safe."        | Containers are one isolation layer; configuration and defense-in-depth still matter. |
| "The shell is just another tool."                      | Shell access can expose powerful system capabilities and requires strong controls.   |
| "Network access can be unrestricted."                  | Network access should be explicitly scoped to task requirements.                     |
| "API keys can be placed in the prompt."                | Secrets require controlled credential handling.                                      |
| "Timeouting the request stops everything."             | Child processes and other resources may continue unless cancellation propagates.     |
| "Checkpointing means saving the chat."                 | Runtime checkpoints must capture actionable execution state.                         |
| "Destroying the sandbox destroys the outputs."         | Durable artifacts should be persisted before environment cleanup.                    |
| "Max token limits control all agent cost."             | Cost also comes from tools, compute, storage, network, and retries.                  |
| "If the agent says stop, it stops."                    | Runtime-level cancellation and termination must be enforceable externally.           |
| "Every task needs the same sandbox."                   | Runtime profiles should match task capability and risk.                              |
| "Logs can contain anything."                           | Logs can accidentally expose secrets or sensitive data and require governance.       |

---

# 14.10 Common Confusions

🔍 **Common Confusions**

| Concept A             | Concept B             | Key Difference                                                                                 |
| --------------------- | --------------------- | ---------------------------------------------------------------------------------------------- |
| Agent                 | Runtime               | Decision-making system vs execution infrastructure                                             |
| Runtime               | Harness               | Often overlapping terms; harness emphasizes control, tooling, and environment around the model |
| Workspace             | Artifact store        | Temporary working area vs durable output storage                                               |
| State                 | Filesystem            | Execution metadata vs file-based working data                                                  |
| Container             | Sandbox               | Container is one isolation mechanism; sandbox is the broader security boundary                 |
| Authentication        | Runtime authorization | Identity verification vs permission to use runtime capabilities                                |
| Tool allowlist        | Domain allowlist      | Which capabilities can be used vs which network destinations can be reached                    |
| CPU limit             | Quota                 | Per-execution resource boundary vs broader aggregate usage limit                               |
| Timeout               | Cancellation          | Time-based termination trigger vs intentional task termination                                 |
| Checkpoint            | Artifact              | Resume-oriented execution snapshot vs durable task output                                      |
| Environment state     | Agent state           | Actual external environment vs runtime representation of task state                            |
| Secret                | Environment variable  | Sensitive credential vs generic process configuration mechanism                                |
| Process isolation     | Workspace isolation   | Prevents process interaction vs limits file/data visibility                                    |
| Ephemeral environment | Durable environment   | Short-lived execution context vs intentionally preserved runtime                               |

---

# 14.11 Practical Applications

🛠️ **Practical Applications**

| Application               | Runtime / Harness Requirements                     |
| ------------------------- | -------------------------------------------------- |
| Coding agent              | Shell, filesystem, process control, test execution |
| Research agent            | Browser/search, network allowlists, artifacts      |
| Data-analysis agent       | Python, files, CPU/memory limits                   |
| Computer-use agent        | GUI environment, process isolation, screenshots    |
| DevOps agent              | Shell, cloud tools, strict permissions             |
| Security-analysis agent   | Isolated network, disposable workspace             |
| Document-processing agent | Filesystem, OCR/tools, artifact storage            |
| CI/CD agent               | Build processes, logs, test results, artifacts     |
| Autonomous workflow       | Long-running state, checkpoints, resume            |
| Enterprise agent          | Tenant isolation, secrets, quotas, auditability    |

---

# 14.12 Important Terms

📌 **Important Terms**

| Term                  | Simple Meaning                                   | Why It Matters                   |
| --------------------- | ------------------------------------------------ | -------------------------------- |
| Agent Runtime         | Infrastructure that executes and controls agents | Enables real-world operation     |
| Harness               | Control layer around the model and tools         | Enforces boundaries              |
| Execution Environment | Compute environment where work happens           | Provides capabilities            |
| Workspace             | Task-specific working directory/environment      | Enables isolation                |
| Filesystem Isolation  | Restricting visible files                        | Protects data                    |
| Shell                 | OS command interface                             | Enables powerful actions         |
| Network Policy        | Rules controlling connectivity                   | Limits external access           |
| Environment Variable  | Process configuration value                      | Controls runtime behavior        |
| Secret                | Sensitive credential                             | Requires protection              |
| Process Management    | Start/monitor/terminate processes                | Controls execution               |
| Sandbox               | Restricted execution environment                 | Limits damage and access         |
| Container             | Isolated runtime unit                            | Useful sandbox mechanism         |
| CPU Limit             | Maximum CPU allocation                           | Prevents resource exhaustion     |
| Memory Limit          | Maximum memory allocation                        | Prevents RAM exhaustion          |
| Time Limit            | Maximum execution duration                       | Prevents runaway tasks           |
| Tool Allowlist        | Allowed tool set                                 | Controls capabilities            |
| Domain Allowlist      | Allowed network destinations                     | Controls connectivity            |
| Quota                 | Aggregate usage limit                            | Controls abuse/cost              |
| Checkpoint            | Persisted execution snapshot                     | Enables resume                   |
| Cancellation          | Intentional task termination                     | Enables user/operator control    |
| Emergency Stop        | Immediate execution shutdown                     | Handles severe incidents         |
| Artifact              | Durable task output                              | Preserves useful results         |
| Artifact Store        | Persistent artifact storage                      | Separates outputs from runtime   |
| Control Plane         | Management layer                                 | Schedules and controls execution |
| Execution Plane       | Actual workload environment                      | Runs agent/processes             |
| Runtime Profile       | Task-specific execution policy                   | Matches capability to risk       |

---

# 14.13 Quick Revision

⚡ **Quick Revision**

1. **Runtime/harness engineering = controlling the environment in which an agent executes.**
2. The runtime provides **workspace, filesystem, shell, network, processes, secrets, tools, and artifacts**.
3. The model should not be the authority for filesystem, network, permission, or resource boundaries.
4. **Sandboxing** combines process, filesystem, network, resource, tool, and workspace controls.
5. Runtime lifecycle:

```text
Create
 ↓
Provision
 ↓
Load
 ↓
Execute
 ↓
Checkpoint
 ↓
Pause / Resume / Approve
 ↓
Persist Artifacts
 ↓
Complete / Fail / Cancel
 ↓
Destroy / Retain
```

6. Runtime controls include **cancellation, interrupts, timeouts, concurrency, quotas, max steps, max tokens, max cost, emergency stop, and human takeover**.
7. **Artifacts should survive sandbox destruction** through durable storage.
8. Long-running agents need **durable state + checkpoints + resumability**.
9. Shell and network access are high-impact capabilities and should be explicitly restricted.
10. A production harness is effectively the **execution control plane for agent autonomy**.

---

# 14.14 Interview Preparation

## 14.14.1 Level 1 — Fundamentals

### Q1. What is an agent runtime?

**Model Answer:**
An agent runtime is the infrastructure that executes and controls an agent. It manages the execution environment, tools, processes, state, resource limits, lifecycle, artifacts, and runtime policies around the model.

### Q2. What is a harness?

**Model Answer:**
A harness is the control layer surrounding the model and its tools. It provides the environment, permissions, resource controls, execution mechanisms, state management, and safeguards required to run an agent reliably.

### Q3. Why does an agent need a runtime?

**Model Answer:**
A model can generate decisions and tool requests, but it does not inherently provide controlled execution. The runtime translates those requests into real actions while enforcing permissions, isolation, resource limits, and lifecycle rules.

### Q4. What is a workspace?

**Model Answer:**
A workspace is a task-specific working environment, often containing files, code, inputs, temporary outputs, and generated artifacts. Isolating workspaces prevents unrelated tasks from interfering with one another.

### Q5. Why is sandboxing needed?

**Model Answer:**
Agents can execute code, read files, start processes, or access networks. Sandboxing limits those capabilities so a faulty or compromised agent cannot freely affect the host, other tasks, or unauthorized external systems.

### Q6. What is an artifact?

**Model Answer:**
An artifact is a durable output generated or consumed by a task, such as a report, image, code package, dataset, test result, or build output.

### Q7. Why are resource limits important?

**Model Answer:**
Without resource limits, agents can consume excessive CPU, memory, time, tokens, or money. Limits protect infrastructure, other tenants, system reliability, and operating costs.

### Q8. Why does a long-running agent need checkpoints?

**Model Answer:**
Checkpoints preserve enough execution state to resume after pauses, approvals, crashes, or infrastructure failures without restarting the task or incorrectly repeating side effects.

---

## 14.14.2 Level 2 — Conceptual Understanding

### Q1. Why is the runtime separate from the model?

**Model Answer:**
The model is probabilistic and generates decisions, while the runtime performs deterministic control and execution. Separating them allows security, authorization, resource limits, and cancellation to remain enforceable even if the model behaves incorrectly.

### Q2. Why aren't containers alone enough for sandboxing?

**Model Answer:**
Containers provide useful isolation, but a secure execution environment also needs appropriate filesystem, process, network, resource, capability, and credential controls. A container with excessive privileges can still expose significant risk.

### Q3. Why should network access be treated as a capability?

**Model Answer:**
Network access can allow agents to communicate with external services, download content, or potentially exfiltrate information. Therefore it should be explicitly scoped according to task requirements.

### Q4. Why are shell tools particularly sensitive?

**Model Answer:**
A shell can invoke many operating-system capabilities indirectly: filesystem access, processes, networking, package installation, and system commands. Its effective capability surface is much larger than a narrowly scoped API.

### Q5. Why do artifacts need separate storage?

**Model Answer:**
The runtime environment is often ephemeral. Durable artifact storage allows useful outputs to survive sandbox destruction and provides versioning, access control, and lifecycle management.

### Q6. Why do runtime cost controls need more than token limits?

**Model Answer:**
Total cost may include model inference, tool calls, compute, storage, network traffic, and retries. Token limits control only one portion of overall resource consumption.

### Q7. Why should task environments be isolated?

**Model Answer:**
Isolation prevents one task from accessing another task's files, processes, credentials, or runtime state. It also reduces interference and improves reproducibility.

### Q8. Why should cancellation propagate to child processes?

**Model Answer:**
Stopping only the top-level agent does not necessarily terminate subprocesses or external work. Those processes can continue consuming resources or modifying state after the user has cancelled the task.

---

## 14.14.3 Level 3 — Practical / Engineering

### Q1. How would you design a sandboxed execution pipeline?

**Model Answer:**

```text
Task
 ↓
Policy Check
 ↓
Provision Sandbox
 ↓
Create Isolated Workspace
 ↓
Apply CPU / Memory / Time Limits
 ↓
Apply Network Policy
 ↓
Apply Tool Allowlist
 ↓
Start Agent
 ↓
Execute Actions
 ↓
Capture Results
 ↓
Checkpoint
 ↓
Persist Artifacts
 ↓
Destroy / Retain
```

The model should never bypass these controls.

### Q2. How would you safely expose shell access to an agent?

**Model Answer:**
Run shell commands inside an isolated environment with a task-specific workspace, bounded CPU/memory/time, controlled network access, restricted credentials, process monitoring, cancellation propagation, and structured output capture.

### Q3. How would you handle a runaway process?

**Model Answer:**

```text
Detect threshold violation
 ↓
Interrupt process
 ↓
Terminate child processes
 ↓
Persist logs / state
 ↓
Mark task recovering or failed
 ↓
Clean environment
```

The runtime should enforce this externally rather than relying on the agent to stop itself.

### Q4. How would you implement human approval in a long-running runtime?

**Model Answer:**

```text
Agent reaches checkpoint
 ↓
Persist execution state
 ↓
WAITING_FOR_APPROVAL
 ↓
Human approves
 ↓
Load checkpoint
 ↓
Resume runtime
```

The environment may be retained or re-provisioned depending on the durability design.

### Q5. How would you handle task cancellation?

**Model Answer:**
Record cancellation intent, stop the agent loop, propagate cancellation to active tools and child processes, persist useful state and artifacts, finalize the task as cancelled, and clean up the execution environment.

### Q6. How would you isolate two users' agent tasks?

**Model Answer:**
Give each task its own identity, workspace, state scope, sandbox, credentials, and artifact namespace. Enforce tenant/user authorization at storage and execution boundaries rather than relying on prompts.

### Q7. How would you design artifact persistence?

**Model Answer:**
After execution, identify approved outputs, validate them, attach metadata such as task and version identifiers, persist them in durable storage, return artifact references, and apply explicit retention/deletion rules.

### Q8. How would you observe an agent runtime?

**Model Answer:**
Correlate request ID → task ID → runtime ID → process/node IDs → tool calls → state transitions → artifact IDs. Record logs, traces, resource usage, errors, latency, and cost.

---

## 14.14.4 Level 4 — Advanced / Deep Understanding

### Q1. Why is the runtime a security boundary?

**Model Answer:**
The runtime determines what the agent can actually execute and access. Even if the model requests an unsafe action, runtime-level controls can reject it, restrict the environment, terminate execution, or prevent access to protected resources.

### Q2. Why is sandboxing a defense-in-depth problem?

**Model Answer:**
No single isolation mechanism addresses every failure mode. Process isolation does not control network access; filesystem isolation does not limit CPU; resource limits do not prevent unauthorized credentials. Multiple independent controls are needed.

### Q3. Why can a task remain dangerous after the model stops?

**Model Answer:**
The model is only one process. Child processes, background jobs, network requests, or scheduled work may continue after the agent itself stops. The runtime must own process and resource lifecycle.

### Q4. Why can an environment be destroyed while its task remains resumable?

**Model Answer:**
A task's durable state and artifacts can be stored separately from its ephemeral compute environment. A future runtime instance can provision a new sandbox and reconstruct the task from persisted state.

### Q5. Why should secrets be capability-scoped?

**Model Answer:**
An agent may need to invoke a service without needing raw access to its credentials. Capability-scoped secret injection reduces credential exposure and makes access easier to revoke or audit.

### Q6. Why should runtime policies be task-specific?

**Model Answer:**
Different tasks need different capabilities and risk levels. A document summarizer may need only file access, while a coding agent may need a shell and compiler. Granting the broadest environment to every task increases risk unnecessarily.

### Q7. Why does artifact persistence need provenance?

**Model Answer:**
Artifacts should be traceable to the task, runtime, source version, and execution that produced them. This supports reproducibility, debugging, evaluation, and auditability.

### Q8. Why can max-step controls fail to control a runaway agent?

**Model Answer:**
A task can consume substantial resources within a small number of steps. One shell command can launch a long-running process or consume large CPU/memory resources. Therefore step limits must complement process, resource, time, and cost controls.

---

## 14.14.5 Level 5 — Scenario-Based Questions

### Scenario 1 — Coding Agent Runs an Infinite Process

The agent executes:

```text
python infinite_loop.py
```

The process consumes CPU indefinitely.

**Question:** What should the runtime do?

**Model Answer:**

```text
Process started
 ↓
Resource monitor
 ↓
CPU/time threshold exceeded
 ↓
Interrupt
 ↓
Terminate process tree
 ↓
Capture logs
 ↓
Checkpoint / record failure
 ↓
Clean sandbox
```

The agent should not be trusted to terminate itself.

---

### Scenario 2 — Agent Needs Internet Access

A research agent needs to search the web but should not access internal enterprise services.

**Question:** How would you design the network policy?

**Model Answer:**

```text
Internet Access
      ↓
Outbound Proxy / Network Policy
      ↓
Allowlisted Research Domains
      ↓
Block Internal Networks
      ↓
Log Requests
```

The policy should be enforced at the network/runtime layer, not solely through instructions.

---

### Scenario 3 — Secret Required for an API Call

An agent needs to call an external service requiring credentials.

**Question:** Should the API key be inserted into the prompt?

**Model Answer:**
No. The runtime should inject or broker the credential at execution time with minimum necessary scope. The secret should not be unnecessarily visible in model context, logs, artifacts, or shell history.

---

### Scenario 4 — User Cancels the Agent

The agent is running several subprocesses when the user presses **Cancel**.

**Question:** What should happen?

**Model Answer:**

```text
Cancel Request
 ↓
Mark cancellation requested
 ↓
Interrupt agent
 ↓
Cancel active tools
 ↓
Terminate child processes
 ↓
Persist useful state/logs
 ↓
Collect artifacts
 ↓
Cleanup sandbox
 ↓
CANCELLED
```

Cancellation must propagate through the entire execution tree.

---

### Scenario 5 — Agent Pauses for Human Approval

The agent has completed analysis but needs approval before executing a high-risk command.

**Question:** How should the runtime behave?

**Model Answer:**

```text
Analysis complete
 ↓
Checkpoint
 ↓
WAITING_FOR_APPROVAL
 ↓
Environment paused / retained according to policy
 ↓
Human approves
 ↓
Restore state
 ↓
Execute approved command
 ↓
Verify
```

The key requirement is durable state and a clear authorization boundary before the side effect.

---

### Scenario 6 — Agent Generates a Large Dataset

The agent creates a 5 GB dataset.

**Question:** Should that dataset remain inside the sandbox?

**Model Answer:**
Not necessarily. The runtime should identify it as a durable artifact, validate the output, move or copy it to appropriate artifact storage, record metadata, and then clean up the ephemeral workspace according to policy.

---

### Scenario 7 — Cross-Task File Access

An agent running Task A attempts to read:

```text
/workspace/task-B/secrets.txt
```

**Question:** Where should this request be blocked?

**Model Answer:**
The filesystem/sandbox boundary should deny access before the file contents reach the model. The model prompt should not be the primary isolation mechanism.

---

# 14.14.6 Knowledge Check

🧠 **Knowledge Check**

If you can explain these naturally in your own words, you understand Layer 12:

* What agent runtime / harness engineering means.
* Why agents need execution environments.
* What a workspace is.
* Why filesystem isolation matters.
* Why shell execution is powerful.
* Why network access must be controlled.
* How environment variables differ from secrets.
* Why process management matters.
* What sandboxing provides.
* Why sandboxing uses defense in depth.
* How containers contribute to isolation.
* Why CPU, memory, and time limits are necessary.
* What tool and domain allowlists do.
* Why workspace boundaries matter.
* The complete runtime lifecycle.
* How checkpoints support long-running tasks.
* How pause/resume works.
* How cancellation differs from interruption.
* How quotas differ from per-task limits.
* Why max steps do not replace resource controls.
* Why max tokens do not represent total system cost.
* What an emergency stop does.
* What human takeover means.
* What artifacts are.
* Why artifacts need durable storage.
* How runtime control plane and execution plane differ.
* How state, sandbox, artifacts, and observability fit together.
* How to design the sandboxed agent runtime project.

---

# 14.14.7 Follow-up Questions

### Basic Question

**What is an agent runtime?**

→ Why is it needed?
→ What executes inside it?
→ What does it control?
→ How is it secured?
→ How does it persist state?

### Basic Question

**What is sandboxing?**

→ Process isolation?
→ Filesystem isolation?
→ Network restrictions?
→ CPU limits?
→ Memory limits?
→ Time limits?
→ Tool allowlists?

### Basic Question

**How does a runtime lifecycle work?**

→ Create?
→ Provision?
→ Execute?
→ Checkpoint?
→ Pause?
→ Resume?
→ Persist artifacts?
→ Cleanup?

### Basic Question

**How do you control an agent?**

→ Cancellation?
→ Interruptibility?
→ Timeouts?
→ Concurrency?
→ Quotas?
→ Max steps?
→ Max tokens?
→ Max cost?
→ Emergency stop?

### Basic Question

**How do you safely execute shell commands?**

→ Isolated workspace?
→ Process control?
→ Network restrictions?
→ Resource limits?
→ Secret handling?
→ Artifact collection?

---

# 14.14.8 Common Confusion Questions

### Q1. Is an agent runtime the same as an LLM?

**Model Answer:**
No. The LLM provides model inference and reasoning capability; the runtime provides controlled execution, resources, state, tools, processes, and lifecycle management.

### Q2. Is a container the same as a sandbox?

**Model Answer:**
Not exactly. A container is one mechanism for isolation. A sandbox is the broader execution security boundary that may include containerization plus network, filesystem, process, resource, capability, and credential controls.

### Q3. Is a workspace the same as artifact storage?

**Model Answer:**
No. A workspace is where active work happens. Artifact storage preserves selected outputs after or beyond the runtime lifetime.

### Q4. Is a timeout the same as cancellation?

**Model Answer:**
No. A timeout is a policy-triggered termination due to exceeding a duration. Cancellation is an intentional request to stop execution.

### Q5. Is a checkpoint an artifact?

**Model Answer:**
Usually they serve different purposes. A checkpoint stores execution state needed to resume; an artifact is a task output or input that should persist independently.

### Q6. Is an environment variable a secret?

**Model Answer:**
Not inherently. Environment variables are a configuration mechanism. Some environment variables contain secrets, but sensitive credentials require additional handling regardless of how they are passed.

---

# 14.14.9 Deep / Trick Questions

### ⚠️ Deeper Question

**If the agent is running inside a sandbox, why do we still need tool permissions?**

**Correct Understanding:**
A sandbox restricts the overall execution environment, but the agent may still have access to multiple capabilities inside it. Tool allowlists further reduce the agent's action surface and enforce least privilege.

---

### ⚠️ Deeper Question

**Why isn't a maximum step count enough to prevent runaway execution?**

**Correct Understanding:**
A single step can start a process that runs indefinitely, consume large memory, or make many internal operations. Step limits must be combined with process, time, resource, network, and cost controls.

---

### ⚠️ Deeper Question

**Why is cancellation harder than setting a cancelled flag?**

**Correct Understanding:**
A task can have an entire execution tree containing child processes, tools, network calls, and queued work. A cancelled flag does not automatically terminate those activities. Cancellation must propagate through the runtime.

---

### ⚠️ Deeper Question

**Why might you destroy the execution environment after completion but retain the task state?**

**Correct Understanding:**
The compute environment may no longer be needed, but the task state can remain useful for audit, reporting, future retrieval, or resumption. Durable state and compute lifetime do not need to be identical.

---

### ⚠️ Deeper Question

**Why should secrets be separated from context?**

**Correct Understanding:**
If a credential is placed in model context, it may become visible to the model, traces, logs, summaries, or generated artifacts. Capability-scoped runtime injection can allow the action without unnecessarily exposing the credential.

---

### ⚠️ Deeper Question

**Why is the network an agent capability rather than just infrastructure?**

**Correct Understanding:**
Network access determines which external systems the agent can observe or affect. It therefore directly expands the agent's action surface and must be governed as a permission.

---

### ⚠️ Deeper Question

**Why does artifact persistence matter for reproducibility?**

**Correct Understanding:**
An ephemeral runtime may disappear after execution. Persistent artifacts, together with task/runtime metadata and provenance, preserve the outputs needed to inspect, evaluate, reproduce, or consume the result.

---

# 14.15 Top Questions You MUST Know

⭐ **Top Questions You MUST Know**

1. What is agent runtime / harness engineering?
2. Why can't the model itself safely manage its execution environment?
3. What components make up an agent runtime?
4. What is sandboxing and why is it necessary?
5. Why are containers only one part of a sandbox?
6. How would you isolate an agent's filesystem and workspace?
7. How would you safely provide shell access?
8. How should network access be controlled?
9. How should secrets be provided to an agent?
10. How do checkpoints and resumability work?
11. How would you implement cancellation across child processes?
12. How do CPU, memory, time, token, step, and cost limits complement each other?
13. What is the difference between the control plane and execution plane?
14. How should artifacts survive sandbox destruction?
15. How would you design a production-grade sandboxed agent runtime?

---

# 14.16 Interview Readiness Checklist

🎯 **Interview Readiness Checklist**

| Skill                    | Can I explain it? |
| ------------------------ | :---------------: |
| Agent runtime definition |         ☐         |
| Harness definition       |         ☐         |
| Agent loop               |         ☐         |
| Execution environment    |         ☐         |
| Workspace                |         ☐         |
| Filesystem isolation     |         ☐         |
| Shell execution          |         ☐         |
| Network access           |         ☐         |
| Environment variables    |         ☐         |
| Secret handling          |         ☐         |
| Artifact storage         |         ☐         |
| Process management       |         ☐         |
| Sandboxing               |         ☐         |
| Containers               |         ☐         |
| Process isolation        |         ☐         |
| Network restrictions     |         ☐         |
| CPU limits               |         ☐         |
| Memory limits            |         ☐         |
| Time limits              |         ☐         |
| Tool allowlists          |         ☐         |
| Domain allowlists        |         ☐         |
| Workspace boundaries     |         ☐         |
| Runtime lifecycle        |         ☐         |
| Task creation            |         ☐         |
| Environment provisioning |         ☐         |
| Context / skill loading  |         ☐         |
| Checkpoints              |         ☐         |
| Pause / resume           |         ☐         |
| Approval flow            |         ☐         |
| Artifact persistence     |         ☐         |
| Task terminal states     |         ☐         |
| Cancellation             |         ☐         |
| Interruptibility         |         ☐         |
| Timeouts                 |         ☐         |
| Concurrency              |         ☐         |
| Quotas                   |         ☐         |
| Max steps                |         ☐         |
| Max tokens               |         ☐         |
| Max cost                 |         ☐         |
| Emergency stop           |         ☐         |
| Human takeover           |         ☐         |
| Artifact lifecycle       |         ☐         |
| Control plane            |         ☐         |
| Execution plane          |         ☐         |
| State layer              |         ☐         |
| Policy layer             |         ☐         |
| Observability            |         ☐         |
| Sandbox architecture     |         ☐         |
| Runtime recovery         |         ☐         |
| Production security      |         ☐         |

---

# 14.17 What You Should Be Able to Explain

🧠 **What You Should Be Able to Explain**

By the end of Layer 12, you should be able to explain:

* What agent runtime and harness engineering mean.
* Why long-running agents need a dedicated execution runtime.
* The difference between the model and the runtime.
* What an execution environment contains.
* How workspaces isolate task data.
* How filesystem isolation protects the host and other tasks.
* Why shell execution is a high-capability operation.
* How network access expands an agent's action surface.
* How environment variables and secrets should be handled.
* Why secret access should be scoped to capabilities.
* How process management works.
* What artifacts are and why they need durable storage.
* What sandboxing means.
* How containers contribute to isolation.
* Why process isolation matters.
* Why filesystem isolation matters.
* How network restrictions reduce risk.
* Why CPU, memory, and time limits are required.
* How tool allowlists restrict capabilities.
* How domain allowlists restrict network destinations.
* Why every task should have an explicit workspace boundary.
* Why sandboxing should use defense in depth.
* How the runtime lifecycle works from task creation to cleanup.
* How environments are provisioned.
* How context and skills are loaded.
* How agent execution is monitored.
* How checkpoints support long-running tasks.
* How human approval pauses and resumes execution.
* How artifacts are persisted before environment destruction.
* How completion, failure, timeout, cancellation, and pause states differ.
* Why environments may be destroyed while durable task state remains.
* How cancellation must propagate to tools and child processes.
* What interruptibility means.
* How timeouts differ from cancellation.
* How concurrency controls prevent resource exhaustion.
* What quotas are.
* Why max steps are useful but insufficient by themselves.
* Why token limits are not equivalent to total cost limits.
* What maximum cost controls accomplish.
* What an emergency stop should do.
* How human takeover works.
* How artifacts move through their lifecycle.
* The difference between ephemeral runtime state and durable outputs.
* The difference between control plane and execution plane.
* How policy/security, state/checkpoints, workspace, artifacts, and observability fit together.
* How to design a sandboxed runtime capable of shell execution.
* How to enforce CPU, memory, time, network, and permission controls.
* How to persist and trace task artifacts.
* How to recover or cancel an agent safely.
* Why **the runtime—not the model—is the ultimate execution boundary**.

## ⚡ Final Mental Model

```text
                          USER TASK
                              │
                              ▼
                       ┌──────────────┐
                       │ TASK CONTROL │
                       │   / API      │
                       └──────┬───────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  RUNTIME CONTROL  │
                    │                   │
                    │ Policy            │
                    │ Scheduling        │
                    │ Quotas            │
                    │ Cancellation      │
                    │ Lifecycle         │
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │ SANDBOX PROVISION │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         Filesystem       Process        Network
         Isolation        Isolation      Policy
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌───────────────────┐
                    │ RESOURCE CONTROL  │
                    │                   │
                    │ CPU               │
                    │ Memory            │
                    │ Time              │
                    │ Steps             │
                    │ Tokens            │
                    │ Cost              │
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │  AGENT EXECUTION  │
                    │                   │
                    │ Model             │
                    │ Planner           │
                    │ Tools             │
                    │ Shell             │
                    └────────┬──────────┘
                             │
                       Action / Result
                             │
                             ▼
                    ┌───────────────────┐
                    │    STATE /        │
                    │    CHECKPOINT     │
                    └────────┬──────────┘
                             │
                   ┌─────────┼─────────┐
                   ▼         ▼         ▼
                Continue    Pause    Approval
                   │         │         │
                   └─────────┼─────────┘
                             ▼
                        VERIFICATION
                             │
                             ▼
                    ┌───────────────────┐
                    │     ARTIFACTS     │
                    │                   │
                    │ Files             │
                    │ Reports           │
                    │ Code              │
                    │ Images            │
                    │ Datasets          │
                    │ Test Results      │
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │ OBSERVABILITY     │
                    │                   │
                    │ Logs              │
                    │ Metrics           │
                    │ Traces            │
                    │ Audit             │
                    └────────┬──────────┘
                             │
                             ▼
                       COMPLETE / FAIL /
                       CANCEL / TIMEOUT
                             │
                             ▼
                    DESTROY / RETAIN ENV
```

> **Core principle:** **An agent runtime is the controlled execution substrate for autonomy. The model supplies intelligence, while the harness supplies boundaries: isolation, permissions, resources, process control, persistence, artifacts, cancellation, recovery, and observability. Production-grade autonomy is therefore not just better reasoning—it is reasoning operating inside a deliberately engineered and enforceable runtime.**
