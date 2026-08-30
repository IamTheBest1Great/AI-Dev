# 📚 Table of Contents

* [16. Layer 14 — Agent Skills](#16-layer-14--agent-skills)

  * [16.1 Skills Architecture](#161-skills-architecture)

    * [16.1.1 Skill Metadata](#1611-skill-metadata)
    * [16.1.2 Skill Instructions](#1612-skill-instructions)
    * [16.1.3 Resources](#1613-resources)
    * [16.1.4 Scripts](#1614-scripts)
    * [16.1.5 Examples](#1615-examples)
    * [16.1.6 Dependencies](#1616-dependencies)
    * [16.1.7 Versioning](#1617-versioning)
    * [16.1.8 Discovery](#1618-discovery)
    * [16.1.9 Progressive Loading](#1619-progressive-loading)
    * [16.1.10 Skills Architecture](#16110-skills-architecture)
  * [16.2 Skill Lifecycle](#162-skill-lifecycle)

    * [16.2.1 Discover](#1621-discover)
    * [16.2.2 Select](#1622-select)
    * [16.2.3 Authorize](#1623-authorize)
    * [16.2.4 Load](#1624-load)
    * [16.2.5 Execute](#1625-execute)
    * [16.2.6 Validate](#1626-validate)
    * [16.2.7 Record Outcome](#1627-record-outcome)
    * [16.2.8 Complete Skill Lifecycle](#1628-complete-skill-lifecycle)
  * [16.3 Skills Engineering](#163-skills-engineering)

    * [16.3.1 Skill Composition](#1631-skill-composition)
    * [16.3.2 Skill Conflicts](#1632-skill-conflicts)
    * [16.3.3 Skill Permissions](#1633-skill-permissions)
    * [16.3.4 Skill Testing](#1634-skill-testing)
    * [16.3.5 Skill Portability](#1635-skill-portability)
    * [16.3.6 Skill Deprecation](#1636-skill-deprecation)
    * [16.3.7 Skill Version Migration](#1637-skill-version-migration)
    * [16.3.8 Skill Governance](#1638-skill-governance)
  * [16.4 Skills vs Tools vs MCP](#164-skills-vs-tools-vs-mcp)

    * [16.4.1 Prompt](#1641-prompt)
    * [16.4.2 Skill](#1642-skill)
    * [16.4.3 Tool](#1643-tool)
    * [16.4.4 MCP](#1644-mcp)
    * [16.4.5 A2A](#1645-a2a)
    * [16.4.6 Conceptual Comparison](#1646-conceptual-comparison)
    * [16.4.7 How They Work Together](#1647-how-they-work-together)
  * [16.5 Agent Skill Architecture](#165-agent-skill-architecture)

    * [16.5.1 Skill Package](#1651-skill-package)
    * [16.5.2 Skill Registry](#1652-skill-registry)
    * [16.5.3 Discovery Layer](#1653-discovery-layer)
    * [16.5.4 Authorization Layer](#1654-authorization-layer)
    * [16.5.5 Loading Layer](#1655-loading-layer)
    * [16.5.6 Execution Layer](#1656-execution-layer)
    * [16.5.7 Validation Layer](#1657-validation-layer)
    * [16.5.8 Outcome and Telemetry Layer](#1658-outcome-and-telemetry-layer)
    * [16.5.9 Skills Runtime Flow](#1659-skills-runtime-flow)
  * [16.6 Agent Skill Pack Project](#166-agent-skill-pack-project)

    * [16.6.1 Project Goal](#1661-project-goal)
    * [16.6.2 Research Skill](#1662-research-skill)
    * [16.6.3 Document Analysis Skill](#1663-document-analysis-skill)
    * [16.6.4 Coding Skill](#1664-coding-skill)
    * [16.6.5 Data Extraction Skill](#1665-data-extraction-skill)
    * [16.6.6 Skill Pack Structure](#1666-skill-pack-structure)
    * [16.6.7 Skill Discovery Flow](#1667-skill-discovery-flow)
    * [16.6.8 Progressive Loading Flow](#1668-progressive-loading-flow)
    * [16.6.9 Skill Validation](#1669-skill-validation)
    * [16.6.10 Skill Pack Versioning](#16610-skill-pack-versioning)
  * [16.7 Key Insights](#167-key-insights)
  * [16.8 Common Mistakes](#168-common-mistakes)
  * [16.9 Common Confusions](#169-common-confusions)
  * [16.10 Practical Applications](#1610-practical-applications)
  * [16.11 Important Terms](#1611-important-terms)
  * [16.12 Quick Revision](#1612-quick-revision)
  * [16.13 Interview Preparation](#1613-interview-preparation)

    * [16.13.1 Level 1 — Fundamentals](#16131-level-1--fundamentals)
    * [16.13.2 Level 2 — Conceptual Understanding](#16132-level-2--conceptual-understanding)
    * [16.13.3 Level 3 — Practical / Engineering](#16133-level-3--practical--engineering)
    * [16.13.4 Level 4 — Advanced / Deep Understanding](#16134-level-4--advanced--deep-understanding)
    * [16.13.5 Level 5 — Scenario-Based Questions](#16135-level-5--scenario-based-questions)
    * [16.13.6 Knowledge Check](#16136-knowledge-check)
    * [16.13.7 Follow-up Questions](#16137-follow-up-questions)
    * [16.13.8 Common Confusion Questions](#16138-common-confusion-questions)
    * [16.13.9 Deep / Trick Questions](#16139-deep--trick-questions)
  * [16.14 Top Questions You MUST Know](#1614-top-questions-you-must-know)
  * [16.15 Interview Readiness Checklist](#1615-interview-readiness-checklist)
  * [16.16 What You Should Be Able to Explain](#1616-what-you-should-be-able-to-explain)

# 16. Layer 14 — Agent Skills

🧠 **Simple Understanding:** Agent Skills are **reusable procedural capabilities** that an agent can discover and load when a task requires them.

A skill is more than a prompt and less than a single tool.

A useful mental model is:

```text
Task
 ↓
Discover Skill
 ↓
Select Skill
 ↓
Authorize
 ↓
Load Instructions + Resources
 ↓
Execute Procedure
 ↓
Validate Result
 ↓
Record Outcome
```

The roadmap defines Agent Skills as reusable procedural capabilities that can be **discovered and loaded by agents**.

⭐ **Core Principle:** A skill packages **how to perform a capability**, while tools provide the executable actions used during that procedure.

---

# 16.1 Skills Architecture

A reusable skill can contain:

```text
Skill
├── Metadata
├── Instructions
├── Resources
├── Scripts
├── Examples
├── Dependencies
├── Version
└── Discovery information
```

The architecture allows an agent to load only the procedure it currently needs.

---

## 16.1.1 Skill Metadata

🧠 **Simple Understanding:** Metadata describes the skill and helps the agent or runtime decide whether it is relevant.

Possible metadata:

```text
Name
Description
Version
Capabilities
Inputs
Outputs
Dependencies
Permissions
Tags
Compatibility
```

Example:

```json
{
  "name": "research",
  "version": "1.2.0",
  "description": "Conduct source-based research and produce cited findings.",
  "tags": ["research", "web", "evidence"]
}
```

### 📌 Quick Info

| Field     | Answer                                |
| --------- | ------------------------------------- |
| **What?** | Descriptive information about a skill |
| **Why?**  | Supports discovery and selection      |
| **How?**  | Structured metadata                   |
| **When?** | Before loading or executing a skill   |

---

## 16.1.2 Skill Instructions

🧠 **Simple Understanding:** Skill instructions describe the procedure the agent should follow.

Example:

```text
1. Define the research question.
2. Search authoritative sources.
3. Compare evidence.
4. Identify contradictions.
5. Produce cited findings.
```

A skill's instructions should define:

* Objective.
* Procedure.
* Constraints.
* Expected outputs.
* Validation criteria.

⭐ **Key Point:** A skill should encode **reusable procedure**, not merely repeat a generic role prompt.

---

## 16.1.3 Resources

🧠 **Simple Understanding:** Resources are supporting materials required by the skill.

Examples:

```text
Templates
Reference documents
Schemas
Checklists
Policies
Domain knowledge
Configuration
```

Example:

```text
Research Skill
├── instructions.md
├── citation_rules.md
├── source_quality.md
└── report_template.md
```

Resources allow the procedural instructions to remain focused while supporting information lives separately.

---

## 16.1.4 Scripts

🧠 **Simple Understanding:** Scripts provide executable helper logic used by a skill.

Examples:

```text
parse_csv.py
validate_citations.py
extract_tables.py
run_tests.sh
```

Scripts are useful when deterministic computation is preferable to asking the model to perform the same operation.

⚠️ **Important:** Scripts are executable components and therefore inherit runtime, permission, and security requirements.

---

## 16.1.5 Examples

🧠 **Simple Understanding:** Examples demonstrate how the skill should be applied.

Examples can show:

```text
Input
 ↓
Expected procedure
 ↓
Expected output
```

Useful for:

* Complex workflows.
* Domain-specific conventions.
* Output formatting.
* Edge cases.

Examples should remain aligned with the current skill version.

---

## 16.1.6 Dependencies

🧠 **Simple Understanding:** Dependencies are capabilities or resources required for the skill to function.

Examples:

```text
Research Skill
 ├── web search
 ├── URL fetch
 └── citation validator

Coding Skill
 ├── filesystem
 ├── shell
 └── test runner
```

Dependencies can include:

* Tools.
* Services.
* Libraries.
* Runtime capabilities.
* Other skills.

---

## 16.1.7 Versioning

🧠 **Simple Understanding:** Skill versioning tracks changes to a procedural capability over time.

Example:

```text
research v1.0
    ↓
research v1.1
    ↓
research v2.0
```

Versioning matters because changes to instructions, scripts, dependencies, or resources can change agent behavior.

A useful package can expose:

```text
Skill Name
Version
Compatibility
Dependencies
Release Notes
```

---

## 16.1.8 Discovery

🧠 **Simple Understanding:** Skill discovery determines which skills are available and which ones might be relevant to the current task.

```text
Task
 ↓
Skill Registry
 ↓
Candidate Skills
 ↓
Relevance Filtering
```

Discovery can use:

* Names.
* Descriptions.
* Tags.
* Capabilities.
* Input/output requirements.

⭐ **Key Point:** Discovery allows the agent to operate over a **library of reusable capabilities** rather than having every procedure permanently loaded into context.

---

## 16.1.9 Progressive Loading

🧠 **Simple Understanding:** Progressive loading means loading only the amount of skill information needed at each stage.

Instead of:

```text
Load entire skill library
```

use:

```text
Skill metadata
      ↓
Relevant skill selected
      ↓
Load instructions
      ↓
Load required resources
      ↓
Load scripts only when needed
```

Benefits:

* Lower context usage.
* Faster startup.
* Better relevance.
* Reduced cognitive load.
* Easier skill management.

---

## 16.1.10 Skills Architecture

```text
                     SKILL REGISTRY
                           │
                           ▼
                       Metadata
                           │
                           ▼
                      Discovery
                           │
                           ▼
                       Selection
                           │
                           ▼
                      Authorization
                           │
                           ▼
                 ┌────────────────────┐
                 │    SKILL PACKAGE   │
                 │                    │
                 │ Instructions       │
                 │ Resources          │
                 │ Scripts            │
                 │ Examples           │
                 │ Dependencies       │
                 └─────────┬──────────┘
                           │
                           ▼
                         Agent
                           │
                           ▼
                       Execution
                           │
                           ▼
                        Validate
                           │
                           ▼
                     Record Outcome
```

---

# 16.2 Skill Lifecycle

The core lifecycle is:

```text
Discover
 ↓
Select
 ↓
Authorize
 ↓
Load
 ↓
Execute
 ↓
Validate
 ↓
Record outcome
```

---

## 16.2.1 Discover

🧠 **Simple Understanding:** Find skills that might help with the task.

Example:

```text
Task:
"Analyze this financial report."

Available skills:
├── Research
├── Document Analysis
├── Coding
└── Data Extraction

Likely candidates:
Document Analysis
Data Extraction
```

---

## 16.2.2 Select

🧠 **Simple Understanding:** Choose the skill or combination of skills that best fits the task.

Selection may consider:

```text
Task intent
Skill description
Required inputs
Available dependencies
User permissions
Skill version
```

A skill can be selected automatically or through explicit workflow logic.

---

## 16.2.3 Authorize

🧠 **Simple Understanding:** Verify that the agent/task is allowed to use the selected skill and its dependencies.

Example:

```text
Coding Skill
 ↓
Requires shell
 ↓
Is shell allowed?
 ├── Yes → Continue
 └── No  → Reject / choose alternate skill
```

Authorization must consider not just the skill itself, but its underlying capabilities.

⭐ **Key Point:** **A skill can be permitted while one of its dependencies is not.**

---

## 16.2.4 Load

🧠 **Simple Understanding:** Load the skill information required for execution.

Possible loading sequence:

```text
Metadata
 ↓
Instructions
 ↓
Required resources
 ↓
Required tools/scripts
```

This supports progressive loading.

---

## 16.2.5 Execute

🧠 **Simple Understanding:** The agent follows the skill procedure using available tools and resources.

Example:

```text
Research Skill
 ↓
Search
 ↓
Read sources
 ↓
Compare evidence
 ↓
Validate
 ↓
Generate findings
```

---

## 16.2.6 Validate

🧠 **Simple Understanding:** Check whether the skill produced the required result.

Validation can inspect:

* Output structure.
* Required fields.
* Evidence.
* Correctness.
* Tool execution.
* Completion criteria.

Example:

```text
Expected:
Cited report

Actual:
Report with no citations

→ Validation fails
```

---

## 16.2.7 Record Outcome

🧠 **Simple Understanding:** Store the result of the skill execution for observability and future evaluation.

Record:

```text
Skill
Version
Task
Inputs
Dependencies
Outcome
Errors
Duration
Artifacts
```

Example:

```json
{
  "skill": "research",
  "version": "1.2.0",
  "status": "completed",
  "sources": 12
}
```

---

## 16.2.8 Complete Skill Lifecycle

```text
                     TASK
                       │
                       ▼
                    DISCOVER
                       │
                       ▼
                     SELECT
                       │
                       ▼
                  AUTHORIZE
                       │
                       ▼
                      LOAD
                       │
                       ▼
                    EXECUTE
                       │
                       ▼
                    VALIDATE
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Success            Failure
              │                 │
              ▼                 ▼
        Record Outcome     Retry / Replan /
                              Escalate
```

---

# 16.3 Skills Engineering

## 16.3.1 Skill Composition

🧠 **Simple Understanding:** Complex tasks can combine multiple reusable skills.

Example:

```text
Research Project
      │
      ├── Research Skill
      │
      ├── Document Analysis Skill
      │
      ├── Data Extraction Skill
      │
      └── Report Generation Skill
```

Composition allows specialization without creating one giant skill.

### When Useful

* Large workflows.
* Specialized procedures.
* Reusable capabilities.
* Team-owned components.

⚠️ **Trade-off:** Too many composable skills can create dependency and coordination complexity.

---

## 16.3.2 Skill Conflicts

🧠 **Simple Understanding:** Two skills may contain incompatible instructions, assumptions, or procedures.

Example:

```text
Skill A:
"Always output JSON."

Skill B:
"Always output Markdown."
```

Potential conflict sources:

* Instructions.
* Tool usage.
* Output formats.
* Policies.
* Dependencies.
* Priorities.

A runtime needs explicit conflict-resolution rules.

---

## 16.3.3 Skill Permissions

A skill can require specific capabilities:

```text
Skill
├── filesystem.read
├── web.search
└── shell.execute
```

Permissions should follow **least privilege**.

Example:

```text
Document Analysis
✓ read document
✓ extract text

✕ delete files
✕ execute shell
```

⭐ **Key Point:** Skill permissions define the capability boundary around reusable procedures.

---

## 16.3.4 Skill Testing

Skills should be tested like reusable software components.

Test:

```text
Input validation
Procedure adherence
Tool selection
Output quality
Failure handling
Permission behavior
Edge cases
Regression cases
```

A useful test structure:

```text
Skill
 ↓
Input fixture
 ↓
Expected behavior
 ↓
Execution
 ↓
Evaluation
```

---

## 16.3.5 Skill Portability

🧠 **Simple Understanding:** A portable skill can be reused across agents, runtimes, or environments with minimal modification.

Portability improves when skills separate:

```text
Procedure
from
Environment-specific implementation
```

For example:

```text
Research procedure
      ↓
Search interface
      ↓
Provider A / Provider B
```

rather than hard-coding one provider directly into the procedural instructions.

---

## 16.3.6 Skill Deprecation

🧠 **Simple Understanding:** Deprecation marks a skill as no longer preferred or supported.

Example:

```text
research-v1
     ↓
DEPRECATED
     ↓
research-v2
```

Deprecation should define:

* Replacement.
* Timeline.
* Compatibility.
* Migration path.

---

## 16.3.7 Skill Version Migration

🧠 **Simple Understanding:** Migration updates agents or workflows from one skill version to another.

Example:

```text
Skill v1
  ↓
Compatibility Check
  ↓
Migration
  ↓
Skill v2
  ↓
Regression Evaluation
```

A migration may require updating:

* Instructions.
* Tool interfaces.
* Resources.
* Scripts.
* Dependencies.
* Expected outputs.

---

## 16.3.8 Skill Governance

A production skill system should govern:

```text
Who can publish?
Who can modify?
Who can use?
Which version is approved?
Which dependencies are allowed?
When is a skill deprecated?
```

Useful governance metadata:

```text
Owner
Version
Status
Permissions
Dependencies
Compatibility
Approval state
Change history
```

---

# 16.4 Skills vs Tools vs MCP

The roadmap defines the conceptual distinction as:

```text
Prompt = instructions
Skill  = reusable procedure/capability
Tool   = executable action/interface
MCP    = protocol for connecting model hosts/agents to tools/data/context
A2A    = protocol for agent-to-agent interaction
```

This distinction is fundamental.

---

## 16.4.1 Prompt

🧠 **Simple Understanding:** A prompt provides instructions or immediate guidance to the model.

Example:

```text
"You are a financial analyst.
Explain the findings clearly."
```

A prompt is primarily about **what the model should do or how it should behave in the current interaction**.

---

## 16.4.2 Skill

🧠 **Simple Understanding:** A skill is a reusable procedure for accomplishing a class of tasks.

Example:

```text
Research Skill:
1. Define question
2. Search sources
3. Evaluate evidence
4. Synthesize
5. Cite
```

A skill is about **how to perform a reusable capability**.

---

## 16.4.3 Tool

🧠 **Simple Understanding:** A tool is an executable action or interface that performs a specific operation.

Examples:

```text
search_web()
read_file()
query_database()
run_tests()
```

A tool is about **doing an operation**.

---

## 16.4.4 MCP

🧠 **Simple Understanding:** MCP is a protocol for connecting model hosts/agents to tools, data, and context.

Conceptually:

```text
Agent / Host
     │
     ▼
    MCP
     │
 ┌───┼────┐
 ▼   ▼    ▼
Tools Data Context
```

The important conceptual distinction is:

```text
Skill → procedure
Tool  → executable capability
MCP   → connection / protocol layer
```

---

## 16.4.5 A2A

🧠 **Simple Understanding:** A2A refers to agent-to-agent interaction.

Conceptually:

```text
Agent A
   │
   ▼
 Agent-to-Agent Protocol
   │
   ▼
Agent B
```

The concern is communication between agents rather than direct tool/data access.

---

## 16.4.6 Conceptual Comparison

| Concept | Primary Role              | Example                                |
| ------- | ------------------------- | -------------------------------------- |
| Prompt  | Instructions              | "Summarize this document"              |
| Skill   | Reusable procedure        | Research workflow                      |
| Tool    | Executable action         | `search_web()`                         |
| MCP     | Connection/protocol layer | Connect host to tools/data/context     |
| A2A     | Agent interaction         | Agent A asks Agent B to perform a task |

⭐ **Key Point:** These concepts operate at different abstraction levels and can be combined.

---

## 16.4.7 How They Work Together

A realistic workflow:

```text
User Request
     ↓
Prompt / Task
     ↓
Skill Discovery
     ↓
Select Research Skill
     ↓
Skill Instructions
     ↓
MCP Connection
     ↓
Available Tools / Data
     ↓
Tool Calls
     ↓
Validation
     ↓
Result
```

For multi-agent execution:

```text
Agent A
  ↓
Research Skill
  ↓
A2A
  ↓
Agent B
  ↓
Document Analysis Skill
  ↓
Tools
```

---

# 16.5 Agent Skill Architecture

## 16.5.1 Skill Package

A practical package may look like:

```text
research-skill/
├── skill.yaml
├── instructions.md
├── examples/
│   ├── basic.md
│   └── advanced.md
├── resources/
│   ├── citation_rules.md
│   └── source_quality.md
├── scripts/
│   └── validate_citations.py
└── tests/
    ├── basic.json
    └── edge_cases.json
```

Possible metadata:

```yaml
name: research
version: 1.2.0
description: Conduct evidence-based research.
dependencies:
  - web_search
  - url_fetch
permissions:
  - internet.read
```

---

## 16.5.2 Skill Registry

🧠 **Simple Understanding:** A skill registry keeps track of available skill packages and metadata.

```text
Registry
├── research
├── document-analysis
├── coding
└── data-extraction
```

The registry can support:

* Discovery.
* Version selection.
* Dependency resolution.
* Status.
* Ownership.
* Permissions.

---

## 16.5.3 Discovery Layer

```text
Task
 ↓
Query Skill Registry
 ↓
Candidate Skills
 ↓
Filter by:
├── Relevance
├── Compatibility
├── Permissions
└── Availability
 ↓
Selected Skills
```

---

## 16.5.4 Authorization Layer

Before loading/executing:

```text
Selected Skill
      ↓
Required Dependencies
      ↓
Permission Check
      ↓
Allowed?
 ├── Yes → Load
 └── No  → Reject / Alternate
```

This prevents a seemingly harmless skill from indirectly gaining unauthorized capabilities.

---

## 16.5.5 Loading Layer

Progressive loading:

```text
Metadata
   ↓
Instructions
   ↓
Required resources
   ↓
Required scripts/tools
```

The system should avoid loading unnecessary components.

---

## 16.5.6 Execution Layer

Execution may involve:

```text
Skill procedure
      ↓
Agent reasoning
      ↓
Tools
      ↓
Resources
      ↓
Scripts
```

The runtime still enforces:

* Permissions.
* Resource limits.
* Tool policies.
* Security boundaries.

---

## 16.5.7 Validation Layer

Validate both **procedure completion** and **result quality**.

```text
Execution
 ↓
Expected outcome?
 ↓
Structural validation
 ↓
Semantic validation
 ↓
External verification when needed
```

---

## 16.5.8 Outcome and Telemetry Layer

Record:

```text
Skill ID
Version
Task ID
Agent ID
Dependencies
Start / end time
Outcome
Errors
Artifacts
Evaluation score
```

This makes skill behavior observable and supports version comparison.

---

## 16.5.9 Skills Runtime Flow

```text
                           TASK
                             │
                             ▼
                    ┌────────────────┐
                    │ SKILL REGISTRY │
                    └───────┬────────┘
                            │
                         Discover
                            │
                            ▼
                         Select
                            │
                            ▼
                       Authorize
                            │
                            ▼
                          Load
                            │
                            ▼
                     ┌──────────────┐
                     │ SKILL        │
                     │ PROCEDURE    │
                     └──────┬───────┘
                            │
                       Execute
                            │
                 ┌──────────┼──────────┐
                 ▼          ▼          ▼
               Tools     Resources   Scripts
                 │          │          │
                 └──────────┼──────────┘
                            ▼
                         Validate
                            │
                            ▼
                    Record Outcome
                            │
                            ▼
                          Result
```

---

# 16.6 Agent Skill Pack Project

## 16.6.1 Project Goal

🧠 **Simple Understanding:** Create a reusable **Agent Skill Pack** containing procedural capabilities for:

* Research.
* Document analysis.
* Coding.
* Data extraction.

The goal is to make these capabilities discoverable, reusable, versioned, testable, and permission-aware.

---

## 16.6.2 Research Skill

### Purpose

Conduct structured research and produce evidence-based findings.

```text
Research Skill
├── Define question
├── Plan search
├── Gather sources
├── Evaluate evidence
├── Resolve contradictions
├── Synthesize
└── Cite
```

### Dependencies

```text
web search
URL fetch
source metadata
citation validation
```

---

## 16.6.3 Document Analysis Skill

### Purpose

Analyze uploaded documents and extract meaningful information.

```text
Document Analysis
├── Identify document type
├── Extract content
├── Identify structure
├── Extract key information
├── Analyze
└── Produce structured findings
```

Possible dependencies:

```text
file access
document parser
OCR when required
table extraction
```

---

## 16.6.4 Coding Skill

### Purpose

Perform structured software-engineering tasks.

```text
Coding Skill
├── Inspect repository
├── Understand task
├── Plan changes
├── Modify files
├── Run tests
├── Inspect failures
└── Produce result
```

Possible dependencies:

```text
filesystem
shell
version control
test runner
```

⚠️ **Important:** Because coding skills may require shell and filesystem access, their permission profile should be stronger than purely informational skills.

---

## 16.6.5 Data Extraction Skill

### Purpose

Extract structured information from semi-structured or unstructured data.

```text
Data Extraction
├── Inspect input
├── Identify schema
├── Extract fields
├── Normalize values
├── Validate
└── Export
```

Possible dependencies:

```text
file access
parsers
OCR
schema validator
```

---

## 16.6.6 Skill Pack Structure

```text
agent-skill-pack/
│
├── research/
│   ├── skill.yaml
│   ├── instructions.md
│   ├── resources/
│   ├── scripts/
│   ├── examples/
│   └── tests/
│
├── document-analysis/
│   ├── skill.yaml
│   ├── instructions.md
│   ├── resources/
│   ├── scripts/
│   ├── examples/
│   └── tests/
│
├── coding/
│   ├── skill.yaml
│   ├── instructions.md
│   ├── resources/
│   ├── scripts/
│   ├── examples/
│   └── tests/
│
└── data-extraction/
    ├── skill.yaml
    ├── instructions.md
    ├── resources/
    ├── scripts/
    ├── examples/
    └── tests/
```

---

## 16.6.7 Skill Discovery Flow

```text
User Task
   ↓
Analyze task
   ↓
Query Skill Registry
   ↓
Candidate Skills
   ↓
Relevance Filter
   ↓
Dependency Check
   ↓
Permission Check
   ↓
Selected Skill
```

Example:

```text
"Extract tables from this PDF."

Candidates:
├── Research
├── Document Analysis
├── Data Extraction

Selected:
Document Analysis
+
Data Extraction
```

---

## 16.6.8 Progressive Loading Flow

```text
Task
 ↓
Load skill metadata
 ↓
Need more information?
 ↓
Load instructions
 ↓
Need support resources?
 ↓
Load relevant resources
 ↓
Need executable helper?
 ↓
Load script/tool
```

⭐ **Key Point:** Progressive loading keeps the agent's active context focused.

---

## 16.6.9 Skill Validation

Each skill should have tests covering:

```text
✓ Normal input
✓ Missing input
✓ Invalid input
✓ Expected output
✓ Tool failures
✓ Dependency failures
✓ Permission rejection
✓ Edge cases
✓ Regression behavior
```

Example:

```text
Skill:
Data Extraction

Input:
CSV with malformed row

Expected:
Detect malformed row
↓
Report validation error
↓
Do not silently corrupt output
```

---

## 16.6.10 Skill Pack Versioning

Version each skill independently where practical:

```text
research       v1.2
coding         v2.0
document       v1.4
data-extraction v1.1
```

Migration flow:

```text
Current Version
      ↓
Read compatibility changes
      ↓
Upgrade
      ↓
Run regression tests
      ↓
Evaluate
      ↓
Promote new version
```

---

# 16.7 Key Insights

💡 **Key Insights**

1. **A skill is a reusable procedure, not merely instructions.** It packages a repeatable way of accomplishing a capability.

2. **Skills sit above tools.** A research skill may use search, URL fetching, retrieval, and citation tools as implementation primitives.

3. **Skills can be dynamically discovered and progressively loaded.** This avoids putting an entire skill library into every model context.

4. **Skill dependencies are important.** A skill's safety depends partly on what tools, resources, and runtime capabilities it can reach.

5. **Skill authorization must include dependencies.** Granting access to a skill without examining its underlying tools can unintentionally grant excessive capabilities.

6. **Versioning matters because procedures change behavior.** Updates to instructions, resources, scripts, or dependencies can change outcomes.

7. **Skills should be tested like software.** Reusable procedural components need regression cases, edge cases, permission tests, and output validation.

---

# 16.8 Common Mistakes

⚠️ **Common Mistakes**

| Mistake                                                | Correct Understanding                                                                                                      |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| "A skill is just a prompt."                            | A skill is a reusable procedural capability that can include instructions, resources, scripts, examples, and dependencies. |
| "A skill is the same as a tool."                       | A skill describes a procedure; a tool performs an executable operation.                                                    |
| "All skills can be loaded at startup."                 | Progressive loading reduces context and complexity.                                                                        |
| "If a skill is authorized, all dependencies are safe." | Dependency permissions must be checked independently.                                                                      |
| "Skill instructions can be versionless."               | Procedure changes can alter behavior and require version tracking.                                                         |
| "Examples are just documentation."                     | They can influence how agents apply the skill and should be version-aligned.                                               |
| "A skill can use any available tool."                  | Skill capabilities should be explicitly scoped.                                                                            |
| "Skill tests only need happy paths."                   | Failure, permission, dependency, and edge cases matter.                                                                    |
| "Deprecated skills can remain indefinitely."           | Deprecation requires migration and lifecycle management.                                                                   |
| "MCP is another kind of skill."                        | MCP is a protocol/connection layer, not a reusable procedure.                                                              |
| "A2A replaces skills."                                 | A2A concerns agent-to-agent interaction; skills describe reusable procedures.                                              |
| "More skills always improve agents."                   | Excessive skill choices can complicate discovery and create conflicts.                                                     |

---

# 16.9 Common Confusions

🔍 **Common Confusions**

| Concept A           | Concept B                 | Key Difference                                                              |
| ------------------- | ------------------------- | --------------------------------------------------------------------------- |
| Prompt              | Skill                     | Immediate instruction vs reusable procedure                                 |
| Skill               | Tool                      | Procedure/capability vs executable action                                   |
| Skill               | Workflow                  | Reusable procedural capability vs broader orchestration flow                |
| Tool                | MCP                       | Executable interface vs protocol connecting hosts to capabilities           |
| MCP                 | A2A                       | Host/tool/data connectivity vs agent-to-agent communication                 |
| Skill               | Memory                    | Procedure for doing something vs retained information                       |
| Skill               | Context                   | Reusable capability definition vs current model-visible information         |
| Skill version       | Tool version              | Version of a procedure/package vs executable interface                      |
| Skill discovery     | Tool discovery            | Finding reusable procedures vs finding executable capabilities              |
| Skill authorization | Tool authorization        | Permission to use a procedure vs permission to execute an underlying action |
| Skill composition   | Multi-agent orchestration | Combining procedures vs coordinating independent agents                     |
| Skill resource      | Tool result               | Supporting input/reference material vs execution output                     |

---

# 16.10 Practical Applications

🛠️ **Practical Applications**

| Application          | Useful Skills                                         |
| -------------------- | ----------------------------------------------------- |
| Research Agent       | Research, evidence validation, report writing         |
| Coding Agent         | Repository analysis, implementation, testing          |
| Document Agent       | Document analysis, table extraction, summarization    |
| Data Agent           | Data extraction, normalization, validation            |
| Enterprise Assistant | Organization-specific procedures                      |
| Customer Support     | Troubleshooting, escalation, policy handling          |
| Finance Automation   | Reconciliation, reporting, approval procedures        |
| Browser Agent        | Research, navigation, form-filling procedures         |
| DevOps Agent         | Deployment, diagnostics, incident-response procedures |
| Multi-Agent System   | Specialized skill packs per agent                     |

---

# 16.11 Important Terms

📌 **Important Terms**

| Term                | Simple Meaning                      | Why It Matters                                |
| ------------------- | ----------------------------------- | --------------------------------------------- |
| Agent Skill         | Reusable procedural capability      | Enables reusable agent behavior               |
| Skill Metadata      | Information describing a skill      | Supports discovery                            |
| Skill Instructions  | Procedure the agent follows         | Defines behavior                              |
| Skill Resource      | Supporting material                 | Extends procedural knowledge                  |
| Skill Script        | Executable helper                   | Provides deterministic operations             |
| Skill Example       | Demonstrates usage                  | Clarifies expected behavior                   |
| Dependency          | Capability required by a skill      | Defines operational requirements              |
| Skill Version       | Specific release of a skill         | Enables safe evolution                        |
| Skill Discovery     | Finding relevant skills             | Enables dynamic capability selection          |
| Progressive Loading | Loading only required skill content | Reduces context overhead                      |
| Skill Composition   | Combining multiple skills           | Supports complex tasks                        |
| Skill Conflict      | Incompatible skill behavior         | Requires resolution                           |
| Skill Permission    | Capability allowed to a skill       | Security boundary                             |
| Skill Testing       | Evaluating skill behavior           | Ensures reliability                           |
| Skill Portability   | Reuse across environments           | Reduces lock-in                               |
| Skill Deprecation   | Retiring an older skill             | Controls lifecycle                            |
| Skill Migration     | Moving to a new skill version       | Maintains compatibility                       |
| Skill Registry      | Catalog of available skills         | Enables discovery/versioning                  |
| Prompt              | Instructions to model               | Directs current behavior                      |
| Tool                | Executable capability               | Performs an action                            |
| MCP                 | Connection protocol                 | Connects hosts/agents with tools/data/context |
| A2A                 | Agent-to-agent protocol             | Enables agent interaction                     |

---

# 16.12 Quick Revision

⚡ **Quick Revision**

1. **Agent Skill = reusable procedure/capability.**
2. A skill can contain **metadata, instructions, resources, scripts, examples, dependencies, and version information**.
3. The lifecycle is:

```text
Discover
 ↓
Select
 ↓
Authorize
 ↓
Load
 ↓
Execute
 ↓
Validate
 ↓
Record Outcome
```

4. **Progressive loading** prevents the entire skill library from entering context.
5. Skill authorization must include **dependency permissions**.
6. Skills should support **composition, testing, portability, deprecation, and version migration**.
7. **Prompt = instructions.**
8. **Skill = reusable procedure.**
9. **Tool = executable action/interface.**
10. **MCP = protocol connecting hosts/agents to tools, data, and context.**
11. **A2A = agent-to-agent interaction protocol.**
12. Skills can be organized into reusable packs for **research, document analysis, coding, and data extraction**.
13. A skill system should behave like a **managed software-component ecosystem**, not an unstructured collection of prompts.

---

# 16.13 Interview Preparation

## 16.13.1 Level 1 — Fundamentals

### Q1. What is an Agent Skill?

**Model Answer:**
An Agent Skill is a reusable procedural capability that an agent can discover and load when a task requires it. A skill can include instructions, resources, scripts, examples, dependencies, metadata, and version information.

### Q2. How is a skill different from a prompt?

**Model Answer:**
A prompt primarily provides instructions for the current model interaction. A skill is a reusable package describing a procedure that can be discovered, versioned, authorized, loaded, executed, and validated.

### Q3. How is a skill different from a tool?

**Model Answer:**
A tool performs a specific executable action, while a skill describes a reusable procedure that may use multiple tools to accomplish a larger capability.

### Q4. What is skill discovery?

**Model Answer:**
Skill discovery is the process of finding available skills that may be relevant to the current task. It typically uses metadata such as descriptions, tags, capabilities, and requirements.

### Q5. What is progressive loading?

**Model Answer:**
Progressive loading means loading skill information incrementally, starting with metadata and loading detailed instructions, resources, and executable components only when necessary.

### Q6. Why do skills need versioning?

**Model Answer:**
Changing instructions, scripts, dependencies, or resources can change agent behavior. Versioning allows systems to identify which procedural definition was used and migrate safely between versions.

### Q7. Why do skills need permissions?

**Model Answer:**
Skills may depend on tools or resources with different risk levels. Permissions define which capabilities the skill can access and prevent reusable procedures from gaining unrestricted access.

### Q8. What is skill validation?

**Model Answer:**
Skill validation checks whether the procedure produced the required result, including output structure, correctness, evidence, and external side effects where applicable.

---

## 16.13.2 Level 2 — Conceptual Understanding

### Q1. Why is a skill more than a prompt?

**Model Answer:**
A skill represents a reusable capability with lifecycle and engineering metadata. It can package procedure, resources, scripts, dependencies, examples, permissions, and versions, whereas a prompt is primarily an instruction mechanism.

### Q2. Why should skills be progressively loaded?

**Model Answer:**
Loading every skill in full increases context size and decision complexity. Progressive loading exposes only the relevant procedure and supporting information required for the current task.

### Q3. Why should skill authorization include dependencies?

**Model Answer:**
A skill may appear informational but depend on powerful tools such as shell execution or filesystem access. The security impact therefore comes from both the skill and the capabilities it can invoke.

### Q4. Why are examples part of a skill package?

**Model Answer:**
Examples make complex procedures and output conventions concrete. They can influence how agents apply the skill, so they should be maintained and versioned with the skill.

### Q5. Why is skill composition useful?

**Model Answer:**
Complex capabilities can be assembled from smaller reusable procedures. This improves modularity, reuse, and maintainability, although too many dependencies can make orchestration harder.

### Q6. Why do skills need testing like software?

**Model Answer:**
Skills are reusable executable procedures whose behavior can regress when instructions, dependencies, resources, or scripts change. Regression and edge-case testing helps maintain reliability.

### Q7. Why does portability matter for skills?

**Model Answer:**
Portable skills can be reused across agents and environments without rewriting their core procedure. Separating procedure from provider-specific implementations reduces coupling and migration cost.

### Q8. Why can two skills conflict?

**Model Answer:**
They may impose contradictory instructions, output formats, policies, assumptions, or tool-use strategies. The runtime needs explicit priorities and conflict-resolution rules.

---

## 16.13.3 Level 3 — Practical / Engineering

### Q1. How would you design a production skill package?

**Model Answer:**

```text
Skill
├── Metadata
├── Version
├── Instructions
├── Resources
├── Examples
├── Scripts
├── Dependencies
├── Permissions
└── Tests
```

The package should also have an owner, lifecycle status, compatibility information, and observable execution outcomes.

### Q2. How would you implement skill discovery?

**Model Answer:**

```text
Task
 ↓
Skill Registry
 ↓
Candidate Skills
 ↓
Relevance Filter
 ↓
Compatibility Check
 ↓
Permission Check
 ↓
Select
```

Discovery should avoid exposing every available skill to every task.

### Q3. How would you safely execute a coding skill?

**Model Answer:**
First authorize the skill and its dependencies, then execute it in a controlled runtime with explicit filesystem, shell, network, and resource permissions. Run validation and tests, capture artifacts, and record the skill/version used.

### Q4. How would you handle a skill dependency failure?

**Model Answer:**
Determine whether the dependency failure is retryable, whether an alternate dependency exists, and whether the skill can proceed safely without it. Otherwise pause, replan, or fail the skill execution explicitly.

### Q5. How would you test a research skill?

**Model Answer:**
Create fixtures covering ordinary research, missing evidence, contradictory sources, tool failures, citation failures, and incomplete outputs. Evaluate both final quality and the procedural behavior.

### Q6. How would you migrate from Skill v1 to Skill v2?

**Model Answer:**

```text
v1
 ↓
Compatibility Analysis
 ↓
Identify behavior changes
 ↓
Update dependencies/resources
 ↓
Run Regression Suite
 ↓
Compare Outcomes
 ↓
Promote v2
 ↓
Deprecate v1
```

### Q7. How would you prevent a skill from gaining excessive capabilities?

**Model Answer:**
Define explicit permission metadata, check permissions for both the skill and its dependencies, use least privilege, and enforce runtime policy outside the model.

### Q8. How would you observe skill usage?

**Model Answer:**
Record skill ID, version, task, agent, dependencies, execution time, status, errors, artifacts, and evaluation results. This makes it possible to compare versions and debug failures.

---

## 16.13.4 Level 4 — Advanced / Deep Understanding

### Q1. Why is a skill best viewed as a software component rather than a prompt?

**Model Answer:**
A skill has a defined lifecycle: discovery, selection, authorization, loading, execution, validation, versioning, and deprecation. It can also have dependencies, tests, permissions, resources, and executable components. These are software-component concerns rather than prompt-only concerns.

### Q2. Why can skill composition become difficult at scale?

**Model Answer:**
As the number of skills grows, dependencies, conflicting instructions, overlapping capabilities, version compatibility, and authorization become more complex. Discovery and composition therefore require explicit metadata and orchestration rules.

### Q3. Why can a skill create security risk even if its instructions are harmless?

**Model Answer:**
The skill may depend on powerful capabilities. For example, a benign coding procedure that can invoke unrestricted shell access effectively inherits the shell's capability surface. Security must therefore be evaluated at the dependency boundary.

### Q4. Why is progressive loading a context-engineering technique?

**Model Answer:**
It controls what information reaches the model at each stage. Instead of loading an entire skill ecosystem, the system selects a relevant skill and loads only the information necessary to execute it.

### Q5. Why is skill versioning more important for agentic systems than for static documentation?

**Model Answer:**
A skill can directly change agent behavior. Updating its instructions or dependencies can alter tool usage, decision paths, outputs, and side effects. Reproducibility therefore requires tracking the exact skill version used.

### Q6. Why should skill validation include external outcomes?

**Model Answer:**
A skill may produce a plausible textual result while failing to accomplish its real objective. Where the skill changes external state, validation should consider whether the intended outcome actually occurred.

### Q7. Why should skills avoid hard-coding providers where practical?

**Model Answer:**
Provider-specific coupling reduces portability. Separating the procedure from the underlying implementation allows the same skill to use alternative providers while preserving the procedural abstraction.

### Q8. Why is deprecation part of skill engineering?

**Model Answer:**
Procedures and dependencies evolve. Without deprecation and migration policies, obsolete skills remain active, create conflicting versions, and make behavior difficult to govern.

---

## 16.13.5 Level 5 — Scenario-Based Questions

### Scenario 1 — Harmless Skill, Dangerous Dependency

A "code formatting" skill requires unrestricted shell access.

**Question:** What would you do and why?

**Model Answer:**

```text
Code Formatting Skill
        ↓
Dependencies
        ↓
Shell Access
        ↓
Risk Assessment
```

I would determine whether unrestricted shell is actually necessary. If not, replace it with a narrowly scoped formatting tool. If shell is required, restrict the runtime capability to the minimum needed.

---

### Scenario 2 — Skill Conflict

Two selected skills specify different output formats:

```text
Skill A → JSON
Skill B → Markdown
```

**Question:** How should the system respond?

**Model Answer:**
The system should not let the model arbitrarily resolve the conflict. Establish explicit precedence based on task requirements, skill roles, or orchestration policy. If the conflict remains unresolved, the workflow should surface it rather than silently producing an invalid result.

---

### Scenario 3 — Skill Version Regression

Skill v2 performs worse than v1 on existing evaluation cases.

**Question:** What would you do?

**Model Answer:**

```text
v2
 ↓
Regression Evaluation
 ↓
Performance drop
 ↓
Inspect behavior changes
 ↓
Fix / rollback / retain v1
 ↓
Re-evaluate
```

I would preserve version traceability and avoid automatically promoting the new version simply because it is newer.

---

### Scenario 4 — Progressive Loading

An agent has 500 available skills.

**Question:** Should the entire skill library be loaded?

**Model Answer:**
No. Load lightweight metadata first, discover relevant candidates, select the skill, and progressively load only its instructions and required resources. This reduces context overhead and selection complexity.

---

### Scenario 5 — Skill Requires Unauthorized Tool

An agent selects a document-analysis skill that requires filesystem access the current task is not permitted to use.

**Question:** What should happen?

**Model Answer:**

```text
Select Skill
 ↓
Inspect Dependencies
 ↓
Filesystem permission denied
 ↓
Reject skill
 ↓
Choose alternate capability
or
Escalate
```

The skill should not bypass the permission boundary merely because it is otherwise relevant.

---

### Scenario 6 — Skill Produces Plausible but Incorrect Result

A data-extraction skill generates a valid-looking JSON file but silently omits 10% of the records.

**Question:** What failed?

**Model Answer:**
Output-format validation passed, but semantic/completeness validation failed. The skill needs stronger correctness criteria, such as record-count checks, source-to-output reconciliation, schema validation, or domain-specific completeness checks.

---

### Scenario 7 — Skill Uses Another Skill

A research skill depends on a document-analysis skill.

**Question:** Is this a problem?

**Model Answer:**
Not inherently. Skill composition can improve modularity, but the dependency graph should be explicit and version-compatible:

```text
Research v2
   ↓
Document Analysis v1.4
   ↓
Required Tools
```

Authorization and failure handling must cover the entire dependency chain.

---

## 16.13.6 Knowledge Check

🧠 **Knowledge Check**

If you can explain these naturally in your own words, you understand Layer 14:

* What an Agent Skill is.
* Why a skill is more than a prompt.
* Why a skill is different from a tool.
* What skill metadata contains.
* What skill instructions represent.
* Why resources are separated from procedural instructions.
* Why scripts can improve deterministic execution.
* Why examples matter.
* What dependencies are.
* Why skills require versioning.
* How skill discovery works.
* Why progressive loading matters.
* The full skill lifecycle.
* Why authorization includes dependencies.
* How skill composition works.
* How skill conflicts arise.
* How skill permissions work.
* How to test skills.
* Why skill portability matters.
* What deprecation means.
* How skill migration works.
* Why governance matters.
* The difference between prompt, skill, tool, MCP, and A2A.
* How skills and tools work together.
* How a skill registry works.
* How to build a reusable Agent Skill Pack.

---

## 16.13.7 Follow-up Questions

### Basic Question

**What is an Agent Skill?**

→ How is it different from a prompt?
→ How is it different from a tool?
→ What does it contain?
→ How is it discovered?
→ How is it authorized?

### Basic Question

**How does the skill lifecycle work?**

→ Discover?
→ Select?
→ Authorize?
→ Load?
→ Execute?
→ Validate?
→ Record?

### Basic Question

**How do you engineer skills?**

→ Composition?
→ Conflicts?
→ Permissions?
→ Testing?
→ Portability?
→ Deprecation?
→ Version migration?

### Basic Question

**How do skills relate to tools and protocols?**

→ Prompt?
→ Skill?
→ Tool?
→ MCP?
→ A2A?

### Basic Question

**How would you build an Agent Skill Pack?**

→ Package format?
→ Registry?
→ Discovery?
→ Loading?
→ Dependencies?
→ Tests?
→ Versioning?

---

## 16.13.8 Common Confusion Questions

### Q1. Is a skill just a prompt file?

**Model Answer:**
No. A skill can contain instructions, resources, scripts, examples, dependencies, permissions, metadata, tests, and versioning. The procedural package is broader than a prompt.

### Q2. Is a skill just a collection of tools?

**Model Answer:**
No. Tools are executable primitives. A skill defines how those primitives should be combined into a reusable procedure.

### Q3. Is MCP a skill format?

**Model Answer:**
No. MCP is a protocol for connecting model hosts/agents to tools, data, and context. A skill is a reusable procedural capability.

### Q4. Is A2A a skill mechanism?

**Model Answer:**
No. A2A concerns communication and interaction between agents. Skills concern reusable procedures within an agent system.

### Q5. Is skill discovery the same as tool discovery?

**Model Answer:**
They are similar patterns at different abstraction levels. Skill discovery finds reusable procedures; tool discovery finds executable capabilities that can implement actions.

---

## 16.13.9 Deep / Trick Questions

### ⚠️ Deeper Question

**If a skill is reusable, why not put every skill into the system prompt permanently?**

**Correct Understanding:**
Permanent loading wastes context and increases the agent's decision space. Dynamic discovery and progressive loading allow the system to expose only relevant capabilities at the appropriate time.

---

### ⚠️ Deeper Question

**If a skill has no executable scripts, does it still need authorization?**

**Correct Understanding:**
Yes. Its dependencies, resources, data access, and underlying tools can still have permission implications. Authorization is about capability access, not merely whether the skill contains code.

---

### ⚠️ Deeper Question

**Why can skill versioning affect reproducibility?**

**Correct Understanding:**
Two runs using different skill versions can follow different procedures, use different resources, or call different tools. Recording the exact version allows engineers to reproduce and explain behavior.

---

### ⚠️ Deeper Question

**Why can a perfectly written skill still fail in production?**

**Correct Understanding:**
Its tools or dependencies may fail, permissions may change, external state may change, resources may become stale, or the procedure may be insufficient for edge cases. Skill quality is therefore a systems problem, not only an instruction-writing problem.

---

### ⚠️ Deeper Question

**Why should skill validation test more than output format?**

**Correct Understanding:**
A structurally valid output can still be incomplete, incorrect, or unsupported. Validation should measure whether the procedure achieved the intended semantic and operational outcome.

---

### ⚠️ Deeper Question

**Why should skill composition be explicit rather than allowing arbitrary skill chaining?**

**Correct Understanding:**
Arbitrary chaining can create hidden dependencies, conflicting instructions, excessive context, permission escalation, and difficult-to-debug behavior. Explicit composition makes dependencies and responsibilities visible.

---

# 16.14 Top Questions You MUST Know

⭐ **Top Questions You MUST Know**

1. What is an Agent Skill?
2. How is a skill different from a prompt?
3. How is a skill different from a tool?
4. What should a skill package contain?
5. How does skill discovery work?
6. What is progressive skill loading?
7. Why must skill authorization include dependencies?
8. How should skills be composed?
9. How do skill conflicts occur and how should they be resolved?
10. How should skills be tested and evaluated?
11. Why is skill versioning important?
12. How should skill deprecation and migration work?
13. What is the difference between skills, tools, MCP, and A2A?
14. How would you design a skill registry and runtime?
15. How would you build a production-grade Agent Skill Pack?

---

# 16.15 Interview Readiness Checklist

🎯 **Interview Readiness Checklist**

| Skill                            | Can I explain it? |
| -------------------------------- | :---------------: |
| Agent Skill definition           |         ☐         |
| Skill vs prompt                  |         ☐         |
| Skill vs tool                    |         ☐         |
| Skill metadata                   |         ☐         |
| Skill instructions               |         ☐         |
| Skill resources                  |         ☐         |
| Skill scripts                    |         ☐         |
| Skill examples                   |         ☐         |
| Skill dependencies               |         ☐         |
| Skill versioning                 |         ☐         |
| Skill discovery                  |         ☐         |
| Progressive loading              |         ☐         |
| Skill lifecycle                  |         ☐         |
| Skill authorization              |         ☐         |
| Dependency authorization         |         ☐         |
| Skill composition                |         ☐         |
| Skill conflicts                  |         ☐         |
| Skill permissions                |         ☐         |
| Skill testing                    |         ☐         |
| Skill portability                |         ☐         |
| Skill deprecation                |         ☐         |
| Version migration                |         ☐         |
| Skill governance                 |         ☐         |
| Skill registry                   |         ☐         |
| Discovery layer                  |         ☐         |
| Loading layer                    |         ☐         |
| Execution layer                  |         ☐         |
| Validation layer                 |         ☐         |
| Outcome telemetry                |         ☐         |
| Prompt vs Skill vs Tool          |         ☐         |
| MCP concepts                     |         ☐         |
| A2A concepts                     |         ☐         |
| Skill-tool interaction           |         ☐         |
| Progressive loading architecture |         ☐         |
| Skill pack structure             |         ☐         |
| Research skill                   |         ☐         |
| Document analysis skill          |         ☐         |
| Coding skill                     |         ☐         |
| Data extraction skill            |         ☐         |
| Production skill security        |         ☐         |
| Skill regression testing         |         ☐         |
| Skill migration strategy         |         ☐         |

---

# 16.16 What You Should Be Able to Explain

🧠 **What You Should Be Able to Explain**

By the end of Layer 14, you should be able to explain:

* What Agent Skills are.
* Why reusable procedural capabilities are useful for agent systems.
* How a skill differs from a prompt.
* How a skill differs from a tool.
* Why a skill is best treated as a managed software component.
* What skill metadata contains.
* How skill instructions encode reusable procedures.
* Why resources can be separated from instructions.
* Why deterministic scripts can be included in a skill package.
* Why examples should be maintained with skill versions.
* What dependencies are.
* Why dependency metadata matters.
* Why skill versioning is necessary.
* How skill discovery works.
* How a skill registry supports discovery.
* Why progressive loading reduces context overhead.
* The complete skill lifecycle:

  * Discover
  * Select
  * Authorize
  * Load
  * Execute
  * Validate
  * Record outcome
* Why authorization must include underlying dependencies.
* How skill composition works.
* How skill conflicts arise.
* How skill permissions should be designed.
* How to test a skill like a software component.
* Why edge cases and permission failures belong in skill tests.
* Why skill portability matters.
* How to separate procedural logic from provider-specific implementations.
* What skill deprecation means.
* How version migration should work.
* Why governance matters in a skill ecosystem.
* The conceptual difference between:

  * Prompt
  * Skill
  * Tool
  * MCP
  * A2A
* How a skill can use multiple tools.
* How MCP can provide access to tools/data/context used by a skill.
* How A2A enables interaction between agents that may each have their own skills.
* How to design a complete Agent Skill runtime.
* How to build a skill registry.
* How to perform skill discovery and selection.
* How to progressively load instructions and resources.
* How to authorize skill dependencies.
* How to validate skill outcomes.
* How to record skill execution telemetry.
* How to build reusable skills for research, document analysis, coding, and data extraction.
* How to version and test an Agent Skill Pack.
* Why **skills turn agent behavior into reusable, discoverable, versioned, testable procedural components rather than one-off prompts**.

## ⚡ Final Mental Model

```text
                         USER TASK
                             │
                             ▼
                    ┌──────────────────┐
                    │  SKILL REGISTRY  │
                    └────────┬─────────┘
                             │
                         DISCOVER
                             │
                             ▼
                       CANDIDATE SKILLS
                             │
                         SELECT
                             │
                             ▼
                       AUTHORIZE
                             │
                    ┌────────┴─────────┐
                    │                  │
                Skill OK?          Dependency
                    │               Check
                    │                  │
                    └────────┬─────────┘
                             ▼
                            LOAD
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
            Instructions  Resources   Scripts
                  │          │          │
                  └──────────┼──────────┘
                             ▼
                           AGENT
                             │
                         EXECUTE
                             │
                  ┌──────────┼──────────┐
                  ▼          ▼          ▼
                Tools      Data       Other
                                      Skills
                  │          │          │
                  └──────────┼──────────┘
                             ▼
                         VALIDATE
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
                Success               Failure
                  │                     │
                  ▼               Retry / Replan /
           Record Outcome            Escalate
                  │
                  ▼
             Artifact / Result
                  │
                  ▼
             Telemetry / Trace
```

> **Core principle:** **Agent Skills are the reusable procedural layer of an agent ecosystem: prompts provide instructions, skills package repeatable procedures, tools perform executable actions, MCP connects agents/hosts to tools and data, and A2A enables agents to interact. A production skill system therefore needs discovery, authorization, progressive loading, execution, validation, versioning, testing, governance, and lifecycle management.**
