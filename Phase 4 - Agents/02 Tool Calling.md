# Module 2 — Tool Calling

## Learning Objective

Teach agents how to interact with external systems, select the correct tools, pass structured inputs, handle responses, recover from failures, and safely execute actions.

---

# Section 1 — Function Calling

This section explains how an LLM invokes tools.

---

### Image 26

# What Is Function Calling?

**Purpose**

Introduce function calling as the foundation of tool use.

**Concepts**

* LLMs calling tools
* Structured inputs
* Structured outputs
* External actions

```text
User
 ↓
LLM
 ↓
Function Call
 ↓
Tool
 ↓
Result
 ↓
Answer
```
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/ed4ca272-6f1f-43a5-bc5a-23ab280af97d" />

---

### Image 27

# Why Function Calling Exists

**Purpose**

Show limitations of pure LLMs.

**Compare**

### Without Function Calling

```text
LLM
 ↓
Guess
```

### With Function Calling

```text
LLM
 ↓
Tool
 ↓
Real Data
```

Examples:

* Weather
* Stock prices
* Databases
* APIs
<img width="1181" height="1331" alt="image" src="https://github.com/user-attachments/assets/3de92bec-3d42-48fc-8167-c54f7e3d0e03" />

---

### Image 28

# Anatomy of a Function Call

**Purpose**

Break down every component.

**Visual**

```text
Function Name
Parameters
Schema
Execution
Response
```

Example:

```json
{
  "name": "get_weather",
  "city": "Mumbai"
}
```
<img width="1199" height="1312" alt="image" src="https://github.com/user-attachments/assets/4bb204e3-fe19-44ab-ae9c-9f7a6eb319f5" />

---

### Image 29

# Function Calling vs ReAct

**Purpose**

Show relationship.

**Compare**

### Function Calling

```text
One Tool
One Result
```

### ReAct

```text
Reason
 ↓
Tool
 ↓
Observe
 ↓
Reason
```
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/a222134b-2f98-4141-968d-d12c30b621a8" />

---

### Image 30

# Real-World Function Calling Examples

**Purpose**

Show practical use cases.

Examples:

* Email agent
* Calendar agent
* CRM agent
* Database agent
* Search agent
<img width="1402" height="1122" alt="image" src="https://github.com/user-attachments/assets/5f098d05-3608-4d69-9eee-60c5468ab7f5" />

---

# Section 2 — Tool Schemas

Most developers underestimate this topic.

---

### Image 31

# What Is a Tool Schema?

**Purpose**

Introduce schemas.

**Visual**

```text
Tool
 ↓
Description
 ↓
Inputs
 ↓
Validation
```
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/2593b4a6-56d2-4343-8b7c-800235439149" />

---

### Image 32

# Anatomy of a Tool Schema

**Purpose**

Explain schema structure.

Components:

```text
Name
Description
Parameters
Types
Required Fields
```

Example JSON schema visualization.
<img width="1122" height="1402" alt="image" src="https://github.com/user-attachments/assets/718a12fd-f4d7-46bc-bb6c-cc53fcfe0e92" />

---

### Image 33

# How Agents Read Tool Schemas

**Purpose**

Show internal decision making.

```text
User Request
 ↓
Read Tool Descriptions
 ↓
Choose Best Tool
 ↓
Execute
```
<img width="1199" height="1312" alt="image" src="https://github.com/user-attachments/assets/90fbd9a8-c7d6-4cfd-8df9-78bdbaf02195" />

---

### Image 34

# Good vs Bad Tool Descriptions

**Purpose**

One of the most important images.

Bad:

```text
Get data
```

Good:

```text
Retrieve customer records
using customer ID.
```

Show:

* ambiguity
* precision
* context
<img width="1341" height="1173" alt="image" src="https://github.com/user-attachments/assets/1c6daa7b-4ad6-467c-aba1-8787887dfaf7" />

---

### Image 35

# Tool Selection Logic

**Purpose**

How agents choose tools.

```text
Request
 ↓
Intent Detection
 ↓
Tool Ranking
 ↓
Selection
```

Examples:

* Search Tool
* Database Tool
* Email Tool
![Uploading image.png…]()

---

### Image 36

# Schema Design Failure Modes

**Purpose**

Common mistakes.

Show:

* Missing parameters
* Ambiguous descriptions
* Wrong types
* Poor validation

---

# Section 3 — Tool Lifecycle

This is how tools actually execute.

---

### Image 37

# Tool Lifecycle Overview

**Purpose**

Introduce full lifecycle.

```text
Need
 ↓
Select
 ↓
Execute
 ↓
Observe
 ↓
Continue
```

---

### Image 38

# Tool Execution Lifecycle

**Purpose**

Detailed breakdown.

```text
User Request
 ↓
Reasoning
 ↓
Tool Selection
 ↓
Parameter Generation
 ↓
Execution
 ↓
Result
 ↓
Next Decision
```

---

### Image 39

# Tool Input Validation

**Purpose**

Prevent bad requests.

Examples:

```text
Email missing
Amount negative
Invalid date
```

Show validation layer.

---

### Image 40

# Tool Responses & Observations

**Purpose**

Understand what comes back.

```text
Tool
 ↓
Response
 ↓
Observation
 ↓
Reasoning
```

Types:

* Success
* Partial Success
* Failure

---

### Image 41

# Error Handling & Recovery

**Purpose**

Production-grade tool usage.

Examples:

```text
API Timeout
 ↓
Retry

Rate Limit
 ↓
Wait

Tool Failure
 ↓
Fallback
```

---

### Image 42

# Multi-Tool Execution

**Purpose**

Show chaining.

Example:

```text
Search
 ↓
Database
 ↓
Email
 ↓
CRM
```

---

### Image 43

# Parallel Tool Execution

**Purpose**

Performance optimization.

```text
Weather
      ↘
Stocks  → Merge Results
      ↗
News
```

---

# Section 4 — Tool Design

Most important section for production agents.

---

### Image 44

# Principles of Good Tool Design

**Purpose**

Foundational best practices.

Principles:

* Simple
* Explicit
* Reliable
* Observable

---

### Image 45

# Designing Agent-Friendly Tools

**Purpose**

How agents think about tools.

Show:

```text
Human Tool
vs
Agent Tool
```

---

### Image 46

# Input Design Best Practices

**Purpose**

Create predictable inputs.

Topics:

* Required fields
* Defaults
* Validation
* Constraints

---

### Image 47

# Output Design Best Practices

**Purpose**

Make responses easy to reason about.

Bad:

```json
{
 "status":"ok"
}
```

Good:

```json
{
 "status":"success",
 "emailSent": true,
 "messageId": "123"
}
```

---

### Image 48

# Tool Observability

**Purpose**

Monitor everything.

Show:

```text
Request
 ↓
Tool
 ↓
Logs
 ↓
Metrics
 ↓
Tracing
```

---

### Image 49

# Tool Security & Permissions

**Purpose**

Safe execution.

Concepts:

* RBAC
* Permissions
* Allow Lists
* Credential Vaults

---

### Image 50

# Production Tool Architecture

**Purpose**

Complete production design.

```text
Agent
 ↓
Tool Registry
 ↓
Validation
 ↓
Execution Layer
 ↓
Monitoring
 ↓
Audit Logs
```

---

# Final Arrangement

```text
MODULE 2 — TOOL CALLING

Section 1 — Function Calling
26. What Is Function Calling?
27. Why Function Calling Exists
28. Anatomy of a Function Call
29. Function Calling vs ReAct
30. Real-World Function Calling Examples

Section 2 — Tool Schemas
31. What Is a Tool Schema?
32. Anatomy of a Tool Schema
33. How Agents Read Tool Schemas
34. Good vs Bad Tool Descriptions
35. Tool Selection Logic
36. Schema Design Failure Modes

Section 3 — Tool Lifecycle
37. Tool Lifecycle Overview
38. Tool Execution Lifecycle
39. Tool Input Validation
40. Tool Responses & Observations
41. Error Handling & Recovery
42. Multi-Tool Execution
43. Parallel Tool Execution

Section 4 — Tool Design
44. Principles of Good Tool Design
45. Designing Agent-Friendly Tools
46. Input Design Best Practices
47. Output Design Best Practices
48. Tool Observability
49. Tool Security & Permissions
50. Production Tool Architecture
```

**Total Module 2 Images: 25**

This mirrors the depth and progression of Module 1, taking the learner from **"What is a tool?" → "How does an agent call it?" → "How does execution work?" → "How do we design production-grade tools?"**.
