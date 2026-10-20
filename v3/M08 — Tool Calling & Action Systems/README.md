# M08 — Tool Calling & Action Systems: Reading Index

**Path:** Core  
**Depth:** L2 foundation; L3 through demonstrated project work  
**Purpose:** Verified action workflow  
**Prerequisites:** M02, M05, M25  
**Completion evidence:** Block unauthorized calls, survive an ambiguous timeout and verify no duplicate write.  
**Prepared:** 20 October 2026

## What is included

This collection contains **eight separate study guides**, one for each section at the level of M08.1.1, M08.1.2 and so on. The smaller numbered subtopics are explained inside the corresponding guide.

The eight guides contain approximately **60,000 words**, including worked examples and code, and **200 answered study questions**. This index is an additional navigation file. The ZIP contains all eight guides and this index as editable Markdown files.

The structure follows the previous M06 notes and the supplied Prompt Engineering sample:

- Linked tables of contents.
- Plain-language definitions and intuition.
- Detailed mechanisms, design decisions and practical examples.
- Comparison tables and worked failure scenarios.
- Scoped Mermaid concept maps.
- Glossaries and final revision sheets.
- Ten answered recall questions in each guide.
- Nine explained multiple-choice questions in each guide, across easy, medium and scenario-based difficulty levels.
- Six interview questions with model answers in each guide.
- L3 exercises with explicit acceptance evidence.
- Primary-source links within each guide and in its reference list.

## Complete file index

| Section | Study guide | Main coverage |
|---|---|---|
| M08.1.1 | [Tool Basics](M08.1.1_Tool_Basics.md) | Schemas, required and optional parameters, typed inputs and outputs, descriptions, examples, constraints and result normalization |
| M08.1.2 | [Tool Routing](M08.1.2_Tool_Routing.md) | Namespaces, grouping, catalogs, discovery, dynamic loading, relevance filtering, tool selection and binding |
| M08.2.1 | [Tool Permissions](M08.2.1_Tool_Permissions.md) | Read-only, low-risk, high-risk and irreversible/financial/sensitive actions; current authorization and risk-based approval |
| M08.3.1 | [Execution Models](M08.3.1_Execution_Models.md) | Single, sequential, parallel and dependent calls; fan-out/fan-in; partial failure and compensation |
| M08.3.2 | [Tool Reliability](M08.3.2_Tool_Reliability.md) | Validation, retries, timeouts, circuit breakers, fallbacks, idempotency, result verification and side-effect classification |
| M08.4.1 | [Action Verification and Safe Side Effects](M08.4.1_Action_Verification_and_Safe_Side_Effects.md) | Preconditions, dry runs, keys, postconditions, receipts, environment verification, duplicate prevention, undo and recovery |
| M08.5.1 | [Approval Validity](M08.5.1_Approval_Validity.md) | Binding approval to parameters and versions, expiry, current permissions, changed external state and approval reservation |
| M08.6.1 | [Worked Example: Create a Task Safely](M08.6.1_Worked_Example.md) | Durable action identity, task creation, timeout recovery, receipt lookup, one-task verification and unresolved-state escalation |

## Suggested study order

### 1. Understand the interface

Read **M08.1.1** and **M08.1.2**.

Learn to describe a tool's accepted arguments and results, organize a catalog, discover relevant tools and bind a proposed call to an actual implementation. Explain why selecting a tool does not establish permission to execute it.

### 2. Define the action boundary

Read **M08.2.1**.

Distinguish authentication, authorization, user intent and approval. Classify the effect in context, including data disclosure from reads, scope of writes and secondary effects such as notifications.

### 3. Execute and recover deliberately

Read **M08.3.1** and **M08.3.2**.

Work through dependencies and partial failures before introducing parallel execution. Study the difference between a failed attempt and a failed business action. Trace why a timeout can leave an unknown outcome and why a new idempotency key can create a duplicate.

### 4. Verify and preserve approval meaning

Read **M08.4.1** and **M08.5.1**.

Define preconditions and postconditions, validate receipts and inspect authoritative state. Learn how an approval becomes invalid when the significant parameters, relevant versions or permissions change. Keep approval validity separate from knowledge of an earlier action's outcome.

### 5. Complete the lifecycle project

Read **M08.6.1** and run its complete Python example.

The local task service loses a response after commit, reloads the original client intent, safely replays it under its explicit contract, looks up the receipt and verifies one resulting task. It also checks unauthorized calls, concurrent requests, changed parameters, stale state, forged receipts, status unavailability and revoked permissions.

Then complete the L3 extension with your own clearly defined service boundary and retain the required evidence.

## How the examples fit together

The examples use fictional task-management and related business workflows. They illustrate software behavior and are not instructions to operate a real organization, financial account or external service.

Each guide has a standalone Python teaching example. Run each example separately; they are not modules intended to be concatenated into one application. Their scopes differ deliberately:

| Guide | Example focus |
|---|---|
| Tool Basics | Input validation and normalized tool results |
| Tool Routing | Catalog filtering, selection and trusted binding |
| Tool Permissions | Policy decisions and denial before an unauthorized action |
| Execution Models | Dependencies, required versus optional reads and fan-in |
| Tool Reliability | Unknown outcomes, replay contracts and bounded recovery decisions |
| Action Verification | Local transaction effects, receipts, postconditions and an outbox record |
| Approval Validity | A pure eligibility gate for action binding, expiry, permissions and versions |
| Worked Example | Separate persistent client/provider SQLite state and the complete task lifecycle |

The notes explain each example's assumptions. A local lock is not a distributed coordination mechanism. A pure approval check is not an atomic executor. A database outbox row is not proof of external delivery. A SQLite lifecycle fixture is not a deployed authenticated API.

The full lifecycle example uses only the standard library and temporary databases. It demonstrates recovery after reconstructing the client and service objects. Its notes distinguish this from a hardware crash test or a production durability guarantee.

## Core distinctions to learn

| Distinction | What you should be able to explain |
|---|---|
| Tool discovery versus authorization | Availability and relevance do not grant access |
| Model arguments versus trusted context | The actor, tenant and approval state come from authenticated application control |
| Read-only versus harmless | A read can disclose sensitive data or consume costly resources |
| Logical action versus attempt | One business action can require several calls without authorizing several effects |
| Timeout versus confirmed failure | Lack of a response may leave the effect unknown |
| Idempotency versus validation | Deduplication does not establish that the action is permitted or correct |
| Receipt versus postcondition | A recorded outcome and the resulting environment provide different evidence |
| Compensation versus rollback | A new recovery action can have its own effects and failures |
| Approval versus current permission | A past confirmation cannot override a present access denial |
| Human review versus outcome proof | Approval does not establish that an unresolved old request cannot still commit |

These distinctions are developed in the guides through original examples and source-linked explanations.

## Module completion portfolio

| Evidence | Minimum useful artifact |
|---|---|
| Tool contract | Input/output schema, result states, constraints and implementation binding |
| Permission enforcement | Tenant/object access matrix and rejected-action tests |
| Authorized intent | Stored normalized proposal and applicable approval evidence |
| Execution plan | Dependencies, parallel groups and partial-failure policy |
| Durable action identity | Saved action ID, key, fingerprint and recovery state |
| Ambiguous timeout | A controlled loss after provider commit, with an unknown client outcome |
| Safe recovery | The original key and parameters reused under an explicit contract, or receipt-based recovery |
| No duplicate write | Authoritative state showing exactly one effect for the logical action |
| Approval validity | Changed-parameter, stale-version, expiry and revocation tests |
| Unresolved outcome | Bounded recovery and an inspectable escalation record |
| Operational reasoning | Retry limits, receipt retention, verification rules and known implementation boundaries |

L2 means you can explain the mechanisms, choose appropriate controls and work through the examples. L3 requires reproducible project results. Memorizing terminology or showing one successful request is insufficient evidence of failure recovery.

## How to revise

For each guide:

1. Read the definitions and explain the central distinction in your own words.
2. Trace a worked example, including what each component knows at every step.
3. Attempt the recall questions before reading their answers.
4. Complete the graded MCQs and explain why the alternatives fail.
5. Practice the interview answers using the concrete task example.
6. Run the teaching program and identify its stated limits.
7. Perform the L3 exercise and keep the required evidence.
8. Return to the revision sheet before moving to the next guide.

When an action fails, identify whether the problem is selection, permission, validation, execution, observation or verification. That diagnosis determines whether a retry is relevant and safe.

## Reading the Markdown files

Open a guide directly, or extract the ZIP and keep the files together so their relative links work. Each file is editable Markdown. The Mermaid diagrams have accompanying explanations for viewers that do not render them.

Primary references are linked inside each guide. Protocol, SDK and provider behavior should be checked against the versions used in your actual project. Tool metadata and an apparently successful response do not establish an end-to-end action guarantee on their own.
