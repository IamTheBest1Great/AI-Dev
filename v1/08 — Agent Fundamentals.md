# 📚 Table of Contents

* [9. Layer 7 — Tool Calling & Action Systems](#9-layer-7--tool-calling--action-systems)

  * [9.1 Tool Basics](#91-tool-basics)

    * [9.1.1 Tool Schema Design](#911-tool-schema-design)
    * [9.1.2 Required vs Optional Parameters](#912-required-vs-optional-parameters)
    * [9.1.3 Typed Inputs](#913-typed-inputs)
    * [9.1.4 Typed Outputs](#914-typed-outputs)
    * [9.1.5 Tool Descriptions](#915-tool-descriptions)
    * [9.1.6 Examples Inside Tool Definitions](#916-examples-inside-tool-definitions)
    * [9.1.7 Tool Constraints](#917-tool-constraints)
    * [9.1.8 Tool Result Normalization](#918-tool-result-normalization)
  * [9.2 Tool Routing](#92-tool-routing)

    * [9.2.1 Selecting Among Tools](#921-selecting-among-tools)
    * [9.2.2 Tool Namespacing](#922-tool-namespacing)
    * [9.2.3 Tool Grouping](#923-tool-grouping)
    * [9.2.4 Dynamic Tool Loading](#924-dynamic-tool-loading)
    * [9.2.5 Tool Catalogs](#925-tool-catalogs)
    * [9.2.6 Tool Discovery](#926-tool-discovery)
    * [9.2.7 Tool Relevance Filtering](#927-tool-relevance-filtering)
  * [9.3 Execution Models](#93-execution-models)

    * [9.3.1 Single Tool Calls](#931-single-tool-calls)
    * [9.3.2 Sequential Tool Calls](#932-sequential-tool-calls)
    * [9.3.3 Parallel Tool Calls](#933-parallel-tool-calls)
    * [9.3.4 Dependent Calls](#934-dependent-calls)
    * [9.3.5 Fan-Out / Fan-In](#935-fan-out--fan-in)
    * [9.3.6 Partial Failure](#936-partial-failure)
    * [9.3.7 Compensation](#937-compensation)
  * [9.4 Tool Reliability](#94-tool-reliability)

    * [9.4.1 Validation](#941-validation)
    * [9.4.2 Retries](#942-retries)
    * [9.4.3 Timeouts](#943-timeouts)
    * [9.4.4 Circuit Breakers](#944-circuit-breakers)
    * [9.4.5 Fallbacks](#945-fallbacks)
    * [9.4.6 Idempotency](#946-idempotency)
    * [9.4.7 Result Verification](#947-result-verification)
    * [9.4.8 Side-Effect Classification](#948-side-effect-classification)
  * [9.5 Tool Permissions](#95-tool-permissions)

    * [9.5.1 Read-Only Tools](#951-read-only-tools)
    * [9.5.2 Low-Risk Writes](#952-low-risk-writes)
    * [9.5.3 High-Risk Writes](#953-high-risk-writes)
    * [9.5.4 Irreversible / Financial / Sensitive Actions](#954-irreversible--financial--sensitive-actions)
    * [9.5.5 Authorization and Approval](#955-authorization-and-approval)
    * [9.5.6 Permission Enforcement Architecture](#956-permission-enforcement-architecture)
  * [9.6 Tool Calling Architecture](#96-tool-calling-architecture)

    * [9.6.1 Tool Definition Layer](#961-tool-definition-layer)
    * [9.6.2 Routing Layer](#962-routing-layer)
    * [9.6.3 Execution Layer](#963-execution-layer)
    * [9.6.4 Reliability Layer](#964-reliability-layer)
    * [9.6.5 Authorization Layer](#965-authorization-layer)
    * [9.6.6 Verification Layer](#966-verification-layer)
  * [9.7 Key Insights](#97-key-insights)
  * [9.8 Common Mistakes](#98-common-mistakes)
  * [9.9 Common Confusions](#99-common-confusions)
  * [9.10 Practical Applications](#910-practical-applications)
  * [9.11 Important Terms](#911-important-terms)
  * [9.12 Quick Revision](#912-quick-revision)
  * [9.13 Interview Preparation](#913-interview-preparation)

    * [9.13.1 Level 1 — Fundamentals](#9131-level-1--fundamentals)
    * [9.13.2 Level 2 — Conceptual Understanding](#9132-level-2--conceptual-understanding)
    * [9.13.3 Level 3 — Practical / Engineering](#9133-level-3--practical--engineering)
    * [9.13.4 Level 4 — Advanced / Deep Understanding](#9134-level-4--advanced--deep-understanding)
    * [9.13.5 Level 5 — Scenario-Based Questions](#9135-level-5--scenario-based-questions)
    * [9.13.6 Knowledge Check](#9136-knowledge-check)
    * [9.13.7 Follow-up Questions](#9137-follow-up-questions)
    * [9.13.8 Common Confusion Questions](#9138-common-confusion-questions)
    * [9.13.9 Deep / Trick Questions](#9139-deep--trick-questions)
  * [9.14 Top Questions You MUST Know](#914-top-questions-you-must-know)
  * [9.15 Interview Readiness Checklist](#915-interview-readiness-checklist)
  * [9.16 What You Should Be Able to Explain](#916-what-you-should-be-able-to-explain)

# 9. Layer 7 — Tool Calling & Action Systems

🧠 **Simple Understanding:** Tool calling gives an AI system controlled access to external capabilities such as APIs, databases, search engines, file systems, and business operations. The model decides **what it wants to do**, while the surrounding system determines **whether and how that action is actually executed**.

The core abstraction is:

```text
User Goal
   ↓
AI / Agent
   ↓
Select Tool
   ↓
Generate Arguments
   ↓
Validate
   ↓
Authorize
   ↓
Execute
   ↓
Normalize Result
   ↓
Verify Outcome
   ↓
Return Result
```

⭐ **Key Point:** A tool is not merely a function available to the model. It is a controlled **boundary between probabilistic AI decisions and deterministic external systems**.

---

## 9.1 Tool Basics

### 9.1.1 Tool Schema Design

🧠 **Simple Understanding:** A tool schema defines the contract between the model and the tool.

📌 **Quick Info**

| Field          | Answer                                                                            |
| -------------- | --------------------------------------------------------------------------------- |
| **What?**      | Structured definition of a tool's interface                                       |
| **Why?**       | Makes model-generated requests understandable and validatable                     |
| **How?**       | Define name, description, parameters, types, constraints, and outputs             |
| **When?**      | Whenever an AI system can invoke a tool                                           |
| **Example**    | `get_order(order_id: string)`                                                     |
| **Trade-offs** | More explicit schemas improve reliability but require more design and maintenance |

Example:

```json
{
  "name": "get_order",
  "description": "Retrieve the current status of an existing order.",
  "input_schema": {
    "type": "object",
    "properties": {
      "order_id": {
        "type": "string"
      }
    },
    "required": ["order_id"]
  }
}
```

A well-designed schema should make invalid or ambiguous tool requests difficult to express.

🔬 **Technical Explanation**

A tool definition typically describes:

```text
Tool Identity
├── Name
├── Description
├── Input contract
├── Output contract
├── Constraints
└── Operational metadata
```

The schema is part of the model-facing interface and therefore influences tool-use behavior.

🎯 **Interview Tip:** Treat schema design as an **interface-design problem**, not just a documentation problem.

---

### 9.1.2 Required vs Optional Parameters

🧠 **Simple Understanding:** Required parameters are necessary for a valid operation; optional parameters modify behavior when supplied.

Example:

```json
{
  "order_id": "ORD-1001",
  "include_history": true
}
```

| Parameter         | Type    | Required? |
| ----------------- | ------- | --------- |
| `order_id`        | string  | Yes       |
| `include_history` | boolean | No        |

Use a parameter as **required** when omission makes the request invalid or ambiguous.

Use it as **optional** when a safe and meaningful default exists.

⚠️ **Common Mistake:** Making everything optional does not make a tool easier to use. It often moves ambiguity into the execution layer.

🧠 **Remember:** Good schemas encode the **minimum information necessary for safe execution**.

---

### 9.1.3 Typed Inputs

🧠 **Simple Understanding:** Typed inputs restrict the kind of data a tool accepts.

Typical types:

```text
string
integer
number
boolean
array
object
enum
```

Example:

```json
{
  "quantity": 3,
  "confirmed": true
}
```

rather than:

```json
{
  "quantity": "three",
  "confirmed": "yes"
}
```

Typed inputs help detect:

* Wrong types.
* Missing fields.
* Invalid structures.
* Malformed arguments.

#### Constraints Beyond Types

Types are not enough.

```text
quantity → integer
```

does not guarantee:

```text
quantity > 0
```

Therefore:

```text
Type Validation
      ↓
Business Validation
```

are separate layers.

---

### 9.1.4 Typed Outputs

🧠 **Simple Understanding:** Typed outputs give downstream systems a predictable structure for tool results.

Example:

```json
{
  "order_id": "ORD-1001",
  "status": "shipped",
  "estimated_delivery": "2026-09-01"
}
```

instead of:

```text
Your order seems to have shipped.
```

Structured outputs are easier to:

* Parse.
* Test.
* Log.
* Evaluate.
* Compare.
* Feed into subsequent tool calls.

⭐ **Key Point:** Structured outputs reduce the amount of interpretation the model must perform.

---

### 9.1.5 Tool Descriptions

🧠 **Simple Understanding:** The tool description tells the model what a tool does and helps it decide when to use it.

Weak:

```text
"Order tool"
```

Better:

```text
"Retrieve the status of an existing customer order.
Use this when the user asks whether an order is processing,
shipped, or delivered. Do not use this tool to cancel orders."
```

A useful description specifies:

* Purpose.
* Intended use.
* Important boundaries.
* Required context.
* Common distinctions from similar tools.

⭐ **Key Insight:** Tool descriptions are part of **routing logic**.

---

### 9.1.6 Examples Inside Tool Definitions

🧠 **Simple Understanding:** Examples demonstrate how the tool should be called.

Example:

```json
{
  "order_id": "ORD-1001"
}
```

Examples are particularly useful for:

* Complex schemas.
* Non-obvious formats.
* Nested objects.
* Domain-specific identifiers.
* Special conventions.

⚠️ **Common Mistake:** Examples must agree with the actual schema. Contradictory examples can teach the model the wrong interface.

---

### 9.1.7 Tool Constraints

🧠 **Simple Understanding:** Constraints define what values and operations are allowed.

Examples:

```text
quantity >= 1
amount <= approved_limit
currency ∈ supported_currencies
date must be valid
resource must belong to user
```

There are several layers:

```text
Schema constraints
      ↓
Business constraints
      ↓
Authorization constraints
      ↓
Risk / policy constraints
```

Example:

```json
{
  "amount": 1000000
}
```

may be syntactically valid but still invalid because:

```text
amount exceeds transaction limit
```

⭐ **Remember:** **Schema-valid ≠ business-valid ≠ authorized.**

---

### 9.1.8 Tool Result Normalization

🧠 **Simple Understanding:** Result normalization converts inconsistent backend outputs into a common representation.

Suppose:

```json
System A:
{"status": "COMPLETE"}
```

and:

```json
System B:
{"state": "completed"}
```

are normalized to:

```json
{
  "status": "completed"
}
```

Benefits:

* Consistent downstream logic.
* Simpler prompts.
* Easier testing.
* Easier evaluation.
* Less vendor-specific coupling.

---

## 9.2 Tool Routing

### 9.2.1 Selecting Among Tools

🧠 **Simple Understanding:** Tool routing decides which capability best matches the user's current goal.

Example:

```text
User:
"Where is my order?"

Potential tools:
├── get_order
├── cancel_order
├── create_order
└── refund_order

Correct route:
get_order
```

Routing can depend on:

* User intent.
* Tool descriptions.
* Available context.
* Current state.
* Permissions.
* Tool availability.
* Required inputs.

🎯 **Interview Tip:** Tool routing is fundamentally a **decision problem over capabilities**.

---

### 9.2.2 Tool Namespacing

🧠 **Simple Understanding:** Namespacing organizes tools by domain.

Example:

```text
payments.get_balance
payments.create_payment
payments.refund_payment

orders.get_order
orders.create_order
orders.cancel_order

calendar.get_event
calendar.create_event
calendar.cancel_event
```

Benefits:

* Reduces naming collisions.
* Makes tool domains explicit.
* Improves discoverability.
* Helps permission organization.

---

### 9.2.3 Tool Grouping

Related tools can be grouped:

```text
CRM
├── find_customer
├── update_customer
└── create_ticket

Payments
├── get_balance
├── create_payment
└── refund_payment

Calendar
├── get_event
├── create_event
└── cancel_event
```

Grouping helps reduce routing complexity as tool counts grow.

---

### 9.2.4 Dynamic Tool Loading

🧠 **Simple Understanding:** Load only the tools relevant to the current task instead of exposing the entire tool universe.

```text
User Request
     ↓
Determine Domain
     ↓
Find Relevant Tools
     ↓
Load Tool Schemas
     ↓
Agent
```

Example:

```text
Travel Request
    ↓
Load:
├── search_flights
├── search_hotels
└── calendar
```

instead of exposing hundreds of unrelated tools.

Benefits:

* Smaller tool-selection space.
* Lower context overhead.
* Better relevance.
* Easier permission control.

---

### 9.2.5 Tool Catalogs

🧠 **Simple Understanding:** A tool catalog is a registry describing available capabilities.

Example:

| Tool             | Domain   | Risk  | Purpose               |
| ---------------- | -------- | ----- | --------------------- |
| `get_order`      | Orders   | Read  | Retrieve order status |
| `cancel_order`   | Orders   | Write | Cancel order          |
| `create_payment` | Payments | High  | Initiate payment      |
| `search_flights` | Travel   | Read  | Search flights        |

A catalog can support:

* Tool discovery.
* Routing.
* Filtering.
* Permissions.
* Versioning.
* Lifecycle management.

---

### 9.2.6 Tool Discovery

🧠 **Simple Understanding:** Tool discovery identifies what capabilities are available for the current goal.

```text
Goal
 ↓
Search Tool Catalog
 ↓
Relevant Capabilities
 ↓
Load Definitions
 ↓
Select Tool
```

Discovery becomes increasingly important as the number of tools grows.

---

### 9.2.7 Tool Relevance Filtering

🧠 **Simple Understanding:** Remove clearly irrelevant tools before presenting candidates to the model.

Example:

```text
User:
"Refund my order."

Remove:
├── weather
├── calendar
├── file_conversion
└── unrelated search tools

Keep:
├── get_order
├── refund_order
└── payment_status
```

⭐ **Key Point:** A smaller relevant tool space can improve routing and reduce accidental tool selection.

---

## 9.3 Execution Models

### 9.3.1 Single Tool Calls

🧠 **Simple Understanding:** One task requires one tool invocation.

```text
User
 ↓
Agent
 ↓
get_order()
 ↓
Result
 ↓
Answer
```

Appropriate when the operation is:

* Simple.
* Independent.
* Easy to verify.

---

### 9.3.2 Sequential Tool Calls

🧠 **Simple Understanding:** One tool must execute before another can run.

Example:

```text
get_customer()
      ↓
customer_id
      ↓
get_orders(customer_id)
      ↓
order_id
      ↓
get_order_details(order_id)
```

Use sequential execution when later operations depend on earlier outputs.

---

### 9.3.3 Parallel Tool Calls

🧠 **Simple Understanding:** Independent tool calls execute concurrently.

```text
                 Request
                    ↓
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Search A    Search B    Search C
        │           │           │
        └───────────┼───────────┘
                    ▼
                  Merge
```

Example:

> Compare shipping prices from three providers.

If each provider can be queried independently, those calls may run in parallel.

**Benefits**

* Lower wall-clock latency.
* Better throughput.

**Trade-offs**

* More simultaneous resource usage.
* More complex failure handling.
* More difficult result coordination.

---

### 9.3.4 Dependent Calls

A dependent call requires a previous result.

Example:

```text
find_customer()
      ↓
customer_id
      ↓
get_subscription()
      ↓
subscription_id
      ↓
cancel_subscription()
```

The dependency should be explicit in orchestration logic.

---

### 9.3.5 Fan-Out / Fan-In

🧠 **Simple Understanding:** Fan-out splits work into multiple independent operations; fan-in combines the results.

```text
                  Request
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
          A          B          C
          │          │          │
          └──────────┼──────────┘
                     ▼
                   Merge
```

Useful for:

* Multi-source search.
* Provider comparison.
* Batch processing.
* Independent data collection.

---

### 9.3.6 Partial Failure

🧠 **Simple Understanding:** A multi-operation workflow can partially succeed.

Example:

```text
Provider A → success
Provider B → success
Provider C → timeout
```

Possible responses:

* Retry C.
* Use a fallback.
* Continue with partial results.
* Ask the user.
* Abort the task.

The correct choice depends on task requirements.

⭐ **Key Insight:** **One failed operation does not automatically mean the whole workflow has failed.**

---

### 9.3.7 Compensation

🧠 **Simple Understanding:** Compensation performs a corrective action after partial success creates an undesirable state.

Example:

```text
Reserve inventory
      ↓
Charge payment
      ↓
Shipment creation fails
      ↓
Compensating actions
      ├── Release inventory
      └── Refund payment
```

Compensation is especially relevant when multiple independent systems participate in one workflow.

⚠️ **Important:** Compensation is not guaranteed to be a perfect rollback.

External operations may be:

* Partially reversible.
* Delayed.
* Irreversible.
* Independently failing.

---

## 9.4 Tool Reliability

### 9.4.1 Validation

🧠 **Simple Understanding:** Validate a tool request before allowing it to reach the external system.

A robust pipeline is:

```text
Model Request
      ↓
Schema Validation
      ↓
Business Validation
      ↓
Authorization
      ↓
Risk / Policy Check
      ↓
Execution
```

Examples:

* Required fields.
* Data types.
* Value ranges.
* Resource existence.
* Resource ownership.
* Current resource state.

---

### 9.4.2 Retries

🧠 **Simple Understanding:** Retry temporarily failed operations when repeating them is safe and useful.

Potential retry cases:

* Temporary network failures.
* Transient service errors.
* Rate limits with appropriate backoff.

Avoid blind retries for:

* Invalid arguments.
* Permission failures.
* Permanent business-rule failures.
* Unknown side-effect states.

Common strategy:

```text
Attempt 1
   ↓
Failure
   ↓
Wait
   ↓
Attempt 2
   ↓
Failure
   ↓
Longer Wait
   ↓
Attempt 3
```

🎯 **Interview Tip:** Always discuss **retry + idempotency + timeout** together.

---

### 9.4.3 Timeouts

🧠 **Simple Understanding:** A timeout puts a limit on how long the system waits for a tool.

Without a timeout:

```text
Agent
 ↓
Tool
 ↓
Hanging dependency
 ↓
Task stalls
```

Timeouts should be designed per tool type because expected latency varies.

A timeout must also define what happens afterward:

```text
Timeout
 ↓
Retry?
Fallback?
Verify state?
Abort?
Escalate?
```

---

### 9.4.4 Circuit Breakers

🧠 **Simple Understanding:** A circuit breaker stops repeatedly calling an unhealthy dependency.

Conceptually:

```text
Healthy
   ↓
Repeated failures
   ↓
Circuit OPEN
   ↓
Fail fast / stop calls
   ↓
Recovery test
   ↓
Circuit CLOSED
```

Benefits:

* Prevents cascading failures.
* Protects system capacity.
* Avoids wasting requests on unhealthy dependencies.

---

### 9.4.5 Fallbacks

🧠 **Simple Understanding:** A fallback provides another way to accomplish a task when the primary mechanism fails.

Example:

```text
Primary Search API
       ↓
    Failure
       ↓
Secondary Search API
       ↓
    Result
```

Fallbacks may be:

* Another provider.
* Cached data.
* A reduced capability.
* A human escalation path.

⚠️ **Common Mistake:** Assuming the fallback is semantically equivalent to the primary tool.

---

### 9.4.6 Idempotency

🧠 **Simple Understanding:** An idempotent operation can be repeated without accidentally producing additional side effects.

Safe example:

```text
set_status("shipped")
```

Potentially dangerous:

```text
charge_card($100)
```

Repeating the second operation might charge twice.

A common mechanism is an idempotency key:

```json
{
  "payment_id": "PAY-1001",
  "idempotency_key": "req-8f32aa"
}
```

The backend can recognize repeated requests representing the same logical operation.

⭐ **Key Point:** **Retries without idempotency can become duplicate-action bugs.**

---

### 9.4.7 Result Verification

🧠 **Simple Understanding:** Verification checks whether the desired external outcome actually occurred.

Example:

```text
create_ticket()
      ↓
Tool says "success"
      ↓
Check ticket system
      ↓
Ticket exists?
 ├── Yes → verified
 └── No  → failure
```

Especially important for:

* Financial actions.
* Irreversible actions.
* High-value workflows.
* Actions with ambiguous responses.

🧠 **Remember:** **Tool response ≠ guaranteed business outcome.**

---

### 9.4.8 Side-Effect Classification

🧠 **Simple Understanding:** Classify tools according to what they can change and how dangerous that change is.

| Category        | Example                     | Risk      |
| --------------- | --------------------------- | --------- |
| Read-only       | `get_order()`               | Low       |
| Low-risk write  | `save_draft()`              | Moderate  |
| High-risk write | `change_permissions()`      | High      |
| Financial       | `transfer_money()`          | Very high |
| Irreversible    | `delete_account()`          | Very high |
| Sensitive       | `access_sensitive_record()` | High      |

Risk classification should influence:

* Authorization.
* Approval.
* Logging.
* Confirmation.
* Verification.
* Recovery.

---

## 9.5 Tool Permissions

The core risk spectrum is:

```text
Read-only
    ↓
Low-risk write
    ↓
High-risk write
    ↓
Irreversible / Financial / Sensitive
```

Higher-risk actions require stronger controls.

---

### 9.5.1 Read-Only Tools

🧠 **Simple Understanding:** Read-only tools retrieve information without changing external state.

Examples:

```text
get_balance()
get_order()
search_documents()
get_customer()
```

Controls may still include:

* Authentication.
* Authorization.
* Tenant filtering.
* Audit logging.
* Data minimization.

---

### 9.5.2 Low-Risk Writes

These modify state but usually have limited consequences.

Examples:

```text
save_draft()
add_internal_note()
update_preference()
```

Controls may include:

* Authorization.
* Validation.
* Audit logging.
* Optional confirmation.

---

### 9.5.3 High-Risk Writes

Examples:

```text
change_permissions()
delete_customer()
modify_billing()
```

Potential controls:

```text
Authentication
      ↓
Authorization
      ↓
Business Validation
      ↓
Risk Check
      ↓
Approval / Confirmation
      ↓
Execution
      ↓
Verification
      ↓
Audit
```

---

### 9.5.4 Irreversible / Financial / Sensitive Actions

🧠 **Simple Understanding:** These actions can cause serious, costly, or irreversible consequences.

Examples:

```text
send_money()
delete_account()
publish_content()
reveal_protected_information()
```

A stronger workflow is:

```text
Agent Intent
    ↓
Authorization
    ↓
Policy Check
    ↓
Explicit Confirmation / Approval
    ↓
Execution
    ↓
Verification
    ↓
Audit Record
```

⭐ **Key Point:** High-risk capabilities should be treated as **controlled operations**, not ordinary function calls.

---

### 9.5.5 Authorization and Approval

🧠 **Simple Understanding:** Authorization asks whether an actor **may** perform the action; approval asks whether an additional required checkpoint has approved the action.

Example:

```text
Agent wants to initiate a large payment
         │
         ├── Authorized?
         │      ↓
         │     Yes
         │
         └── Approval required?
                ↓
               Yes
                ↓
          Human approval
```

They are distinct:

| Concept        | Question                              |
| -------------- | ------------------------------------- |
| Authentication | Who are you?                          |
| Authorization  | Are you allowed?                      |
| Approval       | Has the required checkpoint approved? |

---

### 9.5.6 Permission Enforcement Architecture

```text
                     Agent
                       │
                       ▼
                  Tool Request
                       │
                       ▼
                Schema Validation
                       │
                       ▼
                Identity / Tenant
                       │
                       ▼
              Authorization Check
                       │
                       ▼
                Risk Classification
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Low Risk          High Risk
              │                 │
              │            Approval Required
              │                 │
              └────────┬────────┘
                       ▼
                   Execution
                       │
                       ▼
                  Verification
                       │
                       ▼
                     Audit
```

⭐ **Key Point:** **The model should never be the ultimate authority on whether an action is permitted.**

---

# 9.6 Tool Calling Architecture

## 9.6.1 Tool Definition Layer

Contains:

```text
Name
Description
Input Schema
Output Schema
Examples
Constraints
Risk Metadata
Permission Metadata
```

This layer defines the tool's contract.

---

## 9.6.2 Routing Layer

Responsible for:

* Understanding the task.
* Discovering tools.
* Filtering irrelevant tools.
* Loading relevant definitions.
* Selecting tools.

```text
User Goal
   ↓
Task Understanding
   ↓
Tool Catalog
   ↓
Relevance Filter
   ↓
Candidate Tools
   ↓
Tool Selection
```

---

## 9.6.3 Execution Layer

Responsible for:

* Argument validation.
* Tool dispatch.
* Sequential execution.
* Parallel execution.
* Dependency handling.
* Result collection.

```text
Tool Request
     ↓
Execution Engine
     ├── Single
     ├── Sequential
     ├── Parallel
     └── Fan-out / Fan-in
```

---

## 9.6.4 Reliability Layer

Provides:

```text
Retries
Timeouts
Circuit Breakers
Fallbacks
Idempotency
Compensation
```

This layer deals with the reality that external dependencies fail.

---

## 9.6.5 Authorization Layer

Responsible for:

* Authentication.
* Authorization.
* Tenant isolation.
* Policy enforcement.
* Risk classification.
* Approval workflows.

---

## 9.6.6 Verification Layer

Responsible for checking:

> Did the intended action actually happen?

```text
Tool Call
   ↓
Execution
   ↓
Reported Result
   ↓
External State Check
   ↓
Verified Outcome
```

---

# 9.7 Key Insights

💡 **Key Insights**

1. **Tool calling is an action boundary.** The model produces intent; infrastructure controls execution.

2. **Schemas reduce ambiguity but do not provide complete safety.** Business validation and authorization remain necessary.

3. **Descriptions influence routing.** Tool definitions are part of the AI's decision environment.

4. **Tool count matters.** Large toolsets benefit from catalogs, grouping, namespacing, dynamic loading, and relevance filtering.

5. **Execution semantics matter.** Sequential, parallel, dependent, and fan-out/fan-in workflows have different correctness and failure characteristics.

6. **Retryability depends on side effects.** A safe retry policy requires understanding idempotency and ambiguous outcomes.

7. **Verification closes the action loop.** An agent's claim that an action succeeded should not automatically be treated as proof.

---

# 9.8 Common Mistakes

⚠️ **Common Mistakes**

| Mistake                                           | Correct Understanding                                                  |
| ------------------------------------------------- | ---------------------------------------------------------------------- |
| "Schema-valid means safe."                        | Schema validity is only one layer of validation.                       |
| "The model can decide whether it is allowed."     | Authorization must be enforced outside the model.                      |
| "Retries are always good."                        | Retries can duplicate side effects.                                    |
| "Timeout means nothing happened."                 | The operation may have succeeded even if the response timed out.       |
| "Tool success means business success."            | Important actions may require state verification.                      |
| "Expose every tool."                              | Large toolsets can increase routing complexity and risk.               |
| "All tools need the same controls."               | Tool risk determines appropriate controls.                             |
| "Parallel calls are always better."               | Dependencies, resource limits, and partial failure matter.             |
| "Compensation equals rollback."                   | External actions may not be fully reversible.                          |
| "Descriptions are documentation only."            | Tool descriptions influence model behavior.                            |
| "Fewer tool calls always means a better agent."   | Reliability and successful completion matter more than raw call count. |
| "User confirmation is the same as authorization." | Authorization and approval solve different problems.                   |

---

# 9.9 Common Confusions

🔍 **Common Confusions**

| Concept A      | Concept B            | Key Difference                                                                 |
| -------------- | -------------------- | ------------------------------------------------------------------------------ |
| Tool schema    | Business validation  | Schema checks structure; business validation checks meaning/rules              |
| Authentication | Authorization        | Identity vs permissions                                                        |
| Authorization  | Approval             | Permission vs required decision checkpoint                                     |
| Tool routing   | Tool execution       | Choosing a capability vs invoking it                                           |
| Retry          | Fallback             | Repeat the same operation vs use another path                                  |
| Validation     | Verification         | Check whether an action is acceptable vs check whether outcome occurred        |
| Idempotency    | Deduplication        | Safe repetition of a logical operation vs preventing duplicate processing/data |
| Sequential     | Parallel             | Dependent ordering vs independent concurrent work                              |
| Compensation   | Rollback             | Corrective external action vs transactional reversal                           |
| Tool discovery | Tool routing         | Find available capabilities vs choose the appropriate capability               |
| Read-only      | Low-risk write       | No state change vs limited state change                                        |
| Agent decision | System authorization | Model proposal vs deterministic permission enforcement                         |

---

# 9.10 Practical Applications

🛠️ **Practical Applications**

| Use Case               | Important Concepts                                       |
| ---------------------- | -------------------------------------------------------- |
| Customer-support agent | Routing, schemas, retries, authorization                 |
| Payment agent          | Idempotency, approval, risk classification, verification |
| Scheduling agent       | Discovery, validation, sequential calls                  |
| E-commerce assistant   | Inventory, order tools, compensation                     |
| Coding agent           | File tools, shell tools, dependency handling             |
| Browser agent          | Sequential execution, state verification                 |
| Research agent         | Search, parallel retrieval, fallback                     |
| Workflow automation    | Fan-out/fan-in, partial failure, compensation            |
| Enterprise assistant   | Namespaces, catalogs, permissions                        |
| Finance operations     | High-risk controls, authorization, auditability          |

---

# 9.11 Important Terms

📌 **Important Terms**

| Term                 | Simple Meaning                        | Why It Matters                          |
| -------------------- | ------------------------------------- | --------------------------------------- |
| Tool                 | External capability an AI can invoke  | Connects AI to external systems         |
| Tool Schema          | Interface contract                    | Enables structured invocation           |
| Tool Description     | Explanation of capability             | Influences tool selection               |
| Typed Input          | Input with explicit type              | Prevents malformed arguments            |
| Typed Output         | Structured result                     | Makes downstream processing predictable |
| Tool Routing         | Selecting a tool                      | Core agent capability                   |
| Tool Catalog         | Registry of tools                     | Supports discovery and management       |
| Tool Discovery       | Finding available capabilities        | Important at scale                      |
| Namespacing          | Domain-based naming                   | Reduces ambiguity                       |
| Dynamic Tool Loading | Loading tools on demand               | Reduces routing complexity              |
| Sequential Call      | Dependency-aware execution            | Preserves required ordering             |
| Parallel Call        | Concurrent independent execution      | Reduces latency                         |
| Fan-Out              | Split one task into independent calls | Enables concurrency                     |
| Fan-In               | Combine parallel results              | Reconstructs final result               |
| Partial Failure      | Some operations succeed, others fail  | Requires explicit recovery strategy     |
| Compensation         | Corrective action                     | Handles distributed side effects        |
| Retry                | Reattempt an operation                | Recovers from transient failures        |
| Timeout              | Maximum wait                          | Prevents indefinite blocking            |
| Circuit Breaker      | Stops calls to unhealthy dependencies | Prevents cascading failures             |
| Fallback             | Alternate execution path              | Improves resilience                     |
| Idempotency          | Safe repetition                       | Critical for side-effectful retries     |
| Verification         | Confirm actual outcome                | Prevents false completion               |
| Side Effect          | External state change                 | Determines risk                         |
| Authentication       | Identify the actor                    | Foundation of access control            |
| Authorization        | Determine permission                  | Security boundary                       |
| Approval             | Required additional checkpoint        | Protects high-risk actions              |
| Audit                | Record important actions              | Supports security and debugging         |

---

# 9.12 Quick Revision

⚡ **Quick Revision**

1. A **tool schema** defines the interface between the AI and external capability.
2. **Descriptions and examples** influence tool routing.
3. **Typed inputs and outputs** make interactions more predictable.
4. **Schema validity is not business validity.**
5. **Tool catalogs and dynamic loading** help control large tool ecosystems.
6. **Sequential calls** handle dependencies.
7. **Parallel calls** handle independent work.
8. **Fan-out/fan-in** supports multi-source and batch workflows.
9. **Partial failure** requires explicit recovery behavior.
10. **Retries must consider idempotency.**
11. **Timeouts** do not necessarily mean the action failed.
12. **Circuit breakers** protect unhealthy dependencies.
13. **Fallbacks** provide alternate execution paths.
14. **Verification** checks whether the desired external outcome occurred.
15. **Risk classification** determines how strongly tools should be controlled.
16. **Authentication, authorization, and approval are different controls.**
17. **The model proposes actions; deterministic infrastructure controls execution.**

---

# 9.13 Interview Preparation

## 9.13.1 Level 1 — Fundamentals

### Q1. What is tool calling?

**Model Answer:**
Tool calling allows an AI model to request an external capability using a structured interface. The surrounding application then validates the request, checks permissions, executes the tool, and returns the result.

### Q2. Why are tool schemas important?

**Model Answer:**
Schemas define the contract of the tool. They specify parameters, types, required fields, and often constraints, allowing model-generated tool requests to be validated before execution.

### Q3. What is tool routing?

**Model Answer:**
Tool routing is the process of selecting the capability that best matches the current task. It depends on user intent, tool descriptions, context, available tools, and permissions.

### Q4. Why use typed inputs?

**Model Answer:**
Typed inputs restrict the structure and data types of model-generated arguments. They reduce malformed requests and make validation more deterministic.

### Q5. Why use typed outputs?

**Model Answer:**
Structured outputs make tool results predictable for the application and agent. They simplify parsing, testing, evaluation, and chaining multiple tool calls.

### Q6. What is a retry?

**Model Answer:**
A retry repeats a failed tool call when the failure may be transient and repeating the operation is safe. Retry policies should consider backoff, timeouts, and idempotency.

### Q7. What is idempotency?

**Model Answer:**
Idempotency means repeating the same logical request does not unintentionally create additional effects. It is especially important when retries are possible for side-effectful operations.

### Q8. Why classify tools by risk?

**Model Answer:**
Different tools produce different consequences. A read operation and a financial transfer should not have identical controls. Risk classification determines the required authorization, approval, confirmation, logging, and verification.

---

## 9.13.2 Level 2 — Conceptual Understanding

### Q1. Why isn't schema validation enough?

**Model Answer:**
A schema verifies structure and types, but not necessarily business rules, authorization, ownership, current state, or risk policy. A request can be structurally valid and still be unsafe or invalid.

### Q2. Why can retries cause duplicate actions?

**Model Answer:**
The original request may have succeeded even though its response timed out or was lost. Retrying can therefore execute the side effect again. Idempotency mechanisms help prevent duplicate execution.

### Q3. Why are timeouts difficult for side-effectful operations?

**Model Answer:**
A timeout tells us that a response was not received in time, not necessarily that the operation did not execute. The system may need to verify external state before retrying.

### Q4. Why use dynamic tool loading?

**Model Answer:**
Loading only relevant tools reduces the tool-selection space, decreases unnecessary context, improves routing, and limits exposure to unrelated capabilities.

### Q5. Why is verification different from validation?

**Model Answer:**
Validation checks whether the request should be allowed. Verification checks whether the intended result actually occurred after execution.

### Q6. Why is compensation needed?

**Model Answer:**
In multi-step workflows, earlier operations may succeed while later operations fail. Compensation attempts to restore an acceptable state through corrective actions when a true rollback is unavailable.

### Q7. Why can parallel tool calls be dangerous?

**Model Answer:**
Independent calls can reduce latency, but parallel execution can increase resource pressure and complicate partial failures, ordering, and aggregation. Dependent actions should not be parallelized merely for speed.

### Q8. Why shouldn't authorization happen entirely inside the prompt?

**Model Answer:**
A model is probabilistic and can be manipulated or make mistakes. Authorization is a security control and therefore needs deterministic enforcement outside the model.

---

## 9.13.3 Level 3 — Practical / Engineering

### Q1. How would you design a production tool interface?

**Model Answer:**

```text
Tool Definition
├── Name
├── Description
├── Input Schema
├── Output Schema
├── Examples
├── Constraints
├── Risk Class
└── Permission Requirements
```

Execution should follow:

```text
Tool Request
 ↓
Schema Validation
 ↓
Business Validation
 ↓
Authorization
 ↓
Approval if required
 ↓
Execution
 ↓
Result Normalization
 ↓
Verification
 ↓
Audit
```

### Q2. How would you manage hundreds of tools?

**Model Answer:**
I would create a tool catalog containing metadata such as domain, description, risk, permissions, and schemas. I would use routing and relevance filtering to identify candidate tools and dynamically load only relevant definitions.

### Q3. How would you handle a tool timeout?

**Model Answer:**
First determine whether the operation is idempotent and whether the timeout leaves its external state unknown. For side-effectful operations, I would verify the external state before retrying. For safely retryable operations, I would use bounded retries with backoff.

### Q4. How would you design a payment tool?

**Model Answer:**
I would classify it as high-risk, validate the amount and currency, authenticate and authorize the actor, enforce limits and policy checks, use idempotency, require confirmation or approval where appropriate, execute through the payment service, verify the transaction state, and audit the operation.

### Q5. How would you handle partial failure in a fan-out workflow?

**Model Answer:**
I would classify the failed branch and determine whether the task can proceed with partial results. Depending on requirements, I might retry, use a fallback, report partial completion, ask the user, or abort. The behavior should be explicit in the workflow policy.

### Q6. How would you debug incorrect tool selection?

**Model Answer:**
I would inspect the original user intent, candidate tools, tool descriptions, routing logic, context, namespaces, and permissions. I would also create evaluation cases for commonly confused tools and measure selection accuracy independently.

### Q7. How would you implement safe compensation?

**Model Answer:**
I would identify which side effects are reversible, define explicit compensating actions, handle compensation failure independently, and use reconciliation for ambiguous states. Compensation actions themselves require authorization and observability.

---

## 9.13.4 Level 4 — Advanced / Deep Understanding

### Q1. Why does a timeout create an "unknown state"?

**Model Answer:**
Because a timeout only establishes that the caller did not receive a response within the expected period. The backend may still have completed the operation. For side-effectful actions, the system therefore needs reconciliation or state verification before blindly retrying.

### Q2. Why can a larger toolset reduce agent quality?

**Model Answer:**
A larger toolset increases the decision space, creates more similar tool descriptions, consumes more model-visible context, and increases opportunities for incorrect selection. Tool retrieval and dynamic loading reduce this complexity.

### Q3. Why is result normalization useful beyond convenience?

**Model Answer:**
Normalization creates a stable abstraction over heterogeneous backend APIs. It reduces the amount of backend-specific reasoning the model must perform and makes evaluation, logging, and downstream orchestration easier.

### Q4. Why is compensation not equivalent to rollback?

**Model Answer:**
Rollback generally assumes transactional control over the state being changed. In distributed action systems, operations may span independent systems where no atomic rollback exists. Compensation instead performs additional operations to mitigate previous effects.

### Q5. Why should risk classification be part of tool metadata?

**Model Answer:**
Risk affects how the tool should be authorized, confirmed, rate-limited, logged, and verified. Without explicit risk metadata, a system may accidentally apply identical controls to actions with radically different consequences.

### Q6. Why can fewer tool calls be worse?

**Model Answer:**
Minimizing calls is only an efficiency objective. An agent that uses fewer calls but skips required verification or retrieves incorrect state can be less reliable than one that uses more calls correctly.

### Q7. What is the relationship between tool routing and tool permissions?

**Model Answer:**
Routing determines which capability is relevant, while permissions determine which capabilities the actor is allowed to use. Permission checks should constrain execution regardless of what the model selects.

---

## 9.13.5 Level 5 — Scenario-Based Questions

### Scenario 1 — Payment Timeout

A payment request times out. The agent does not know whether the charge succeeded.

**Question:** What should happen?

**Model Answer:**

```text
Timeout
  ↓
State Unknown
  ↓
Check payment status / idempotency record
  ↓
Known success?
 ├── Yes → Return verified result
 ├── No  → Retry if safe
 └── Unknown → Reconcile / escalate
```

The critical mistake would be blindly issuing another charge.

---

### Scenario 2 — Wrong Tool Selection

The user says:

> "Cancel my order."

The agent selects `get_order()`.

**Question:** How would you improve the system?

**Model Answer:**
I would inspect tool descriptions, names, candidate-tool filtering, and routing behavior. I would distinguish similar tools explicitly in descriptions and examples, reduce irrelevant candidate tools, and add evaluation cases separating:

```text
get_order
cancel_order
refund_order
modify_order
```

---

### Scenario 3 — 500 Available Tools

An enterprise agent has hundreds of tools across HR, finance, IT, sales, and operations.

**Question:** Would you expose all of them on every request?

**Model Answer:**
Usually not. I would use a catalog and route the request to a domain, filter tools by relevance and permissions, and dynamically load only the required tool definitions.

---

### Scenario 4 — Partial Failure

An agent compares three travel providers:

```text
Provider A → success
Provider B → success
Provider C → failure
```

**Question:** What should the workflow do?

**Model Answer:**
The workflow should follow an explicit partial-failure policy. It might retry C, use a fallback, provide a partial comparison, ask the user, or abort. The choice depends on whether complete comparison is required.

---

### Scenario 5 — Irreversible Action

An agent wants to delete a customer account.

**Question:** What controls should be added?

**Model Answer:**

```text
Agent Request
     ↓
Schema Validation
     ↓
Identity / Tenant Check
     ↓
Authorization
     ↓
Risk Check
     ↓
Explicit Confirmation / Approval
     ↓
Execution
     ↓
Verification
     ↓
Audit
```

The model should not have unilateral authority to perform the deletion.

---

# 9.13.6 Knowledge Check

🧠 **Knowledge Check**

If you can explain these naturally in your own words, you understand Layer 7:

* What a tool is.
* What a tool schema defines.
* Why descriptions influence routing.
* Why typed inputs and outputs matter.
* Why schema validation is not sufficient.
* How tool routing works.
* Why tool catalogs become necessary at scale.
* What dynamic tool loading accomplishes.
* Why relevance filtering matters.
* When calls should be sequential.
* When calls can be parallel.
* What dependent calls are.
* What fan-out/fan-in means.
* What partial failure means.
* Why compensation exists.
* How retries interact with idempotency.
* Why timeouts produce ambiguous states.
* What circuit breakers do.
* What fallbacks do.
* Why result verification matters.
* How tools should be classified by risk.
* The difference between authentication, authorization, and approval.
* Why the model should not be the ultimate security authority.

---

# 9.13.7 Follow-up Questions

### Basic Question

**What is tool calling?**

→ Why is it needed?
→ How is the tool described?
→ How is the tool selected?
→ How are arguments validated?
→ How is execution authorized?
→ How is the result verified?

### Basic Question

**What is tool routing?**

→ How are tools discovered?
→ How are irrelevant tools filtered?
→ Why use namespaces?
→ Why use dynamic loading?
→ How do permissions affect routing?

### Basic Question

**How are tools executed?**

→ Single or multiple calls?
→ Sequential or parallel?
→ Which calls are dependent?
→ What is fan-out/fan-in?
→ What happens if one call fails?

### Basic Question

**How do you make tool execution reliable?**

→ Validation?
→ Retry?
→ Timeout?
→ Circuit breaker?
→ Fallback?
→ Idempotency?
→ Compensation?
→ Verification?

### Basic Question

**How do you secure tool execution?**

→ Authentication?
→ Authorization?
→ Tenant isolation?
→ Risk classification?
→ Confirmation?
→ Approval?
→ Audit?

---

# 9.13.8 Common Confusion Questions

### Q1. Is a tool schema a security mechanism?

**Model Answer:**
No. A schema checks the structure of the request. Security requires authorization, policy enforcement, resource ownership checks, and other controls.

### Q2. Is authentication the same as authorization?

**Model Answer:**
No. Authentication identifies the actor; authorization determines which actions that actor is permitted to perform.

### Q3. Is user confirmation the same as authorization?

**Model Answer:**
No. Authorization establishes permission. Confirmation is a separate checkpoint that may be required before a permitted high-risk action is executed.

### Q4. Is a retry the same as a fallback?

**Model Answer:**
No. A retry repeats the same operation. A fallback uses an alternative operation, provider, or execution path.

### Q5. Is verification the same as validation?

**Model Answer:**
No. Validation checks whether a request is acceptable before execution. Verification checks whether the desired outcome occurred after execution.

### Q6. Is compensation the same as rollback?

**Model Answer:**
No. Compensation uses corrective operations to mitigate an earlier effect. Rollback generally refers to reversing state within transactional control.

---

# 9.13.9 Deep / Trick Questions

### ⚠️ Deeper Question

**The model selected the correct tool and generated schema-valid arguments. Can the system safely execute it?**

**Correct Understanding:**
Not necessarily. Business rules, authorization, resource ownership, current state, risk policies, and approval requirements still need to be checked.

---

### ⚠️ Deeper Question

**The tool timed out. Why shouldn't the system immediately retry?**

**Correct Understanding:**
Because the operation may already have succeeded. Retrying a non-idempotent side effect can duplicate the action.

---

### ⚠️ Deeper Question

**Why isn't a successful HTTP/API response always enough?**

**Correct Understanding:**
The response may indicate request acceptance or processing rather than the final business state. Important workflows may require independent verification.

---

### ⚠️ Deeper Question

**Why can exposing more tools make an agent worse?**

**Correct Understanding:**
It expands the selection space, introduces more similar capabilities, increases context usage, and raises the chance of incorrect routing.

---

### ⚠️ Deeper Question

**If an agent has permission to perform an action, why can approval still be necessary?**

**Correct Understanding:**
Authorization and approval solve different problems. The actor may be allowed to perform an action, while policy still requires explicit user or human approval because the action is high-risk.

---

### ⚠️ Deeper Question

**Why is compensation hard in distributed systems?**

**Correct Understanding:**
Different systems may not share a common transaction boundary. There may be no atomic rollback, and compensating actions can themselves fail or produce different effects.

---

# 9.14 Top Questions You MUST Know

⭐ **Top Questions You MUST Know**

1. What is tool calling?
2. How do you design a reliable tool schema?
3. Why are typed inputs and outputs important?
4. How do tool descriptions affect routing?
5. How would you route an agent among hundreds of tools?
6. Why use tool catalogs, grouping, and namespaces?
7. What is dynamic tool loading?
8. When should tool calls be sequential versus parallel?
9. What are dependent calls?
10. What are fan-out and fan-in?
11. How should partial failures be handled?
12. What is compensation and how does it differ from rollback?
13. How do retries interact with idempotency?
14. How should a timed-out side-effectful operation be handled?
15. How would you secure and verify a high-risk tool?

---

# 9.15 Interview Readiness Checklist

🎯 **Interview Readiness Checklist**

| Skill                           | Can I explain it? |
| ------------------------------- | :---------------: |
| Tool calling                    |         ☐         |
| Tool schema design              |         ☐         |
| Required vs optional parameters |         ☐         |
| Typed inputs                    |         ☐         |
| Typed outputs                   |         ☐         |
| Tool descriptions               |         ☐         |
| Examples                        |         ☐         |
| Tool constraints                |         ☐         |
| Result normalization            |         ☐         |
| Tool routing                    |         ☐         |
| Namespacing                     |         ☐         |
| Tool grouping                   |         ☐         |
| Dynamic tool loading            |         ☐         |
| Tool catalogs                   |         ☐         |
| Tool discovery                  |         ☐         |
| Relevance filtering             |         ☐         |
| Single calls                    |         ☐         |
| Sequential calls                |         ☐         |
| Parallel calls                  |         ☐         |
| Dependent calls                 |         ☐         |
| Fan-out / fan-in                |         ☐         |
| Partial failure                 |         ☐         |
| Compensation                    |         ☐         |
| Validation                      |         ☐         |
| Retries                         |         ☐         |
| Timeouts                        |         ☐         |
| Circuit breakers                |         ☐         |
| Fallbacks                       |         ☐         |
| Idempotency                     |         ☐         |
| Result verification             |         ☐         |
| Side-effect classification      |         ☐         |
| Authentication                  |         ☐         |
| Authorization                   |         ☐         |
| Approval                        |         ☐         |
| Risk classification             |         ☐         |
| Auditability                    |         ☐         |
| Tenant isolation                |         ☐         |
| High-risk action controls       |         ☐         |
| Production recovery             |         ☐         |
| Tool observability              |         ☐         |

---

# 9.16 What You Should Be Able to Explain

🧠 **What You Should Be Able to Explain**

By the end of Layer 7, you should be able to explain:

* What tool calling is.
* Why AI systems need tools to interact with external systems.
* How tool schemas define interfaces.
* How required and optional parameters should be chosen.
* Why strong input and output typing matters.
* Why tool descriptions influence model behavior.
* How examples improve understanding of non-obvious interfaces.
* Why schema constraints are only one layer of validation.
* How tool results can be normalized.
* How tool routing works.
* Why namespacing and grouping are useful.
* Why tool catalogs matter as capability counts grow.
* How tool discovery works.
* Why relevance filtering can improve routing.
* When calls should execute sequentially.
* When calls can safely execute in parallel.
* How dependent calls work.
* How fan-out/fan-in patterns work.
* How to design for partial failure.
* Why compensation is needed in distributed workflows.
* Why compensation is not necessarily rollback.
* How validation protects the execution boundary.
* When retries are appropriate.
* Why retries can create duplicate side effects.
* What idempotency means.
* Why timeouts create ambiguous states.
* How circuit breakers protect unhealthy services.
* When fallbacks are appropriate.
* Why result verification matters.
* How side effects should influence tool risk.
* The difference between read-only, low-risk, high-risk, financial, sensitive, and irreversible actions.
* The difference between authentication and authorization.
* The difference between authorization and approval.
* Why high-risk actions require stronger controls.
* Why authorization must be enforced outside the model.
* How routing, execution, reliability, authorization, and verification fit together.
* How tool calling becomes a production-grade action system rather than a simple function-call mechanism.

## ⚡ Final Mental Model

```text
                           USER GOAL
                               │
                               ▼
                     ┌───────────────────┐
                     │    AI / AGENT     │
                     │                   │
                     │ Understand Goal   │
                     │ Decide Action     │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │   TOOL DISCOVERY  │
                     │                   │
                     │ Catalog           │
                     │ Grouping          │
                     │ Namespaces        │
                     │ Relevance Filter  │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │   TOOL SELECTION  │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │  TOOL REQUEST     │
                     │                   │
                     │ Name              │
                     │ Arguments         │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │    VALIDATION     │
                     │                   │
                     │ Types             │
                     │ Schema            │
                     │ Business Rules    │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │   AUTHORIZATION   │
                     │                   │
                     │ Identity          │
                     │ Tenant            │
                     │ Permissions       │
                     │ Risk              │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ APPROVAL / POLICY │
                     │  WHEN REQUIRED    │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │    EXECUTION      │
                     │                   │
                     │ Single            │
                     │ Sequential        │
                     │ Parallel          │
                     │ Fan-Out/Fan-In    │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │    RELIABILITY    │
                     │                   │
                     │ Retry             │
                     │ Timeout           │
                     │ Circuit Breaker   │
                     │ Fallback          │
                     │ Idempotency       │
                     │ Compensation      │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ RESULT NORMALIZER  │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │    VERIFICATION   │
                     │                   │
                     │ Did the intended  │
                     │ outcome happen?   │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │    AUDIT / TRACE  │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │  RESULT TO AGENT  │
                     └───────────────────┘
```

> **Core principle:** **A production tool-calling system is the controlled conversion of AI intent into real-world action: the model selects and proposes, schemas constrain, validation checks, authorization permits, reliability mechanisms recover, execution performs, and verification confirms the actual outcome.**
