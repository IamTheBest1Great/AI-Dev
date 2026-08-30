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
    * [9.5.4 Irreversible, Financial, and Sensitive Actions](#954-irreversible-financial-and-sensitive-actions)
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

🧠 **Simple Understanding:** Tool calling gives an AI system controlled ways to interact with the outside world. The model decides **what action may be needed**, while the surrounding application validates, authorizes, executes, and verifies that action.

A useful mental model is:

```text
User Goal
   ↓
LLM / Agent
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
Return Result to Agent
```

The central engineering challenge is that a tool is not merely a function the model can call. It is an **action boundary** between probabilistic model behavior and deterministic external systems.

---

## 9.1 Tool Basics

### 9.1.1 Tool Schema Design

🧠 **Simple Understanding:** A tool schema is the machine-readable contract describing what a tool accepts and returns.

📌 **Quick Info**

| Field         | Answer                                                                         |
| ------------- | ------------------------------------------------------------------------------ |
| **What?**     | Formal definition of a tool's interface                                        |
| **Why?**      | Gives the model and application a shared contract                              |
| **How?**      | Define name, description, parameters, types, constraints, and output structure |
| **When?**     | Every model-accessible tool should have a clear schema                         |
| **Example**   | `get_order(order_id: string)`                                                  |
| **Trade-off** | More expressive schemas improve validation but increase complexity             |

Example:

```json
{
  "name": "get_order",
  "description": "Retrieve the current status of an order.",
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

A good schema makes invalid states difficult to represent.

⭐ **Key Point:** The schema is part of the **model-to-system interface**, so poor schema design can directly cause tool-use failures.

### 9.1.2 Required vs Optional Parameters

🧠 **Simple Understanding:** Required parameters are necessary for the tool to work correctly; optional parameters modify behavior when supplied.

Example:

```json
{
  "order_id": "ORD-123",
  "include_history": true
}
```

Here:

* `order_id` → required.
* `include_history` → optional.

Use **required parameters** when omission makes the operation ambiguous or invalid.

Use **optional parameters** when the tool has a safe, sensible default.

⚠️ **Common Mistake:** Making parameters optional merely to make the schema easier for the model. Ambiguity often moves the failure from validation into execution.

### 9.1.3 Typed Inputs

🧠 **Simple Understanding:** Typed inputs constrain what the model is allowed to send.

Examples:

```text
order_id       → string
quantity       → integer
temperature    → number
confirmed      → boolean
items          → array
customer       → object
```

Typed inputs help catch:

* Wrong data types.
* Missing fields.
* Malformed structures.
* Invalid values.

Example:

```json
{
  "quantity": 3
}
```

is structurally different from:

```json
{
  "quantity": "three"
}
```

### 9.1.4 Typed Outputs

🧠 **Simple Understanding:** Typed outputs give downstream components a predictable representation of tool results.

Example:

```json
{
  "order_id": "ORD-123",
  "status": "shipped",
  "estimated_delivery": "2026-08-31"
}
```

rather than returning arbitrary unstructured text such as:

```text
Looks like your package is probably shipping soon.
```

Structured outputs make it easier for:

* Agents to reason over results.
* Applications to validate results.
* Observability systems to record events.
* Tests to assert expected behavior.

### 9.1.5 Tool Descriptions

🧠 **Simple Understanding:** A tool description tells the model what the tool does, when to use it, and what its important boundaries are.

A useful description answers:

* What does the tool do?
* What does it not do?
* When should it be used?
* What information does it need?
* What important constraints apply?

Poor:

```text
"Order tool"
```

Better:

```text
"Retrieve the status of an existing customer order.
Use this when the user wants to know whether an order has shipped,
been delivered, or is still processing. Do not use it to cancel orders."
```

⭐ **Key Point:** Tool descriptions influence tool selection, so descriptions are part of **routing behavior**, not just documentation.

### 9.1.6 Examples Inside Tool Definitions

Examples can demonstrate correct tool usage.

```text
Example:
Input:
{
  "order_id": "ORD-123"
}
```

They are useful when the interface contains:

* Non-obvious fields.
* Special formats.
* Domain-specific values.
* Complex nested structures.

However, examples should reinforce the actual contract rather than contradict it.

### 9.1.7 Tool Constraints

🧠 **Simple Understanding:** Constraints define what the tool is allowed to accept or do.

Examples:

```text
quantity >= 1
currency ∈ supported currencies
date must be valid ISO format
amount <= configured limit
user must own target resource
```

Constraints belong at the execution boundary whenever possible.

For example, the model may produce:

```json
{
  "amount": 1000000
}
```

but a downstream authorization or business-rule layer may reject it even if the schema is syntactically valid.

⭐ **Remember:** **Schema validity does not imply business validity.**

### 9.1.8 Tool Result Normalization

🧠 **Simple Understanding:** Normalization converts different raw tool responses into a consistent representation for the agent.

Suppose two backend systems return:

```text
System A → {"status": "COMPLETE"}
System B → {"state": "completed"}
```

Normalize both to:

```json
{
  "status": "completed"
}
```

Benefits:

* Consistent agent behavior.
* Simpler prompts.
* Easier evaluation.
* Easier logging.
* Reduced provider-specific complexity.

---

## 9.2 Tool Routing

### 9.2.1 Selecting Among Tools

🧠 **Simple Understanding:** Tool routing determines which available capability should handle the user's current need.

Example:

```text
User:
"Where is my order?"

        ↓

Potential tools:
├── get_order
├── cancel_order
├── create_order
└── refund_order

        ↓

Correct route:
get_order
```

Routing can depend on:

* User intent.
* Required information.
* Tool descriptions.
* Current state.
* Permissions.
* Context.
* Tool availability.

### 9.2.2 Tool Namespacing

🧠 **Simple Understanding:** Namespacing organizes tools into explicit domains.

Example:

```text
payments.get_balance
payments.create_payment
payments.refund_payment

orders.get_order
orders.cancel_order
orders.create_order
```

This is useful when many tools exist.

Benefits:

* Reduces naming collisions.
* Improves discoverability.
* Clarifies domains.
* Helps organize permissions.

### 9.2.3 Tool Grouping

Group related tools:

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
├── find_event
├── create_event
└── cancel_event
```

Tool grouping can reduce cognitive and routing complexity when the toolset becomes large.

### 9.2.4 Dynamic Tool Loading

🧠 **Simple Understanding:** Instead of exposing every tool to the model on every request, load relevant tools when needed.

```text
User Request
    ↓
Determine Domain
    ↓
Load Relevant Tools
    ↓
Agent
```

For example:

```text
Travel request
   ↓
Load:
flight_search
hotel_search
calendar
```

instead of loading hundreds of unrelated tools.

Potential benefits:

* Lower tool-selection complexity.
* Smaller model-visible tool space.
* Better routing precision.
* Easier permission enforcement.

### 9.2.5 Tool Catalogs

A tool catalog stores information about available tools.

Example:

| Tool             | Domain   | Risk      | Description           |
| ---------------- | -------- | --------- | --------------------- |
| `get_order`      | Orders   | Read      | Retrieve order status |
| `cancel_order`   | Orders   | Write     | Cancel an order       |
| `create_payment` | Payments | High-risk | Initiate payment      |
| `search_flights` | Travel   | Read      | Search flights        |

A catalog can support:

* Discovery.
* Filtering.
* Routing.
* Permission checks.
* Tool lifecycle management.

### 9.2.6 Tool Discovery

🧠 **Simple Understanding:** Tool discovery is the process through which the agent or orchestration layer finds which capabilities are available and relevant.

```text
Goal
 ↓
Search Catalog
 ↓
Relevant Tools
 ↓
Load Tool Schemas
 ↓
Agent Decides
```

This becomes increasingly important as tool counts grow.

### 9.2.7 Tool Relevance Filtering

Before presenting tools to the model, filter obviously irrelevant ones.

Example:

```text
User asks about:
"Refund my order"

Remove:
├── calendar tools
├── weather tools
├── file-formatting tools
└── irrelevant search tools

Keep:
├── get_order
├── refund_order
└── payment_status
```

⭐ **Key Insight:** Reducing the candidate tool set can improve routing quality and lower unnecessary tool-selection complexity.

---

## 9.3 Execution Models

### 9.3.1 Single Tool Calls

🧠 **Simple Understanding:** One user task requires one tool invocation.

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

Best when the action is:

* Simple.
* Independent.
* Easily verifiable.

### 9.3.2 Sequential Tool Calls

Some tasks require one call before another.

```text
get_customer()
      ↓
get_orders(customer_id)
      ↓
get_order_details(order_id)
```

The output of one operation becomes the input to the next.

### 9.3.3 Parallel Tool Calls

Independent operations can run concurrently.

```text
                Request
                   ↓
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   Search A    Search B    Search C
        │          │          │
        └──────────┼──────────┘
                   ▼
                Merge
```

Example:

> "Compare prices from three providers."

The searches may be independent and therefore parallelizable.

Benefits:

* Lower total latency.
* Better throughput.

Trade-offs:

* Higher instantaneous resource usage.
* More complex failure handling.
* More difficult ordering.

### 9.3.4 Dependent Calls

A dependent call cannot run until an earlier result exists.

Example:

```text
Search customer
      ↓
Customer ID
      ↓
Retrieve subscriptions
      ↓
Subscription ID
      ↓
Cancel subscription
```

Dependencies should be explicit rather than accidentally encoded through fragile model behavior.

### 9.3.5 Fan-Out / Fan-In

🧠 **Simple Understanding:** Fan-out sends work to multiple independent operations; fan-in collects the results.

```text
                Request
                   │
          ┌────────┼────────┐
          ▼        ▼        ▼
         A         B        C
          │        │        │
          └────────┼────────┘
                   ▼
                  Merge
```

Useful for:

* Multi-source search.
* Comparison.
* Batch operations.
* Parallel data gathering.

### 9.3.6 Partial Failure

Not every parallel operation necessarily succeeds.

```text
A → success
B → success
C → timeout
```

The system must decide whether:

* Continue with partial information.
* Retry C.
* Use a fallback.
* Abort the overall task.
* Ask the user.

⭐ **Key Point:** "One tool failed" does not automatically mean "the whole task failed."

### 9.3.7 Compensation

🧠 **Simple Understanding:** Compensation is a corrective action used when an earlier action succeeded but a later action failed.

Example:

```text
Reserve inventory
      ↓
Charge payment
      ↓
Shipment creation fails
      ↓
Compensate
      ↓
Release inventory / refund payment
```

Compensation is not identical to a database rollback. An external action may not be perfectly reversible.

⚠️ **Important:** Never assume a compensating action is harmless or exactly equivalent to undoing the original action.

---

## 9.4 Tool Reliability

### 9.4.1 Validation

Validation should occur at multiple layers.

```text
Model Output
    ↓
Schema Validation
    ↓
Business Validation
    ↓
Authorization
    ↓
Execution
```

Examples:

* Type validation.
* Required fields.
* Range validation.
* Resource existence.
* User ownership.
* State constraints.

### 9.4.2 Retries

🧠 **Simple Understanding:** Retries repeat a failed operation when the failure is potentially temporary.

Appropriate retry candidates may include:

* Transient network failures.
* Temporary service unavailability.
* Rate limiting, with suitable backoff.

Avoid blind retries for:

* Invalid input.
* Authorization failures.
* Permanent business-rule failures.
* Unknown side effects.

A common strategy is exponential backoff:

```text
Retry 1 → short delay
Retry 2 → longer delay
Retry 3 → longer delay
```

### 9.4.3 Timeouts

🧠 **Simple Understanding:** A timeout prevents a tool call from waiting indefinitely.

Without a timeout:

```text
Agent
 ↓
Tool
 ↓
... hangs ...
 ↓
Entire task stalls
```

Timeout policies should consider:

* Tool type.
* Expected latency.
* User experience.
* Retry behavior.
* Downstream capacity.

### 9.4.4 Circuit Breakers

🧠 **Simple Understanding:** A circuit breaker temporarily stops sending calls to an unhealthy dependency.

Conceptually:

```text
Healthy
  ↓
Failures increase
  ↓
Circuit opens
  ↓
Calls rejected / fast-failed
  ↓
Dependency recovers
  ↓
Circuit closes
```

Benefits:

* Protects the system from cascading failures.
* Prevents repeated calls to an unhealthy service.
* Reduces wasted latency.

### 9.4.5 Fallbacks

A fallback provides an alternate path when a primary tool fails.

Example:

```text
Primary search API
      ↓
Failure
      ↓
Secondary search API
      ↓
Result
```

A fallback should be evaluated for semantic equivalence.

A fallback that returns different or lower-quality information can itself introduce correctness problems.

### 9.4.6 Idempotency

🧠 **Simple Understanding:** An idempotent operation can safely be repeated without unintentionally producing additional effects.

Example:

```text
set_status("shipped")
```

may be safely repeated.

Compare with:

```text
charge_card($100)
```

Repeated execution could charge the customer twice.

⭐ **Key Point:** Retry policy and idempotency must be designed together.

A common pattern for side-effectful operations is an idempotency key:

```json
{
  "payment_id": "PAY-123",
  "idempotency_key": "req-abc-789"
}
```

Repeated requests using the same key can be recognized as the same logical operation by a supporting backend.

### 9.4.7 Result Verification

🧠 **Simple Understanding:** Verification checks whether the tool's reported result corresponds to the desired real-world state.

Example:

```text
Agent → create_ticket()
          ↓
Response: "success"
          ↓
Verify ticket exists
          ↓
Confirmed
```

This is especially important for high-value or side-effectful actions.

### 9.4.8 Side-Effect Classification

Before allowing a tool to run, understand what it can change.

| Category        | Example                               | Risk      |
| --------------- | ------------------------------------- | --------- |
| Read            | `get_order`                           | Low       |
| Low-risk write  | Add a draft note                      | Moderate  |
| High-risk write | Modify account settings               | High      |
| Financial       | Transfer money                        | Very high |
| Irreversible    | Delete data                           | Very high |
| Sensitive       | Access protected personal information | High      |

⭐ **Key Point:** Risk classification should influence authorization, confirmation, logging, and recovery behavior.

---

## 9.5 Tool Permissions

The action system should classify tools according to risk:

```text
Read-only
   ↓
Low-risk write
   ↓
High-risk write
   ↓
Irreversible / Financial / Sensitive
```

The higher the risk, the stronger the control requirements should become.

### 9.5.1 Read-Only Tools

🧠 **Simple Understanding:** Read-only tools retrieve information without changing external state.

Examples:

```text
get_balance()
get_order()
search_documents()
get_weather()
```

Typical controls:

* Authentication.
* Authorization.
* Data filtering.
* Audit logging.

### 9.5.2 Low-Risk Writes

These modify state but have limited consequences.

Examples:

```text
save_draft()
add_internal_note()
update_preference()
```

Possible controls:

* Permission checks.
* Input validation.
* Audit logs.
* Optional confirmation depending on the application.

### 9.5.3 High-Risk Writes

Examples:

```text
change_permissions()
delete_customer()
modify_billing()
```

These generally require stronger:

* Authorization.
* Validation.
* Logging.
* Confirmation.
* Rate limits.
* Rollback/compensation planning.

### 9.5.4 Irreversible, Financial, and Sensitive Actions

These represent the strongest risk class.

Examples:

```text
send_money()
delete_account()
publish_public_content()
reveal_sensitive_information()
```

The system may require:

```text
User Intent
   ↓
Authorization
   ↓
Policy Check
   ↓
Explicit Confirmation
   ↓
Tool Execution
   ↓
Verification
   ↓
Audit Record
```

### 9.5.5 Authorization and Approval

🧠 **Simple Understanding:** Authorization answers **"Is this actor allowed to perform this action?"** Approval answers **"Has the required human or policy checkpoint approved this action?"**

These are different controls.

Example:

```text
Agent wants to refund $5,000

Authorization?
→ User/service may have permission.

Approval?
→ Policy may require human approval above a threshold.
```

### 9.5.6 Permission Enforcement Architecture

```text
                    Agent
                      │
                      ▼
                 Tool Request
                      │
                      ▼
                Schema Check
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
                ┌─────┴─────┐
                ▼           ▼
             Low Risk    High Risk
                │           │
                │      Approval Required
                │           │
                └─────┬─────┘
                      ▼
                  Execution
                      │
                      ▼
                  Verification
                      │
                      ▼
                   Audit
```

⭐ **Key Point:** The model should not be the final authority on whether an action is allowed.

---

## 9.6 Tool Calling Architecture

### 9.6.1 Tool Definition Layer

Contains:

```text
Tool name
Description
Input schema
Output schema
Examples
Constraints
Risk metadata
Permission metadata
```

### 9.6.2 Routing Layer

Responsible for:

* Identifying candidate tools.
* Filtering irrelevant tools.
* Loading relevant tools.
* Selecting tools.
* Building tool plans.

```text
Goal
 ↓
Intent / Task Understanding
 ↓
Tool Catalog
 ↓
Relevance Filter
 ↓
Candidate Tools
 ↓
Tool Selection
```

### 9.6.3 Execution Layer

Responsible for:

* Validating arguments.
* Dispatching calls.
* Managing dependencies.
* Parallel execution.
* Timeouts.
* Retries.
* Result capture.

### 9.6.4 Reliability Layer

Provides:

```text
Validation
Retries
Timeouts
Circuit breakers
Fallbacks
Idempotency
Compensation
```

### 9.6.5 Authorization Layer

Responsible for:

* Authentication.
* Authorization.
* Tenant isolation.
* Policy enforcement.
* Approval workflows.
* Risk controls.

### 9.6.6 Verification Layer

Responsible for asking:

> Did the intended action actually occur?

```text
Tool Request
    ↓
Execute
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

1. **Tool calling is an action boundary.** The LLM proposes an action; application code should decide whether that action is valid and permitted.

2. **Schemas reduce ambiguity but do not provide complete safety.** A request can satisfy the type schema while still violating business rules or authorization requirements.

3. **Tool descriptions affect routing quality.** A vague description can cause the correct tool to be ignored or the wrong tool to be selected.

4. **Risk should be attached to tools.** Read-only retrieval and financial operations should not use identical execution controls.

5. **Retries can be dangerous for side effects.** Retry semantics must account for idempotency and the possibility that the first call actually succeeded but its response was lost.

6. **Verification matters after execution.** A successful API response may not be sufficient evidence that the intended business outcome exists.

7. **Tool systems should degrade gracefully.** Partial failure, fallback, compensation, and human escalation are part of production action-system design.

---

# 9.8 Common Mistakes

⚠️ **Common Mistakes**

| Mistake                                              | Correct Understanding                                                              |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| "The model can safely call any valid tool."          | Valid syntax does not imply authorization or business validity.                    |
| "Schema validation is enough."                       | Business rules, permissions, and state constraints require additional validation.  |
| "Retries are always good."                           | Retries can duplicate side effects unless the operation is safely retryable.       |
| "The tool said success, so we're done."              | High-value actions should be verified against actual state where appropriate.      |
| "All tools can be exposed at once."                  | Large tool sets increase routing complexity and unnecessary exposure.              |
| "Descriptions are documentation only."               | Tool descriptions influence model routing behavior.                                |
| "Parallel calls are always faster."                  | Dependencies and resource constraints may prevent safe parallel execution.         |
| "Rollback is always possible."                       | Many external actions are only partially reversible or irreversible.               |
| "Authorization can be done by the prompt."           | Security controls must be enforced outside the model's natural-language reasoning. |
| "A fallback is automatically equivalent."            | Different tools may have different semantics or data quality.                      |
| "Fewer tool calls always means better."              | Correctness and reliable completion matter more than minimizing calls alone.       |
| "A successful final message proves task completion." | The external environment should be checked for side-effectful workflows.           |

---

# 9.9 Common Confusions

🔍 **Common Confusions**

| Concept A      | Concept B            | Key Difference                                                                                                                 |
| -------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Tool schema    | Business validation  | Schema checks interface structure; business validation checks whether the operation is allowed/meaningful.                     |
| Authentication | Authorization        | Authentication identifies the actor; authorization determines what the actor may do.                                           |
| Authorization  | Approval             | Authorization permits an action; approval represents an additional required decision/checkpoint.                               |
| Tool routing   | Tool execution       | Routing chooses the capability; execution actually invokes it.                                                                 |
| Retry          | Fallback             | Retry repeats the same path; fallback uses an alternate path.                                                                  |
| Rollback       | Compensation         | Rollback reverses a transaction within a system; compensation performs another action to mitigate an external effect.          |
| Sequential     | Parallel execution   | Sequential operations wait on prior results; parallel operations run independent work concurrently.                            |
| Tool discovery | Tool routing         | Discovery finds available capabilities; routing chooses the appropriate one.                                                   |
| Idempotency    | Deduplication        | Idempotency makes repeated logical requests safe; deduplication prevents duplicate processing/data.                            |
| Verification   | Validation           | Validation checks whether an action/request is acceptable; verification checks whether the intended outcome actually occurred. |
| Read-only      | Low-risk write       | Read-only changes no state; low-risk write modifies state with comparatively limited consequences.                             |
| Agent decision | System authorization | The agent can propose an action; the system determines whether that action is permitted.                                       |

---

# 9.10 Practical Applications

🛠️ **Practical Applications**

| Use Case                   | Useful Tool-System Concepts                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| Customer-support agent     | Tool routing, typed arguments, retries, permissions                     |
| Payment agent              | Risk classification, idempotency, authorization, approval, verification |
| Scheduling agent           | Tool discovery, sequential calls, conflict validation                   |
| E-commerce assistant       | Order tools, inventory checks, compensation                             |
| Enterprise knowledge agent | Search tools, dynamic loading, tenant-aware authorization               |
| Coding agent               | File tools, shell tools, dependency ordering, sandboxing                |
| Browser agent              | Navigation tools, sequential execution, environment verification        |
| Workflow automation        | Fan-out/fan-in, retries, partial failure, compensation                  |
| Data operations agent      | Typed inputs, validation, high-risk action controls                     |
| Finance operations         | Strong authorization, approvals, auditability, idempotency              |

---

# 9.11 Important Terms

📌 **Important Terms**

| Term             | Simple Meaning                               | Why It Matters                          |
| ---------------- | -------------------------------------------- | --------------------------------------- |
| Tool             | External capability callable by an AI system | Gives the model access to actions/data  |
| Tool Schema      | Formal interface contract                    | Makes inputs structured and validatable |
| Tool Description | Human/model-facing explanation               | Influences tool selection               |
| Typed Input      | Input constrained by a type                  | Prevents malformed arguments            |
| Typed Output     | Structured result contract                   | Makes downstream processing predictable |
| Tool Routing     | Selecting an appropriate tool                | Core to agent behavior                  |
| Tool Catalog     | Registry of available tools                  | Enables discovery and management        |
| Tool Discovery   | Finding relevant capabilities                | Important with large toolsets           |
| Namespacing      | Organizing tools by domain                   | Reduces ambiguity and collisions        |
| Sequential Call  | Call after a previous result                 | Handles dependencies                    |
| Parallel Call    | Independent calls run together               | Reduces latency                         |
| Fan-Out          | Split work across multiple operations        | Useful for parallel workflows           |
| Fan-In           | Merge outputs from parallel operations       | Recombines distributed work             |
| Partial Failure  | Some operations succeed while others fail    | Requires recovery policy                |
| Compensation     | Corrective action after partial success      | Handles external side effects           |
| Retry            | Repeat a failed call                         | Handles transient failures              |
| Timeout          | Maximum wait duration                        | Prevents indefinite stalls              |
| Circuit Breaker  | Temporarily stop unhealthy dependency calls  | Prevents cascading failures             |
| Fallback         | Alternate execution path                     | Improves resilience                     |
| Idempotency      | Safe repeated logical execution              | Critical for retries                    |
| Verification     | Confirm real-world outcome                   | Prevents false completion               |
| Side Effect      | Change caused outside the model              | Determines risk                         |
| Authentication   | Identify actor                               | Foundation for access control           |
| Authorization    | Determine permitted actions                  | Security boundary                       |
| Approval         | Additional required decision                 | Controls high-risk operations           |
| Audit Log        | Record action history                        | Supports security and debugging         |
| Compensation     | Mitigate a completed external action         | Important for distributed workflows     |

---

# 9.12 Quick Revision

⚡ **Quick Revision**

1. A **tool schema** defines the machine-readable interface between the model and application.
2. **Required/optional parameters** determine what information the tool needs.
3. **Typed inputs/outputs** make tool interactions predictable and validatable.
4. **Tool descriptions and examples influence routing behavior.**
5. **Tool routing** chooses the appropriate capability from available tools.
6. **Dynamic loading and relevance filtering** help manage large toolsets.
7. **Sequential execution** handles dependencies; **parallel execution** handles independent work.
8. **Fan-out/fan-in** supports parallel multi-source workflows.
9. **Retries require awareness of idempotency**, especially for side-effectful operations.
10. **Timeouts, circuit breakers, and fallbacks** improve reliability.
11. **Validation is not authorization.**
12. **Authorization is not approval.**
13. **Tool risk classification** should influence execution controls.
14. **Verification checks whether the intended real-world outcome actually happened.**
15. The model should **propose actions**, but deterministic system controls should decide whether those actions can execute.

---

# 9.13 Interview Preparation

## 9.13.1 Level 1 — Fundamentals

### Q1. What is tool calling?

**Model Answer:**
Tool calling is a mechanism that allows an AI model to request execution of a defined external capability. The model produces a structured tool request, and the application validates, authorizes, executes, and returns the result to the model.

### Q2. Why are tool schemas important?

**Model Answer:**
Schemas define the contract between the AI system and the tool. They specify fields, types, required parameters, and constraints, allowing the application to validate model-generated arguments before execution.

### Q3. What is tool routing?

**Model Answer:**
Tool routing is the process of determining which tool should handle a particular user goal. It depends on the user's intent, tool descriptions, available capabilities, context, permissions, and sometimes current environment state.

### Q4. What is the difference between required and optional parameters?

**Model Answer:**
Required parameters must be supplied for the operation to be meaningful or valid. Optional parameters modify behavior when present and have safe defaults when omitted.

### Q5. Why are typed inputs useful?

**Model Answer:**
Typed inputs constrain the structure of model-generated arguments. They help detect malformed requests such as passing a string where an integer is required or omitting required fields.

### Q6. What is a retry?

**Model Answer:**
A retry repeats a failed tool invocation when the failure may be transient. Retries should be bounded and designed around timeout behavior, backoff, and the tool's idempotency characteristics.

### Q7. What is idempotency?

**Model Answer:**
Idempotency means that repeating the same logical request does not unintentionally create additional effects. It is important for safely retrying operations such as payments or record creation.

### Q8. Why classify tools by risk?

**Model Answer:**
Different actions have different consequences. A read-only search does not require the same controls as deleting data or transferring money. Risk classification allows stronger authorization, confirmation, audit, and verification requirements for more dangerous tools.

### Q9. What is tool result normalization?

**Model Answer:**
Result normalization transforms different raw backend responses into a consistent structure for the agent and application. This reduces backend-specific complexity and makes downstream reasoning and evaluation more predictable.

---

## 9.13.2 Level 2 — Conceptual Understanding

### Q1. Why is schema validation not enough for safe tool execution?

**Model Answer:**
Schema validation checks whether the request has the correct structure and types. It does not necessarily establish that the action is permitted, that the requested resource belongs to the user, or that the operation is valid under business rules.

### Q2. What is the difference between authentication and authorization?

**Model Answer:**
Authentication determines who the actor is. Authorization determines what that actor is allowed to do. An authenticated user can still be unauthorized to perform a particular tool action.

### Q3. Why can retries create duplicate actions?

**Model Answer:**
The initial request may have succeeded at the backend while its response was lost or timed out. A retry could then execute the action a second time. Idempotency keys or other deduplication mechanisms can prevent duplicate effects when supported.

### Q4. When should tools execute sequentially rather than in parallel?

**Model Answer:**
Use sequential execution when a later call depends on the output or state established by an earlier call. Parallel execution is appropriate when operations are independent.

### Q5. What is compensation?

**Model Answer:**
Compensation is a corrective action used when part of a multi-step workflow has succeeded but a later operation fails. It attempts to mitigate the resulting inconsistent state, although external actions may not be perfectly reversible.

### Q6. Why is result verification important?

**Model Answer:**
A tool may return a success response without guaranteeing that the final business outcome exists or remains valid. Verification checks the actual external state, which is particularly important for side-effectful actions.

### Q7. Why dynamically load tools?

**Model Answer:**
Large toolsets increase routing complexity and expose the model to many irrelevant capabilities. Dynamic loading limits the visible tool space to relevant capabilities and can improve routing precision while simplifying permission management.

### Q8. Why should authorization exist outside the model?

**Model Answer:**
The model is probabilistic and can generate incorrect or maliciously influenced decisions. Security controls need deterministic enforcement so an unsafe model output cannot bypass authorization.

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
├── Constraints
├── Risk Classification
└── Permission Requirements
```

Then enforce:

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

This separates model-generated intent from actual execution authority.

### Q2. How would you safely implement a payment tool?

**Model Answer:**
I would treat payment as a high-risk operation. The system would validate amount and currency, authenticate the actor, authorize the specific operation, enforce business limits, require approval or confirmation where policy requires it, use idempotency to prevent duplicate charges, execute through a controlled payment service, verify the result, and record an audit event.

### Q3. How would you handle a tool that times out?

**Model Answer:**
First determine whether the operation is retryable and whether it is idempotent. Then apply a bounded retry policy with appropriate backoff when safe. If the result remains unknown, do not blindly execute another side effect; instead check the external state or use a reconciliation mechanism.

### Q4. How would you handle three parallel tool calls where one fails?

**Model Answer:**
I would classify the failed operation and determine whether the overall task can proceed with partial information. Depending on business requirements, the system could retry the failed call, use a fallback, mark the result partial, ask the user, or abort the workflow. The policy should be explicit rather than left entirely to the model.

### Q5. How would you manage hundreds of tools?

**Model Answer:**
I would use a catalog with domains, descriptions, schemas, risk metadata, and permissions. Then use routing or retrieval to identify relevant tools and dynamically load only the necessary schemas. Namespacing and grouping would further reduce ambiguity.

### Q6. How would you debug incorrect tool selection?

**Model Answer:**
Inspect the user's intent, candidate tools presented to the model, tool descriptions, tool names, routing logic, previous context, permissions, and the final tool choice. I would also build evaluation cases for common confusions and measure tool-selection accuracy independently from overall task success.

### Q7. How would you design compensation for a multi-step transaction?

**Model Answer:**
First identify which steps are externally observable and reversible. Then define explicit compensation actions for recoverable failures and reconciliation procedures for ambiguous or irreversible states. Compensation should itself be authorized and monitored because it is another external action.

---

## 9.13.4 Level 4 — Advanced / Deep Understanding

### Q1. Why can a timeout create uncertainty instead of a simple failure?

**Model Answer:**
A timeout means the caller did not receive a result within the expected period. It does not necessarily mean the backend did not execute the action. The operation may have succeeded but the response may have been delayed or lost. Therefore side-effectful timeout handling often requires reconciliation or state verification before retrying.

### Q2. Why is tool output normalization important for agent reliability?

**Model Answer:**
Agents perform better when equivalent concepts are represented consistently. If different backends use different field names, status values, or nesting structures, the model must interpret unnecessary variations. Normalization moves that complexity into deterministic application code.

### Q3. Why can exposing more tools reduce agent performance?

**Model Answer:**
A larger candidate set increases the decision space and creates more opportunities for ambiguous or incorrect tool selection. Irrelevant tools also consume model-visible context and can make descriptions compete with one another.

### Q4. Why should risk metadata be part of the tool system?

**Model Answer:**
The same routing and execution machinery should not blindly treat a read operation and a financial transaction as equivalent. Risk metadata allows the orchestration layer to apply different authorization, confirmation, logging, rate limiting, and verification policies.

### Q5. What happens if the model selects the correct tool but gives incorrect arguments?

**Model Answer:**
The tool selection is correct, but execution can still fail or cause the wrong effect. This is why argument correctness must be evaluated independently and why tool schemas, validation, business rules, and state checks are necessary.

### Q6. Why is compensation not equivalent to rollback?

**Model Answer:**
Rollback usually assumes transactional control over the state being changed. External tools may span independent systems where no atomic transaction exists. Compensation instead performs a new action intended to mitigate an earlier action, and the compensating action itself can fail.

### Q7. Why should tool execution be observable?

**Model Answer:**
Tool interactions are critical parts of agent behavior. Logging tool selection, arguments, results, latency, retries, failures, and authorization decisions makes it possible to debug incorrect actions, evaluate trajectories, investigate security incidents, and optimize reliability.

---

## 9.13.5 Level 5 — Scenario-Based Questions

### Scenario 1 — Double Payment Risk

A payment tool times out after the agent requests a charge. The agent does not know whether the payment succeeded.

**Question:** What should happen next?

**Model Answer:**

Do not blindly retry.

```text
Timeout
  ↓
Payment state unknown
  ↓
Check transaction using idempotency key / payment status
  ↓
Known successful?
 ├── Yes → return verified success
 ├── No  → safely retry if policy allows
 └── Unknown → reconcile / escalate
```

The key issue is distinguishing **request failure** from **operation failure**.

---

### Scenario 2 — Wrong Tool Selected

The user says:

> "Cancel my order."

The agent calls `get_order()` instead of `cancel_order()`.

**Question:** How would you improve the system?

**Model Answer:**

I would inspect:

* Tool descriptions.
* Tool names.
* Candidate tool set.
* Routing logic.
* User intent examples.
* Confusing neighboring tools.

Then improve the routing layer and create evaluation cases specifically distinguishing:

```text
get_order
cancel_order
refund_order
modify_order
```

Tool-selection accuracy should be measured independently.

---

### Scenario 3 — Large Tool Ecosystem

An enterprise agent has 500 tools across HR, finance, sales, IT, and operations.

**Question:** Should all 500 tools be exposed to the model?

**Model Answer:**
Usually not. I would maintain a structured tool catalog and use domain classification, discovery, permissions, and relevance filtering to identify a much smaller candidate set. Then dynamically load the relevant tool schemas.

This reduces routing complexity and limits unnecessary tool exposure.

---

### Scenario 4 — Partial Failure

An agent compares three shipping providers.

```text
Provider A → success
Provider B → success
Provider C → timeout
```

**Question:** Should the entire task fail?

**Model Answer:**
Not necessarily. The orchestration layer should apply a defined partial-failure policy. It could retry C, use a fallback, return a partial comparison, or ask the user whether incomplete results are acceptable. The correct behavior depends on the task requirements.

---

### Scenario 5 — High-Risk Action

An agent wants to delete a customer's account.

**Question:** How should the workflow differ from a read-only search?

**Model Answer:**

```text
Agent proposes delete_account
        ↓
Schema validation
        ↓
Identity / tenant check
        ↓
Authorization
        ↓
Risk classification
        ↓
Explicit confirmation / approval
        ↓
Execution
        ↓
Verification
        ↓
Audit record
```

The model should never be the sole authorization mechanism.

---

# 9.13.6 Knowledge Check

🧠 **Knowledge Check**

If you can explain these naturally in your own words, you understand Layer 7:

* What a tool is.
* What a tool schema provides.
* Why descriptions affect routing.
* Why typed inputs and outputs matter.
* Why schema validity does not imply business validity.
* How tool routing works.
* Why large toolsets benefit from catalogs and dynamic loading.
* When to execute tools sequentially.
* When to execute tools in parallel.
* What fan-out and fan-in mean.
* What partial failure means.
* Why compensation exists.
* Why retries can be dangerous.
* How idempotency protects side-effectful operations.
* Why timeouts are necessary.
* What circuit breakers do.
* Why result verification matters.
* Why read and write operations need different controls.
* The difference between authentication, authorization, and approval.
* Why authorization must be enforced outside the model.

---

# 9.13.7 Follow-up Questions

### Basic Question

**What is tool calling?**

→ Why is it needed?
→ How is a tool described?
→ How is the tool selected?
→ How are arguments validated?
→ Who authorizes execution?
→ How is the result verified?

### Basic Question

**What is tool routing?**

→ How are tools discovered?
→ How do you reduce the candidate set?
→ What is namespacing?
→ What is dynamic loading?
→ How do permissions affect routing?

### Basic Question

**How do tools execute?**

→ Sequential or parallel?
→ Which calls are dependent?
→ What is fan-out/fan-in?
→ What happens on partial failure?
→ What is compensation?

### Basic Question

**How do you make tools reliable?**

→ Validation?
→ Retries?
→ Timeouts?
→ Circuit breakers?
→ Fallbacks?
→ Idempotency?
→ Verification?

### Basic Question

**How do you secure tools?**

→ Authentication?
→ Authorization?
→ Tenant isolation?
→ Risk classification?
→ Approval?
→ Audit logging?

---

# 9.13.8 Common Confusion Questions

### Q1. Is a tool schema a security mechanism?

**Model Answer:**
No. It validates the structure of a request but does not replace authorization, policy enforcement, or resource ownership checks.

### Q2. Is a retry the same as a fallback?

**Model Answer:**
No. A retry repeats the same operation. A fallback changes the execution path to another mechanism or provider.

### Q3. Is authorization the same as user confirmation?

**Model Answer:**
No. Authorization determines whether the actor may perform the action. Confirmation is an additional user or policy checkpoint indicating that the action should proceed.

### Q4. Is tool selection the same as tool execution?

**Model Answer:**
No. Selection determines which capability is appropriate; execution performs the actual external operation.

### Q5. Is verification the same as validation?

**Model Answer:**
No. Validation checks whether a proposed operation is acceptable. Verification checks whether the intended result actually happened.

---

# 9.13.9 Deep / Trick Questions

### ⚠️ Deeper Question

**The model selected the correct tool and generated a schema-valid request. Can you safely execute it?**

**Correct Understanding:**
Not necessarily. The request may violate business rules, authorization, tenant boundaries, resource ownership, rate limits, or risk policies. Schema validity is only one layer of validation.

---

### ⚠️ Deeper Question

**Why can retrying a timed-out request be dangerous even when the API returned no response?**

**Correct Understanding:**
Because the backend may have executed the action but failed to return a response. Retrying a non-idempotent operation can therefore duplicate its side effect.

---

### ⚠️ Deeper Question

**Why is "success" from a tool not always enough?**

**Correct Understanding:**
The tool response may confirm request processing rather than the final real-world state. For important actions, the system should verify the resulting state independently where feasible.

---

### ⚠️ Deeper Question

**Why can a bigger tool catalog make an agent worse?**

**Correct Understanding:**
More tools increase the routing decision space, create more similar descriptions, consume model-visible context, and increase opportunities for incorrect selection. Retrieval and dynamic loading can reduce this complexity.

---

### ⚠️ Deeper Question

**If an action is authorized, why might approval still be required?**

**Correct Understanding:**
Authorization answers whether the actor is permitted to act. Policy may still require a human or explicit user approval for high-risk actions such as large financial transactions or irreversible operations.

---

### ⚠️ Deeper Question

**Why is compensation difficult in distributed action systems?**

**Correct Understanding:**
Different tools may operate in separate systems with different transactional guarantees. There may be no atomic rollback across them, and the compensating action itself can fail or have different semantics from the original action.

---

# 9.14 Top Questions You MUST Know

⭐ **Top Questions You MUST Know**

1. What is tool calling and why is it needed?
2. How do you design a good tool schema?
3. Why are required/optional parameters and strong typing important?
4. How do tool descriptions influence routing?
5. How would you route an agent among hundreds of tools?
6. Why use tool catalogs, namespacing, and dynamic loading?
7. When should tool calls execute sequentially vs in parallel?
8. What are fan-out/fan-in and partial failure?
9. What is compensation, and how does it differ from rollback?
10. How do retries interact with idempotency?
11. How would you handle a timeout on a side-effectful tool?
12. Why are validation, authorization, and verification separate concerns?
13. How would you classify tools by risk?
14. How would you secure a high-risk or financial tool?
15. How would you verify that an agent's external action actually succeeded?

---

# 9.15 Interview Readiness Checklist

🎯 **Interview Readiness Checklist**

| Skill                           | Can I explain it? |
| ------------------------------- | :---------------: |
| Tool calling fundamentals       |         ☐         |
| Tool schema design              |         ☐         |
| Required vs optional parameters |         ☐         |
| Typed inputs                    |         ☐         |
| Typed outputs                   |         ☐         |
| Tool descriptions               |         ☐         |
| Tool examples                   |         ☐         |
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
| Approval workflows              |         ☐         |
| Risk classification             |         ☐         |
| Audit logging                   |         ☐         |
| Tenant isolation                |         ☐         |
| Production recovery             |         ☐         |
| Tool observability              |         ☐         |
| High-risk action design         |         ☐         |

---

# 9.16 What You Should Be Able to Explain

🧠 **What You Should Be Able to Explain**

By the end of Layer 7, you should be able to explain:

* What tool calling is and why agentic systems need it.
* How a tool schema acts as an interface contract.
* How required and optional parameters should be chosen.
* Why typed inputs and outputs reduce ambiguity.
* How descriptions and examples influence model tool selection.
* Why schema constraints are not a replacement for business validation.
* Why tool results should often be normalized.
* How routing works across multiple tools.
* Why namespaces and tool groups help organize large tool ecosystems.
* How tool catalogs support discovery and lifecycle management.
* Why dynamic tool loading can improve routing quality.
* Why relevance filtering matters.
* When tools should run sequentially.
* When they can run in parallel.
* How dependent tool calls work.
* What fan-out/fan-in means.
* How to design for partial failure.
* What compensation means in distributed action workflows.
* How validation protects the execution boundary.
* When retries are appropriate.
* Why retries can cause duplicate side effects.
* How idempotency makes retries safer.
* Why timeouts are essential.
* How circuit breakers protect unhealthy dependencies.
* When fallbacks are appropriate.
* Why result verification matters.
* How to classify tools by side-effect risk.
* The difference between read-only, low-risk write, high-risk write, financial, sensitive, and irreversible operations.
* The difference between authentication and authorization.
* The difference between authorization and approval.
* Why high-risk actions require stronger controls.
* Why the model should not be the final security authority.
* How a production tool-calling architecture separates routing, execution, authorization, reliability, and verification.
* How to design tool systems that are reliable under retries, timeouts, failures, and ambiguous outcomes.
* How to evaluate tool selection and tool argument correctness.
* How to build safe action systems where AI proposes actions but deterministic infrastructure controls execution.

## ⚡ Final Mental Model

```text
                         USER GOAL
                             │
                             ▼
                    ┌─────────────────┐
                    │   AI / AGENT    │
                    │                 │
                    │ Reason / Decide │
                    └────────┬────────┘
                             │
                     Tool Selection
                             │
                             ▼
                    ┌─────────────────┐
                    │ TOOL CATALOG    │
                    │                 │
                    │ Discover        │
                    │ Filter          │
                    │ Load            │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ TOOL REQUEST    │
                    │                 │
                    │ Name            │
                    │ Arguments       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   VALIDATION    │
                    │                 │
                    │ Schema          │
                    │ Types           │
                    │ Business Rules  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ AUTHORIZATION   │
                    │                 │
                    │ Identity        │
                    │ Tenant          │
                    │ Permissions     │
                    │ Risk            │
                    └────────┬────────┘
                             │
                    Approval Required?
                       ┌─────┴─────┐
                       │           │
                      Yes          No
                       │           │
                       ▼           │
                 Human / Policy    │
                   Approval        │
                       │           │
                       └─────┬─────┘
                             ▼
                    ┌─────────────────┐
                    │   EXECUTION     │
                    │                 │
                    │ Single          │
                    │ Sequential      │
                    │ Parallel        │
                    │ Fan-out/Fan-in  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  RELIABILITY    │
                    │                 │
                    │ Retry           │
                    │ Timeout         │
                    │ Fallback        │
                    │ Circuit Breaker │
                    │ Idempotency     │
                    │ Compensation    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    RESULT       │
                    │  NORMALIZATION  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  VERIFICATION   │
                    │                 │
                    │ Did the desired │
                    │ state occur?    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   AUDIT / LOG   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ RETURN RESULT   │
                    │    TO AGENT     │
                    └─────────────────┘
```

> **Core principle:** **Tool calling is not simply "letting an LLM call functions." It is the engineering discipline of converting probabilistic model decisions into controlled external actions through schemas, routing, validation, authorization, reliable execution, risk management, and outcome verification.**
