# 📚 Table of Contents

* [19. Layer 17 — Agent Protocol Landscape](#19-layer-17--agent-protocol-landscape)

  * [19.1 Protocol Landscape Overview](#191-protocol-landscape-overview)

    * [19.1.1 MCP](#1911-mcp)
    * [19.1.2 A2A](#1912-a2a)
    * [19.1.3 A2UI](#1913-a2ui)
    * [19.1.4 AG-UI](#1914-ag-ui)
    * [19.1.5 Commerce Protocols](#1915-commerce-protocols)
    * [19.1.6 Payment Protocols](#1916-payment-protocols)
    * [19.1.7 Protocol Landscape Map](#1917-protocol-landscape-map)
  * [19.2 Agent Identity & Trust](#192-agent-identity--trust)

    * [19.2.1 Why Agent Identity Matters](#1921-why-agent-identity-matters)
    * [19.2.2 User Identity vs Agent Identity vs Client Identity](#1922-user-identity-vs-agent-identity-vs-client-identity)
    * [19.2.3 Agent Identity vs Workload / Service Identity](#1923-agent-identity-vs-workload--service-identity)
    * [19.2.4 Delegated Authority](#1924-delegated-authority)
    * [19.2.5 Credential Binding](#1925-credential-binding)
    * [19.2.6 Agent Provenance](#1926-agent-provenance)
    * [19.2.7 Authentication vs Authorization](#1927-authentication-vs-authorization)
    * [19.2.8 Trust Chains](#1928-trust-chains)
    * [19.2.9 Capability-Based Authorization](#1929-capability-based-authorization)
    * [19.2.10 Credential Rotation and Revocation](#19210-credential-rotation-and-revocation)
    * [19.2.11 Identity Discovery and Metadata](#19211-identity-discovery-and-metadata)
    * [19.2.12 Merchant-Side Agent Verification](#19212-merchant-side-agent-verification)
    * [19.2.13 Cross-System Identity Propagation](#19213-cross-system-identity-propagation)
    * [19.2.14 Identity and Authorization Audit Trails](#19214-identity-and-authorization-audit-trails)
    * [19.2.15 CIMD](#19215-cimd)
    * [19.2.16 Web Bot Auth](#19216-web-bot-auth)
    * [19.2.17 Identity & Trust Architecture](#19217-identity--trust-architecture)
  * [19.3 Agentic Commerce & Payments](#193-agentic-commerce--payments)

    * [19.3.1 Why Agentic Commerce Exists](#1931-why-agentic-commerce-exists)
    * [19.3.2 Agentic Commerce Lifecycle](#1932-agentic-commerce-lifecycle)
    * [19.3.3 Product / Catalog Discovery](#1933-product--catalog-discovery)
    * [19.3.4 Merchant and Agent Discovery](#1934-merchant-and-agent-discovery)
    * [19.3.5 Offer and Pricing Retrieval](#1935-offer-and-pricing-retrieval)
    * [19.3.6 Cart Construction](#1936-cart-construction)
    * [19.3.7 Checkout Orchestration](#1937-checkout-orchestration)
    * [19.3.8 Purchase Authorization](#1938-purchase-authorization)
    * [19.3.9 Delegated Payment Authority](#1939-delegated-payment-authority)
    * [19.3.10 Mandates and Pre-Authorization](#19310-mandates-and-pre-authorization)
    * [19.3.11 Spending Limits and Budgets](#19311-spending-limits-and-budgets)
    * [19.3.12 Per-Transaction and Cumulative Limits](#19312-per-transaction-and-cumulative-limits)
    * [19.3.13 User Consent and Revocation](#19313-user-consent-and-revocation)
    * [19.3.14 Payment Intent vs Payment Execution](#19314-payment-intent-vs-payment-execution)
    * [19.3.15 Merchant-Side Agent Verification](#19315-merchant-side-agent-verification)
    * [19.3.16 Transaction Confirmation](#19316-transaction-confirmation)
    * [19.3.17 Idempotency for Financial Actions](#19317-idempotency-for-financial-actions)
    * [19.3.18 Fraud and Abuse Controls](#19318-fraud-and-abuse-controls)
    * [19.3.19 Payment Risk Classification](#19319-payment-risk-classification)
    * [19.3.20 Refunds and Reversals](#19320-refunds-and-reversals)
    * [19.3.21 Disputes and Reconciliation](#19321-disputes-and-reconciliation)
    * [19.3.22 Settlement Concepts](#19322-settlement-concepts)
    * [19.3.23 Payment Auditability](#19323-payment-auditability)
    * [19.3.24 Human Approval for High-Risk Transactions](#19324-human-approval-for-high-risk-transactions)
    * [19.3.25 Safe Fallback for Failed or Ambiguous Transactions](#19325-safe-fallback-for-failed-or-ambiguous-transactions)
    * [19.3.26 Privacy and Data Minimization](#19326-privacy-and-data-minimization)
    * [19.3.27 AP2](#19327-ap2)
    * [19.3.28 ACP](#19328-acp)
    * [19.3.29 x402](#19329-x402)
    * [19.3.30 MPP](#19330-mpp)
    * [19.3.31 Visa TAP](#19331-visa-tap)
    * [19.3.32 Mastercard Agent Pay](#19332-mastercard-agent-pay)
    * [19.3.33 Agentic Commerce Architecture](#19333-agentic-commerce-architecture)
    * [19.3.34 Security Requirements](#19334-security-requirements)
  * [19.4 Agentic AI Foundation (AAIF) & Protocol Governance](#194-agentic-ai-foundation-aaif--protocol-governance)

    * [19.4.1 What AAIF Is](#1941-what-aaif-is)
    * [19.4.2 Linux Foundation Governance Context](#1942-linux-foundation-governance-context)
    * [19.4.3 Neutral Stewardship](#1943-neutral-stewardship)
    * [19.4.4 Specification Ownership](#1944-specification-ownership)
    * [19.4.5 Contribution Models](#1945-contribution-models)
    * [19.4.6 Technical Steering and Working Groups](#1946-technical-steering-and-working-groups)
    * [19.4.7 Versioning and Compatibility](#1947-versioning-and-compatibility)
    * [19.4.8 Interoperability Testing](#1948-interoperability-testing)
    * [19.4.9 Release Cadence and Ecosystem Coordination](#1949-release-cadence-and-ecosystem-coordination)
    * [19.4.10 Vendor-Neutral Standards vs Vendor APIs](#19410-vendor-neutral-standards-vs-vendor-apis)
    * [19.4.11 How Governance Affects Adoption](#19411-how-governance-affects-adoption)
    * [19.4.12 AAIF Governance Flow](#19412-aaif-governance-flow)
  * [19.5 Putting the Protocols Together](#195-putting-the-protocols-together)

    * [19.5.1 Agent Stack](#1951-agent-stack)
    * [19.5.2 End-to-End Interaction](#1952-end-to-end-interaction)
    * [19.5.3 Which Protocol Solves Which Problem?](#1953-which-protocol-solves-which-problem)
    * [19.5.4 Architecture Decision Guide](#1954-architecture-decision-guide)
  * [19.6 Key Insights](#196-key-insights)
  * [19.7 Common Mistakes](#197-common-mistakes)
  * [19.8 Common Confusions](#198-common-confusions)
  * [19.9 Practical Applications](#199-practical-applications)
  * [19.10 Important Terms](#1910-important-terms)
  * [19.11 Quick Revision](#1911-quick-revision)
  * [19.12 Interview Preparation](#1912-interview-preparation)

    * [19.12.1 Level 1 — Fundamentals](#19121-level-1--fundamentals)
    * [19.12.2 Level 2 — Conceptual Understanding](#19122-level-2--conceptual-understanding)
    * [19.12.3 Level 3 — Practical / Engineering](#19123-level-3--practical--engineering)
    * [19.12.4 Level 4 — Advanced / Deep Understanding](#19124-level-4--advanced--deep-understanding)
    * [19.12.5 Level 5 — Scenario-Based Questions](#19125-level-5--scenario-based-questions)
    * [19.12.6 Knowledge Check](#19126-knowledge-check)
    * [19.12.7 Follow-up Questions](#19127-follow-up-questions)
    * [19.12.8 Common Confusion Questions](#19128-common-confusion-questions)
    * [19.12.9 Deep / Trick Questions](#19129-deep--trick-questions)
  * [19.13 Top Questions You MUST Know](#1913-top-questions-you-must-know)
  * [19.14 Interview Readiness Checklist](#1914-interview-readiness-checklist)
  * [19.15 What You Should Be Able to Explain](#1915-what-you-should-be-able-to-explain)

# 19. Layer 17 — Agent Protocol Landscape

🧠 **Simple Understanding:** The Agent Protocol Landscape is the collection of protocols and standards that solve **different communication boundaries around an agent**: tools and data, agent-to-agent collaboration, user interfaces, commerce, and payments.

The most important lesson is not memorizing protocol names. It is recognizing **which architectural problem each family solves**.

The roadmap frames the landscape as:

| Protocol / Concept     | Main Problem                                |
| ---------------------- | ------------------------------------------- |
| **MCP**                | Agent/model ↔ tools/data/context            |
| **A2A**                | Agent ↔ agent interoperability              |
| **A2UI**               | Agent ↔ dynamically generated UI            |
| **AG-UI concepts**     | Agent ↔ user-interface event streaming      |
| **Commerce protocols** | Agent ↔ commerce/catalog/checkout workflows |
| **Payment protocols**  | Agent-authorized payment interactions       |

This role-based view is the foundation of the layer.

---

# 19.1 Protocol Landscape Overview

## 19.1.1 MCP

🧠 **Simple Understanding:** MCP standardizes how an AI host or agent interacts with external **tools, resources, and other exposed capabilities**.

```text
Agent / Host
     ↓
    MCP
     ↓
Tools / Data / Resources
```

Think:

> **"Give the agent access to capabilities."**

---

## 19.1.2 A2A

🧠 **Simple Understanding:** A2A standardizes communication and collaboration **between independent agents**.

```text
Agent A
   ↓
  A2A
   ↓
Agent B
```

Think:

> **"Let one agent delegate work to another agent."**

The A2A project is currently hosted by the Linux Foundation and describes itself as an open protocol for communication and interoperability between opaque agentic applications. ([GitHub][1])

---

## 19.1.3 A2UI

🧠 **Simple Understanding:** A2UI is concerned with agents driving or generating **rich user-interface experiences** rather than returning only plain text.

```text
Agent
 ↓
A2UI
 ↓
UI Renderer
 ↓
User Interface
```

The current A2UI site describes A2UI as a protocol for agent-driven interfaces, with a production release in the 0.9.x family and a v1.0 candidate also available. ([A2UI][2])

A2UI v1.0's candidate specification describes a JSON-based streaming UI protocol with messages for creating surfaces and updating components/data models. ([GitHub][3])

---

## 19.1.4 AG-UI

🧠 **Simple Understanding:** AG-UI focuses on the **agent ↔ frontend/user interaction channel**, especially event-based, real-time interaction.

```text
Agent Backend
     ↓
   AG-UI
     ↓
Frontend Application
     ↓
User
```

The AG-UI project describes itself as an open, lightweight, event-based protocol connecting AI agents to user-facing applications, including agent state, UI intents, and user interactions. ([GitHub][4])

### A2UI vs AG-UI

This distinction is important:

| A2UI                              | AG-UI                              |
| --------------------------------- | ---------------------------------- |
| Generates/renders UI structure    | Connects agent backend to frontend |
| UI specification                  | Event-based interaction protocol   |
| Agent → UI rendering              | Agent ↔ user-facing application    |
| Focus on surfaces/components/data | Focus on events/state/interactions |

The AG-UI documentation explicitly distinguishes the two and notes that they can work together. ([GitHub][4])

---

## 19.1.5 Commerce Protocols

🧠 **Simple Understanding:** Commerce protocols govern how an agent participates in commercial workflows such as discovering products, retrieving offers, constructing carts, and orchestrating checkout.

```text
Agent
 ↓
Product Discovery
 ↓
Offer
 ↓
Cart
 ↓
Checkout
```

The key architectural distinction is:

```text
Commerce ≠ Payment
```

Commerce answers:

> **What are we buying, from whom, at what offer, and how do we complete checkout?**

---

## 19.1.6 Payment Protocols

🧠 **Simple Understanding:** Payment protocols govern the authorization and execution of financial transactions.

```text
Authorized Intent
      ↓
Payment Protocol
      ↓
Transaction
      ↓
Confirmation / Settlement
```

Payment concerns:

* Authority.
* Limits.
* Transaction execution.
* Verification.
* Fraud control.
* Reconciliation.

---

## 19.1.7 Protocol Landscape Map

```text
                         AGENTIC SYSTEM
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
       CAPABILITIES          AGENTS               USERS
          │                    │                    │
          │                    │                    │
         MCP                  A2A                AG-UI
          │                    │                    │
          ▼                    ▼                    ▼
      Tools / Data        Other Agents        Frontend
                                                   │
                                                  A2UI
                                                   │
                                                   ▼
                                               Dynamic UI

                               │
                               ▼
                          COMMERCE
                               │
                    Catalog / Offer / Cart
                               │
                               ▼
                           PAYMENTS
                               │
                    Authorization / Execute
```

⭐ **Key Insight:** Protocols should be selected by **boundary**, not by popularity.

---

# 19.2 Agent Identity & Trust

The roadmap correctly emphasizes that interoperability requires more than protocol compatibility: production systems must identify clients and agents, bind authority, establish trust, and verify who is acting on whose behalf.

---

## 19.2.1 Why Agent Identity Matters

🧠 **Simple Understanding:** Once agents can act autonomously, the system needs to answer:

```text
Who is acting?
For whom?
With whose authority?
Doing what?
Against which resource?
```

Without identity:

```text
"some agent"
```

is not sufficient for production security.

---

## 19.2.2 User Identity vs Agent Identity vs Client Identity

These identities should not be conflated.

| Identity              | Represents                 | Example                              |
| --------------------- | -------------------------- | ------------------------------------ |
| **User identity**     | Human principal            | Alice                                |
| **Agent identity**    | Autonomous agent           | procurement-agent-7                  |
| **Client identity**   | Calling application/client | enterprise-ui                        |
| **Workload identity** | Running service/workload   | service-account / workload principal |

Example:

```text
Alice
  ↓
User Identity
  ↓
Enterprise Assistant
  ↓
Agent Identity
  ↓
A2A / MCP Client
  ↓
Remote Service
```

---

## 19.2.3 Agent Identity vs Workload / Service Identity

🧠 **Simple Understanding:** The identity of an **agent as an actor** is not necessarily identical to the infrastructure identity of the process hosting it.

Example:

```text
Agent:
payment-agent

Hosted by:
Kubernetes workload
```

This creates two useful questions:

```text
Which software workload is running?
Which logical agent is acting?
```

A production system may need to bind both.

---

## 19.2.4 Delegated Authority

🧠 **Simple Understanding:** Delegated authority means an agent acts using authority granted by another principal, typically a user or organization.

```text
User
 ↓
Grant limited authority
 ↓
Agent
 ↓
Perform allowed action
```

Example:

```text
User:
"Buy office supplies up to ₹20,000."

Agent authority:
purchase.office_supplies
limit = ₹20,000
```

The important concept is:

```text
Delegation ≠ Unlimited impersonation
```

---

## 19.2.5 Credential Binding

🧠 **Simple Understanding:** Credential binding links an authorization credential to the correct actor, client, task, or resource context.

A system may need to answer:

```text
Credential
   ↓
Who issued it?
Who may use it?
For what scope?
For which resource?
For how long?
```

This helps prevent credential replay or use outside the intended context.

---

## 19.2.6 Agent Provenance

🧠 **Simple Understanding:** Agent provenance records the lineage of an action.

Example:

```text
User
 ↓
User Agent
 ↓
Research Agent
 ↓
MCP Tool
 ↓
External API
```

Provenance can capture:

* Agent identity.
* Caller identity.
* Delegation relationship.
* Tool used.
* Resource accessed.
* Timestamp.
* Authorization context.

---

## 19.2.7 Authentication vs Authorization

| Concept            | Question                    |
| ------------------ | --------------------------- |
| **Authentication** | Who are you?                |
| **Authorization**  | What are you allowed to do? |

Example:

```text
Agent A
 ↓
Authentication ✓
 ↓
Authorization
 ↓
payments.write ✗
```

The agent can be legitimate but unauthorized for the requested action.

⭐ **Interview Tip:** This distinction appears across MCP, A2A, commerce, and payment architectures.

---

## 19.2.8 Trust Chains

🧠 **Simple Understanding:** A trust chain allows one identity or authority decision to be connected to another.

Example:

```text
User Identity
      ↓
Trusted Application
      ↓
Authenticated Agent
      ↓
Authorized Delegation
      ↓
Remote Agent
      ↓
Backend
```

The system should preserve enough information to determine how authority propagated.

---

## 19.2.9 Capability-Based Authorization

🧠 **Simple Understanding:** Capability-based authorization grants narrowly scoped permissions for specific operations or resources.

Example:

```text
Agent Capability
├── customer.read
├── ticket.read
└── ticket.create
```

rather than:

```text
Agent = "admin"
```

This supports least privilege.

---

## 19.2.10 Credential Rotation and Revocation

🧠 **Simple Understanding:** Credentials need lifecycle controls.

```text
Issue
 ↓
Use
 ↓
Rotate
 ↓
Revoke / Expire
```

Rotation reduces the lifetime of exposed credentials.

Revocation is important when:

* User withdraws authority.
* Agent is compromised.
* Credential leaks.
* Role changes.
* Task completes.

---

## 19.2.11 Identity Discovery and Metadata

🧠 **Simple Understanding:** Systems need a way to discover identity-related metadata and authorization information.

Possible metadata:

```text
Agent ID
Issuer
Organization
Endpoints
Capabilities
Scopes
Validity
Keys / verification metadata
```

This supports federation between independent systems.

---

## 19.2.12 Merchant-Side Agent Verification

🧠 **Simple Understanding:** In agentic commerce, the merchant may need to determine whether a request genuinely comes from the claimed agent and whether the agent is authorized to act for a user.

```text
Agent
 ↓
Merchant
 ↓
Verify:
Identity
Authority
Scope
Transaction
```

This is more demanding than ordinary bot detection because the agent may be taking commercially consequential actions.

---

## 19.2.13 Cross-System Identity Propagation

When a request crosses multiple systems:

```text
User
 ↓
Agent A
 ↓
Agent B
 ↓
MCP Server
 ↓
Payment Service
```

the system may need to preserve:

```text
Original principal
Caller agent
Delegation chain
Scopes
Task ID
Trace ID
```

Otherwise downstream systems may lose the original authorization context.

---

## 19.2.14 Identity and Authorization Audit Trails

A production audit record may contain:

```json
{
  "actor": "payment-agent-7",
  "on_behalf_of": "user-123",
  "capability": "payments.write",
  "resource": "order-881",
  "decision": "allow",
  "timestamp": "..."
}
```

This supports:

* Security investigation.
* Dispute handling.
* Compliance.
* Incident response.
* Transaction reconstruction.

---

## 19.2.15 CIMD

🧠 **Simple Understanding:** Client ID Metadata Documents (CIMD) are an implementation-layer identity/metadata mechanism relevant to modern authorization and MCP deployments.

The roadmap treats CIMD as an **evolving implementation-layer standard**, not as a fundamental replacement for the identity architecture itself.

⭐ **Key Point:** Learn the underlying concepts first:

```text
Identity
 ↓
Metadata
 ↓
Authorization
 ↓
Delegation
```

Then track the concrete specification and adoption state.

---

## 19.2.16 Web Bot Auth

🧠 **Simple Understanding:** Web Bot Auth refers to emerging mechanisms for cryptographically verifiable agent/bot identity at the web or merchant edge.

Conceptually:

```text
Agent Request
 ↓
Cryptographic identity proof
 ↓
Merchant / Web Service
 ↓
Verify agent
```

The roadmap explicitly positions this as an emerging mechanism to track rather than a timeless architectural primitive.

⚠️ **Important:** For fast-moving identity standards, implementation details should always be checked against the current specification before deployment.

---

## 19.2.17 Identity & Trust Architecture

```text
                              USER
                                │
                         User Identity
                                │
                                ▼
                       ┌────────────────┐
                       │  USER AGENT    │
                       └───────┬────────┘
                               │
                        Agent Identity
                               │
                      Delegated Authority
                               │
                               ▼
                       ┌────────────────┐
                       │ REMOTE AGENT   │
                       └───────┬────────┘
                               │
                         A2A / MCP
                               │
                               ▼
                        Backend Service
                               │
                         Authorization
                               │
                               ▼
                          Resource / Action

         ─────────────── Audit / Provenance ───────────────►
```

---

# 19.3 Agentic Commerce & Payments

🧠 **Simple Understanding:** Agentic commerce connects autonomous reasoning to **real commercial actions**.

The critical architecture is:

```text
Intent
 ↓
Authority
 ↓
Policy
 ↓
Commerce
 ↓
Payment
 ↓
Confirmation
 ↓
Settlement
```

The roadmap explicitly emphasizes distinguishing **intent, authorization, transaction construction, execution, confirmation, and settlement**.

---

## 19.3.1 Why Agentic Commerce Exists

Traditional ecommerce assumes:

```text
Human
 ↓
Browse
 ↓
Choose
 ↓
Checkout
 ↓
Pay
```

Agentic commerce introduces:

```text
User
 ↓
Goal
 ↓
Agent
 ↓
Discover
 ↓
Compare
 ↓
Select
 ↓
Checkout
 ↓
Pay
```

The agent becomes an active participant in the purchasing workflow.

---

## 19.3.2 Agentic Commerce Lifecycle

```text
User Intent
   ↓
Product / Service Discovery
   ↓
Offer / Pricing Retrieval
   ↓
Merchant Selection
   ↓
Cart Construction
   ↓
Checkout
   ↓
Purchase Authorization
   ↓
Payment Intent
   ↓
Transaction Authorization
   ↓
Payment Execution
   ↓
Confirmation
   ↓
Settlement
   ↓
Reconciliation / Audit
```

⭐ **Key Point:** The most dangerous mistake is collapsing all of these stages into:

```text
"Agent clicked buy."
```

---

## 19.3.3 Product / Catalog Discovery

🧠 **Simple Understanding:** The agent must discover what products or services are available.

Potential information:

* Product.
* SKU.
* Availability.
* Price.
* Currency.
* Attributes.
* Seller.
* Delivery information.

---

## 19.3.4 Merchant and Agent Discovery

The ecosystem may need to establish:

```text
Which merchant?
Which agent?
Who represents whom?
```

This becomes particularly important when multiple independent agents and merchants interact.

---

## 19.3.5 Offer and Pricing Retrieval

🧠 **Simple Understanding:** The agent must determine the actual offer before constructing a transaction.

A price can depend on:

* Quantity.
* Location.
* Currency.
* Discounts.
* Taxes.
* Shipping.
* Time.
* User eligibility.

Therefore:

```text
Displayed price
   ≠
Guaranteed final transaction amount
```

---

## 19.3.6 Cart Construction

The agent constructs the commercial intent:

```text
Cart
├── Product A × 2
├── Product B × 1
└── Shipping option
```

This is still different from executing payment.

---

## 19.3.7 Checkout Orchestration

Checkout may involve:

```text
Address
Shipping
Tax
Discount
Inventory
Merchant confirmation
Payment selection
```

The agent coordinates these steps while respecting policy.

---

## 19.3.8 Purchase Authorization

🧠 **Simple Understanding:** Before a transaction occurs, the agent must establish whether it has authority to make that purchase.

Example:

```text
User Authority
├── Merchant = approved
├── Category = office supplies
├── Max transaction = ₹20,000
└── Valid until = ...
```

---

## 19.3.9 Delegated Payment Authority

The user or organization may delegate limited payment capability:

```text
User
 ↓
Payment Authority
 ↓
Agent
 ↓
Transaction
```

Good delegation is:

```text
Bounded
Scoped
Revocable
Auditable
```

---

## 19.3.10 Mandates and Pre-Authorization

🧠 **Simple Understanding:** A mandate/pre-authorization can represent permission established in advance under defined conditions.

Example:

```text
Authorize:
Recurring software subscription
≤ ₹5,000/month
Approved merchant
```

The agent can then operate within that authority rather than requesting unrestricted authorization each time.

---

## 19.3.11 Spending Limits and Budgets

Possible controls:

```text
Monthly budget
Daily budget
Category budget
Merchant budget
Per-agent budget
```

Example:

```text
Travel Agent
Budget = ₹100,000/month
```

Budgets should be enforced by trusted infrastructure, not merely by model instructions.

---

## 19.3.12 Per-Transaction and Cumulative Limits

Two different controls:

| Limit              | Example           |
| ------------------ | ----------------- |
| Per transaction    | ≤ ₹20,000         |
| Daily cumulative   | ≤ ₹50,000/day     |
| Monthly cumulative | ≤ ₹2,00,000/month |

A transaction may satisfy the per-transaction limit while still exceeding the cumulative budget.

---

## 19.3.13 User Consent and Revocation

🧠 **Simple Understanding:** Users need a way to grant and withdraw authority.

```text
Grant
 ↓
Use
 ↓
Review
 ↓
Revoke
```

Revocation should take effect across relevant systems as appropriate.

---

## 19.3.14 Payment Intent vs Payment Execution

This distinction is fundamental.

```text
Payment Intent
= "I intend / authorize this payment."

Payment Execution
= "Actually perform the transaction."
```

Example:

```text
Intent:
Pay ₹8,000 to Merchant X

Execution:
Payment network processes ₹8,000
```

Separating the two enables policy checks before real financial side effects.

---

## 19.3.15 Merchant-Side Agent Verification

A merchant may need to verify:

```text
Agent identity
User authority
Transaction scope
Policy validity
```

This is the merchant-side counterpart to buyer-side authorization.

---

## 19.3.16 Transaction Confirmation

🧠 **Simple Understanding:** Confirmation establishes what actually happened.

Possible result:

```text
AUTHORIZED
PROCESSING
SUCCEEDED
FAILED
REQUIRES_ACTION
UNKNOWN
```

"Request accepted" should not automatically mean "payment completed."

---

## 19.3.17 Idempotency for Financial Actions

🧠 **Simple Understanding:** Repeating a financial request must not accidentally create duplicate transactions.

Example:

```text
Agent
 ↓
Charge ₹10,000
 ↓
Response lost
 ↓
Retry
```

Without idempotency:

```text
₹10,000
+
₹10,000
=
Duplicate charge
```

Use mechanisms such as:

```text
Idempotency Key
Transaction ID
Provider-side reconciliation
```

⭐ **Key Point:** Financial actions should be designed assuming ambiguous outcomes are possible.

---

## 19.3.18 Fraud and Abuse Controls

Agentic systems create new abuse surfaces:

* Automated purchasing.
* Credential misuse.
* Account takeover.
* Prompt/manipulation attacks.
* Fake merchant interactions.
* Transaction replay.
* Excessive purchasing.

Controls may include:

```text
Velocity limits
Risk scoring
Merchant verification
Behavioral detection
Step-up authentication
Transaction review
```

---

## 19.3.19 Payment Risk Classification

Not all actions carry equal risk.

```text
Read-only
   ↓
Low-value purchase
   ↓
High-value purchase
   ↓
Sensitive / irreversible transaction
```

Risk level should drive:

* Authorization strength.
* Human approval.
* Transaction limits.
* Monitoring.

---

## 19.3.20 Refunds and Reversals

Payment systems must support cases where a transaction needs correction.

```text
Payment
 ↓
Refund
```

or:

```text
Authorization
 ↓
Reversal
```

The exact semantics depend on the payment architecture.

---

## 19.3.21 Disputes and Reconciliation

🧠 **Simple Understanding:** Reconciliation compares internal records with external transaction records.

```text
Internal Record
       +
Merchant Record
       +
Payment Network Record
       ↓
Reconciliation
```

A mismatch must be detected and investigated.

---

## 19.3.22 Settlement Concepts

🧠 **Simple Understanding:** Settlement is the process by which transaction obligations are ultimately completed between relevant parties.

Conceptually:

```text
Transaction
 ↓
Authorization
 ↓
Capture / Processing
 ↓
Settlement
```

Do not confuse:

```text
Authorization
with
Settlement
```

---

## 19.3.23 Payment Auditability

Every consequential transaction should be traceable:

```text
User
 ↓
Agent
 ↓
Authority
 ↓
Merchant
 ↓
Transaction
 ↓
Payment
 ↓
Result
```

Audit records should support reconstruction of what occurred.

---

## 19.3.24 Human Approval for High-Risk Transactions

Example:

```text
Agent wants to purchase ₹250,000
            ↓
Risk threshold exceeded
            ↓
Human approval
            ↓
Execute
```

Human approval should be policy-driven rather than arbitrary.

---

## 19.3.25 Safe Fallback for Failed or Ambiguous Transactions

If payment status is unknown:

```text
UNKNOWN
  ↓
Do not blindly retry
  ↓
Query authoritative provider
  ↓
Reconcile
  ↓
Retry safely or escalate
```

This is critical for financial correctness.

---

## 19.3.26 Privacy and Data Minimization

🧠 **Simple Understanding:** Agentic commerce should expose only the data required for the transaction.

Prefer:

```text
Minimum necessary data
```

over:

```text
Entire customer profile
```

This reduces:

* Privacy risk.
* Data leakage.
* Unnecessary exposure.

---

## 19.3.27 AP2

🧠 **Simple Understanding:** AP2, as described in the roadmap, is an agent-payment protocol focused on agent-authorized payment flows using cryptographic mandates and authorization concepts.

Use it as an **implementation-layer example**, not as the definition of agentic payments.

The durable architecture remains:

```text
Identity
+
Authority
+
Policy
+
Transaction
+
Verification
```

---

## 19.3.28 ACP

🧠 **Simple Understanding:** ACP, as described in the roadmap, refers to agentic-commerce / checkout integration patterns associated with the OpenAI + Stripe ecosystem.

Architecturally, think:

```text
Agent
 ↓
Commerce / Checkout
 ↓
Payment
```

rather than treating ACP as the entire agentic-payment stack.

---

## 19.3.29 x402

🧠 **Simple Understanding:** x402, as described in the roadmap, is an HTTP-oriented payment signaling approach associated with the HTTP 402 concept and stablecoin-oriented payment flows.

The architecture is conceptually:

```text
HTTP Request
 ↓
Payment Required
 ↓
Payment
 ↓
Retry / Continue
```

It is most useful to understand as a **payment signaling/execution mechanism**, not as a replacement for the full commerce architecture.

---

## 19.3.30 MPP

🧠 **Simple Understanding:** MPP, as described in the roadmap, represents machine/session-oriented payment authorization and spending patterns associated with Stripe/Tempo.

The conceptual interest is:

```text
Machine / Agent
 ↓
Session / Authority
 ↓
Spending Controls
 ↓
Payment
```

Treat protocol details as evolving implementation knowledge.

---

## 19.3.31 Visa TAP

🧠 **Simple Understanding:** Visa TAP is a network-level example the roadmap highlights for tracking agent identity/authentication and payment-oriented infrastructure.

Architecturally, it belongs closer to:

```text
Agent Identity
+
Network Payment Infrastructure
```

than to general agent orchestration.

---

## 19.3.32 Mastercard Agent Pay

🧠 **Simple Understanding:** Mastercard Agent Pay is another network-level ecosystem example the roadmap highlights for tracking agentic commerce/payment infrastructure.

Its architectural role is associated with:

```text
Agent identity
+
Commerce
+
Payment network
```

rather than generic agent-to-agent communication.

---

## 19.3.33 Agentic Commerce Architecture

```text
                              USER
                                │
                         Purchase Request
                                │
                                ▼
                              AGENT
                                │
                         Intent Formation
                                │
                                ▼
                       Identity + Authority
                                │
                                ▼
                      Policy / Spending Rules
                                │
                                ▼
                  Commerce Discovery / Checkout
                                │
                                ▼
                       Payment Intent
                                │
                                ▼
                     Payment Authorization
                                │
                                ▼
                       Payment Execution
                                │
                                ▼
                     Confirmation / Settlement
                                │
                                ▼
                     Audit / Reconciliation
```

---

## 19.3.34 Security Requirements

The roadmap's security requirements can be organized as:

| Requirement                                | Purpose                             |
| ------------------------------------------ | ----------------------------------- |
| Explicit transaction authorization         | Prevent unauthorized purchases      |
| Least-privilege payment authority          | Limit agent capability              |
| Spending ceilings                          | Limit financial exposure            |
| Merchant/recipient verification            | Prevent wrong-party payment         |
| Replay protection                          | Prevent duplicate use               |
| Idempotency keys                           | Prevent duplicate financial effects |
| Transaction signing / mandate verification | Validate authority                  |
| Credential isolation                       | Reduce credential exposure          |
| Strong audit records                       | Reconstruct actions                 |
| Human approval                             | Control high-risk actions           |
| Emergency cancellation                     | Stop dangerous activity             |

---

# 19.4 Agentic AI Foundation (AAIF) & Protocol Governance

## 19.4.1 What AAIF Is

🧠 **Simple Understanding:** AAIF is the governance ecosystem intended to provide a neutral home for open standards and shared infrastructure around agentic AI.

The Linux Foundation describes AAIF as a neutral home for open standards powering agentic AI systems. ([Linux Foundation][5])

Current 2026 reporting shows continued expansion of AAIF membership and ecosystem activity, including organizations from enterprise technology and financial services. ([Linux Foundation][6])

---

## 19.4.2 Linux Foundation Governance Context

AAIF's relevance is that open protocols benefit from governance that is not controlled solely by one commercial vendor.

The goal is to create a shared environment for:

```text
Specifications
Implementations
Contributions
Interoperability
Ecosystem coordination
```

---

## 19.4.3 Neutral Stewardship

🧠 **Simple Understanding:** Neutral stewardship means shared infrastructure and specifications can evolve under broader community governance rather than being controlled entirely by one implementation vendor.

This matters because adoption depends on trust.

---

## 19.4.4 Specification Ownership

A protocol ecosystem needs clarity around:

```text
Who maintains the specification?
Who approves changes?
Who handles security issues?
Who publishes releases?
```

Without ownership, protocols can fragment.

---

## 19.4.5 Contribution Models

Open protocol ecosystems need mechanisms for:

* Contributions.
* Proposals.
* Reviews.
* Issue tracking.
* Reference implementations.
* Security reporting.

These processes determine how quickly and safely standards evolve.

---

## 19.4.6 Technical Steering and Working Groups

🧠 **Simple Understanding:** Technical steering structures provide a way to coordinate protocol evolution and technical decisions.

Conceptually:

```text
Community
   ↓
Proposals
   ↓
Technical Review
   ↓
Working Group / Steering
   ↓
Specification
```

---

## 19.4.7 Versioning and Compatibility

As protocols evolve:

```text
v1
 ↓
v1.x
 ↓
v2
```

the ecosystem must maintain:

* Compatibility guidance.
* Migration paths.
* Deprecation policies.
* Conformance tests.

Without this, interoperability breaks as soon as implementations evolve independently.

---

## 19.4.8 Interoperability Testing

🧠 **Simple Understanding:** A specification is not enough. Independent implementations must actually communicate correctly.

Testing should cover:

```text
Discovery
Schemas
Authentication
Authorization
Requests
Responses
Errors
Version behavior
Edge cases
```

The current A2A project, for example, exposes a Technology Compatibility Kit (TCK) and an inspector for implementation validation. ([GitHub][1])

---

## 19.4.9 Release Cadence and Ecosystem Coordination

Protocol ecosystems need coordinated releases so:

```text
Specification
 ↓
SDKs
 ↓
Implementations
 ↓
Tests
 ↓
Documentation
```

do not drift apart.

---

## 19.4.10 Vendor-Neutral Standards vs Vendor APIs

| Vendor-Neutral Standard       | Vendor-Specific API                |
| ----------------------------- | ---------------------------------- |
| Shared ecosystem              | Vendor ecosystem                   |
| Multiple implementations      | Usually one provider               |
| Interoperability goal         | Product integration goal           |
| Broader portability           | Potentially higher vendor coupling |
| Community governance possible | Provider-controlled evolution      |

Neither is inherently better for every situation.

### Use vendor-specific APIs when:

* You intentionally depend on one provider.
* Provider-specific functionality gives strong value.
* Interoperability is not a requirement.

### Use open protocols when:

* Multiple vendors must interoperate.
* You want portability.
* Independent agents/services need a common interface.

---

## 19.4.11 How Governance Affects Adoption

The roadmap's core chain is:

```text
Protocol Specification
        ↓
Governance / Stewardship
        ↓
Release Process
        ↓
Compatibility
        ↓
Ecosystem Adoption
        ↓
Production Stability
```

This is a crucial systems insight.

A technically excellent protocol can still fail if:

* Governance is unclear.
* Versions fragment.
* Implementations diverge.
* Compatibility testing is weak.
* Vendors lose confidence.

---

## 19.4.12 AAIF Governance Flow

```text
                  COMMUNITY
                      │
                      ▼
                  Proposal
                      │
                      ▼
              Technical Review
                      │
                      ▼
              Working Groups /
              Technical Steering
                      │
                      ▼
                Specification
                      │
                ┌─────┴─────┐
                ▼           ▼
          Reference       Tests
        Implementations
                │           │
                └─────┬─────┘
                      ▼
               Release / Version
                      │
                      ▼
              Ecosystem Adoption
                      │
                      ▼
               Production Use
```

---

# 19.5 Putting the Protocols Together

## 19.5.1 Agent Stack

A useful conceptual stack is:

```text
┌─────────────────────────────────────┐
│               User                 │
├─────────────────────────────────────┤
│          AG-UI / UI Layer          │
├─────────────────────────────────────┤
│        A2UI / Dynamic UI           │
├─────────────────────────────────────┤
│          Agent Layer               │
├─────────────────────────────────────┤
│              A2A                   │
│     Agent ↔ Agent Collaboration    │
├─────────────────────────────────────┤
│              MCP                   │
│      Agent ↔ Tools / Data          │
├─────────────────────────────────────┤
│ Commerce / Payment Protocols       │
├─────────────────────────────────────┤
│ Identity / Authorization / Trust   │
├─────────────────────────────────────┤
│ Runtime / Infrastructure           │
└─────────────────────────────────────┘
```

⚠️ This is a **conceptual stack**, not a requirement that every system use every layer.

---

## 19.5.2 End-to-End Interaction

Consider:

> "Find the best laptop under my budget, ask the shopping agent to compare options, and buy the chosen one."

Possible architecture:

```text
User
 ↓
User-facing Agent
 ↓
AG-UI
 ↓
User interaction
 ↓
A2A
 ↓
Shopping Agent
 ↓
MCP
 ↓
Catalog / Merchant Tools
 ↓
Commerce Workflow
 ↓
Identity + Delegated Authority
 ↓
Payment Protocol
 ↓
Transaction
 ↓
Confirmation
 ↓
AG-UI
 ↓
User
```

This demonstrates why protocol landscape literacy matters: **multiple protocol families can participate in one end-to-end agentic workflow.**

---

## 19.5.3 Which Protocol Solves Which Problem?

| Question                                           | Protocol / Layer            |
| -------------------------------------------------- | --------------------------- |
| "How does my agent access a tool?"                 | **MCP**                     |
| "How does my agent access a resource?"             | **MCP**                     |
| "How does one agent delegate to another?"          | **A2A**                     |
| "How does the frontend receive live agent events?" | **AG-UI**                   |
| "How can an agent drive a dynamic UI?"             | **A2UI**                    |
| "How does an agent participate in checkout?"       | **Commerce protocols**      |
| "How does an agent authorize/pay?"                 | **Payment protocols**       |
| "Who is this agent acting for?"                    | **Identity / trust layer**  |
| "Who governs the open protocol ecosystem?"         | **AAIF / governance layer** |

---

## 19.5.4 Architecture Decision Guide

### Need tools/data?

```text
Use MCP
```

### Need agent delegation?

```text
Use A2A
```

### Need real-time agent/frontend interaction?

```text
Use AG-UI concepts
```

### Need generated UI surfaces?

```text
Use A2UI
```

### Need purchasing workflows?

```text
Use commerce protocols
```

### Need financial execution?

```text
Use payment infrastructure/protocols
```

### Need cross-system authority?

```text
Use identity + delegated authorization
```

### Need open ecosystem governance?

```text
Understand AAIF / protocol stewardship
```

---

# 19.6 Key Insights

💡 **Key Insights**

1. **Protocol literacy is architecture literacy.** You do not need to implement every protocol, but you must know which system boundary each addresses.

2. **MCP, A2A, A2UI, and AG-UI are complementary rather than interchangeable.** MCP connects agents to capabilities, A2A connects agents, A2UI handles agent-driven UI rendering, and AG-UI connects agents to user-facing applications. ([GitHub][4])

3. **Identity is a cross-cutting layer.** An interoperable protocol without reliable identity and delegated authority is insufficient for consequential production actions.

4. **Commerce and payment should remain conceptually separate.** Finding and buying an item is a commerce workflow; authorizing and executing the financial transaction is a payment workflow.

5. **Financial agent actions need stronger guarantees than ordinary tool calls.** Idempotency, reconciliation, authority limits, merchant verification, and emergency controls become critical.

6. **Open protocols need governance.** A protocol is only as useful as the ecosystem's ability to maintain compatibility, testing, releases, and trust. AAIF's current role reflects this ecosystem-level need. ([Linux Foundation][5])

7. **The landscape is evolving rapidly.** Current MCP work has introduced a stateless protocol core, multi-round-trip requests, header-based routing, cacheable list results, tasks, authorization hardening, and extensions; this is exactly why durable architectural concepts should be learned before memorizing implementation details. ([Model Context Protocol Blog][7])

---

# 19.7 Common Mistakes

⚠️ **Common Mistakes**

| Mistake                                                    | Correct Understanding                                                                        |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| "MCP, A2A, and AG-UI all solve the same problem."          | They address different boundaries.                                                           |
| "A2A means all agents must share the same implementation." | Interoperability is specifically about collaborating across independent implementations.     |
| "Protocol compatibility means security."                   | Identity, authorization, trust, and policy still need to be enforced.                        |
| "Agent identity is the same as user identity."             | User, agent, client, and workload identities can be different principals.                    |
| "Delegation means giving the agent my full credentials."   | Delegated authority should be scoped and revocable.                                          |
| "Commerce and payment are one protocol problem."           | Commerce covers the purchase workflow; payment covers financial authorization/execution.     |
| "A successful authorization means the payment succeeded."  | Authorization and settlement are different stages.                                           |
| "Retrying a payment is safe."                              | Financial retries can duplicate transactions unless protected by idempotency/reconciliation. |
| "Latest protocol version is automatically safest."         | Version adoption requires compatibility and production validation.                           |
| "Open standard = universally compatible implementation."   | Implementations can still differ in semantics, schemas, security, and edge-case behavior.    |
| "Governance is organizational overhead."                   | Governance directly affects interoperability and ecosystem stability.                        |
| "AI agents can self-enforce spending limits."              | Limits should be enforced by trusted infrastructure.                                         |

---

# 19.8 Common Confusions

🔍 **Common Confusions**

| Concept A        | Concept B          | Key Difference                                                  |
| ---------------- | ------------------ | --------------------------------------------------------------- |
| MCP              | A2A                | Agent ↔ capability vs agent ↔ agent                             |
| A2A              | AG-UI              | Agent ↔ agent vs agent ↔ user-facing application                |
| A2UI             | AG-UI              | Dynamic UI generation vs frontend interaction/event protocol    |
| Commerce         | Payment            | Purchase workflow vs financial transaction                      |
| User identity    | Agent identity     | Human principal vs autonomous actor                             |
| Agent identity   | Workload identity  | Logical agent vs infrastructure/service principal               |
| Authentication   | Authorization      | Who are you? vs what may you do?                                |
| Trust            | Authorization      | Confidence vs permission                                        |
| Intent           | Execution          | Desired action vs actual side effect                            |
| Authorization    | Settlement         | Permission to transact vs completion of financial settlement    |
| Mandate          | Transaction        | Delegated permission vs individual transaction                  |
| Protocol         | SDK                | Interoperability specification vs implementation library        |
| Open standard    | Vendor API         | Shared ecosystem protocol vs provider-specific interface        |
| Protocol version | Capability version | Protocol compatibility vs functionality compatibility           |
| Artifact         | Message            | Durable work product vs communication payload                   |
| Audit            | Observability      | Security/accountability record vs broader operational telemetry |

---

# 19.9 Practical Applications

🛠️ **Practical Applications**

| Application                   | Relevant Protocol Families               |
| ----------------------------- | ---------------------------------------- |
| Enterprise assistant          | MCP + A2A + identity                     |
| Coding agent                  | MCP + A2A + runtime                      |
| Research platform             | MCP + A2A + AG-UI                        |
| Dynamic AI dashboard          | A2UI + AG-UI                             |
| Multi-agent enterprise        | A2A + MCP + identity                     |
| Agent marketplace             | A2A + discovery + identity               |
| Shopping agent                | A2A + commerce + identity                |
| Autonomous purchasing         | Commerce + payment + delegated authority |
| Financial agent               | A2A + payment + strong authorization     |
| Cross-company agent ecosystem | A2A + identity + governance              |
| Agent-driven frontend         | AG-UI + A2UI                             |
| Enterprise protocol gateway   | MCP/A2A + identity + policy              |

---

# 19.10 Important Terms

📌 **Important Terms**

| Term                     | Simple Meaning                                     | Why It Matters                    |
| ------------------------ | -------------------------------------------------- | --------------------------------- |
| Protocol Landscape       | Ecosystem of agent communication standards         | Helps choose correct architecture |
| MCP                      | Agent-to-capability protocol                       | Tools/data access                 |
| A2A                      | Agent-to-agent interoperability                    | Delegation/collaboration          |
| A2UI                     | Agent-driven UI protocol                           | Dynamic UI generation             |
| AG-UI                    | Agent-user interaction protocol                    | Frontend/event integration        |
| Commerce Protocol        | Commercial workflow mechanism                      | Catalog/checkout                  |
| Payment Protocol         | Financial transaction mechanism                    | Authorization/execution           |
| Agent Identity           | Identity of autonomous actor                       | Security/audit                    |
| Client Identity          | Identity of calling application                    | Authentication/policy             |
| Workload Identity        | Infrastructure identity                            | Runtime security                  |
| Delegated Authority      | Granted power to act                               | Controlled autonomy               |
| Credential Binding       | Linking credentials to context                     | Prevents misuse                   |
| Agent Provenance         | Action lineage                                     | Accountability                    |
| Trust Chain              | Chain of authenticated/delegated relationships     | Federation                        |
| Capability Authorization | Permission for specific capability                 | Least privilege                   |
| Credential Rotation      | Replacing credentials periodically                 | Limits exposure                   |
| Revocation               | Removing active authority                          | Emergency/security control        |
| Agent Verification       | Proving agent identity/authority                   | Merchant/federated trust          |
| Commerce Discovery       | Finding products/services                          | Purchase workflow                 |
| Checkout                 | Finalizing commercial order                        | Purchase execution                |
| Payment Intent           | Authorized intent to pay                           | Separates intent from execution   |
| Payment Execution        | Actual transaction processing                      | Financial side effect             |
| Idempotency              | Safe repeat behavior                               | Prevents duplicate payment        |
| Reconciliation           | Compare records with reality                       | Handles ambiguity                 |
| Settlement               | Final completion of payment obligations            | Financial lifecycle               |
| AP2                      | Agent-payment protocol example                     | Implementation-layer knowledge    |
| ACP                      | Agentic commerce protocol example                  | Checkout integration              |
| x402                     | HTTP-oriented payment signaling example            | Machine payments                  |
| MPP                      | Machine/session payment example                    | Agent spending                    |
| Visa TAP                 | Network-level agent payment infrastructure         | Identity/payment ecosystem        |
| Mastercard Agent Pay     | Network-level agentic payment infrastructure       | Commerce/payment                  |
| AAIF                     | Governance ecosystem for open agent infrastructure | Protocol stewardship              |
| Interoperability Testing | Testing independent implementations together       | Production compatibility          |

---

# 19.11 Quick Revision

⚡ **Quick Revision**

1. The protocol landscape is about **different system boundaries**, not a list of competing technologies.
2. **MCP → agent/capability access.**
3. **A2A → agent/agent collaboration.**
4. **A2UI → agent-driven UI generation.**
5. **AG-UI → agent/frontend/user interaction events.** ([GitHub][4])
6. **Commerce → product/offer/cart/checkout.**
7. **Payment → financial authorization/execution/settlement.**
8. Identity answers:

```text
Who?
For whom?
With whose authority?
For what scope?
```

9. **Authentication ≠ authorization.**
10. **Trust ≠ permission.**
11. Agentic payments require:

    * Least privilege
    * Spending limits
    * Idempotency
    * Reconciliation
    * Auditability
    * Human approval where required
12. A2A can sit above MCP:

```text
A2A
 ↓
Specialized Agent
 ↓
MCP
 ↓
Tools / Data
```

13. Open protocols need governance, testing, versioning, and compatibility discipline.
14. AAIF matters because protocol adoption depends on ecosystem stewardship, not just specification quality. ([Linux Foundation][5])

---

# 19.12 Interview Preparation

## 19.12.1 Level 1 — Fundamentals

### Q1. What is the Agent Protocol Landscape?

**Model Answer:**
It is the collection of protocol families that address different boundaries around AI agents, such as connecting agents to tools and data, connecting agents to other agents, connecting agents to user interfaces, and enabling commerce and payment interactions.

### Q2. What is the difference between MCP and A2A?

**Model Answer:**
MCP connects an agent or host to tools, resources, and data. A2A connects independent agents so they can communicate and delegate work. An agent can use MCP internally while collaborating with another agent through A2A. ([GitHub][1])

### Q3. What is A2UI?

**Model Answer:**
A2UI is an agent-driven UI protocol intended to let agents produce structured, dynamically rendered interfaces. Its current ecosystem includes a production 0.9.x release family and a v1.0 candidate. ([A2UI][2])

### Q4. What is AG-UI?

**Model Answer:**
AG-UI is an event-based protocol connecting AI agents to user-facing applications, including agent state, UI intents, and user interactions. ([GitHub][4])

### Q5. What is agent identity?

**Model Answer:**
Agent identity is the identity of the autonomous software actor performing or requesting an action. It should be distinguishable from user identity, client identity, and infrastructure workload identity.

### Q6. What is delegated authority?

**Model Answer:**
Delegated authority is permission granted by one principal, such as a user, for an agent to perform specific actions within defined boundaries. Good delegation is scoped, revocable, and auditable.

### Q7. Why are commerce and payment different?

**Model Answer:**
Commerce handles the purchase workflow—discovering products, offers, carts, and checkout—while payment handles authorization and execution of the financial transaction.

### Q8. What is AAIF?

**Model Answer:**
AAIF is an ecosystem and governance initiative under the Linux Foundation focused on open standards and shared infrastructure for agentic AI. ([Linux Foundation][5])

---

## 19.12.2 Level 2 — Conceptual Understanding

### Q1. Why can't protocol compatibility alone establish trust?

**Model Answer:**
A protocol only establishes how systems communicate. It does not prove who is calling, whether the caller is authorized, or whether the request is acting on behalf of a valid principal. Identity and authorization must be handled separately.

### Q2. Why should user identity and agent identity be separated?

**Model Answer:**
A human may delegate only limited authority to an agent. Treating the agent as the user would make revocation, least privilege, auditing, and transaction restrictions much harder.

### Q3. Why is workload identity different from agent identity?

**Model Answer:**
Workload identity identifies the running software infrastructure, while agent identity identifies the logical autonomous actor. One agent may be recreated across multiple workloads, so the relationship should be explicit rather than assumed.

### Q4. Why is payment intent different from execution?

**Model Answer:**
Separating intent from execution allows policy, authorization, risk checks, and user approval to occur before an irreversible financial side effect.

### Q5. Why is idempotency essential for agentic payments?

**Model Answer:**
Network failures can make an operation's outcome ambiguous. A retry can therefore duplicate a transaction unless the financial action has idempotent semantics or the system reconciles with the payment provider before retrying.

### Q6. Why can A2UI and AG-UI be used together?

**Model Answer:**
A2UI can represent the dynamically generated UI experience, while AG-UI can carry the event and interaction flow between the agent backend and the user-facing frontend. The two operate at different layers. ([GitHub][4])

### Q7. Why is governance important for open protocols?

**Model Answer:**
Without clear stewardship, versioning, compatibility testing, and contribution processes, independent implementations can diverge and interoperability can degrade. Governance turns a specification into an evolving ecosystem.

### Q8. Why should commerce and payment remain separate abstractions?

**Model Answer:**
The commercial decision and the financial execution have different data, policies, risks, and lifecycle states. Separating them makes architecture and authorization clearer.

---

## 19.12.3 Level 3 — Practical / Engineering

### Q1. How would you choose between MCP and A2A?

**Model Answer:**

```text
Need external tool/data access?
        ↓
       MCP

Need another autonomous agent?
        ↓
       A2A
```

A2A and MCP can be combined when the delegated agent itself needs tools or data.

### Q2. How would you design agent identity in a multi-agent enterprise?

**Model Answer:**
Maintain explicit identities for the user, logical agent, calling client, and runtime workload. Bind them through signed or otherwise verifiable authorization context and propagate delegation metadata downstream for auditing and authorization.

### Q3. How would you implement delegated purchasing authority?

**Model Answer:**

```text
User
 ↓
Grant authority
 ↓
Define:
merchant
category
amount
time window
currency
 ↓
Agent
 ↓
Policy Check
 ↓
Transaction
```

The authority should be stored and enforced by trusted infrastructure.

### Q4. How would you design payment retry behavior?

**Model Answer:**
Use an idempotency key or transaction identifier, check the authoritative payment provider when an outcome is ambiguous, and retry only when the system can establish that doing so cannot create a duplicate side effect.

### Q5. How would you build an agent protocol gateway?

**Model Answer:**
Place a controlled gateway in front of remote capabilities and agents, then centralize identity verification, authorization, routing, rate limiting, audit, compatibility policies, and observability while preserving protocol semantics.

### Q6. How would you test cross-vendor agent interoperability?

**Model Answer:**
Use contract and compatibility tests for discovery, capabilities, task lifecycle, authentication, authorization, schemas, errors, version negotiation, artifacts, timeouts, and edge cases. The A2A ecosystem's TCK is an example of this testing mindset. ([GitHub][1])

### Q7. How would you integrate commerce and payment into an agent?

**Model Answer:**

```text
Agent
 ↓
Discover Product
 ↓
Retrieve Offer
 ↓
Construct Cart
 ↓
Checkout
 ↓
Check Delegated Authority
 ↓
Payment Intent
 ↓
Payment Authorization
 ↓
Payment Execution
 ↓
Reconcile
```

Each stage should have explicit state and authorization boundaries.

### Q8. How would you support human approval?

**Model Answer:**

```text
Agent
 ↓
Risk classification
 ↓
Threshold exceeded?
 ↓
Human approval
 ↓
Authorize
 ↓
Execute
```

Approval should be tied to the specific action and authority context rather than being a generic confirmation.

---

## 19.12.4 Level 4 — Advanced / Deep Understanding

### Q1. Why is identity a cross-cutting concern rather than one more protocol?

**Model Answer:**
Identity spans MCP, A2A, commerce, payment, UI, and backend systems. Every one of those boundaries may need to know who is acting, on whose behalf, with which permissions. Identity therefore provides the security context around protocol interactions rather than being just another message format.

### Q2. Why can protocol layering reduce complexity rather than increase it?

**Model Answer:**
Each protocol can own one boundary. MCP handles capability connectivity, A2A handles delegation, UI protocols handle frontend interaction, and payment protocols handle financial operations. Clear separation prevents one protocol from becoming a giant monolithic interface.

### Q3. Why can the same agent participate in several protocol layers simultaneously?

**Model Answer:**
Protocols address different relationships. An agent may use AG-UI to interact with a frontend, A2A to delegate work, MCP to access tools, and payment protocols to execute authorized financial actions.

### Q4. Why is delegated authority preferable to broad credentials?

**Model Answer:**
Delegated authority enables least privilege, bounded transactions, revocation, and clear audit trails. Broad credentials allow too much power and make it difficult to reason about what an agent was actually authorized to do.

### Q5. Why does merchant-side agent verification matter if the payment provider authenticates the transaction?

**Model Answer:**
The merchant may need to verify that the requesting agent legitimately represents the user and that the agent's delegated authority covers the purchase. Payment authorization alone does not answer all merchant-side identity questions.

### Q6. Why is governance a technical concern?

**Model Answer:**
Governance determines how specifications evolve, how versions remain compatible, how security issues are handled, and how independent implementations are tested. These directly affect whether a protocol remains interoperable in production.

### Q7. Why is open-source implementation not the same thing as open governance?

**Model Answer:**
A project can publish source code while still having centralized control over specification changes. Open governance involves broader contribution and decision-making structures, not simply public source code.

### Q8. Why should an agent protocol landscape be learned by architectural problem rather than by protocol name?

**Model Answer:**
Protocol names change, merge, evolve, and gain extensions. The durable skill is recognizing the boundary: capability access, agent collaboration, user interaction, UI generation, commerce, payment, or identity. That lets an engineer adapt when implementations evolve.

---

## 19.12.5 Level 5 — Scenario-Based Questions

### Scenario 1 — Wrong Identity Binding

A user-authorized shopping agent runs inside a generic service workload used by many agents.

**Question:** How do you know which logical agent is acting?

**Model Answer:**
Separate workload identity from logical agent identity and bind them explicitly in the authorization context. The transaction record should preserve user, agent, client, and workload identities.

---

### Scenario 2 — Agent Has Valid Credentials but Wrong Scope

A research agent has a valid credential but requests:

```text
payments.write
```

**Question:** Should it succeed?

**Model Answer:**
No. Authentication proves the identity; authorization determines permitted capabilities. The missing scope should cause denial, with the decision recorded for audit.

---

### Scenario 3 — Payment Response Lost

The agent sends a payment request but the response is lost.

**Question:** What is the safest next step?

**Model Answer:**

```text
Unknown result
 ↓
Do not blindly retry
 ↓
Query authoritative provider
 ↓
Find transaction status
 ↓
Record result
 ↓
Retry only if safely permitted
```

This avoids duplicate charges.

---

### Scenario 4 — User Grants a ₹50,000 Budget

An agent tries to make two purchases:

```text
₹30,000
₹25,000
```

Both individually satisfy the per-transaction limit of ₹50,000.

**Question:** Can the second transaction proceed?

**Model Answer:**
Not necessarily. A cumulative budget must also be enforced. If ₹50,000 is the total budget, the second purchase exceeds the remaining authority.

---

### Scenario 5 — Shopping Agent Finds a Better Merchant

The user authorized Merchant A only. The agent discovers Merchant B with a lower price.

**Question:** Should it buy from Merchant B?

**Model Answer:**
Not unless the delegated authority permits merchant substitution. The agent should not expand the authority scope simply because another offer is better.

---

### Scenario 6 — A2A Task Crosses Organizations

A Company A agent delegates research to Company B's agent.

**Question:** What additional concerns arise?

**Model Answer:**

```text
Identity
Trust
Authorization
Data-sharing boundaries
Version compatibility
Audit
Task ownership
Failure handling
```

The organizational boundary becomes part of the security architecture.

---

### Scenario 7 — Dynamic UI Request

An agent needs to present a live approval interface containing changing fields and actions.

**Question:** Which protocol family should you think about?

**Model Answer:**
Think separately about the two concerns: AG-UI for the event-based agent/frontend interaction channel and A2UI for agent-driven dynamic UI representation. They can work together. ([GitHub][4])

---

### Scenario 8 — Protocol Version Drift

Two vendors claim to support the same protocol, but one implementation interprets a capability differently.

**Question:** What does this tell you?

**Model Answer:**
Protocol compatibility is not sufficient by itself. The ecosystem needs conformance tests, semantic compatibility, version negotiation, and clear specification governance.

---

# 19.12.6 Knowledge Check

🧠 **Knowledge Check**

If you can explain these naturally in your own words, you understand Layer 17:

* What the agent protocol landscape is.
* Why protocol families solve different system-boundary problems.
* The difference between MCP and A2A.
* The difference between A2UI and AG-UI.
* Why commerce and payment are different abstractions.
* Why identity cuts across all protocol layers.
* The difference between user, agent, client, and workload identities.
* What delegated authority means.
* Why credential binding matters.
* What agent provenance means.
* Authentication vs authorization.
* Trust vs authorization.
* Trust chains.
* Capability-based authorization.
* Credential rotation and revocation.
* Identity discovery and metadata.
* Merchant-side agent verification.
* Cross-system identity propagation.
* Identity audit trails.
* Why CIMD and Web Bot Auth should be treated as evolving implementation-layer standards.
* The agentic commerce lifecycle.
* The difference between payment intent and execution.
* Why spending limits need both per-transaction and cumulative controls.
* Why idempotency matters.
* Why reconciliation matters.
* What payment risk classification means.
* When human approval is needed.
* Why payment failures can be ambiguous.
* What AP2, ACP, x402, MPP, Visa TAP, and Mastercard Agent Pay represent at a high level.
* What AAIF is.
* Why governance affects interoperability.
* Why compatibility testing is essential.
* How multiple protocol layers can participate in one agentic workflow.
* How to select the correct protocol family for an architecture.

---

# 19.12.7 Follow-up Questions

### Basic Question

**What is MCP?**

→ What boundary does it solve?
→ What does it expose?
→ How does authorization work?

### Basic Question

**What is A2A?**

→ Why is it different from MCP?
→ How are tasks represented?
→ How is identity handled?

### Basic Question

**What is A2UI vs AG-UI?**

→ Dynamic UI or event transport?
→ Can they work together?
→ Where do they sit architecturally?

### Basic Question

**What is agent identity?**

→ User identity?
→ Agent identity?
→ Client identity?
→ Workload identity?
→ Delegation?

### Basic Question

**What is agentic commerce?**

→ Discovery?
→ Offer?
→ Cart?
→ Checkout?
→ Authority?
→ Payment?

### Basic Question

**Why are payments special?**

→ Financial side effects?
→ Idempotency?
→ Reconciliation?
→ Risk?
→ Human approval?

### Basic Question

**Why does protocol governance matter?**

→ Versioning?
→ Compatibility?
→ Testing?
→ Adoption?
→ Stability?

---

# 19.12.8 Common Confusion Questions

### Q1. Is A2A just MCP between two agents?

**Model Answer:**
No. MCP is capability access, while A2A models independent agents as collaborators. An agent can use MCP internally while using A2A externally.

### Q2. Is AG-UI a UI rendering protocol?

**Model Answer:**
Primarily it is an agent-user interaction/event protocol connecting agent backends to user-facing applications. A2UI is the more directly UI-generation-oriented layer. ([GitHub][4])

### Q3. Is A2UI a replacement for the frontend?

**Model Answer:**
No. It defines a way for agents to describe/render interactive UI structures; a compatible renderer/application still provides the actual user experience. ([GitHub][3])

### Q4. Is authorization part of A2A/MCP?

**Model Answer:**
Protocol interactions can carry or support authorization concepts, but application-specific identity and authorization policy still need to be implemented and enforced by the deployment.

### Q5. Is payment intent the same as payment success?

**Model Answer:**
No. Intent indicates an authorized desire to transact; execution and later confirmation determine whether the financial side effect actually occurred.

### Q6. Is AAIF another protocol?

**Model Answer:**
No. AAIF is a governance/ecosystem layer around open agent infrastructure and standards.

---

# 19.12.9 Deep / Trick Questions

### ⚠️ Deeper Question

**If A2A standardizes agent communication, why is agent identity still difficult?**

**Correct Understanding:**
Communication syntax does not establish authority. Production systems must determine which principal is acting, who delegated authority, what scopes apply, which workload is executing the request, and how downstream systems should trust the chain.

---

### ⚠️ Deeper Question

**Why can two agents be protocol-compatible but still unsafe to connect?**

**Correct Understanding:**
They may speak the same protocol while having incompatible security assumptions, excessive capabilities, weak identity binding, different interpretations of data, or insufficient authorization. Interoperability is not equivalent to trust.

---

### ⚠️ Deeper Question

**Why should a payment protocol not also define the entire commerce experience?**

**Correct Understanding:**
Commerce includes discovery, offers, carts, shipping, checkout, and merchant interaction, while payment includes financial authorization and execution. Keeping those boundaries separate allows independent evolution and clearer risk controls.

---

### ⚠️ Deeper Question

**Why is cumulative spending control necessary if each transaction has a limit?**

**Correct Understanding:**
An attacker or malfunctioning agent can perform many individually valid transactions and still exceed the intended overall budget. Per-transaction and cumulative controls protect against different failure modes.

---

### ⚠️ Deeper Question

**Why is delegated authority safer than simply giving the agent an access token?**

**Correct Understanding:**
A broad token may grant more power than the task requires. Delegated authority can be scoped by action, resource, merchant, amount, time, and other constraints, making autonomous behavior easier to control and revoke.

---

### ⚠️ Deeper Question

**Why is governance part of interoperability engineering?**

**Correct Understanding:**
Protocol interoperability depends on stable specifications, version rules, compatibility tests, change processes, and ecosystem coordination. Without governance, implementations can diverge even when they claim to implement the same protocol.

---

### ⚠️ Deeper Question

**Why is protocol knowledge less durable than architectural knowledge?**

**Correct Understanding:**
Specific protocols evolve, gain extensions, change versions, or lose adoption. The durable skill is recognizing architectural boundaries—capabilities, agents, users, UI, commerce, payments, identity—and selecting appropriate mechanisms for each.

---

# 19.13 Top Questions You MUST Know

⭐ **Top Questions You MUST Know**

1. What is the Agent Protocol Landscape?
2. What problem does MCP solve?
3. What problem does A2A solve?
4. What is the difference between A2UI and AG-UI?
5. Why are commerce and payment separate protocol domains?
6. Why is agent identity distinct from user and workload identity?
7. What is delegated authority?
8. What is the difference between authentication, authorization, and trust?
9. Why is capability-based authorization useful for agents?
10. Why are idempotency and reconciliation essential for agentic payments?
11. How do per-transaction and cumulative spending limits differ?
12. How do agentic commerce and payment workflows fit together?
13. What do AP2, ACP, x402, MPP, Visa TAP, and Mastercard Agent Pay represent conceptually?
14. What is AAIF and why does governance matter?
15. How do multiple protocol families combine into one end-to-end agentic architecture?

---

# 19.14 Interview Readiness Checklist

🎯 **Interview Readiness Checklist**

| Skill                           | Can I explain it? |
| ------------------------------- | :---------------: |
| Protocol landscape definition   |         ☐         |
| MCP role                        |         ☐         |
| A2A role                        |         ☐         |
| A2UI role                       |         ☐         |
| AG-UI role                      |         ☐         |
| Commerce protocols              |         ☐         |
| Payment protocols               |         ☐         |
| Protocol selection by boundary  |         ☐         |
| User identity                   |         ☐         |
| Agent identity                  |         ☐         |
| Client identity                 |         ☐         |
| Workload identity               |         ☐         |
| Delegated authority             |         ☐         |
| Credential binding              |         ☐         |
| Agent provenance                |         ☐         |
| Authentication                  |         ☐         |
| Authorization                   |         ☐         |
| Trust chains                    |         ☐         |
| Capability authorization        |         ☐         |
| Credential rotation             |         ☐         |
| Credential revocation           |         ☐         |
| Identity metadata               |         ☐         |
| Merchant-side verification      |         ☐         |
| Identity propagation            |         ☐         |
| Authorization audit trails      |         ☐         |
| CIMD concepts                   |         ☐         |
| Web Bot Auth concepts           |         ☐         |
| Commerce lifecycle              |         ☐         |
| Product discovery               |         ☐         |
| Merchant discovery              |         ☐         |
| Offer/pricing retrieval         |         ☐         |
| Cart construction               |         ☐         |
| Checkout orchestration          |         ☐         |
| Purchase authorization          |         ☐         |
| Delegated payment authority     |         ☐         |
| Mandates                        |         ☐         |
| Spending budgets                |         ☐         |
| Cumulative limits               |         ☐         |
| User consent/revocation         |         ☐         |
| Payment intent                  |         ☐         |
| Payment execution               |         ☐         |
| Merchant verification           |         ☐         |
| Transaction confirmation        |         ☐         |
| Financial idempotency           |         ☐         |
| Fraud controls                  |         ☐         |
| Payment risk classification     |         ☐         |
| Refunds/reversals               |         ☐         |
| Reconciliation                  |         ☐         |
| Settlement                      |         ☐         |
| Payment auditability            |         ☐         |
| Human approval                  |         ☐         |
| Safe failure handling           |         ☐         |
| Data minimization               |         ☐         |
| AP2                             |         ☐         |
| ACP                             |         ☐         |
| x402                            |         ☐         |
| MPP                             |         ☐         |
| Visa TAP                        |         ☐         |
| Mastercard Agent Pay            |         ☐         |
| AAIF                            |         ☐         |
| Open governance                 |         ☐         |
| Versioning                      |         ☐         |
| Compatibility testing           |         ☐         |
| Vendor-neutral standards        |         ☐         |
| End-to-end protocol composition |         ☐         |

---

# 19.15 What You Should Be Able to Explain

🧠 **What You Should Be Able to Explain**

By the end of Layer 17, you should be able to explain:

* What the agent protocol landscape is.
* Why modern agent systems need multiple protocol families.
* Why protocol selection should be based on architectural boundary.
* What MCP solves.
* What A2A solves.
* What A2UI solves.
* What AG-UI solves.
* How A2UI and AG-UI differ and can work together. ([GitHub][4])
* Why commerce and payment are separate domains.
* How identity cuts across all these protocol layers.
* Why user identity differs from agent identity.
* Why client identity differs from workload identity.
* How to model an autonomous agent as a security principal.
* What delegated authority means.
* Why delegated authority should be scoped.
* How credential binding works conceptually.
* What agent provenance means.
* The difference between authentication and authorization.
* The difference between trust and authorization.
* What a trust chain is.
* How capability-based authorization supports least privilege.
* Why credential rotation and revocation matter.
* How identity metadata supports discovery and federation.
* Why merchant-side agent verification matters.
* How identity and delegation propagate across multiple systems.
* How identity-related actions should be audited.
* What CIMD represents at a high level.
* Why Web Bot Auth should be tracked as an evolving implementation-layer mechanism.
* What agentic commerce is.
* The complete commerce lifecycle.
* How product and merchant discovery work.
* How offers and pricing are retrieved.
* How an agent constructs a cart.
* How checkout orchestration works.
* How purchase authorization differs from payment execution.
* What delegated payment authority is.
* What mandates and pre-authorizations represent.
* Why spending budgets are necessary.
* Why both per-transaction and cumulative limits matter.
* Why users need consent and revocation controls.
* The distinction between payment intent and actual payment execution.
* Why merchant-side verification is required.
* Why transaction confirmation must be explicit.
* Why payment retries require idempotency.
* Why ambiguous transaction outcomes require reconciliation.
* What fraud and abuse controls look like.
* How payment risk classification can drive stronger controls.
* Why refunds and reversals are separate lifecycle operations.
* Why disputes and reconciliation matter.
* What settlement means.
* Why transaction auditability is essential.
* When human approval should be required.
* How to fail safely when payment status is unknown.
* Why data minimization matters in agentic commerce.
* The conceptual role of AP2.
* The conceptual role of ACP.
* The conceptual role of x402.
* The conceptual role of MPP.
* The conceptual role of Visa TAP.
* The conceptual role of Mastercard Agent Pay.
* What AAIF is.
* Why Linux Foundation governance matters to open agent infrastructure.
* Why neutral stewardship can improve ecosystem confidence.
* How specifications are contributed to and maintained.
* Why technical steering and working groups matter.
* Why versioning is necessary.
* Why interoperability testing is necessary.
* How governance affects adoption and production stability.
* The difference between vendor-neutral standards and vendor-specific APIs.
* How MCP, A2A, A2UI, AG-UI, commerce, payment, and identity can coexist in a single system.
* How to choose the right protocol for a given architectural boundary.
* Why protocol names and exact specifications evolve quickly.
* Why architectural concepts are more durable than individual protocol versions.
* Why **the real skill is not memorizing protocols but understanding the boundaries, trust relationships, authority models, and failure semantics that those protocols standardize**.

## ⚡ Final Mental Model

```text
                         AGENTIC ECOSYSTEM
                                │
                                ▼
                              USER
                                │
                         User Identity
                                │
                                ▼
                      ┌──────────────────┐
                      │ USER-FACING APP  │
                      └────────┬─────────┘
                               │
                         AG-UI / UI Events
                               │
                               ▼
                         ┌──────────────┐
                         │    AGENT     │
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
            A2A                MCP              A2UI
              │                 │                 │
              │                 │                 ▼
              │                 │          Dynamic UI
              │                 │
              │                 ▼
              │            Tools / Data
              │
              ▼
      SPECIALIZED AGENTS
        │        │       │
        ▼        ▼       ▼
     Research  Coding  Finance
        │        │       │
       MCP      MCP     MCP
        │        │       │
        └────────┼───────┘
                 ▼
          Agentic Workflow
                 │
                 ▼
             COMMERCE
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
    Catalog    Offer      Checkout
       │         │          │
       └─────────┼──────────┘
                 ▼
          DELEGATED AUTHORITY
                 │
                 ▼
          PAYMENT INTENT
                 │
                 ▼
        PAYMENT AUTHORIZATION
                 │
                 ▼
         PAYMENT EXECUTION
                 │
                 ▼
          CONFIRMATION
                 │
                 ▼
            SETTLEMENT
                 │
                 ▼
         RECONCILIATION
                 │
                 ▼
              AUDIT

        ─────────────────────────
          CROSS-CUTTING LAYERS
        ─────────────────────────
        Identity
        Trust
        Authorization
        Provenance
        Policy
        Observability
        Versioning
        Compatibility
        Governance
```

> **Core principle:** **The agent protocol landscape is not a collection of competing protocols that you must memorize. It is a set of architectural boundaries: MCP connects agents to capabilities, A2A connects agents to agents, A2UI and AG-UI connect agents to user interfaces, commerce protocols connect agents to commercial workflows, payment protocols connect authorized agent intent to financial execution, and identity/trust infrastructure determines who is acting and on whose authority. The production engineer's job is to compose these boundaries safely—with least privilege, explicit delegation, strong identity, idempotent side effects, reconciliation, auditability, compatibility testing, and sound protocol governance.**

