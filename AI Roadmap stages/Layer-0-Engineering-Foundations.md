# 🔩 Layer 0 — Engineering Foundations
> **Goal:** Build the non-AI engineering foundation that makes production AI systems reliable, maintainable, and scalable
> **Timeline:** Months 1–2 (parallel with early AI phases)
> **Outcome:** You write TypeScript like a senior engineer, design systems under pressure, understand databases deeply, and ship code that doesn't break at 2am.

---

## 📚 Table of Contents

1. [0.1 — TypeScript Strict Mode](#01--typescript-strict-mode)
2. [0.2 — System Design for AI Systems](#02--system-design-for-ai-systems)
3. [0.3 — Databases Beyond CRUD](#03--databases-beyond-crud)
4. [0.4 — Distributed Systems Basics](#04--distributed-systems-basics)
5. [0.5 — Observability](#05--observability)
6. [0.6 — CI/CD for AI Systems](#06--cicd-for-ai-systems)
7. [Layer 0 Project](#-layer-0-project)
8. [Master Checklist](#-master-checklist)

---

## Why Engineering Foundations Beat AI Knowledge Every Time

```
┌─────────────────────────────────────────────────────────────────────┐
│              THE HIDDEN REASON SENIOR AI ENGINEERS EARN 3-5X       │
│                                                                     │
│  Junior AI Engineer thinks:                                         │
│  "I know RAG, LangGraph, and Claude API. I'm ready."              │
│                                                                     │
│  Senior AI Engineer thinks:                                         │
│  "My RAG system needs:                                             │
│   - Idempotent embedding jobs (BullMQ retries safely)             │
│   - Circuit breakers on vector DB calls                            │
│   - Structured logs with tenant + trace IDs on every operation    │
│   - HNSW index tuned for our vector count + query pattern         │
│   - Canary deployment when we change chunking strategy            │
│   - EXPLAIN ANALYZE to verify index is used on similarity search  │
│   - Feature flag so we can rollback instantly if quality drops"   │
│                                                                     │
│  None of that is AI knowledge. All of it is engineering.          │
│  That's the gap most tutorials never mention.                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 0.1 — TypeScript Strict Mode

### 0.1.1 — Why Strict Mode Matters for AI Backends

LLM outputs are runtime JSON blobs. Without strict TypeScript, every LLM response can silently pass malformed data to downstream code.

```typescript
// WITHOUT strict mode — silent bug factory
function processLLMResponse(response: any) {
  return response.data.items.map(item => item.name.toUpperCase());
  // If response.data is null → TypeError at runtime
  // If items is undefined → TypeError
  // If name is a number → toUpperCase doesn't exist
  // ALL SILENT at compile time with 'any'
}

// WITH strict mode — caught at compile time
interface LLMResponse {
  data: {
    items: Array<{ name: string; confidence: number }>;
  } | null;
}

function processLLMResponse(response: LLMResponse): string[] {
  if (!response.data) return []; // forced to handle null
  return response.data.items.map(item => item.name.toUpperCase());
}
```

**tsconfig.json for production AI projects:**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    
    // STRICT MODE — all of these
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noImplicitAny": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noImplicitOverride": true,
    
    // QUALITY
    "skipLibCheck": false,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 0.1.2 — Generics & Conditional Types

```typescript
// GENERIC: Type-safe LLM wrapper
async function callLLMTyped<T>(
  prompt: string,
  schema: z.ZodType<T>,
  model = "claude-sonnet-4-20250514"
): Promise<T> {
  const raw = await callLLM(prompt, model);
  const result = schema.safeParse(JSON.parse(raw));
  if (!result.success) {
    throw new Error(`LLM output validation failed: ${result.error.message}`);
  }
  return result.data;
}

// CONDITIONAL TYPES: Handle different response shapes
type APIResponse<T> = T extends string
  ? { type: "text"; content: T }
  : T extends object
  ? { type: "json"; data: T }
  : never;

// INFER: Extract type from inside generic
type ExtractArrayItem<T> = T extends Array<infer U> ? U : never;
type ChunkItem = ExtractArrayItem<Chunk[]>; // = Chunk

// MAPPED TYPES: Transform every field
type WithTimestamp<T> = {
  [K in keyof T]: T[K];
} & { createdAt: Date; updatedAt: Date };

type PromptWithTimestamp = WithTimestamp<Prompt>;
// = { name: string; content: string; createdAt: Date; updatedAt: Date }

// TEMPLATE LITERAL TYPES: Type-safe event names
type AIEvent = `ai.${"llm" | "embed" | "rag"}.${"started" | "completed" | "failed"}`;
// Valid: "ai.llm.started", "ai.rag.completed"
// Invalid: "ai.something.else" → TypeScript error
```

### 0.1.3 — Mapped Types for AI Schemas

```typescript
// Add confidence to every extracted field
type WithConfidence<T extends Record<string, unknown>> = {
  [K in keyof T]: {
    value: T[K];
    confidence: number; // 0-1
    source: string;     // Which chunk supported this
  };
};

interface ExtractedInvoice {
  invoiceNumber: string;
  totalAmount: number;
  vendorName: string;
}

type ConfidentInvoice = WithConfidence<ExtractedInvoice>;
// {
//   invoiceNumber: { value: string; confidence: number; source: string };
//   totalAmount: { value: number; confidence: number; source: string };
//   vendorName: { value: string; confidence: number; source: string };
// }

// Partial extraction for when some fields are missing
type PartialExtraction<T> = {
  [K in keyof T]?: T[K] | null;
};
```

### 0.1.4 — Recursive Types for Nested JSON

```typescript
// Type for arbitrary LLM JSON output
type JSONValue =
  | string
  | number
  | boolean
  | null
  | JSONValue[]
  | { [key: string]: JSONValue };

// Typed nested extraction result
interface KnowledgeGraphNode {
  id: string;
  type: "person" | "company" | "product" | "concept";
  name: string;
  attributes: Record<string, string | number | boolean>;
  relationships: {
    targetId: string;
    relationshipType: string;
    confidence: number;
    bidirectional: boolean;
  }[];
  metadata: JSONValue; // Flexible additional data
}
```

### 0.1.5 — Zod for Runtime Validation

```typescript
import { z } from "zod";

// FULL PRODUCTION EXAMPLE: Invoice extraction schema
const LineItemSchema = z.object({
  description: z.string().min(1).max(500),
  quantity: z.number().positive(),
  unitPrice: z.number().positive(),
  total: z.number().positive(),
}).refine(
  (item) => Math.abs(item.quantity * item.unitPrice - item.total) < 0.01,
  { message: "Total must equal quantity × unitPrice" }
);

const InvoiceSchema = z.object({
  invoiceNumber: z.string().regex(/^[A-Z0-9\-]+$/, "Invalid invoice number format"),
  vendor: z.object({
    name: z.string().min(1),
    address: z.string().optional(),
    taxId: z.string().optional(),
    email: z.string().email().optional(),
  }),
  lineItems: z.array(LineItemSchema).min(1),
  subtotal: z.number().positive(),
  taxRate: z.number().min(0).max(1).optional(),
  taxAmount: z.number().min(0),
  total: z.number().positive(),
  currency: z.enum(["USD", "EUR", "GBP", "INR", "SGD"]).default("USD"),
  invoiceDate: z.string().datetime().optional(),
  dueDate: z.string().datetime().optional(),
  notes: z.string().max(1000).optional(),
}).refine(
  (inv) => Math.abs(inv.subtotal + inv.taxAmount - inv.total) < 0.01,
  { message: "total must equal subtotal + taxAmount" }
);

type Invoice = z.infer<typeof InvoiceSchema>; // Auto-generated TypeScript type!

// TRANSFORM: Normalize LLM output
const FlexibleAmountSchema = z.union([
  z.number(),
  z.string().transform((s) => {
    const cleaned = s.replace(/[$,₹€£\s]/g, "");
    const num = parseFloat(cleaned);
    if (isNaN(num)) throw new Error(`Cannot parse amount: ${s}`);
    return num;
  }),
]);

// DISCRIMINATED UNION: Handle multi-intent responses
const IntentResponseSchema = z.discriminatedUnion("intent", [
  z.object({
    intent: z.literal("search"),
    query: z.string(),
    filters: z.record(z.string()).optional(),
  }),
  z.object({
    intent: z.literal("create"),
    entityType: z.enum(["ticket", "task", "note"]),
    data: z.record(z.unknown()),
  }),
  z.object({
    intent: z.literal("clarify"),
    questions: z.array(z.string()),
    context: z.string(),
  }),
]);
```

### 0.1.6 — Discriminated Unions for LLM Responses

```typescript
// Model every possible LLM response state
type LLMCallResult<T> =
  | { success: true; data: T; tokens: number; latencyMs: number }
  | { success: false; error: "rate_limited"; retryAfterMs: number }
  | { success: false; error: "content_filtered"; reason: string }
  | { success: false; error: "timeout"; timeoutMs: number }
  | { success: false; error: "provider_error"; code: number; message: string }
  | { success: false; error: "validation_failed"; issues: string[] };

async function handleLLMResult<T>(result: LLMCallResult<T>): Promise<T | null> {
  if (result.success) {
    return result.data;
  }

  switch (result.error) {
    case "rate_limited":
      await sleep(result.retryAfterMs);
      return null; // Caller should retry

    case "content_filtered":
      logger.warn({ reason: result.reason }, "Content filtered");
      return null;

    case "timeout":
      throw new Error(`LLM timeout after ${result.timeoutMs}ms`);

    case "provider_error":
      if (result.code >= 500) throw new Error(`Provider error: ${result.message}`);
      return null;

    case "validation_failed":
      logger.error({ issues: result.issues }, "LLM output validation failed");
      return null;

    default:
      // TypeScript ensures this is unreachable if all cases handled
      const _exhaustive: never = result;
      throw new Error("Unhandled result type");
  }
}
```

### 0.1.7 — Branded Types for ID Safety

```typescript
// WITHOUT branded types — easy to mix up IDs
function getUserDocuments(userId: string, tenantId: string) { ... }
getUserDocuments(tenantId, userId); // WRONG ORDER — TypeScript can't catch this!

// WITH branded types — compile-time protection
type UserId = string & { readonly __brand: "UserId" };
type TenantId = string & { readonly __brand: "TenantId" };
type DocumentId = string & { readonly __brand: "DocumentId" };
type SessionId = string & { readonly __brand: "SessionId" };

// Constructors (the only way to create these types)
const UserId = (id: string): UserId => id as UserId;
const TenantId = (id: string): TenantId => id as TenantId;

// Now the function is type-safe:
function getUserDocuments(userId: UserId, tenantId: TenantId) { ... }

getUserDocuments(
  UserId("usr_123"),
  TenantId("ten_456")
); // ✅ Works

getUserDocuments(
  TenantId("ten_456"),  // ❌ TypeScript error! TenantId ≠ UserId
  UserId("usr_123")
);
```

---

## 0.2 — System Design for AI Systems

### 0.2.1 — CAP Theorem Applied to AI

```
┌─────────────────────────────────────────────────────────────────────┐
│              CAP THEOREM FOR AI SYSTEM COMPONENTS                  │
│                                                                     │
│  CONSISTENCY (C): Every read gets the most recent write            │
│  AVAILABILITY (A): Every request gets a response (no errors)       │
│  PARTITION TOLERANCE (P): Works despite network splits             │
│  (You can only guarantee 2 of 3)                                   │
│                                                                     │
│  Component              Need C?  Need A?  Choose   Why             │
│  ─────────────────────  ───────  ───────  ──────── ─────────────── │
│  Vector store           No       Yes      AP       Stale embedding  │
│                                                     ok for search   │
│  Token usage counter    Yes      No       CP       Must be accurate │
│                                                     for billing      │
│  Conversation history   No       Yes      AP       Slight sync lag  │
│                                                     is acceptable    │
│  Rate limit counters    Yes      No       CP       Must be accurate │
│                                                     to prevent abuse │
│  Prompt cache           No       Yes      AP       Stale cache ok  │
│  Session state          No       Yes      AP       Best effort      │
│  Audit logs             Yes      No       CP       Legal compliance │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.2.2 — Rate Limiting Patterns

```typescript
// TOKEN BUCKET — best for user-facing APIs
// Has a "bucket" that fills at a constant rate
// Allows short bursts (bucket drains quickly)
class TokenBucket {
  private tokens: number;
  private lastRefill: number;

  constructor(
    private capacity: number,         // Max tokens
    private refillRate: number,        // Tokens per second
    private refillInterval: number = 1000 // Refill every N ms
  ) {
    this.tokens = capacity;
    this.lastRefill = Date.now();
  }

  consume(tokensRequired = 1): boolean {
    this.refill();
    if (this.tokens < tokensRequired) return false;
    this.tokens -= tokensRequired;
    return true;
  }

  private refill() {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    const newTokens = elapsed * this.refillRate;
    this.tokens = Math.min(this.capacity, this.tokens + newTokens);
    this.lastRefill = now;
  }
}

// SLIDING WINDOW — most accurate, what Redis uses
// Redis implementation:
async function slidingWindowRateLimit(
  redis: Redis,
  key: string,
  maxRequests: number,
  windowMs: number
): Promise<{ allowed: boolean; remaining: number; resetIn: number }> {
  const now = Date.now();
  const windowStart = now - windowMs;

  const pipeline = redis.pipeline();
  pipeline.zremrangebyscore(key, 0, windowStart);     // Remove old entries
  pipeline.zadd(key, now, `${now}-${Math.random()}`); // Add current
  pipeline.zcard(key);                                  // Count total
  pipeline.expire(key, Math.ceil(windowMs / 1000) * 2);

  const results = await pipeline.exec();
  const requestCount = results?.[2]?.[1] as number;

  return {
    allowed: requestCount <= maxRequests,
    remaining: Math.max(0, maxRequests - requestCount),
    resetIn: windowMs,
  };
}

// LEAKY BUCKET — best for smoothing traffic to downstream APIs
// Process at a constant rate regardless of input rate
// Good for: controlling calls to OpenAI API when you have many users
class LeakyBucket {
  private queue: Array<() => void> = [];
  private processing = false;
  private readonly rateMs: number; // Minimum ms between requests

  constructor(requestsPerSecond: number) {
    this.rateMs = 1000 / requestsPerSecond;
  }

  add<T>(fn: () => Promise<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      this.queue.push(async () => {
        try { resolve(await fn()); }
        catch (err) { reject(err); }
      });
      if (!this.processing) this.process();
    });
  }

  private async process() {
    this.processing = true;
    while (this.queue.length > 0) {
      const task = this.queue.shift()!;
      await task();
      await sleep(this.rateMs); // Rate limiting happens here
    }
    this.processing = false;
  }
}
```

### 0.2.3 — Database Sharding for AI

```typescript
// When to shard: 100M+ rows, CPU > 70% sustained
// For AI: conversation history at scale

// HASH-BASED SHARDING for multi-tenant data
class ShardedDB {
  private shards: Database[];
  private numShards: number;

  constructor(shards: Database[]) {
    this.shards = shards;
    this.numShards = shards.length;
  }

  // Deterministic: same tenantId always goes to same shard
  private getShardForTenant(tenantId: string): Database {
    const hash = this.murmurHash(tenantId);
    const shardIndex = hash % this.numShards;
    return this.shards[shardIndex];
  }

  async getConversations(tenantId: string): Promise<Conversation[]> {
    const shard = this.getShardForTenant(tenantId);
    return shard.query(`SELECT * FROM conversations WHERE tenant_id = $1`, [tenantId]);
  }

  // PROBLEM: Cross-shard queries don't work
  // "Show me all conversations from all tenants" requires querying all shards
  // Design your data model to NEVER need cross-shard queries

  private murmurHash(str: string): number {
    let hash = 2166136261;
    for (let i = 0; i < str.length; i++) {
      hash ^= str.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return Math.abs(hash);
  }
}
```

### 0.2.4 — Event-Driven Architecture for AI Pipelines

```
┌─────────────────────────────────────────────────────────────────────┐
│              EVENT-DRIVEN AI PIPELINE                              │
│                                                                     │
│  SYNCHRONOUS (bad for AI):                                         │
│  User: Upload PDF                                                   │
│  API: [wait 30 seconds while embedding 100 chunks]                │
│  User: [times out]                                                  │
│                                                                     │
│  EVENT-DRIVEN (production):                                         │
│  User: Upload PDF                                                   │
│  API: "Got it! Job ID: abc123" (returns IMMEDIATELY)              │
│       ↓ publishes event                                             │
│  Queue: document.uploaded { documentId, tenantId, filePath }      │
│       ↓ worker consumes                                             │
│  Worker 1: extract text → publishes document.extracted            │
│       ↓                                                             │
│  Worker 2: chunk text → publishes document.chunked                │
│       ↓                                                             │
│  Worker 3: embed chunks → publishes document.embedded             │
│       ↓                                                             │
│  Worker 4: index vectors → publishes document.indexed             │
│       ↓                                                             │
│  Notification: "Your document is ready!"                           │
│                                                                     │
│  User polls /api/jobs/abc123 or receives WebSocket push           │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// Redis Streams implementation for AI pipeline events
class AIEventBus {
  constructor(private redis: Redis) {}

  async publish(
    stream: string,
    event: Record<string, string>
  ): Promise<string> {
    const messageId = await this.redis.xadd(
      stream,
      "*", // Auto-generate ID
      ...Object.entries(event).flat()
    );
    return messageId;
  }

  async consume(
    stream: string,
    group: string,
    consumer: string,
    handler: (event: Record<string, string>) => Promise<void>
  ): Promise<void> {
    // Create consumer group if doesn't exist
    try {
      await this.redis.xgroup("CREATE", stream, group, "$", "MKSTREAM");
    } catch {
      // Group already exists
    }

    while (true) {
      const messages = await this.redis.xreadgroup(
        "GROUP", group, consumer,
        "COUNT", "10",
        "BLOCK", "5000",
        "STREAMS", stream, ">"
      );

      if (!messages) continue;

      for (const [, entries] of messages) {
        for (const [messageId, fields] of entries) {
          const event: Record<string, string> = {};
          for (let i = 0; i < fields.length; i += 2) {
            event[fields[i]] = fields[i + 1];
          }

          try {
            await handler(event);
            await this.redis.xack(stream, group, messageId);
          } catch (error) {
            // Leave unacknowledged — will be redelivered
            logger.error({ messageId, error }, "Event processing failed");
          }
        }
      }
    }
  }
}

// Usage
const eventBus = new AIEventBus(redis);

// Publisher (in upload handler)
await eventBus.publish("ai-pipeline", {
  type: "document.uploaded",
  documentId: doc.id,
  tenantId: tenant.id,
  filePath: uploadedPath,
  timestamp: Date.now().toString(),
});

// Consumers (in worker processes)
await eventBus.consume(
  "ai-pipeline",
  "embedding-workers",
  "worker-1",
  async (event) => {
    if (event.type !== "document.uploaded") return;
    await embedDocument(event.documentId, event.tenantId, event.filePath);
  }
);
```

### 0.2.5 — Idempotency in AI Pipelines

```typescript
// The problem: BullMQ/Redis Streams deliver at-least-once
// If a worker crashes mid-job, the job runs again
// You must make every operation idempotent!

class IdempotentEmbeddingWorker {
  async processDocument(documentId: string, tenantId: string): Promise<void> {
    // Check if already processed (idempotency check)
    const existing = await db.documentChunks.findFirst({
      where: { documentId, tenantId },
    });

    if (existing) {
      logger.info({ documentId }, "Document already embedded, skipping");
      return; // IDEMPOTENT: safe to call multiple times
    }

    // Use distributed lock to prevent concurrent processing of same document
    await withDistributedLock(`embed:${documentId}`, 300, async () => {
      // Double-check inside lock (race condition protection)
      const existingInLock = await db.documentChunks.findFirst({
        where: { documentId },
      });
      if (existingInLock) return;

      // Actually process
      const chunks = await getDocumentChunks(documentId);
      const embeddings = await embedBatch(chunks.map(c => c.content));

      // Upsert (not insert) — safe if called multiple times
      await db.documentChunks.createMany({
        data: chunks.map((c, i) => ({
          documentId,
          tenantId,
          content: c.content,
          embedding: embeddings[i],
          chunkIndex: i,
        })),
        skipDuplicates: true, // Postgres ON CONFLICT DO NOTHING
      });
    });
  }
}
```

### 0.2.6 — Back Pressure

```typescript
// What happens when 10,000 documents are uploaded at once?
// Without back pressure: worker overloaded, out of memory, crashes
// With back pressure: queue depth monitored, workers scaled, load shed

class BackPressureQueue {
  private queue: Queue;
  private readonly maxQueueDepth: number;

  constructor(queue: Queue, maxQueueDepth = 10_000) {
    this.queue = queue;
    this.maxQueueDepth = maxQueueDepth;
  }

  async add(
    job: JobData,
    options: { priority?: number } = {}
  ): Promise<{ accepted: boolean; jobId?: string; retryAfterMs?: number }> {
    const waiting = await this.queue.getWaitingCount();
    const active = await this.queue.getActiveCount();
    const depth = waiting + active;

    if (depth >= this.maxQueueDepth) {
      // Load shedding — reject new work when overwhelmed
      const estimatedProcessingMs =
        depth * (await this.getAverageProcessingTime());

      return {
        accepted: false,
        retryAfterMs: Math.min(estimatedProcessingMs, 5 * 60 * 1000), // Max 5 min
      };
    }

    // Priority queuing — paid users before free users
    const jobId = await this.queue.add("process", job, {
      priority: options.priority ?? 5,
    });

    return { accepted: true, jobId: jobId.id };
  }

  private async getAverageProcessingTime(): Promise<number> {
    const completed = await this.queue.getCompleted(0, 100);
    if (completed.length === 0) return 5000; // Default: 5s

    const times = completed.map(job => {
      const processingTime = (job.finishedOn ?? 0) - (job.processedOn ?? 0);
      return processingTime;
    });

    return times.reduce((a, b) => a + b, 0) / times.length;
  }
}

// API handler with back pressure
app.post("/api/documents", async (req, res) => {
  const result = await backPressureQueue.add(
    { documentId: req.body.documentId, tenantId: req.tenant.id },
    { priority: req.user.plan === "pro" ? 1 : 5 } // Pro users get priority
  );

  if (!result.accepted) {
    return res.status(503).json({
      error: "Service temporarily busy",
      retryAfter: Math.ceil(result.retryAfterMs! / 1000),
    });
  }

  res.status(202).json({ jobId: result.jobId, status: "queued" });
});
```

---

## 0.3 — Databases Beyond CRUD

### 0.3.1 — PostgreSQL Deep Dive

```sql
-- UNDERSTANDING INDEXES — this is the most important DB skill

-- B-TREE: Default. Best for equality and range queries.
CREATE INDEX idx_messages_user ON messages(user_id);
CREATE INDEX idx_messages_created ON messages(created_at DESC);
-- Use for: WHERE user_id = ?, ORDER BY created_at

-- COMPOSITE INDEX: Order matters!
CREATE INDEX idx_messages_user_created ON messages(user_id, created_at DESC);
-- Useful for: WHERE user_id = ? ORDER BY created_at DESC
-- NOT useful for: WHERE created_at > ? (leading column must be in WHERE)

-- PARTIAL INDEX: Only index rows meeting a condition
CREATE INDEX idx_active_users ON users(email) WHERE deleted_at IS NULL;
-- Much smaller index, much faster for WHERE deleted_at IS NULL

-- GIN INDEX: For arrays, JSONB, and full-text search
CREATE INDEX idx_messages_metadata ON messages USING gin(metadata);
-- Use for: WHERE metadata @> '{"type": "support"}'
CREATE INDEX idx_messages_search ON messages USING gin(to_tsvector('english', content));
-- Use for: WHERE search_vector @@ to_tsquery('english', 'refund policy')

-- BRIN: For naturally ordered columns (like created_at with UUID7)
CREATE INDEX idx_events_created_brin ON events USING brin(created_at);
-- Very small index, good for time-series data
-- Only useful if data is physically ordered by this column

-- HOW TO DIAGNOSE SLOW QUERIES:
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT m.id, m.content, c.title
FROM messages m
JOIN conversations c ON c.id = m.conversation_id
WHERE c.tenant_id = '123e4567-e89b-12d3-a456-426614174000'
  AND m.created_at > NOW() - INTERVAL '7 days'
ORDER BY m.created_at DESC
LIMIT 50;

-- READING THE OUTPUT:
-- "Seq Scan" = BAD for large tables (reading every row)
-- "Index Scan" = GOOD (using our index)
-- "actual rows vs expected rows" = large difference means stale statistics
-- Run: ANALYZE messages; -- to refresh statistics
-- "Buffers: shared hit=X" = data from cache (fast)
-- "Buffers: shared read=X" = data from disk (slow)

-- COMMON FIX PATTERNS:
-- Seq Scan on large table → Create index on WHERE column
-- Sort spill to disk → Add ORDER BY column to index
-- Hash Join → Create index on join column (usually FK)
-- Nested Loop on large table → Usually missing index

-- TABLE PARTITIONING for large AI datasets
CREATE TABLE token_usage (
  id UUID DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL,
  tokens INTEGER NOT NULL,
  cost_usd NUMERIC(10,6) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE token_usage_2025_01 PARTITION OF token_usage
  FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE token_usage_2025_02 PARTITION OF token_usage
  FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- Queries for "this month" only scan ONE partition = 12x faster!
SELECT SUM(cost_usd) FROM token_usage
WHERE tenant_id = $1
  AND created_at >= DATE_TRUNC('month', NOW());
```

### 0.3.2 — Redis Mastery

```typescript
// REDIS DATA STRUCTURE DECISION TREE:
//
// Single value + TTL? → String
// Key-value pairs for one entity? → Hash
// Ordered by score? → Sorted Set
// Message queue with acknowledgment? → Stream
// Unique membership? → Set
// Sequential list? → List (but Stream is usually better for queues)

// PATTERN 1: Pub/Sub for real-time job progress
class JobProgressBroadcaster {
  constructor(private redis: Redis) {}

  async publishProgress(jobId: string, progress: {
    step: string;
    percent: number;
    message?: string;
  }): Promise<void> {
    const channel = `job:progress:${jobId}`;
    await this.redis.publish(channel, JSON.stringify({
      ...progress,
      timestamp: Date.now(),
    }));
  }

  subscribeToJob(
    jobId: string,
    onProgress: (progress: any) => void,
    onComplete: () => void
  ): () => void {
    const subscriber = this.redis.duplicate();
    const channel = `job:progress:${jobId}`;

    subscriber.subscribe(channel);
    subscriber.on("message", (_, message) => {
      const data = JSON.parse(message);
      if (data.percent === 100) {
        onComplete();
        subscriber.unsubscribe(channel);
        subscriber.quit();
      } else {
        onProgress(data);
      }
    });

    // Return cleanup function
    return () => {
      subscriber.unsubscribe(channel);
      subscriber.quit();
    };
  }
}

// PATTERN 2: Sorted Set for leaderboard / priority queue
class TenantPriorityQueue {
  constructor(private redis: Redis) {}

  // Add with score = Unix timestamp (lower score = processed first)
  async enqueue(queueName: string, item: string, priority: "high" | "normal" | "low"): Promise<void> {
    const priorityMultiplier = { high: 0.5, normal: 1.0, low: 2.0 };
    const score = Date.now() * priorityMultiplier[priority];
    await this.redis.zadd(queueName, score, item);
  }

  // Dequeue: get lowest score (highest priority, oldest)
  async dequeue(queueName: string): Promise<string | null> {
    const result = await this.redis.zpopmin(queueName, 1);
    return result.length > 0 ? result[0] : null;
  }

  // Peek without removing
  async peek(queueName: string, count = 10): Promise<string[]> {
    return this.redis.zrange(queueName, 0, count - 1);
  }
}
```

### 0.3.3 — MongoDB Aggregation Pipelines

```javascript
// REAL EXAMPLES from production AI analytics

// Query 1: Average quality score per model per week
db.llmCalls.aggregate([
  // Stage 1: Filter to last 90 days
  { $match: {
    createdAt: { $gte: new Date(Date.now() - 90 * 86400000) },
    tenantId: "ten_123"
  }},

  // Stage 2: Group by week + model
  { $group: {
    _id: {
      week: { $isoWeek: "$createdAt" },
      year: { $isoWeekYear: "$createdAt" },
      model: "$model"
    },
    avgQuality: { $avg: "$qualityScore" },
    totalCost: { $sum: "$costUsd" },
    callCount: { $sum: 1 },
    avgLatency: { $avg: "$latencyMs" }
  }},

  // Stage 3: Reshape output
  { $project: {
    _id: 0,
    week: "$_id.week",
    year: "$_id.year",
    model: "$_id.model",
    avgQuality: { $round: ["$avgQuality", 3] },
    totalCost: { $round: ["$totalCost", 4] },
    callCount: 1,
    avgLatency: { $round: ["$avgLatency", 0] }
  }},

  // Stage 4: Sort chronologically
  { $sort: { year: 1, week: 1, model: 1 } }
]);

// Query 2: Most common failed queries (for improving the system)
db.llmCalls.aggregate([
  { $match: { status: "failed", tenantId: "ten_123" }},
  { $group: {
    _id: "$errorType",
    count: { $sum: 1 },
    firstSeen: { $min: "$createdAt" },
    lastSeen: { $max: "$createdAt" },
    exampleErrors: { $push: { $substr: ["$errorMessage", 0, 100] } }
  }},
  { $sort: { count: -1 } },
  { $limit: 10 },
  { $project: {
    errorType: "$_id",
    count: 1,
    firstSeen: 1,
    lastSeen: 1,
    exampleErrors: { $slice: ["$exampleErrors", 3] } // Just first 3 examples
  }}
]);

// Query 3: User engagement cohort analysis
db.conversations.aggregate([
  // Get cohort (which week they joined) and weekly usage
  { $lookup: {
    from: "users",
    localField: "userId",
    foreignField: "_id",
    as: "user"
  }},
  { $unwind: "$user" },
  { $group: {
    _id: {
      cohortWeek: { $isoWeek: "$user.createdAt" },
      activityWeek: { $isoWeek: "$createdAt" }
    },
    activeUsers: { $addToSet: "$userId" }
  }},
  { $project: {
    cohortWeek: "$_id.cohortWeek",
    activityWeek: "$_id.activityWeek",
    retainedUsers: { $size: "$activeUsers" }
  }},
  { $sort: { cohortWeek: 1, activityWeek: 1 } }
]);
```

### 0.3.4 — pgvector Indexes

```sql
-- HNSW vs IVFFlat: The definitive guide

-- HNSW (Hierarchical Navigable Small World)
-- ✅ Best recall (finds most relevant results)
-- ✅ Fast queries (O(log n))
-- ✅ Can insert new vectors without rebuilding
-- ❌ More memory (about 8-16 bytes per vector dimension)
-- USE WHEN: < 5M vectors, memory available, need best accuracy

CREATE INDEX ON document_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (
    m = 16,              -- Connections per node. Higher = better recall, more memory
                         -- Range: 4-64. Default: 16. For production: 16-32
    ef_construction = 64 -- Build quality. Higher = better index, slower build
                         -- Range: 4-512. Default: 64. For production: 64-128
  );

-- At query time, can set ef_search for accuracy/speed tradeoff:
SET hnsw.ef_search = 100; -- Higher = more accurate but slower (default: 40)


-- IVFFlat (Inverted File with Flat quantization)  
-- ✅ Less memory (flat compression)
-- ✅ Works well for very large datasets (10M+ vectors)
-- ❌ Needs data to train on (39x lists minimum rows recommended)
-- ❌ Must rebuild when data grows significantly
-- ❌ Slightly lower recall than HNSW
-- USE WHEN: > 5M vectors, memory constrained

-- IMPORTANT: Don't create IVFFlat until you have ~39k rows!
-- (Need at least 39 * lists rows for meaningful training)
CREATE INDEX ON document_chunks
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100); -- Usually: sqrt(total_rows). For 1M rows: lists=1000

-- Set probes at query time (more probes = higher recall, slower)
SET ivfflat.probes = 10; -- Check 10 of 100 lists (default: 1)


-- PRACTICAL BENCHMARK FOR YOUR DATA:
-- Run this query, check if it uses the index:
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, content,
  1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM document_chunks
WHERE tenant_id = 'ten_123'
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;

-- If you see "Index Scan using document_chunks_embedding_idx" = GOOD
-- If you see "Seq Scan" = index not being used (probably too few rows)
-- pgvector only uses index when it estimates index is faster
```

---

## 0.4 — Distributed Systems Basics

### 0.4.1 — Exactly-Once vs At-Least-Once

```
┌─────────────────────────────────────────────────────────────────────┐
│              DELIVERY GUARANTEES                                   │
│                                                                     │
│  EXACTLY-ONCE: Message processed exactly one time                  │
│  ├── Very hard to guarantee (requires distributed transactions)    │
│  ├── Expensive (2-phase commit, consensus protocols)               │
│  └── Used for: financial transactions, inventory decrement         │
│                                                                     │
│  AT-LEAST-ONCE: Message processed one or more times               │
│  ├── Easy to implement (just retry on failure)                     │
│  ├── Efficient (no coordination needed)                            │
│  ├── BullMQ default behavior                                       │
│  └── REQUIRES: all operations to be idempotent                    │
│                                                                     │
│  AT-MOST-ONCE: Message processed zero or one times                │
│  ├── Simple but risky                                              │
│  └── Used for: fire-and-forget notifications                       │
│                                                                     │
│  For AI embedding pipelines:                                       │
│  → Use AT-LEAST-ONCE + make embedding idempotent                  │
│  → Check "already embedded" before embedding                       │
│  → Upsert (not insert) when storing embeddings                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.4.2 — Saga Pattern

```typescript
// Multi-step AI pipeline where any step might fail
// Saga ensures we can roll back or compensate

interface SagaStep {
  name: string;
  execute: (context: SagaContext) => Promise<SagaContext>;
  compensate: (context: SagaContext) => Promise<void>; // Undo
}

class DocumentProcessingSaga {
  private steps: SagaStep[] = [
    {
      name: "extract-text",
      execute: async (ctx) => {
        const text = await extractTextFromFile(ctx.filePath);
        // Save extraction so we can clean up on failure
        await db.processingArtifacts.create({
          data: { documentId: ctx.documentId, type: "extracted_text", path: ctx.filePath },
        });
        return { ...ctx, extractedText: text };
      },
      compensate: async (ctx) => {
        // Nothing to undo — text extraction is read-only
      },
    },
    {
      name: "create-chunks",
      execute: async (ctx) => {
        const chunks = await createChunks(ctx.extractedText!);
        const saved = await db.documentChunks.createMany({ data: chunks });
        return { ...ctx, chunkIds: saved };
      },
      compensate: async (ctx) => {
        // Undo: delete the chunks we created
        await db.documentChunks.deleteMany({ where: { documentId: ctx.documentId } });
      },
    },
    {
      name: "embed-chunks",
      execute: async (ctx) => {
        const embeddings = await generateEmbeddings(ctx.chunkIds!);
        await storeEmbeddings(embeddings);
        return { ...ctx, embeddingsStored: true };
      },
      compensate: async (ctx) => {
        // Undo: delete from vector DB
        await vectorDB.deleteByDocumentId(ctx.documentId, ctx.tenantId);
      },
    },
    {
      name: "update-status",
      execute: async (ctx) => {
        await db.documents.update({
          where: { id: ctx.documentId },
          data: { status: "indexed", indexedAt: new Date() },
        });
        return { ...ctx, success: true };
      },
      compensate: async (ctx) => {
        await db.documents.update({
          where: { id: ctx.documentId },
          data: { status: "failed" },
        });
      },
    },
  ];

  async execute(initialContext: SagaContext): Promise<SagaContext> {
    const completedSteps: SagaStep[] = [];
    let context = initialContext;

    for (const step of this.steps) {
      try {
        context = await step.execute(context);
        completedSteps.push(step);
      } catch (error: any) {
        // Step failed — compensate all completed steps in reverse
        console.error(`Saga step "${step.name}" failed: ${error.message}`);

        for (const completedStep of [...completedSteps].reverse()) {
          try {
            await completedStep.compensate(context);
          } catch (compensateError: any) {
            console.error(`Compensation failed for "${completedStep.name}":`, compensateError);
            // Log for manual review — this is an inconsistent state
          }
        }

        throw error;
      }
    }

    return context;
  }
}
```

### 0.4.3 — Outbox Pattern

```typescript
// Problem: We need to save to DB AND publish event atomically.
// If server crashes between save and publish, we lose the event.

// OUTBOX PATTERN: Write event to DB table in same transaction,
// then a separate process polls and publishes.

// Database schema:
// CREATE TABLE outbox_events (
//   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
//   aggregate_type TEXT NOT NULL,
//   aggregate_id TEXT NOT NULL,
//   event_type TEXT NOT NULL,
//   payload JSONB NOT NULL,
//   created_at TIMESTAMPTZ DEFAULT NOW(),
//   published_at TIMESTAMPTZ, -- NULL until published
//   retry_count INTEGER DEFAULT 0
// );

async function saveDocumentAndQueueEmbedding(
  document: Document,
  tenantId: string
): Promise<void> {
  // ATOMIC: both operations succeed or both fail
  await db.$transaction([
    // Save the document
    db.documents.create({ data: { ...document, tenantId } }),

    // Write to outbox (same transaction!)
    db.outboxEvents.create({
      data: {
        aggregateType: "document",
        aggregateId: document.id,
        eventType: "document.created",
        payload: { documentId: document.id, tenantId, filePath: document.filePath },
      },
    }),
  ]);
  // If any of the above fails, NEITHER happens — no lost events!
}

// Outbox processor (runs separately, continuously)
async function processOutbox(): Promise<void> {
  while (true) {
    const events = await db.outboxEvents.findMany({
      where: {
        publishedAt: null,
        retryCount: { lt: 5 },
      },
      take: 100,
      orderBy: { createdAt: "asc" },
    });

    for (const event of events) {
      try {
        // Publish to message queue
        await embeddingQueue.add(event.eventType, event.payload);

        // Mark as published
        await db.outboxEvents.update({
          where: { id: event.id },
          data: { publishedAt: new Date() },
        });
      } catch (error) {
        await db.outboxEvents.update({
          where: { id: event.id },
          data: { retryCount: { increment: 1 } },
        });
      }
    }

    await sleep(1000); // Poll every second
  }
}
```

---

## 0.5 — Observability

### 0.5.1 — Structured Logging

See Phase 6 (Observability section) for complete Pino setup and the LLM call logging template.

**Quick reference — what to log at each level:**

```typescript
// INFO: Normal operations
logger.info({ event: "llm_call_completed", latencyMs: 1200 });
logger.info({ event: "document_indexed", chunkCount: 45 });
logger.info({ event: "user_signed_up", userId: "usr_123" });

// WARN: Recoverable issues  
logger.warn({ event: "cache_miss_high_rate", rate: 0.85, threshold: 0.5 });
logger.warn({ event: "slow_query", queryMs: 5000, query: "search_embeddings" });
logger.warn({ event: "fallback_provider_used", provider: "openai", reason: "anthropic_timeout" });

// ERROR: Unexpected failures
logger.error({ event: "embedding_failed", error: err.message, documentId: "doc_123" });
logger.error({ event: "payment_failed", error: err.message, userId: "usr_456" });

// FATAL: System-level failures
logger.fatal({ event: "database_connection_failed", error: err.message });
```

### 0.5.2 — Prometheus Metrics

See Phase 6 for complete metric definitions. Quick summary:

```typescript
// THE 4 GOLDEN SIGNALS for AI services:
// 1. LATENCY: How long do requests take? (histogram p50/p95/p99)
// 2. TRAFFIC: How many requests/second? (counter per endpoint)
// 3. ERRORS: What is the error rate? (counter with error type label)
// 4. SATURATION: How full is the system? (queue depth, connection pool)
```

---

## 0.6 — CI/CD for AI Systems

### 0.6.1 — The Complete Pipeline

See Phase 6 for the complete GitHub Actions implementation.

**The minimum viable AI CI pipeline (implement this today):**

```yaml
# .github/workflows/ci.yml — minimum viable version
name: CI
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - run: npm ci
      - run: npm run type-check  # tsc --noEmit
      - run: npm run lint         # eslint
      - run: npm run test         # vitest

  evals:
    runs-on: ubuntu-latest
    if: contains(github.event.head_commit.modified, 'prompts/')
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npx promptfoo eval --output evals/results.json
        env: { ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }} }
      - run: |
          SCORE=$(cat evals/results.json | jq '.stats.successes / .stats.total')
          if (( $(echo "$SCORE < 0.85" | bc -l) )); then exit 1; fi
```

---

## 🔨 Layer 0 Project

**Build a TypeScript AI client library (`@yourname/ai-client`):**

```typescript
// What it must include:

// 1. Typed client wrapping OpenAI and Anthropic
// 2. Middleware composition (logging, caching, rate limiting, PII scrubbing)
// 3. Zod validation on ALL outputs
// 4. Discriminated union error handling
// 5. Branded types for IDs
// 6. Exponential backoff retry
// 7. Cost tracking
// 8. Published to npm

// The API should look like:
const client = createAIClient({
  providers: { anthropic: process.env.ANTHROPIC_API_KEY },
  middleware: [withLogging(logger), withSemanticCache(redis), withRateLimit(redis)],
  defaultModel: "claude-sonnet-4-20250514",
});

const result = await client.complete<InvoiceData>({
  prompt: "Extract invoice data...",
  schema: InvoiceSchema,
  maxRetries: 3,
});

// result is fully typed as InvoiceData — guaranteed by Zod at runtime
```

---

## ✅ Master Checklist

Before these are truly complete:

**TypeScript**
- [ ] All projects use `strict: true` with zero `any`
- [ ] Every LLM output validated with Zod schema
- [ ] All critical IDs are branded types
- [ ] Error handling uses discriminated unions
- [ ] LLM client uses generic type parameter for response type

**System Design**
- [ ] Can explain CAP theorem with a real example from your project
- [ ] Can implement token bucket, sliding window, and leaky bucket
- [ ] Designed at least one sharded data model on paper
- [ ] Can draw an event-driven AI pipeline diagram
- [ ] All AI jobs are idempotent (safe to run twice)
- [ ] Back pressure implemented (queue depth limit + 503 response)

**Databases**
- [ ] Can read `EXPLAIN ANALYZE` output and identify Seq Scans
- [ ] Created an HNSW index and set ef_search for accuracy tradeoff
- [ ] Implemented at least 5 different Redis patterns from scratch
- [ ] Written a MongoDB aggregation pipeline with $lookup and $group
- [ ] Created table partitioning for a time-series table

**Distributed Systems**
- [ ] All queue consumers are idempotent
- [ ] Implemented the saga pattern for one multi-step workflow
- [ ] Implemented the outbox pattern for at-least-once event delivery
- [ ] Distributed lock using Redis SET NX EX implemented

**Observability**
- [ ] Pino structured logging with all required fields
- [ ] Prometheus metrics exposed at /metrics
- [ ] Langfuse traces connected to structured logs via traceId

**CI/CD**
- [ ] GitHub Actions pipeline: lint → type-check → unit tests → evals
- [ ] Eval gate blocking PRs on quality regression
- [ ] Docker multi-stage build for production

---

*Layer 0 complete. This is the foundation that makes everything in Phases 1-8 actually work at production scale. The engineers who skip this layer hit a ceiling at mid-level. The engineers who master it become staff engineers.*
