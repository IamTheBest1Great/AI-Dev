# 🟡 Phase 2 — Core AI Patterns
> **Goal:** Build the backbone of real AI products — search, memory, retrieval, and structured AI output
> **Timeline:** Weeks 4–8
> **Outcome:** You can build production RAG systems, implement hybrid search, validate structured LLM output, and measure quality with RAGAS.

---

## 📚 Table of Contents

1. [2.1 — Embeddings & Vector Search](#21--embeddings--vector-search)
2. [2.2 — Vector Databases](#22--vector-databases)
3. [2.3 — RAG (Retrieval-Augmented Generation)](#23--rag-retrieval-augmented-generation)
4. [2.4 — Structured Outputs & Validation](#24--structured-outputs--validation)
5. [2.5 — Frameworks](#25--frameworks)
6. [2.6 — Tool Calling / Function Calling](#26--tool-calling--function-calling)
7. [2.7 — Knowledge Graphs (Advanced)](#27--knowledge-graphs-advanced)
8. [2.8 — What Most Developers Miss](#28--what-most-developers-miss)
9. [Phase 2 Projects](#-phase-2-projects)
10. [Master Checklist](#-master-checklist)

---

## The Big Picture: Why RAG Exists

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE KNOWLEDGE PROBLEM                           │
│                                                                     │
│  LLM Training                     Your Business                    │
│  ─────────────                     ────────────                    │
│  Cutoff: Aug 2024                  Has docs from today            │
│  Public internet data              Has private internal docs       │
│  No access to your data            Has customer-specific data      │
│  Hallucinate when uncertain        Needs accurate answers          │
│                                                                     │
│  WITHOUT RAG:                      WITH RAG:                       │
│  "Q: What is our refund policy?"   "Q: What is our refund policy?" │
│  "A: Most companies offer 30..."   "A: Per your policy doc: full   │
│  (hallucination — made up!)         refund within 14 days..."      │
│                                    (grounded in YOUR documents!)   │
│                                                                     │
│  RAG = Let LLM use YOUR documents to answer questions              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2.1 — Embeddings & Vector Search

### 2.1.1 — What Embeddings Are

An embedding converts text into a list of numbers (a vector) that captures the **meaning** of the text, not just the words.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TEXT → VECTOR TRANSFORMATION                    │
│                                                                     │
│  "The dog chased the cat"                                          │
│         │                                                           │
│         ▼  [Embedding Model]                                       │
│  [0.23, -0.45, 0.12, 0.78, ..., -0.34]  ← 1536 numbers            │
│                                                                     │
│  "A puppy ran after a kitten"                                      │
│         │                                                           │
│         ▼  [Embedding Model]                                       │
│  [0.21, -0.43, 0.14, 0.76, ..., -0.32]  ← very similar numbers!   │
│                                                                     │
│  "Quantum physics equations"                                        │
│         │                                                           │
│         ▼  [Embedding Model]                                       │
│  [-0.67, 0.89, -0.23, 0.11, ..., 0.55]  ← very different numbers! │
│                                                                     │
│  KEY INSIGHT: Similar meanings → similar vectors → close in space  │
└─────────────────────────────────────────────────────────────────────┘
```

**The vector space intuition:**

```
┌────────────────────────────────────────────────────────────────────┐
│              SEMANTIC VECTOR SPACE (2D simplified)                │
│                                                                    │
│         animals                    technology                      │
│            │                           │                           │
│     cat ●  │  ● dog                    │  ● computer               │
│            │                           │                           │
│     kitten ●  ● puppy              phone ●  ● laptop               │
│                                                                    │
│  "cat" and "kitten" are CLOSE (similar meaning)                   │
│  "cat" and "computer" are FAR APART (different meaning)           │
│                                                                    │
│  Vector arithmetic:                                               │
│  king - man + woman ≈ queen                                       │
│  Paris - France + Italy ≈ Rome                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 2.1.2 — Generating Embeddings

```typescript
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Single embedding
async function embedText(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",  // 1536 dimensions, $0.02/1M tokens
    // model: "text-embedding-3-large",  // 3072 dimensions, $0.13/1M tokens
    input: text.trim().slice(0, 8191), // max 8192 tokens
  });
  return response.data[0].embedding;
}

// ALWAYS batch embed when processing documents — much cheaper
async function embedBatch(texts: string[]): Promise<number[][]> {
  // OpenAI allows up to 2048 inputs per call
  const BATCH_SIZE = 100;
  const results: number[][] = [];

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    const batch = texts.slice(i, i + BATCH_SIZE);

    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: batch,
    });

    // Sort by index to maintain order
    const sorted = response.data.sort((a, b) => a.index - b.index);
    results.push(...sorted.map((d) => d.embedding));

    // Rate limit protection
    if (i + BATCH_SIZE < texts.length) {
      await sleep(200); // 200ms between batches
    }
  }

  return results;
}

// Cost calculation:
// text-embedding-3-small: $0.02 per 1M tokens
// A 10-page PDF ≈ 5000 tokens
// 1000 PDFs × 5000 tokens = 5M tokens = $0.10 total
// Embedding is CHEAP. Don't optimize prematurely.
```

**Alternative embedding providers:**

```typescript
// Cohere (better for search retrieval tasks)
import { CohereClient } from "cohere-ai";
const cohere = new CohereClient({ token: process.env.COHERE_API_KEY });

async function embedWithCohere(
  texts: string[],
  type: "search_document" | "search_query"  // CRITICAL: use correct type!
): Promise<number[][]> {
  const response = await cohere.embed({
    texts,
    model: "embed-english-v3.0",
    inputType: type,
    // "search_document" → for content being stored
    // "search_query" → for the user's question
    // Using wrong type reduces accuracy by ~10-15%!
  });
  return response.embeddings as number[][];
}

// Local embeddings with Ollama (free, private)
const ollama = new OpenAI({
  baseURL: "http://localhost:11434/v1",
  apiKey: "ollama",
});

async function embedLocally(text: string): Promise<number[]> {
  const response = await ollama.embeddings.create({
    model: "nomic-embed-text",  // Pull with: ollama pull nomic-embed-text
    input: text,
  });
  return response.data[0].embedding;
}
```

### 2.1.3 — Similarity Metrics

```typescript
// COSINE SIMILARITY — most common, works for normalized vectors
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) throw new Error("Vectors must be same dimension");

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  if (normA === 0 || normB === 0) return 0;

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  // Returns: -1 (opposite) to 1 (identical)
  // Good threshold for RAG: > 0.75
}

// DOT PRODUCT — faster, but requires normalized vectors
function dotProduct(vecA: number[], vecB: number[]): number {
  return vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
  // If vectors are L2-normalized, dot product = cosine similarity
}

// EUCLIDEAN DISTANCE — less common for text similarity
function euclideanDistance(vecA: number[], vecB: number[]): number {
  return Math.sqrt(
    vecA.reduce((sum, val, i) => sum + Math.pow(val - vecB[i], 2), 0)
  );
  // Lower = more similar (opposite of cosine)
}

// Practical usage:
const queryEmbedding = await embedText("What is the refund policy?");
const docEmbedding = await embedText("We offer full refunds within 14 days.");

const similarity = cosineSimilarity(queryEmbedding, docEmbedding);
console.log(`Similarity: ${similarity.toFixed(4)}`); // ~0.85 (very similar)
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              SIMILARITY SCORES REFERENCE                           │
│                                                                     │
│  Score    Meaning                         Use case                 │
│  ───────  ──────────────────────────────  ──────────────────────── │
│  0.90+    Near identical meaning          Duplicate detection       │
│  0.80-0.9 Strongly related               High-confidence RAG hit   │
│  0.70-0.8 Related topic                  Good RAG result           │
│  0.60-0.7 Loosely related               Use with caution          │
│  < 0.60   Not meaningfully related      Filter out                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1.4 — Chunking Strategies

This is the #1 place RAG systems fail. Poor chunking = poor retrieval = bad answers.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHY CHUNKING MATTERS                            │
│                                                                     │
│  Document: 50-page technical manual                                │
│                                                                     │
│  BAD: Embed the whole 50-page document as one chunk               │
│  Problem: When you search "how to reset password", you get        │
│  the entire manual back. The LLM drowns in irrelevant context.    │
│                                                                     │
│  BAD: Split into 50-character chunks                               │
│  Problem: "To reset your pass" — loses context. Meaning broken.   │
│                                                                     │
│  GOOD: Split into 400-token semantic chunks with overlap          │
│  "To reset your password: 1) Go to Settings 2) Click Security    │
│  3) Select Reset Password 4) Enter your email..."                 │
│  → Retrievable, meaningful, complete thought                      │
└─────────────────────────────────────────────────────────────────────┘
```

**All chunking strategies with code:**

```typescript
import { encode } from "gpt-tokenizer";

// STRATEGY 1: Fixed-size chunking (simple, but often wrong)
function chunkFixed(
  text: string,
  chunkSize = 500,      // tokens
  overlap = 50          // token overlap between chunks
): string[] {
  const tokens = encode(text);
  const chunks: string[] = [];

  for (let i = 0; i < tokens.length; i += chunkSize - overlap) {
    const chunkTokens = tokens.slice(i, i + chunkSize);
    // Convert token IDs back to text
    const chunkText = decode(chunkTokens);
    chunks.push(chunkText.trim());
  }

  return chunks.filter((c) => c.length > 0);
}
// Problem: May split in the middle of a sentence or paragraph

// STRATEGY 2: Recursive chunking (much better — respects structure)
function chunkRecursive(
  text: string,
  maxTokens = 400,
  separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
): string[] {
  // Count tokens in text
  const tokenCount = encode(text).length;

  // Base case: text fits in one chunk
  if (tokenCount <= maxTokens) {
    return text.trim() ? [text.trim()] : [];
  }

  // Try splitting by each separator in order
  for (const separator of separators) {
    const parts = text.split(separator);

    // Check if splitting actually helped
    if (parts.length <= 1) continue;

    // Merge small parts back together up to maxTokens
    const chunks: string[] = [];
    let currentChunk = "";

    for (const part of parts) {
      const candidate = currentChunk
        ? currentChunk + separator + part
        : part;

      if (encode(candidate).length <= maxTokens) {
        currentChunk = candidate;
      } else {
        if (currentChunk) chunks.push(currentChunk.trim());
        // Part itself might be too long — recurse
        if (encode(part).length > maxTokens) {
          chunks.push(...chunkRecursive(part, maxTokens, separators.slice(1)));
        } else {
          currentChunk = part;
        }
      }
    }

    if (currentChunk) chunks.push(currentChunk.trim());
    return chunks.filter((c) => c.length > 0);
  }

  // Last resort: character split
  return chunkFixed(text, maxTokens);
}

// STRATEGY 3: Semantic chunking (best quality, more expensive)
async function chunkSemantic(
  text: string,
  maxTokens = 400
): Promise<string[]> {
  // Split into sentences first
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];

  // Embed each sentence
  const embeddings = await embedBatch(sentences);

  // Find semantic boundaries (where topic changes significantly)
  const chunks: string[] = [];
  let currentChunk = sentences[0];
  let currentChunkTokens = encode(sentences[0]).length;

  for (let i = 1; i < sentences.length; i++) {
    const similarity = cosineSimilarity(embeddings[i - 1], embeddings[i]);
    const isTopicChange = similarity < 0.85; // Threshold for topic shift
    const wouldExceedLimit =
      currentChunkTokens + encode(sentences[i]).length > maxTokens;

    if (isTopicChange || wouldExceedLimit) {
      chunks.push(currentChunk.trim());
      currentChunk = sentences[i];
      currentChunkTokens = encode(sentences[i]).length;
    } else {
      currentChunk += " " + sentences[i];
      currentChunkTokens += encode(sentences[i]).length;
    }
  }

  if (currentChunk) chunks.push(currentChunk.trim());
  return chunks;
}

// STRATEGY 4: Parent-child chunking (best for production)
interface ChunkPair {
  parentId: string;
  parent: string;           // Large chunk (800-1000 tokens) — for context
  children: {
    id: string;
    content: string;        // Small chunk (100-200 tokens) — for retrieval
    embedding: number[];    // Embed the small chunk
  }[];
}

async function chunkParentChild(
  text: string,
  parentSize = 800,
  childSize = 150
): Promise<ChunkPair[]> {
  const parents = chunkRecursive(text, parentSize);
  const result: ChunkPair[] = [];

  for (const parent of parents) {
    const parentId = generateId();
    const children = chunkRecursive(parent, childSize);
    const childEmbeddings = await embedBatch(children);

    result.push({
      parentId,
      parent,
      children: children.map((content, i) => ({
        id: `${parentId}-${i}`,
        content,
        embedding: childEmbeddings[i],
      })),
    });
  }

  return result;
}
// Usage:
// 1. Store all parent chunks in DB
// 2. Embed and store child chunks in vector DB
// 3. On query: search child chunks → retrieve their parent → send parent to LLM
// This gives precise retrieval (small chunks) + full context (large parent)

// STRATEGY 5: Contextual chunking (2025 state of the art)
async function chunkContextual(
  text: string,
  document: string, // The full document for context
  chunkSize = 400
): Promise<{ content: string; contextualContent: string }[]> {
  const rawChunks = chunkRecursive(text, chunkSize);

  const contextualChunks = await Promise.all(
    rawChunks.map(async (chunk) => {
      // Ask Claude to add context to each chunk
      const context = await anthropic.messages.create({
        model: "claude-haiku-4", // Fast cheap model for this
        max_tokens: 100,
        system: "You add brief context to document chunks for search purposes.",
        messages: [
          {
            role: "user",
            content: `Document beginning:
${document.slice(0, 1000)}...

Chunk to contextualize:
<chunk>${chunk}</chunk>

In 1-2 sentences, explain what section this chunk is from and what it covers.
Be concise. Output only the contextual description.`,
          },
        ],
      });

      const contextText = context.content[0].text;
      return {
        content: chunk,
        contextualContent: `${contextText}\n\n${chunk}`, // Context + chunk combined
      };
    })
  );

  return contextualChunks;
}
// Contextual chunking improves retrieval accuracy by ~35% (Anthropic research)
```

### 2.1.5 — Chunk Quality Validation

```typescript
function validateChunk(chunk: string): {
  isValid: boolean;
  issues: string[];
} {
  const issues: string[] = [];
  const tokenCount = encode(chunk).length;

  if (tokenCount < 50) {
    issues.push(`Too small: ${tokenCount} tokens (minimum 50)`);
  }

  if (tokenCount > 1000) {
    issues.push(`Too large: ${tokenCount} tokens (maximum 1000)`);
  }

  if (chunk.trim().length < 30) {
    issues.push("Chunk is mostly whitespace");
  }

  // Check if it ends mid-sentence (sign of bad splitting)
  const lastChar = chunk.trim().slice(-1);
  if (![".", "!", "?", '"', "'", ")"].includes(lastChar)) {
    issues.push("Chunk may end mid-sentence");
  }

  return { isValid: issues.length === 0, issues };
}
```

---

## 2.2 — Vector Databases

### Choosing the Right Vector DB

```
┌─────────────────────────────────────────────────────────────────────┐
│              VECTOR DATABASE DECISION MATRIX                       │
│                                                                     │
│  Question                   Best Choice                            │
│  ──────────────────────────  ──────────────────────────────────    │
│  MERN stack, < 10M vectors   pgvector (Supabase)                   │
│  Need SQL joins on metadata  pgvector                              │
│  Scale > 10M vectors         Pinecone                              │
│  Need managed service        Pinecone                              │
│  Self-hosted, privacy-first  Qdrant                                │
│  GraphQL interface needed    Weaviate                              │
│  Just prototyping            Chroma (local)                        │
│  Already on Supabase         pgvector (free with Supabase)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2.1 — pgvector (PostgreSQL)

The best default choice. Already have PostgreSQL? Just add the extension.

```sql
-- Enable the extension
CREATE EXTENSION vector;

-- Create table with embedding column
CREATE TABLE document_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  document_id UUID NOT NULL REFERENCES documents(id),
  content TEXT NOT NULL,
  embedding vector(1536),         -- 1536 for text-embedding-3-small
  metadata JSONB DEFAULT '{}',
  search_vector tsvector           -- For hybrid search (full-text)
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  chunk_index INTEGER,             -- Position in document
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index (best for most cases)
CREATE INDEX ON document_chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
-- m = connections per node (higher = more accurate, more memory)
-- ef_construction = build quality (higher = better, slower to build)

-- IVFFlat index (for very large datasets > 1M vectors)
-- CREATE INDEX ON document_chunks
--   USING ivfflat (embedding vector_cosine_ops)
--   WITH (lists = 100);  -- sqrt(total_rows) is good default
-- IMPORTANT: IVFFlat needs ~39k rows before it's useful

-- Full-text search index for hybrid search
CREATE INDEX ON document_chunks USING gin(search_vector);

-- Always index tenant_id (for multi-tenant isolation)
CREATE INDEX ON document_chunks(tenant_id);
CREATE INDEX ON document_chunks(document_id);
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              HNSW vs IVFFlat — WHEN TO USE WHICH                   │
│                                                                     │
│  HNSW (Hierarchical Navigable Small World)                        │
│  ├── Better recall (finds more relevant results)                  │
│  ├── Faster queries                                                │
│  ├── Higher memory usage (~1.6 bytes per dim per vector)          │
│  ├── Can be built incrementally (insert one-by-one)               │
│  └── USE WHEN: < 1M vectors, memory is available                  │
│                                                                     │
│  IVFFlat (Inverted File with Flat vectors)                        │
│  ├── Lower memory (~0.5 bytes per dim per vector)                 │
│  ├── Requires re-training when data grows significantly           │
│  ├── Needs 39x nlist rows before building (~39k for lists=100)    │
│  └── USE WHEN: > 1M vectors, memory-constrained                   │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// TypeScript with pgvector using Drizzle ORM
import { pgTable, uuid, text, jsonb, integer } from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core";
import { db } from "@/lib/db";
import { cosineDistance, desc, gt, sql } from "drizzle-orm";

const documentChunks = pgTable("document_chunks", {
  id: uuid("id").primaryKey().defaultRandom(),
  tenantId: uuid("tenant_id").notNull(),
  documentId: uuid("document_id").notNull(),
  content: text("content").notNull(),
  embedding: vector("embedding", { dimensions: 1536 }),
  metadata: jsonb("metadata").default({}),
  chunkIndex: integer("chunk_index"),
});

// INSERT a chunk with embedding
async function insertChunk(
  tenantId: string,
  documentId: string,
  content: string,
  chunkIndex: number,
  metadata: Record<string, unknown> = {}
) {
  const embedding = await embedText(content);

  await db.insert(documentChunks).values({
    tenantId,
    documentId,
    content,
    embedding,
    metadata,
    chunkIndex,
  });
}

// SEARCH — vector similarity with tenant isolation
async function searchSimilar(
  tenantId: string,
  query: string,
  topK = 5,
  similarityThreshold = 0.7
) {
  const queryEmbedding = await embedText(query);

  const results = await db
    .select({
      id: documentChunks.id,
      content: documentChunks.content,
      metadata: documentChunks.metadata,
      similarity: sql<number>`1 - (${cosineDistance(documentChunks.embedding, queryEmbedding)})`,
    })
    .from(documentChunks)
    .where(
      sql`
        ${documentChunks.tenantId} = ${tenantId}
        AND 1 - (${cosineDistance(documentChunks.embedding, queryEmbedding)}) > ${similarityThreshold}
      `
    )
    .orderBy(cosineDistance(documentChunks.embedding, queryEmbedding))
    .limit(topK);

  return results;
}
```

### 2.2.2 — Pinecone

Best choice when you need managed scale (> 10M vectors) or simple namespace isolation.

```typescript
import { Pinecone } from "@pinecone-database/pinecone";

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
const index = pinecone.index("your-index-name");

// MULTI-TENANT: Use namespace per tenant
function getTenantIndex(tenantId: string) {
  return index.namespace(tenantId); // Each tenant completely isolated!
}

// UPSERT chunks
async function upsertChunks(
  tenantId: string,
  chunks: { id: string; content: string; metadata: Record<string, any> }[]
) {
  const embeddings = await embedBatch(chunks.map((c) => c.content));
  const tenantIndex = getTenantIndex(tenantId);

  // Upsert in batches of 100 (Pinecone limit)
  const BATCH_SIZE = 100;
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);

    await tenantIndex.upsert(
      batch.map((chunk, j) => ({
        id: chunk.id,
        values: embeddings[i + j],
        metadata: {
          content: chunk.content, // Store content in metadata for retrieval
          ...chunk.metadata,
        },
      }))
    );
  }
}

// QUERY
async function queryPinecone(
  tenantId: string,
  query: string,
  topK = 5,
  filter?: Record<string, any>
) {
  const queryEmbedding = await embedText(query);
  const tenantIndex = getTenantIndex(tenantId);

  const results = await tenantIndex.query({
    vector: queryEmbedding,
    topK,
    includeMetadata: true,
    filter, // e.g., { documentType: { $eq: "policy" } }
  });

  return results.matches.map((match) => ({
    id: match.id,
    score: match.score,
    content: match.metadata?.content as string,
    metadata: match.metadata,
  }));
}

// DELETE all chunks for a document
async function deleteDocumentChunks(tenantId: string, documentId: string) {
  const tenantIndex = getTenantIndex(tenantId);
  await tenantIndex.deleteMany({ filter: { documentId: { $eq: documentId } } });
}
```

---

## 2.3 — RAG (Retrieval-Augmented Generation)

### 2.3.1 — Basic RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPLETE RAG PIPELINE                           │
│                                                                     │
│  INDEXING PHASE (runs once per document):                          │
│  ─────────────────────────────────────                             │
│  Document                                                           │
│     │                                                               │
│     ▼                                                               │
│  [Pre-process] → Clean text, extract tables, OCR scanned images    │
│     │                                                               │
│     ▼                                                               │
│  [Chunk] → Split into 400-token chunks with overlap               │
│     │                                                               │
│     ▼                                                               │
│  [Embed] → Convert each chunk to 1536-dim vector                  │
│     │                                                               │
│     ▼                                                               │
│  [Store] → Vector DB (pgvector) + metadata in PostgreSQL           │
│                                                                     │
│  RETRIEVAL PHASE (runs on every query):                            │
│  ──────────────────────────────────────                            │
│  User Query                                                         │
│     │                                                               │
│     ▼                                                               │
│  [Embed Query] → Same model as document embeddings!               │
│     │                                                               │
│     ▼                                                               │
│  [Search] → Find top-5 most similar chunks                        │
│     │                                                               │
│     ▼                                                               │
│  [Build Prompt] → Inject retrieved chunks as context               │
│     │                                                               │
│     ▼                                                               │
│  [Generate] → LLM answers using ONLY provided context              │
│     │                                                               │
│     ▼                                                               │
│  [Validate] → Check answer is grounded in context                  │
│     │                                                               │
│     ▼                                                               │
│  Response to User                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Complete production RAG implementation:**

```typescript
class RAGSystem {
  private vectorDB: VectorDB;
  private llm: OpenAI;

  constructor(vectorDB: VectorDB, llm: OpenAI) {
    this.vectorDB = vectorDB;
    this.llm = llm;
  }

  // INDEX: Process and store a document
  async indexDocument(
    tenantId: string,
    documentId: string,
    content: string,
    metadata: Record<string, unknown> = {}
  ): Promise<{ chunksCreated: number }> {
    // Step 1: Chunk the document
    const chunks = chunkRecursive(content, 400);

    // Step 2: Validate chunks
    const validChunks = chunks.filter((chunk) => {
      const { isValid, issues } = validateChunk(chunk);
      if (!isValid) console.warn(`Invalid chunk: ${issues.join(", ")}`);
      return isValid;
    });

    // Step 3: Embed all chunks in batch (efficient)
    const embeddings = await embedBatch(validChunks);

    // Step 4: Store in vector DB
    await this.vectorDB.upsertMany(
      validChunks.map((content, i) => ({
        id: `${documentId}-chunk-${i}`,
        tenantId,
        documentId,
        content,
        embedding: embeddings[i],
        chunkIndex: i,
        metadata,
      }))
    );

    return { chunksCreated: validChunks.length };
  }

  // RETRIEVE: Find relevant chunks for a query
  async retrieve(
    tenantId: string,
    query: string,
    options: {
      topK?: number;
      similarityThreshold?: number;
      filter?: Record<string, unknown>;
    } = {}
  ): Promise<RetrievedChunk[]> {
    const { topK = 5, similarityThreshold = 0.7, filter } = options;

    const queryEmbedding = await embedText(query);

    const results = await this.vectorDB.search(
      queryEmbedding,
      tenantId,
      topK * 2, // Retrieve extra, filter below threshold
      filter
    );

    return results
      .filter((r) => r.similarity >= similarityThreshold)
      .slice(0, topK);
  }

  // GENERATE: Use retrieved context to answer the query
  async query(
    tenantId: string,
    query: string,
    conversationHistory: Message[] = []
  ): Promise<{ answer: string; sources: string[]; tokensUsed: number }> {
    // Step 1: Retrieve relevant chunks
    const chunks = await this.retrieve(tenantId, query);

    if (chunks.length === 0) {
      return {
        answer:
          "I don't have information about that in my knowledge base. Please contact support.",
        sources: [],
        tokensUsed: 0,
      };
    }

    // Step 2: Build context string with source citations
    const context = chunks
      .map(
        (chunk, i) =>
          `[Source ${i + 1}] (Relevance: ${(chunk.similarity * 100).toFixed(0)}%)\n${chunk.content}`
      )
      .join("\n\n---\n\n");

    // Step 3: Build the RAG prompt
    const systemPrompt = `You are a helpful assistant with access to a knowledge base.

INSTRUCTIONS:
- Answer questions using ONLY the provided context below
- If the answer isn't in the context, say "I don't have that information"
- Always cite your sources using [Source N] notation
- Do not hallucinate or add information not in the context
- Be concise and direct

CONTEXT:
${context}`;

    // Step 4: Call LLM
    const response = await this.llm.chat.completions.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 500,
      messages: [
        ...conversationHistory,
        { role: "user", content: query },
      ],
      system: systemPrompt,
    });

    const answer = response.choices[0].message.content || "";

    return {
      answer,
      sources: chunks.map((c) => c.metadata?.sourceUrl || c.id),
      tokensUsed: response.usage?.total_tokens || 0,
    };
  }
}
```

### 2.3.2 — Advanced RAG Techniques

#### HyDE — Hypothetical Document Embeddings

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHY HyDE WORKS                                  │
│                                                                     │
│  PROBLEM: User queries are short and vague.                        │
│  "password reset" → doesn't embed similarly to                     │
│  "To reset your password, navigate to Settings > Security..."     │
│                                                                     │
│  SOLUTION: Generate a hypothetical answer, then embed THAT.        │
│                                                                     │
│  Query: "password reset"                                           │
│     ↓                                                               │
│  Generate hypothetical answer:                                      │
│  "To reset your password: 1) Go to Settings                        │
│   2) Click Security 3) Enter email..."                             │
│     ↓                                                               │
│  Embed the hypothetical answer                                      │
│     ↓                                                               │
│  Search with that embedding → finds the REAL document!             │
│                                                                     │
│  Why it works: Hypothetical answer is in the SAME semantic space   │
│  as the real documentation. Query alone is in user-speak space.   │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
async function hydeSearch(
  tenantId: string,
  query: string,
  topK = 5
): Promise<RetrievedChunk[]> {
  // Step 1: Generate a hypothetical answer
  const hypotheticalAnswer = await callLLM(`
You are helping with a document search task.
Generate a short paragraph (3-5 sentences) that would be a perfect answer to this question.
Write it as if it were extracted from a help document or knowledge base.
Focus on the type of content that would realistically exist in documentation.

Question: ${query}

Hypothetical document excerpt:`);

  // Step 2: Embed the HYPOTHETICAL ANSWER (not the query!)
  const hydeEmbedding = await embedText(hypotheticalAnswer);

  // Step 3: Search with hypothetical embedding
  const results = await vectorDB.search(hydeEmbedding, tenantId, topK);

  return results;
}
```

#### Multi-Query Retrieval

```typescript
async function multiQuerySearch(
  tenantId: string,
  query: string,
  topK = 5
): Promise<RetrievedChunk[]> {
  // Step 1: Generate query variants
  const variantsResponse = await callLLM(`
Generate 3 different search queries that would help answer this question.
Each query should approach the topic from a different angle.
Return as JSON array: ["query1", "query2", "query3"]

Original question: ${query}

JSON:`);

  let variants: string[];
  try {
    variants = JSON.parse(variantsResponse);
  } catch {
    variants = [query]; // Fallback to original
  }

  const allQueries = [query, ...variants.slice(0, 3)];

  // Step 2: Run all queries in parallel
  const allResults = await Promise.all(
    allQueries.map((q) => vectorDB.search(
      await embedText(q),
      tenantId,
      topK
    ))
  );

  // Step 3: Merge with Reciprocal Rank Fusion
  return reciprocalRankFusion(allResults, topK);
}

// RRF: Robust way to merge multiple ranked lists
function reciprocalRankFusion(
  resultSets: RetrievedChunk[][],
  topK: number,
  k = 60  // RRF constant — 60 is the standard
): RetrievedChunk[] {
  const scores = new Map<string, number>();
  const chunkMap = new Map<string, RetrievedChunk>();

  for (const results of resultSets) {
    results.forEach((chunk, rank) => {
      const current = scores.get(chunk.id) || 0;
      scores.set(chunk.id, current + 1 / (k + rank + 1));
      chunkMap.set(chunk.id, chunk);
    });
  }

  return [...chunkMap.values()]
    .sort((a, b) => (scores.get(b.id) || 0) - (scores.get(a.id) || 0))
    .slice(0, topK);
}
```

#### Cohere Re-ranking

```typescript
import { CohereClient } from "cohere-ai";
const cohere = new CohereClient({ token: process.env.COHERE_API_KEY! });

async function rerankResults(
  query: string,
  chunks: RetrievedChunk[],
  topN = 5
): Promise<RetrievedChunk[]> {
  if (chunks.length === 0) return [];

  // Cohere's cross-encoder compares query+document TOGETHER
  // Much more accurate than cosine similarity alone (~15% improvement)
  const response = await cohere.rerank({
    model: "rerank-english-v3.0",
    query,
    documents: chunks.map((c) => c.content),
    topN,
    returnDocuments: false,
  });

  // Map reranked indices back to original chunks
  return response.results.map((result) => ({
    ...chunks[result.index],
    similarity: result.relevanceScore, // Replace cosine score with Cohere score
  }));
}

// PRODUCTION PATTERN: Vector search → Rerank
async function searchWithRerank(
  tenantId: string,
  query: string,
  finalTopK = 5
): Promise<RetrievedChunk[]> {
  // Step 1: Get more candidates than needed
  const candidates = await vectorDB.search(
    await embedText(query),
    tenantId,
    finalTopK * 4 // Get 20 candidates for 5 final results
  );

  // Step 2: Rerank with Cohere's cross-encoder
  const reranked = await rerankResults(query, candidates, finalTopK);

  return reranked;
}
```

#### Hybrid Search (Vector + BM25)

```
┌─────────────────────────────────────────────────────────────────────┐
│              WHY HYBRID SEARCH BEATS PURE VECTOR SEARCH            │
│                                                                     │
│  Vector Search (Semantic):                                         │
│  ✓ "What is Anthropic's refund policy?" → finds "money back"       │
│  ✗ "API key SKxxxxxx" → doesn't find exact API key reference       │
│                                                                     │
│  Keyword Search (BM25):                                            │
│  ✓ "SKxxxxxx" → finds exact match                                  │
│  ✗ "money back" → doesn't find "refund"                            │
│                                                                     │
│  Hybrid (both combined):                                           │
│  ✓ Semantic understanding + exact keyword matching                 │
│  ✓ Best for production RAG systems                                  │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
async function hybridSearch(
  tenantId: string,
  query: string,
  topK = 5
): Promise<RetrievedChunk[]> {
  const queryEmbedding = await embedText(query);

  // Run both searches in parallel
  const [semanticResults, keywordResults] = await Promise.all([
    // Vector search (semantic)
    vectorDB.search(queryEmbedding, tenantId, topK * 2),

    // Full-text search (BM25 in PostgreSQL)
    db.execute(sql`
      SELECT
        id,
        content,
        metadata,
        ts_rank(search_vector, plainto_tsquery('english', ${query})) as bm25_score
      FROM document_chunks
      WHERE
        tenant_id = ${tenantId}
        AND search_vector @@ plainto_tsquery('english', ${query})
      ORDER BY bm25_score DESC
      LIMIT ${topK * 2}
    `),
  ]);

  // Combine with RRF
  const combined = reciprocalRankFusion(
    [semanticResults, keywordResults],
    topK
  );

  return combined;
}
```

#### Contextual Compression

```typescript
async function compressContext(
  query: string,
  chunks: RetrievedChunk[]
): Promise<string[]> {
  // For each chunk, extract ONLY the relevant parts
  const compressed = await Promise.all(
    chunks.map(async (chunk) => {
      const response = await callLLMCheap(`
Extract only the parts of this text that are directly relevant to answering the question.
If nothing is relevant, respond with exactly "IRRELEVANT".
Keep your response to 2-3 sentences maximum.

Question: ${query}

Text:
${chunk.content}

Relevant parts:`);

      return response === "IRRELEVANT" ? null : response;
    })
  );

  return compressed.filter((c): c is string => c !== null);
}
// This reduces context window usage by 40-60%
// Makes LLM more accurate by removing noise
```

#### Self-Querying Retrievers

```typescript
const MetadataFilterSchema = z.object({
  dateRange: z.enum(["today", "this_week", "this_month", "this_year"]).nullable(),
  documentType: z.enum(["policy", "tutorial", "faq", "contract"]).nullable(),
  department: z.enum(["hr", "engineering", "finance", "legal"]).nullable(),
  author: z.string().nullable(),
});

async function selfQueryingSearch(
  tenantId: string,
  query: string
): Promise<RetrievedChunk[]> {
  // Let the LLM extract metadata filters from natural language
  const filterResponse = await callLLM(`
Extract any metadata filters from this search query.
If no specific filter is mentioned, use null.
Return ONLY valid JSON.

Query: "${query}"

Available filters:
- dateRange: "today" | "this_week" | "this_month" | "this_year" | null
- documentType: "policy" | "tutorial" | "faq" | "contract" | null
- department: "hr" | "engineering" | "finance" | "legal" | null
- author: string | null

JSON:`);

  const filters = MetadataFilterSchema.safeParse(JSON.parse(filterResponse));

  // Build filter for vector DB
  const metadataFilter: Record<string, any> = {};
  if (filters.success) {
    if (filters.data.documentType)
      metadataFilter.documentType = filters.data.documentType;
    if (filters.data.department)
      metadataFilter.department = filters.data.department;
    // Add date filter logic here
  }

  // Search with extracted filters
  return await vectorDB.search(
    await embedText(query),
    tenantId,
    5,
    metadataFilter
  );
}
// Example: "Show me HR policies from last month"
// Extracts: { department: "hr", dateRange: "this_month" }
// Only searches in HR documents from this month — much faster + more accurate
```

---

## 2.4 — Structured Outputs & Validation

### 2.4.1 — Zod for LLM Output Validation

```typescript
import { z } from "zod";

// Define the schema FIRST, derive the type from it
const InvoiceSchema = z.object({
  invoiceNumber: z.string().regex(/^INV-\d+$/, "Invalid invoice format"),
  vendor: z.object({
    name: z.string().min(1),
    address: z.string().optional(),
    taxId: z.string().optional(),
  }),
  lineItems: z.array(
    z.object({
      description: z.string(),
      quantity: z.number().positive(),
      unitPrice: z.number().positive(),
      total: z.number().positive(),
    })
  ).min(1, "Must have at least one line item"),
  subtotal: z.number().positive(),
  taxAmount: z.number().min(0),
  total: z.number().positive(),
  dueDate: z.string().datetime().optional(),
  currency: z.enum(["USD", "EUR", "GBP", "INR"]).default("USD"),
}).refine(
  // Custom validation: total = subtotal + tax
  (data) => Math.abs(data.total - (data.subtotal + data.taxAmount)) < 0.01,
  "Total must equal subtotal + tax"
);

// TypeScript type is automatically derived!
type Invoice = z.infer<typeof InvoiceSchema>;

// Use it:
const prompt = `
Extract invoice data from the following text and return ONLY valid JSON.
The JSON must match this exact structure:
{
  "invoiceNumber": "INV-XXXXX",
  "vendor": { "name": "...", "address": "...", "taxId": "..." },
  "lineItems": [{ "description": "...", "quantity": 1, "unitPrice": 10.00, "total": 10.00 }],
  "subtotal": 0.00,
  "taxAmount": 0.00,
  "total": 0.00,
  "dueDate": "ISO datetime string or null",
  "currency": "USD|EUR|GBP|INR"
}

Invoice text:
${invoiceText}

JSON:`;

async function extractInvoice(invoiceText: string): Promise<Invoice> {
  const MAX_ATTEMPTS = 3;
  let lastError = "";

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const fullPrompt = attempt === 0
      ? prompt
      : prompt + `\n\nPrevious attempt failed: ${lastError}. Please fix the JSON.`;

    const raw = await callLLM(fullPrompt);

    // Try to extract JSON from response (model might add text around it)
    const jsonMatch = raw.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      lastError = "Response contained no JSON object";
      continue;
    }

    const parsed = InvoiceSchema.safeParse(JSON.parse(jsonMatch[0]));

    if (parsed.success) return parsed.data;

    // Format Zod errors for the next prompt
    lastError = parsed.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join(", ");
  }

  throw new Error(`Invoice extraction failed after ${MAX_ATTEMPTS} attempts`);
}
```

### 2.4.2 — Instructor.js

A wrapper library that handles retry logic automatically.

```typescript
import Instructor from "@instructor-ai/instructor";
import OpenAI from "openai";

const client = Instructor({
  client: new OpenAI({ apiKey: process.env.OPENAI_API_KEY! }),
  mode: "TOOLS", // Most reliable mode
  // Other modes: "JSON", "MD_JSON", "JSON_SCHEMA"
});

// Instructor handles retries and validation automatically!
const invoice = await client.chat.completions.create({
  model: "gpt-4o",
  response_model: {
    schema: InvoiceSchema,
    name: "Invoice",
  },
  max_retries: 3,  // Automatically retries with error feedback!
  messages: [
    {
      role: "user",
      content: `Extract invoice data from: ${invoiceText}`,
    },
  ],
});

// invoice is fully typed as Invoice!
console.log(invoice.total);     // TypeScript knows this is a number
console.log(invoice.vendor.name); // TypeScript knows this is a string
```

### 2.4.3 — JSON Mode vs Tool Calling

```typescript
// JSON MODE — forces valid JSON, but no schema enforcement
const jsonMode = await openai.chat.completions.create({
  model: "gpt-4o",
  response_format: { type: "json_object" }, // MUST mention JSON in prompt too
  messages: [
    {
      role: "user",
      content: `Return a JSON object with name and age. User: "John is 25".`,
    },
  ],
});
// Guaranteed to be valid JSON, but fields might be missing or wrong type

// TOOL CALLING — schema enforced, most reliable
const toolCalling = await openai.chat.completions.create({
  model: "gpt-4o",
  tools: [
    {
      type: "function",
      function: {
        name: "extract_person",
        description: "Extract person information from text",
        parameters: {
          type: "object",
          properties: {
            name: { type: "string", description: "Person's full name" },
            age: { type: "integer", description: "Person's age in years" },
          },
          required: ["name", "age"],
        },
      },
    },
  ],
  tool_choice: { type: "function", function: { name: "extract_person" } },
  messages: [{ role: "user", content: "John is 25 years old." }],
});

const toolArgs = JSON.parse(
  toolCalling.choices[0].message.tool_calls![0].function.arguments
);
// { name: "John", age: 25 } — schema enforced!

// RULE: Use JSON mode for simple outputs, tool calling for complex schemas
```

---

## 2.5 — Frameworks

### 2.5.1 — LangChain.js

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";

const model = new ChatOpenAI({ model: "gpt-4o" });
const outputParser = new StringOutputParser();

// LCEL (LangChain Expression Language) — composable chains with |
const chain = RunnableSequence.from([
  PromptTemplate.fromTemplate("Tell me a joke about {topic}"),
  model,
  outputParser,
]);

const result = await chain.invoke({ topic: "programming" });

// WHEN TO USE LangChain:
// ✓ Need pre-built document loaders (PDF, website, Notion, GitHub)
// ✓ Rapid prototyping with pre-built components
// ✓ Need pre-built retrievers, memory, etc.

// WHEN NOT TO USE LangChain:
// ✗ Performance-critical paths (abstractions add overhead)
// ✗ Need full control over prompts
// ✗ Complex debugging (abstractions hide what's happening)
// ✗ Simple single-LLM calls (just use OpenAI SDK directly)
```

### 2.5.2 — Framework Selection Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│              FRAMEWORK DECISION FLOWCHART                          │
│                                                                     │
│  Simple LLM call?                                                  │
│    YES → Use raw OpenAI/Anthropic SDK                              │
│    NO ↓                                                             │
│                                                                     │
│  Need stateful agent workflow with branching?                      │
│    YES → LangGraph                                                  │
│    NO ↓                                                             │
│                                                                     │
│  Heavy document processing (PDFs, websites, Notion)?              │
│    YES → LlamaIndex.ts                                             │
│    NO ↓                                                             │
│                                                                     │
│  Multi-agent with roles (researcher, writer, editor)?              │
│    YES → CrewAI                                                    │
│    NO ↓                                                             │
│                                                                     │
│  Need pre-built chains, retrievers, integrations?                 │
│    YES → LangChain.js                                              │
│    NO → Raw API + your own utilities                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2.6 — Tool Calling / Function Calling

### 2.6.1 — The Tool Call Cycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE TOOL CALL LOOP                              │
│                                                                     │
│  Your App                    LLM                                    │
│  ────────                    ───                                   │
│                                                                     │
│  1. Send messages + tool     "I need to call                       │
│     definitions    ───────►  search_docs({query: 'policy'})"       │
│                                                                     │
│  2. Receive tool_use block   ◄─────── stop_reason: "tool_use"      │
│     Append to history                                              │
│                                                                     │
│  3. Execute the tool                                               │
│     "search_docs called"                                           │
│     Returns: [doc1, doc2]                                          │
│                                                                     │
│  4. Send tool result         "Based on these docs:                 │
│     back to LLM   ───────►   The policy states..."                │
│                              ◄─────── stop_reason: "end_turn"      │
│                                                                     │
│  5. Final answer to user                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.6.2 — Tool Schema Design

```typescript
// Tool descriptions ARE prompts — bad descriptions = bad tool selection!

// BAD tool schema:
const badTool = {
  name: "search",
  description: "Search for stuff",
  input_schema: {
    type: "object",
    properties: {
      q: { type: "string" },
    },
  },
};
// Model won't know when to use this or what "q" means

// GOOD tool schema:
const goodTool = {
  name: "search_knowledge_base",
  description: `Search the company knowledge base for relevant documentation.
  
  USE THIS TOOL when:
  - User asks about company policies, procedures, or products
  - User needs information from internal documentation
  - User references a specific feature or process
  
  DO NOT USE when:
  - Question requires real-time data (prices, news)
  - User asks general knowledge questions
  - Question is about external products you don't document`,

  input_schema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description:
          "Natural language search query. Be specific. Max 200 characters. " +
          "Example: 'password reset process for enterprise accounts'",
      },
      filter_category: {
        type: "string",
        enum: ["hr", "engineering", "finance", "legal", "general"],
        description:
          "Document category to filter search. Use 'general' if unsure which department.",
      },
      max_results: {
        type: "integer",
        description: "Maximum number of results to return. Default: 5, Max: 10.",
        default: 5,
        minimum: 1,
        maximum: 10,
      },
    },
    required: ["query"],
  },
};
```

### 2.6.3 — Complete Tool Execution Loop

```typescript
async function runAgentWithTools(
  userMessage: string,
  tools: Tool[],
  maxIterations = 10
): Promise<string> {
  const messages: Message[] = [{ role: "user", content: userMessage }];
  let iterations = 0;

  while (iterations < maxIterations) {
    iterations++;

    const response = await anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      tools,
      messages,
    });

    // Add assistant response to history
    messages.push({ role: "assistant", content: response.content });

    // If done — return the text response
    if (response.stop_reason === "end_turn") {
      const textBlock = response.content.find((b) => b.type === "text");
      return textBlock?.text || "";
    }

    // If tool call — execute and continue
    if (response.stop_reason === "tool_use") {
      const toolUseBlocks = response.content.filter(
        (b) => b.type === "tool_use"
      );

      // Execute ALL tool calls in PARALLEL (faster!)
      const toolResults = await Promise.all(
        toolUseBlocks.map(async (toolUse) => {
          if (toolUse.type !== "tool_use") return null;

          logger.info(
            { tool: toolUse.name, input: toolUse.input },
            "Executing tool"
          );

          let result: unknown;
          try {
            result = await executeToolSafely(toolUse.name, toolUse.input);
          } catch (error: any) {
            // Don't let tool errors crash the loop
            // Return error to LLM — it can try a different approach
            result = {
              error: true,
              message: error.message,
              suggestion: "Try with different parameters or use a different approach",
            };
          }

          return {
            type: "tool_result" as const,
            tool_use_id: toolUse.id,
            content: JSON.stringify(result),
          };
        })
      );

      // Add all tool results to history
      messages.push({
        role: "user",
        content: toolResults.filter(Boolean),
      });
    }
  }

  throw new Error(`Agent exceeded max iterations (${maxIterations})`);
}

// Safe tool executor with validation
async function executeToolSafely(
  name: string,
  input: unknown
): Promise<unknown> {
  const toolRegistry: Record<string, (input: unknown) => Promise<unknown>> = {
    search_knowledge_base: (i) => searchKnowledgeBase(i as SearchInput),
    get_weather: (i) => getWeather(i as WeatherInput),
    create_ticket: (i) => createTicket(i as TicketInput),
    send_email: async (i) => {
      // ALWAYS validate before side effects!
      const validated = SendEmailSchema.parse(i);
      return sendEmail(validated);
    },
  };

  const toolFn = toolRegistry[name];
  if (!toolFn) throw new Error(`Unknown tool: ${name}`);

  return await toolFn(input);
}
```

---

## 2.7 — Knowledge Graphs (Advanced)

### 2.7.1 — Neo4j Basics

```
┌─────────────────────────────────────────────────────────────────────┐
│              KNOWLEDGE GRAPH vs VECTOR SEARCH                      │
│                                                                     │
│  Vector Search:                                                    │
│  "Who is the CEO of Acme?" → finds similar text → answers         │
│  Can't answer: "Who manages the team that builds the API           │
│  that powers the feature that caused the outage?"                  │
│  (Multi-hop reasoning fails)                                       │
│                                                                     │
│  Knowledge Graph:                                                  │
│  CEO → leads → Company                                            │
│  Company → has → Team                                              │
│  Team → builds → Feature                                           │
│  Feature → had → Outage                                            │
│  → Can traverse relationships! Multi-hop = easy                   │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
import neo4j from "neo4j-driver";

const driver = neo4j.driver(
  process.env.NEO4J_URI!,
  neo4j.auth.basic(process.env.NEO4J_USER!, process.env.NEO4J_PASSWORD!)
);

// Extract entities and relationships with LLM
async function extractKnowledgeGraph(text: string): Promise<GraphData> {
  const response = await callLLM(`
Extract entities and relationships from this text.
Return as JSON:
{
  "entities": [
    { "id": "unique_id", "type": "Person|Company|Product|Concept", "name": "...", "attributes": {} }
  ],
  "relationships": [
    { "from": "entity_id", "to": "entity_id", "type": "WORKS_AT|CREATES|MANAGES|...", "attributes": {} }
  ]
}

Text: ${text}

JSON:`);

  return JSON.parse(response);
}

// Store in Neo4j
async function storeInNeo4j(graphData: GraphData) {
  const session = driver.session();

  try {
    await session.executeWrite(async (tx) => {
      // Create entities
      for (const entity of graphData.entities) {
        await tx.run(
          `MERGE (e:${entity.type} {id: $id})
           SET e.name = $name, e += $attributes`,
          { id: entity.id, name: entity.name, attributes: entity.attributes }
        );
      }

      // Create relationships
      for (const rel of graphData.relationships) {
        await tx.run(
          `MATCH (a {id: $from}), (b {id: $to})
           MERGE (a)-[r:${rel.type}]->(b)
           SET r += $attributes`,
          { from: rel.from, to: rel.to, attributes: rel.attributes }
        );
      }
    });
  } finally {
    await session.close();
  }
}

// Query with natural language via LLM
async function queryKnowledgeGraph(naturalLanguageQuery: string): Promise<string> {
  // Step 1: Convert NL to Cypher
  const cipherQuery = await callLLM(`
Convert this question to a Cypher query for a knowledge graph about companies, people, and products.
Return ONLY the Cypher query, no explanation.

Schema:
- (Person {name, role})
- (Company {name, industry})
- (Product {name, type})
- Relationships: WORKS_AT, MANAGES, CREATES, USES, COMPETES_WITH

Question: ${naturalLanguageQuery}

Cypher:`);

  // Step 2: Execute Cypher
  const session = driver.session();
  try {
    const result = await session.run(cipherQuery);
    const records = result.records.map((r) => r.toObject());

    // Step 3: Summarize results
    return await callLLM(`
Answer this question based on the database results.
Question: ${naturalLanguageQuery}
Results: ${JSON.stringify(records)}
Answer:`);
  } finally {
    await session.close();
  }
}
```

---

## 2.8 — What Most Developers Miss

### 2.8.1 — A/B Testing Chunk Strategies

```typescript
// Don't assume which chunking strategy is best — MEASURE IT
async function benchmarkChunkingStrategies(
  testDocuments: string[],
  testQueries: { query: string; expectedAnswer: string }[]
): Promise<ChunkingBenchmark> {
  const strategies = {
    fixed_400: (doc: string) => chunkFixed(doc, 400, 40),
    recursive_400: (doc: string) => chunkRecursive(doc, 400),
    fixed_200: (doc: string) => chunkFixed(doc, 200, 20),
    parent_child: async (doc: string) => {
      const pairs = await chunkParentChild(doc);
      return pairs.flatMap((p) => p.children.map((c) => c.content));
    },
  };

  const results: Record<string, number> = {};

  for (const [strategyName, chunkFn] of Object.entries(strategies)) {
    let totalScore = 0;

    // Test each strategy against all documents and queries
    for (const doc of testDocuments) {
      const chunks = await Promise.resolve(chunkFn(doc));
      const embeddings = await embedBatch(chunks);

      // Store in temp vector DB
      await tempDB.clear();
      await tempDB.insert(chunks.map((c, i) => ({ content: c, embedding: embeddings[i] })));

      for (const { query, expectedAnswer } of testQueries) {
        const retrieved = await tempDB.search(await embedText(query), 5);
        const context = retrieved.map((r) => r.content).join("\n");

        // Score: does context contain expected answer?
        const containsAnswer = context
          .toLowerCase()
          .includes(expectedAnswer.toLowerCase());
        totalScore += containsAnswer ? 1 : 0;
      }
    }

    results[strategyName] =
      totalScore / (testDocuments.length * testQueries.length);
  }

  return results;
}
// Results might look like:
// { fixed_400: 0.72, recursive_400: 0.84, fixed_200: 0.68, parent_child: 0.91 }
// → Use parent-child chunking!
```

### 2.8.2 — RAGAS Evaluation Framework

```typescript
// RAGAS measures 4 key metrics for RAG quality

// METRIC 1: Faithfulness (0-1)
// "Does the answer only contain information from the context?"
async function measureFaithfulness(
  question: string,
  answer: string,
  context: string[]
): Promise<number> {
  const score = await callLLM(`
Rate how faithfully this answer is grounded in the context (0-10).
10 = every claim is directly supported by context
0 = answer contains information not in context (hallucination!)

Question: ${question}
Context: ${context.join("\n")}
Answer: ${answer}

Score (0-10, nothing else):`);

  return parseInt(score.trim()) / 10;
}

// METRIC 2: Answer Relevancy (0-1)
// "Does the answer actually address the question?"
async function measureAnswerRelevancy(
  question: string,
  answer: string
): Promise<number> {
  // Generate fake questions that the answer would address
  const generatedQuestions = await callLLM(`
Generate 3 questions that this answer would perfectly answer.
Return as JSON array: ["question1", "question2", "question3"]

Answer: ${answer}

JSON:`);

  const fakeQuestions: string[] = JSON.parse(generatedQuestions);

  // Measure similarity between real question and generated questions
  const realEmbedding = await embedText(question);
  const fakeEmbeddings = await embedBatch(fakeQuestions);

  const avgSimilarity =
    fakeEmbeddings.reduce(
      (sum, emb) => sum + cosineSimilarity(realEmbedding, emb),
      0
    ) / fakeEmbeddings.length;

  return avgSimilarity;
}

// METRIC 3: Context Precision (0-1)
// "Are the retrieved chunks actually relevant?"
async function measureContextPrecision(
  question: string,
  context: string[]
): Promise<number> {
  let relevantCount = 0;

  for (const chunk of context) {
    const isRelevant = await callLLM(`
Is this context chunk useful for answering the question?
Answer with only "yes" or "no".

Question: ${question}
Chunk: ${chunk}

Answer:`);

    if (isRelevant.trim().toLowerCase() === "yes") relevantCount++;
  }

  return relevantCount / context.length;
}

// METRIC 4: Context Recall (0-1)
// "Did we retrieve all the information needed?"
async function measureContextRecall(
  question: string,
  groundTruthAnswer: string,
  context: string[]
): Promise<number> {
  const sentences = groundTruthAnswer.match(/[^.!?]+[.!?]+/g) || [];
  let supportedCount = 0;

  for (const sentence of sentences) {
    const isSupported = await callLLM(`
Can this statement be attributed to the provided context?
Answer with only "yes" or "no".

Statement: ${sentence}
Context: ${context.join("\n")}

Answer:`);

    if (isSupported.trim().toLowerCase() === "yes") supportedCount++;
  }

  return sentences.length > 0 ? supportedCount / sentences.length : 0;
}

// Run full RAGAS evaluation:
async function runRAGAS(testCases: {
  question: string;
  groundTruth: string;
}[]): Promise<RAGASReport> {
  const scores = { faithfulness: 0, answerRelevancy: 0, contextPrecision: 0, contextRecall: 0 };

  for (const { question, groundTruth } of testCases) {
    const { answer, chunks } = await ragSystem.query("test-tenant", question);
    const context = chunks.map((c) => c.content);

    scores.faithfulness += await measureFaithfulness(question, answer, context);
    scores.answerRelevancy += await measureAnswerRelevancy(question, answer);
    scores.contextPrecision += await measureContextPrecision(question, context);
    scores.contextRecall += await measureContextRecall(question, groundTruth, context);
  }

  const n = testCases.length;
  return {
    faithfulness: scores.faithfulness / n,
    answerRelevancy: scores.answerRelevancy / n,
    contextPrecision: scores.contextPrecision / n,
    contextRecall: scores.contextRecall / n,
    overallScore: Object.values(scores).reduce((a, b) => a + b, 0) / (4 * n),
  };
}

// Target scores:
// faithfulness: > 0.85 (low = hallucination)
// answerRelevancy: > 0.80 (low = off-topic answers)
// contextPrecision: > 0.75 (low = retrieving irrelevant chunks)
// contextRecall: > 0.80 (low = missing important chunks)
```

### 2.8.3 — Document Pre-Processing

```typescript
import pdf from "pdf-parse";
import mammoth from "mammoth";

// PDF extraction with structure preservation
async function extractFromPDF(buffer: Buffer): Promise<ProcessedDocument> {
  const pdfData = await pdf(buffer);
  let text = pdfData.text;

  // Clean common PDF artifacts
  text = text
    .replace(/\f/g, "\n\n")          // Form feeds → paragraph breaks
    .replace(/[ \t]+\n/g, "\n")       // Trailing whitespace
    .replace(/\n{3,}/g, "\n\n")       // Excessive newlines
    .replace(/([a-z])-\n([a-z])/g, "$1$2") // Hyphenated word breaks
    .trim();

  return {
    text,
    pageCount: pdfData.numpages,
    metadata: pdfData.info,
  };
}

// DOCX extraction
async function extractFromDOCX(buffer: Buffer): Promise<ProcessedDocument> {
  const result = await mammoth.extractRawText({ buffer });
  return { text: result.value, warnings: result.messages };
}

// For scanned PDFs (images) — need OCR
// Options:
// 1. AWS Textract (cloud, accurate, handles tables and forms)
// 2. Azure Document Intelligence (cloud, handles complex layouts)
// 3. Tesseract.js (local, free, less accurate)

// Caption images before embedding (multimodal)
async function captionImage(imageBase64: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: "claude-haiku-4", // Cheap vision model
    max_tokens: 200,
    messages: [{
      role: "user",
      content: [
        {
          type: "image",
          source: { type: "base64", media_type: "image/jpeg", data: imageBase64 },
        },
        {
          type: "text",
          text: "Describe this image in 2-3 sentences for search purposes. Include all visible text, objects, and context. Be factual and concise.",
        },
      ],
    }],
  });

  return response.content[0].text;
}
```

### 2.8.4 — Incremental Indexing

```typescript
// DON'T re-embed everything when documents update
// DO track what changed and only re-embed changes

interface DocumentVersion {
  documentId: string;
  contentHash: string;  // MD5 of content
  indexedAt: Date;
  chunkCount: number;
}

async function incrementalIndex(
  tenantId: string,
  documentId: string,
  content: string
): Promise<{ action: "skipped" | "indexed"; reason: string }> {
  // Compute hash of new content
  const contentHash = md5(content);

  // Check if already indexed with same content
  const existing = await db.documentVersions.findFirst({
    where: { documentId, tenantId },
  });

  if (existing?.contentHash === contentHash) {
    return { action: "skipped", reason: "Content unchanged" };
  }

  // Delete old chunks if updating
  if (existing) {
    await vectorDB.deleteByDocumentId(tenantId, documentId);
  }

  // Index new content
  await ragSystem.indexDocument(tenantId, documentId, content);

  // Save version record
  await db.documentVersions.upsert({
    where: { documentId },
    create: { documentId, tenantId, contentHash, indexedAt: new Date() },
    update: { contentHash, indexedAt: new Date() },
  });

  return { action: "indexed", reason: existing ? "Content updated" : "New document" };
}

// Event-driven re-indexing (real-time freshness)
// When a document changes → publish event → worker re-indexes
// Set up with: Redis Pub/Sub, Kafka, or BullMQ events
```

---

## 🔨 Phase 2 Projects

### Project 1: Production RAG System (PDF Q&A Bot)

**Features to implement:**
- Multi-PDF upload with preprocessing pipeline
- Per-document namespace isolation (different tenants can't see each other's docs)
- Hybrid search (vector + BM25)
- Source citation with similarity scores
- Streaming response with cited sources highlighted
- RAGAS dashboard showing faithfulness and relevancy scores

**Architecture:**
```
PDF Upload → Pre-process → Chunk (recursive, 400 tokens)
     → Embed (batch) → Store (pgvector + metadata)
     → Query → Hybrid search → Rerank (Cohere)
     → Build prompt → Stream response → Show sources
```

**Stand-out feature:** Show the user WHICH chunk from WHICH page answered their question, with a similarity score.

### Project 2: RAG Quality Dashboard

**What to build:** A monitoring dashboard for your RAG system.

**Metrics to display:**
- Daily/weekly RAGAS scores (faithfulness, relevancy, precision, recall)
- Average retrieval latency
- Cache hit rate
- Most common queries
- Queries that triggered "I don't know" responses
- Cost per query

**Tech stack:** Next.js + Chart.js + PostgreSQL (storing eval results)

### Project 3: Semantic Search Engine

**What to build:** Product search for an e-commerce site that understands meaning.

**Features:**
- Search "comfortable office chair" → finds "ergonomic mesh back chair" (semantic match!)
- Hybrid search: "red nike shoes size 10" → exact match for "nike" + semantic for "red shoes"
- Show relevance score for each result
- Filter by category using self-querying retriever

---

## ✅ Master Checklist

Before moving to Phase 3, verify you can:

**Embeddings**
- [ ] Generate embeddings with OpenAI, Cohere, and local Ollama
- [ ] Explain cosine similarity and what score = good retrieval
- [ ] Implement all 5 chunking strategies from memory
- [ ] Choose the right chunking strategy for a given document type
- [ ] Validate chunk quality programmatically

**Vector Databases**
- [ ] Set up pgvector with HNSW index on PostgreSQL
- [ ] Set up Pinecone with namespace isolation per tenant
- [ ] Explain HNSW vs IVFFlat and when to use each
- [ ] Implement metadata filtering in vector search

**RAG**
- [ ] Build a complete RAG pipeline from scratch (no LangChain)
- [ ] Implement HyDE and explain why it improves retrieval
- [ ] Implement multi-query retrieval with RRF fusion
- [ ] Add Cohere re-ranking to a pipeline
- [ ] Implement hybrid search (vector + BM25)
- [ ] Implement self-querying retrievers
- [ ] Run RAGAS evaluation and interpret the scores

**Structured Outputs**
- [ ] Write complex nested Zod schemas
- [ ] Implement retry-on-validation-failure pattern
- [ ] Explain when to use JSON mode vs tool calling
- [ ] Use Instructor.js for automatic retry

**Tool Calling**
- [ ] Write tool schemas with high-quality descriptions
- [ ] Implement the full tool execution loop
- [ ] Execute multiple tool calls in parallel
- [ ] Handle tool errors gracefully (return error to LLM, let it adapt)

---

*Phase 2 complete. You now understand RAG at a production level that 90% of developers never reach. The chunking, hybrid search, and RAGAS evaluation skills alone put you ahead of most AI developers.*
