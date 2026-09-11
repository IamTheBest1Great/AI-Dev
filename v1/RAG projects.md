# RAG Architecture & Portfolio Project Guide

## Part 1: RAG Architecture Types

Here's a breakdown of the major RAG (Retrieval-Augmented Generation) architectures companies use, organized by the problem each one is built to solve:

### 1. Naive/Simple RAG
**How it works:** Query → embed → retrieve top-k chunks from a vector store → stuff into prompt → generate.

**Use case:** FAQ bots, internal documentation search, simple customer support. Good starting point but struggles with complex queries, multi-hop reasoning, or noisy retrieval.

### 2. Advanced RAG (with query/retrieval optimization)
Adds pre- and post-processing steps around the naive pipeline:
- **Query rewriting/expansion** — rephrasing vague user queries into better search queries
- **Hybrid search** — combining dense (vector) + sparse (BM25/keyword) retrieval
- **Re-ranking** — using a cross-encoder to reorder retrieved chunks by relevance
- **Contextual compression** — trimming irrelevant parts of retrieved chunks before generation

**Use case:** Legal/financial document search, enterprise knowledge bases where precision matters a lot.

### 3. Modular RAG
A flexible pipeline where components (retrievers, rankers, generators) are swappable and can be routed conditionally — e.g., different retrievers for structured vs. unstructured data.

**Use case:** Companies with heterogeneous data sources (SQL databases + PDFs + wikis) that need different retrieval strategies per data type.

### 4. Agentic RAG
An LLM agent decides *when*, *whether*, and *how* to retrieve — it can issue multiple retrieval calls, use tools (calculators, APIs, code execution), and iteratively refine its search based on intermediate results.

**Use case:** Complex research assistants, multi-step customer workflows (e.g., "check my order status, then compare it to the refund policy, then draft a response").

### 5. Graph RAG
Instead of (or alongside) vector search, it builds/queries a knowledge graph of entities and relationships, useful for multi-hop questions like "what products did the team that shipped Feature X also work on?"

**Use case:** Pharma/biotech (drug-gene-disease relationships), fraud detection, supply chain analysis — anywhere relationships matter more than raw text similarity.

### 6. Hierarchical / Multi-level RAG (e.g., RAPTOR)
Builds a tree of summaries at different abstraction levels (chunk → section → document → corpus summary), so retrieval can pull either fine detail or high-level context depending on the query.

**Use case:** Long-document analysis — contracts, research papers, technical manuals where questions range from "what's this document about" to "what's the exact clause in section 4.2."

### 7. Corrective RAG (CRAG)
Adds a grading/self-check step: retrieved documents are evaluated for relevance, and if they're poor, the system falls back to web search or query reformulation instead of generating on bad context.

**Use case:** Customer-facing chatbots where hallucination risk is costly (banking, healthcare, insurance).

### 8. Self-RAG
The model itself decides during generation whether it needs more retrieval, and critiques its own output against retrieved evidence using special reflection tokens.

**Use case:** High-stakes generation tasks (medical/legal drafting) where factual grounding needs continuous verification, not just a one-shot retrieval.

### 9. Multi-modal RAG
Retrieves and reasons over text, images, tables, and sometimes audio/video together (e.g., embedding chart images alongside text for retrieval).

**Use case:** Manufacturing/engineering (retrieving diagrams + specs together), retail (product images + descriptions), medical imaging + reports.

### 10. Federated / Multi-source RAG
Queries multiple separate knowledge bases or vector stores (e.g., per-department, per-client, or per-region) and merges results, often with access-control-aware retrieval.

**Use case:** Large enterprises with siloed data (HR, legal, engineering each with their own store) or SaaS products serving multiple tenants where data isolation matters.

**How companies typically choose:**

| Priority | Common architecture |
|---|---|
| Simplicity, low latency | Naive RAG |
| Precision on dense docs | Advanced RAG (hybrid + re-ranking) |
| Multi-step reasoning/tool use | Agentic RAG |
| Relationship-heavy data | Graph RAG |
| Long documents, varying granularity | Hierarchical RAG |
| Low hallucination tolerance | Corrective/Self-RAG |
| Mixed data types | Multi-modal RAG |
| Data silos/multi-tenant | Federated RAG |

In practice, production systems are rarely "pure" — most companies combine 2–3 of these (e.g., hybrid search + re-ranking + agentic query routing) rather than picking one off the shelf.

---

## Part 2: Deployment Modes — Cloud / Hybrid / Local

Before diving into project implementations, it's worth understanding the three deployment modes you'll implement across these projects, since this is as important a skill as the retrieval architecture itself.

### Why this matters
Many companies (healthcare, finance, government, legal, and any org with strict data governance) cannot send data to external APIs due to compliance requirements (HIPAA, SOC 2, data residency laws) or simply don't trust third parties with proprietary data. This shapes real-world RAG architecture decisions as much as the retrieval pattern does.

### The three modes

**Cloud mode** — everything via external API
- LLM: OpenAI / Anthropic API
- Embeddings: OpenAI `text-embedding-3-small` or similar API
- Vector store: Pinecone or managed Qdrant Cloud
- Fastest to build, least data control

**Local mode** — everything self-hosted
- LLM: Llama 3.x / Mistral / Qwen served via **Ollama**, **vLLM**, or **TGI**
- Embeddings: local models like `bge-large`, `nomic-embed`, `e5-large` (via `sentence-transformers`)
- Vector store: self-hosted Qdrant/Weaviate/Milvus, or Chroma (in-process, zero setup)
- No data leaves the network; usually a quality tradeoff vs. cloud LLMs

**Hybrid mode** — the realistic enterprise pattern
- Embeddings + vector store stay **local** (raw sensitive chunks never leave your infra)
- Generation call goes to a **cloud LLM API**, but only receives the already-retrieved, already-approved context + question — not the whole dataset
- This is a genuine risk-reduction pattern many enterprises actually run

### Implementation: the abstraction layer pattern

Build your pipeline so the LLM, embedder, and vector store are all swappable behind a config, not hardcoded:

```python
# config.py
MODE = "cloud"  # or "hybrid" or "local"

if MODE == "cloud":
    llm = ChatOpenAI(model="gpt-4o")  # or Anthropic
    embedder = OpenAIEmbeddings()
    vectorstore = Pinecone(...)

elif MODE == "hybrid":
    llm = ChatOpenAI(model="gpt-4o")        # generation still external
    embedder = HuggingFaceEmbeddings(model="BAAI/bge-large-en")  # local
    vectorstore = Qdrant(location="local")   # self-hosted, data stays put

elif MODE == "local":
    llm = ChatOllama(model="llama3.1:8b")    # fully local
    embedder = HuggingFaceEmbeddings(model="BAAI/bge-large-en")
    vectorstore = Qdrant(location="local")
```

Everything downstream (retrieval logic, re-ranking, agent loop) stays identical — only the backing service per component changes. This mirrors how real companies structure this.

### Setting up local mode (step by step)
1. **Install Ollama**: `curl -fsSL https://ollama.com/install.sh | sh` (Linux/Mac) or download the Windows installer
2. **Pull a model**: `ollama pull llama3.1:8b` — runs fine on 16GB RAM, no GPU strictly required (slow on CPU-only)
3. **Serve it**: Ollama auto-runs a local API server at `localhost:11434` — LangChain's `ChatOllama` or LlamaIndex's `Ollama` class connects to it like any API
4. **Local embeddings**: `pip install sentence-transformers`, load `BAAI/bge-large-en-v1.5` — runs on CPU, just slower
5. **Local vector store**: run Qdrant via Docker (`docker run -p 6333:6333 qdrant/qdrant`) or use Chroma in-process for zero setup
6. **No GPU?** Use a quantized model — `llama3.1:8b-instruct-q4_0` runs reasonably on a modern laptop CPU. Any NVIDIA GPU with 8GB+ VRAM speeds this up significantly.

### Which mode to use for each project

| Project | Recommended mode(s) | Why |
|---|---|---|
| Project 1 (Advanced Q&A) | **All 3 modes** | Simplest pipeline — best place to learn the swap mechanics cleanly |
| Project 2 (Long-doc/Graph) | Cloud only | Complex enough already; local LLMs struggle more with multi-hop graph reasoning |
| Project 3 (Corrective RAG) | **Hybrid** | Realistic "support bot for a company with compliance needs" narrative |
| Project 4 (Agentic) | Cloud only | Agentic tool-calling reliability drops noticeably with smaller local models |
| Project 5 (Enterprise) | **Local or hybrid** | Fits the narrative — an internal enterprise assistant is exactly the use case that demands data stay in-house |

### The comparison artifact (build this for Project 1)
Since Project 1 runs in all three modes, measure and document:

| Metric | Cloud | Hybrid | Local |
|---|---|---|---|
| Answer quality (RAGAS score) | | | |
| Latency (avg response time) | | | |
| Cost per 1000 queries | | | |
| Data leaves network? | Yes | Partially | No |

This table, built from your own measurements, is a strong portfolio artifact — it shows you understand RAG as an engineering tradeoff space, not just a technique to implement once.

---

## Part 3: Project Implementations

### Project 1: Advanced Document Q&A
**What it does:** User uploads/points to a document corpus, asks questions in natural language, gets accurate answers with sources — with a toggle between retrieval modes to compare quality.

**Architecture covered:** Naive RAG → Advanced RAG (hybrid search, re-ranking, contextual compression)
**Deployment modes:** Cloud, Hybrid, Local (see Part 2)

#### Naive RAG (baseline)
1. **Chunk** documents — ~500-token pieces with ~50-token overlap (`RecursiveCharacterTextSplitter` or custom)
2. **Embed** each chunk (OpenAI `text-embedding-3-small`, or open-source `bge-large`)
3. **Store** vectors in a vector DB (Chroma for local dev, Qdrant/Pinecone for production)
4. **Retrieve**: embed the query, cosine similarity search, pull top-5 chunks
5. **Generate**: stuff those 5 chunks into a prompt template with the question, send to the LLM

This is your control group — works okay but fails on queries needing exact keyword matches (names, codes, numbers).

#### + Hybrid Search
1. Run the query through **BM25** (keyword-based, `rank_bm25` library) alongside vector search
2. Get top-k from each method
3. Combine using **Reciprocal Rank Fusion (RRF)**
4. Numeric/exact-term queries now work much better

#### + Re-ranking
1. Over-fetch ~20 candidates from hybrid search
2. Score each (query, chunk) pair with a **cross-encoder** (`ms-marco-MiniLM-L-6-v2` or Cohere Rerank)
3. Keep only the top-5 after re-ranking
4. Usually gives the biggest single quality jump in the pipeline

#### + Contextual Compression
1. Run each final chunk through a smaller/cheaper LLM: "extract only the sentences relevant to this question"
2. Shrinks context, reduces cost, prevents distraction from irrelevant sentences

#### Proving it worked
Build a ~30-50 question eval set by hand. Run all 4 pipeline variants through **RAGAS** metrics (faithfulness, answer relevancy, context precision). Plot the improvement — this chart is your portfolio centerpiece.

---

### Project 2: Long-Doc Intelligence Tool
**What it does:** Handles broad questions ("what's the overall risk profile?"), narrow questions ("exact penalty clause in section 4.2"), and relationship questions ("which vendor is liable if Party A breaches?").

**Architecture covered:** Hierarchical RAG (RAPTOR-style) + Graph RAG
**Deployment mode:** Cloud only

#### Hierarchical RAG (RAPTOR-style)
1. Chunk the document normally (leaf-level nodes)
2. **Cluster** semantically similar chunks (k-means or GMM on embeddings)
3. Generate an LLM **summary** per cluster → "level 2" nodes
4. Cluster level-2 summaries again, summarize again → "level 3" (repeat to a single root summary)
5. Tree: leaf chunks → cluster summaries → document summary
6. At query time: **route** based on question type (whole-document view vs. specific detail) to the right tree level

#### Graph RAG
1. Extract entities/relationships from every chunk as (subject, relation, object) triples via LLM prompt
2. Load triples into **Neo4j** (nodes = entities, edges = relationships)
3. Translate the question into a **Cypher query** via LLM (or `LangChain's GraphCypherQAChain`)
4. Run the Cypher query, feed structured results to the LLM for a natural language answer
5. For mixed questions, combine the graph subpath with relevant text chunks in the final generation step

#### The routing logic
An initial LLM call classifies the question as "summary-level," "detail-level," or "relationship-level," and routes to the tree, leaf chunks, or graph accordingly.

---

### Project 3: Reliable Support Bot
**What it does:** Answers product questions from documentation while explicitly avoiding confident fabrication — flags uncertainty and searches the web when its docs don't have the answer.

**Architecture covered:** Corrective RAG + Self-RAG-inspired critique
**Deployment mode:** Hybrid (local embeddings/vector store, cloud generation — fits the compliance-driven support bot narrative)

#### Corrective RAG
1. Standard retrieval (hybrid + re-rank, reusing Project 1's pipeline)
2. Add a **grading step**: for each retrieved chunk, ask an LLM "is this relevant? yes/no/partial"
3. **Decision logic:**
   - Mostly relevant → generate normally
   - Mostly irrelevant → fallback to **web search** (Tavily/SerpAPI), generate from fresh results
   - Mixed → combine both, flag it in the answer
4. Small addition (one extra LLM call per chunk), big reliability payoff

#### Self-RAG-inspired critique
1. After drafting an answer, run a second pass: "identify any claim NOT supported by the sources"
2. Strip unsupported claims or regenerate with an explicit "only use supported claims" instruction
3. Optionally output a **groundedness score** (% of sentences traceable to a source) — nice UX/portfolio feature

#### Proving it worked
Write ~15 adversarial test questions (some with no good answer in your docs, some tempting fabrication of specifics). Show graceful handling vs. a naive RAG bot confidently making things up.

---

### Project 4: Multi-Tool Research Agent (flagship)
**What it does:** Handles compound questions needing multiple steps — retrieving from your knowledge base, searching the live web, doing calculations — deciding the sequence itself.

**Architecture covered:** Agentic RAG
**Deployment mode:** Cloud only (agentic tool-calling reliability drops noticeably with smaller local models)

#### Implementation
1. Define **tools** as LLM-callable functions:
   - `search_vector_db(query)` — internal knowledge base
   - `web_search(query)` — live web results
   - `calculator(expression)` — math
   - domain-specific API tools as relevant
2. Use **LangGraph** to manage the loop (models the agent as a state graph, easier to debug than a black-box loop)
3. **Core loop:**
   - LLM decides which tool to call first (structured tool call output)
   - Tool executes, result returns to context
   - LLM decides: enough info, or another tool call?
   - Repeat until ready to answer
4. Build in a **max-iteration cap** and error handling — agents can loop indefinitely or call tools with malformed inputs
5. Example: "compare R&D spend across 3 companies and explain the trend using recent news" → agent calls `search_vector_db` 3x, `calculator` for growth rates, `web_search` for news context, synthesizes a final answer

#### Why this impresses
You can show the **trace** — the sequence of tool calls — demonstrating actual reasoning about what information is needed, not a fixed pipeline. LangGraph gives you this trace for free.

---

### Project 5: Enterprise Knowledge Assistant
**What it does:** Simulates a multi-department company assistant where a user's role determines what they can search, and includes documents with images/charts that need to be searchable.

**Architecture covered:** Federated RAG with access control + Multi-modal RAG
**Deployment mode:** Local or Hybrid (fits the "data stays in-house" enterprise narrative)

#### Federated RAG with access control
1. Create **separate vector store collections** per department (HR, Engineering, Sales)
2. Attach **metadata** at ingestion: `{"department": "HR", "sensitivity": "internal"}`
3. Apply a **metadata filter** based on the logged-in user's role before running the vector search — enforced at the retrieval layer, not the prompt layer (prompt-based restrictions aren't reliable security)

#### Multi-modal RAG
1. Use a **multi-modal embedding model** for charts/tables/images — CLIP for image-text, or **ColPali** (embeds entire page images, avoiding messy table-to-text extraction)
2. Store image embeddings alongside text embeddings
3. For visual questions ("what does the revenue chart show for Q3"), retrieve the relevant page/image and pass it to a multi-modal LLM (GPT-4o or Claude) with the question
4. Preserves table/chart structure that text extraction would lose

---

## Practical build tip across all projects
Reuse the same underlying "retrieval toolkit" (chunking, embedding, vector store wrapper, eval harness, and the cloud/hybrid/local config layer from Part 2) across projects instead of rewriting it each time — put it in a shared library/repo. This is faster to build and signals to reviewers that you think about reusable infrastructure, not just one-off scripts.

## Suggested build order
1. **Project 1** — foundation, build in all 3 deployment modes
2. **Project 4** — agentic, highest hiring signal, do this while fresh
3. **Project 3** — reliability, relatively fast to add once Project 1 exists
4. **Project 2 and 5** — specializations, pick based on target industry
