Here's a breakdown of the major RAG (Retrieval-Augmented Generation) architectures companies use, organized by the problem each one is built to solve:

## 1. Naive/Simple RAG
**How it works:** Query → embed → retrieve top-k chunks from a vector store → stuff into prompt → generate.

**Use case:** FAQ bots, internal documentation search, simple customer support. Good starting point but struggles with complex queries, multi-hop reasoning, or noisy retrieval.

## 2. Advanced RAG (with query/retrieval optimization)
Adds pre- and post-processing steps around the naive pipeline:
- **Query rewriting/expansion** — rephrasing vague user queries into better search queries
- **Hybrid search** — combining dense (vector) + sparse (BM25/keyword) retrieval
- **Re-ranking** — using a cross-encoder to reorder retrieved chunks by relevance
- **Contextual compression** — trimming irrelevant parts of retrieved chunks before generation

**Use case:** Legal/financial document search, enterprise knowledge bases where precision matters a lot.

## 3. Modular RAG
A flexible pipeline where components (retrievers, rankers, generators) are swappable and can be routed conditionally — e.g., different retrievers for structured vs. unstructured data.

**Use case:** Companies with heterogeneous data sources (SQL databases + PDFs + wikis) that need different retrieval strategies per data type.

## 4. Agentic RAG
An LLM agent decides *when*, *whether*, and *how* to retrieve — it can issue multiple retrieval calls, use tools (calculators, APIs, code execution), and iteratively refine its search based on intermediate results.

**Use case:** Complex research assistants, multi-step customer workflows (e.g., "check my order status, then compare it to the refund policy, then draft a response").

## 5. Graph RAG
Instead of (or alongside) vector search, it builds/queries a knowledge graph of entities and relationships, useful for multi-hop questions like "what products did the team that shipped Feature X also work on?"

**Use case:** Pharma/biotech (drug-gene-disease relationships), fraud detection, supply chain analysis — anywhere relationships matter more than raw text similarity.

## 6. Hierarchical / Multi-level RAG (e.g., RAPTOR)
Builds a tree of summaries at different abstraction levels (chunk → section → document → corpus summary), so retrieval can pull either fine detail or high-level context depending on the query.

**Use case:** Long-document analysis — contracts, research papers, technical manuals where questions range from "what's this document about" to "what's the exact clause in section 4.2."

## 7. Corrective RAG (CRAG)
Adds a grading/self-check step: retrieved documents are evaluated for relevance, and if they're poor, the system falls back to web search or query reformulation instead of generating on bad context.

**Use case:** Customer-facing chatbots where hallucination risk is costly (banking, healthcare, insurance).

## 8. Self-RAG
The model itself decides during generation whether it needs more retrieval, and critiques its own output against retrieved evidence using special reflection tokens.

**Use case:** High-stakes generation tasks (medical/legal drafting) where factual grounding needs continuous verification, not just a one-shot retrieval.

## 9. Multi-modal RAG
Retrieves and reasons over text, images, tables, and sometimes audio/video together (e.g., embedding chart images alongside text for retrieval).

**Use case:** Manufacturing/engineering (retrieving diagrams + specs together), retail (product images + descriptions), medical imaging + reports.

## 10. Federated / Multi-source RAG
Queries multiple separate knowledge bases or vector stores (e.g., per-department, per-client, or per-region) and merges results, often with access-control-aware retrieval.

**Use case:** Large enterprises with siloed data (HR, legal, engineering each with their own store) or SaaS products serving multiple tenants where data isolation matters.

---

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

In practice, production systems are rarely "pure" — most companies combine 2–3 of these (e.g., hybrid search + re-ranking + agentic query routing) rather than picking one off the shelf. If you're building toward a specific use case, I can go deeper on what would fit best.
---

## Project 1: Advanced Document Q&A

**What it actually does:** User uploads/points to a document corpus, asks questions in natural language, gets accurate answers with sources — and you can toggle between retrieval modes to compare quality.

### Implementation: Naive RAG (baseline)
1. **Chunk** documents — split into ~500-token pieces with ~50-token overlap (use `RecursiveCharacterTextSplitter` from LangChain, or write your own)
2. **Embed** each chunk using an embedding model (OpenAI `text-embedding-3-small`, or open-source `bge-large`)
3. **Store** vectors in a vector DB (Chroma for local dev, Qdrant/Pinecone for something more "production")
4. **Retrieve**: embed the user's query, do cosine similarity search, pull top-5 chunks
5. **Generate**: stuff those 5 chunks into a prompt template with the question, send to the LLM

This is your control group — it'll work okay but fail on queries needing exact keyword matches (names, codes, numbers) since embeddings are bad at exact-match.

### Implementation: add Hybrid Search
1. Run the same query through **BM25** (keyword-based, use `rank_bm25` library) alongside vector search
2. Get top-k from each method
3. Combine results using **Reciprocal Rank Fusion (RRF)** — a simple formula that merges two ranked lists by score, weighting items that appear in both highly
4. Now numeric/exact-term queries ("what was the 2023 revenue figure") work much better because BM25 catches literal matches vectors miss

### Implementation: add Re-ranking
1. Take the ~20 candidates from hybrid search (over-fetch instead of just top-5)
2. Pass each (query, chunk) pair through a **cross-encoder** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`, or Cohere's Rerank API) — this scores relevance much more accurately than embedding similarity because it looks at query and chunk *together*, not as separate vectors
3. Keep only the top-5 after re-ranking, discard the rest
4. This step alone typically gives the biggest quality jump in the whole pipeline

### Implementation: add Contextual Compression
1. Before sending the final 5 chunks to the LLM, run each through a smaller/cheaper LLM call: "extract only the sentences relevant to this question"
2. This shrinks your context, reduces cost, and stops the main LLM from getting distracted by irrelevant sentences inside an otherwise-relevant chunk

### How to prove it worked
Build a small eval set (~30-50 question/answer pairs you write by hand from the docs). Run all 4 pipeline variants against it, score with **RAGAS** metrics (faithfulness, answer relevancy, context precision). Plot the improvement — this chart is your portfolio centerpiece for this project.

---

## Project 2: Long-Doc Intelligence Tool

**What it actually does:** Handles two very different question types well — broad ("what's the overall risk profile in this contract?") and narrow ("what's the exact penalty clause in section 4.2?") — plus relationship questions ("which vendor is liable if Party A breaches?").

### Implementation: Hierarchical RAG (RAPTOR-style)
1. Chunk the document normally (leaf-level nodes)
2. **Cluster** semantically similar chunks together (use k-means or GMM clustering on the embeddings)
3. For each cluster, generate an LLM **summary** — this becomes a "level 2" node
4. Cluster the level-2 summaries again, summarize again → "level 3" node (repeat until you have one root summary)
5. You now have a tree: leaf chunks → cluster summaries → document summary
6. At query time: **route** based on question type — a classifier (or simple heuristic/LLM call) decides "does this need the whole-document view or a specific detail?" and retrieves from the appropriate tree level (or searches across all levels and lets re-ranking sort it out)

### Implementation: Graph RAG
1. Run every chunk through an LLM extraction prompt: "extract entities and relationships as (subject, relation, object) triples" — e.g., ("Party A", "must indemnify", "Party B")
2. Load these triples into **Neo4j** (nodes = entities, edges = relationships)
3. At query time, use an LLM to translate the natural language question into a **Cypher query** (Neo4j's query language) — or use a library like `LangChain's GraphCypherQAChain` that does this translation for you
4. Run the Cypher query against the graph, get structured results, feed them to the LLM to generate a natural language answer
5. For questions requiring both structure and text (e.g., "explain why Party A is liable"), combine: pull the relevant graph subpath AND the source text chunks that mention those entities, feed both into the final generation step

### The key routing logic
This is what ties the project together: an initial LLM call classifies the question as "summary-level," "detail-level," or "relationship-level," and routes to the tree, the leaf chunks, or the graph accordingly.

---

## Project 3: Reliable Support Bot

**What it actually does:** Answers product questions from documentation, but explicitly avoids confidently making things up — it flags uncertainty and searches the web when its own docs don't have the answer.

### Implementation: Corrective RAG
1. Standard retrieval (hybrid + re-rank, reusing Project 1's pipeline)
2. Add a **grading step**: for each retrieved chunk, ask an LLM (cheap/fast model is fine) "is this chunk actually relevant to answering the question? yes/no/partial"
3. **Decision logic:**
   - If most chunks graded "relevant" → generate answer normally
   - If most graded "irrelevant" → trigger a **web search fallback** (Tavily or SerpAPI), retrieve fresh results, generate from those instead
   - If mixed → combine both sources, but flag it in the answer ("Based on our docs and a web search...")
4. This is a small addition (one extra LLM call per chunk) with a big reliability payoff

### Implementation: Self-RAG-inspired critique
1. After the LLM generates a draft answer, run a **second pass**: "here's the draft answer and the source chunks — identify any claim in the draft NOT supported by the sources"
2. If unsupported claims are found, either strip them out or regenerate with an explicit instruction to only use supported claims
3. Optionally output a **groundedness score** (e.g., % of sentences in the answer traceable to a source) and display it to the user — this transparency is itself a nice UX/portfolio feature

### How to prove it worked
Write ~15 adversarial test questions — some with no good answer in your docs, some that tempt the model to hallucinate specifics (dates, numbers). Show your bot handling these gracefully ("I don't have information on that in our docs, but here's what I found on the web...") compared to a naive RAG bot confidently making something up.

---

## Project 4: Multi-Tool Research Agent (your flagship)

**What it actually does:** Handles compound questions that need multiple steps — retrieving from your knowledge base, searching the live web, and doing calculations — deciding the sequence itself rather than following a fixed pipeline.

### Implementation: Agentic RAG
1. Define your **tools** as functions the LLM can call:
   - `search_vector_db(query)` — your internal knowledge base
   - `web_search(query)` — live web results
   - `calculator(expression)` — for math
   - `fetch_stock_data(ticker)` or similar domain-specific API, if relevant
2. Use a framework to manage the loop — **LangGraph** is the current standard for this (it models the agent as a state graph, easier to debug than a black-box agent loop)
3. **The core loop:**
   - LLM receives the question, decides which tool to call first (this is a structured "tool call" the LLM outputs, not free text)
   - Tool executes, result goes back into the LLM's context
   - LLM decides: do I have enough info now, or do I need another tool call?
   - Repeat until the LLM decides it can answer, then generates the final response
4. **Important implementation detail:** build in a max-iteration cap and error handling — agents can loop indefinitely or call tools with malformed inputs, so you need guardrails
5. For the example question ("compare R&D spend across 3 companies and explain the trend using recent news"):
   - Agent calls `search_vector_db` three times (once per company's 10-K)
   - Agent calls `calculator` to compute growth rates
   - Agent calls `web_search` for recent news context
   - Agent synthesizes all of this into a final answer

### Why this impresses in interviews
You can literally show the **trace** — the sequence of tool calls the agent made — which demonstrates it's not just pattern-matching a fixed pipeline but actually reasoning about what information it needs. LangGraph gives you this trace for free, which is great for a demo.

---

## Project 5: Enterprise Knowledge Assistant

**What it actually does:** Simulates a multi-department company assistant where a user's role determines what they can search, and includes documents with images/charts that need to be searchable too.

### Implementation: Federated RAG with access control
1. Create **separate vector store collections** per department (HR, Engineering, Sales) — in Qdrant/Chroma this is just separate named collections
2. Attach **metadata** to each chunk at ingestion time: `{"department": "HR", "sensitivity": "internal"}`
3. At query time, apply a **metadata filter** based on the logged-in user's role before running the vector search — e.g., a "sales" role only ever queries the Sales collection, or a filter excludes `sensitivity: "confidential"` chunks entirely
4. This is enforced at the retrieval layer, not the prompt layer — critical distinction, since prompt-based restrictions ("please don't share HR data") are not reliable security

### Implementation: Multi-modal RAG
1. For documents containing charts/tables/images (sales decks, financial reports), use a **multi-modal embedding model** — CLIP for image-text, or **ColPali** which embeds entire page images (avoiding messy table-to-text extraction entirely)
2. Store image embeddings alongside text embeddings in the same or a parallel vector store
3. At query time, if the question seems visual ("what does the revenue chart show for Q3"), retrieve the relevant page/image directly and pass it to a multi-modal LLM (GPT-4o or Claude, both accept images) along with the question
4. This sidesteps a huge pain point in traditional RAG — tables and charts converted to text lose their structure and meaning, whereas showing the model the actual image preserves it

---

### A practical build tip across all 5 projects
Reuse the same underlying "retrieval toolkit" (chunking, embedding, vector store wrapper, eval harness) across projects instead of rewriting it each time — put it in a shared library/repo. This is both faster for you and shows a reviewer you think about reusable infrastructure, not just one-off scripts.

Want me to help you pick the actual dataset/domain for Project 1 so you can start building this week?
