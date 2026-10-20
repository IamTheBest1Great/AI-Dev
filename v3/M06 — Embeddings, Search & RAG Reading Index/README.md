# M06 — Embeddings, Search & RAG: Reading Index

**Path:** Core  
**Depth:** L2 foundation; L3 through demonstrated project work  
**Purpose:** Knowledge application  
**Prerequisites:** M03, M05, M07  
**Completion evidence:** Compare retrieval baselines; cite evidence; verify permissions and source deletion.  
**Prepared:** 20 October 2026

## What is included

This collection contains **19 separate study guides**, one for each section at the level of M06.1.1, M06.1.2 and so on. The smaller numbered subtopics are taught inside their corresponding guide.

The 19 guides contain approximately **114,500 words** in total. This index is an additional navigation file. The ZIP includes this index and all 19 guides as individual Markdown files.

The teaching structure follows the supplied Prompt Engineering sample:

- Linked tables of contents.
- Plain-language definitions and intuition.
- Detailed mechanisms and practical decision rules.
- Worked examples, numerical calculations and comparison tables.
- Benefits, limitations, failure modes and common mistakes.
- Concept maps and source-linked explanations.
- Glossaries and final revision sheets.
- Recall questions with answers.
- Easy, medium and scenario-based multiple-choice questions with explained answers.
- Interview questions with model answers.
- Practical exercises and L3 completion evidence.

## Complete file index

| Section | Study guide | Main coverage |
|---|---|---|
| M06.1.1 | [Document Ingestion](M06.1.1_Document_Ingestion.md) | Uploads, MIME detection, formats, OCR, data quality, IDs, versions and provenance |
| M06.1.2 | [Chunking](M06.1.2_Chunking.md) | All chunking strategies, context-preserving retrieval units, overlap and boundaries |
| M06.1.3 | [Embeddings](M06.1.3_Embeddings.md) | Vector representations, similarity mathematics, model fit, caching and re-embedding |
| M06.2.1 | [Retrieval and Information Retrieval Fundamentals](M06.2.1_Retrieval_and_Information_Retrieval_Fundamentals.md) | Classical search, BM25, lexical/dense/hybrid retrieval, fusion and query improvement |
| M06.2.2 | [Vector Databases and ANN Search](M06.2.2_Vector_Databases_and_ANN_Search.md) | Exact versus ANN, HNSW, IVF, quantization, filtering, storage and scaling |
| M06.2.3 | [Query Processing and Retrieval Routing](M06.2.3_Query_Processing_and_Retrieval_Routing.md) | Query understanding, all router destinations and source-selection decisions |
| M06.3.1 | [Advanced RAG](M06.3.1_Advanced_RAG.md) | Retrieval refinements, reasoning methods, specialized RAG and evidence quality |
| M06.3.2 | [Knowledge Graphs](M06.3.2_Knowledge_Graphs.md) | Entities, relationships, Cypher, traversal, resolution and graph provenance |
| M06.4.1 | [Context Engineering for RAG](M06.4.1_Context_Engineering_for_RAG.md) | Evidence selection, ordering, budgets, compression, contradictions and abstention |
| M06.4.2 | [RAG Evaluation](M06.4.2_RAG_Evaluation.md) | Retrieval/context/answer metrics, worked calculations, datasets, judges and regression |
| M06.5.1 | [RAG Failure Modes](M06.5.1_RAG_Failure_Modes.md) | Retrieval, knowledge and generation failures; diagnosis and remedies |
| M06.5.2 | [RAG Security](M06.5.2_RAG_Security.md) | Tenant isolation, ACLs, injections, leakage, cache isolation and audit traces |
| M06.6.1 | [Incremental Indexing](M06.6.1_Incremental_Indexing.md) | Change detection, updates, deletion, idempotency, recovery and reconciliation |
| M06.6.2 | [Production RAG](M06.6.2_Production_RAG.md) | Performance, caching, costs, observability, freshness, gates and rollback |
| M06.7.1 | [Knowledge Architecture Comparison](M06.7.1_Knowledge_Architecture_Comparison.md) | Seven architecture choices and the full Enterprise Knowledge Platform extension |
| M06.8.1 | [Candidate Refinement](M06.8.1_Candidate_Refinement.md) | Required baseline comparison, candidate generation and basic reranking |
| M06.8.2 | [Offline versus Online Pipelines](M06.8.2_Offline_versus_Online_Pipelines.md) | Preparation and serving stages, interfaces, dependencies and lifecycle correctness |
| M06.9.1 | [Evidence-Based Adoption](M06.9.1_Evidence_Based_Adoption.md) | Elective-method experiments, fair comparisons, uncertainty and adoption decisions |
| M06.10.1 | [Permission-Aware Document Answer](M06.10.1_Permission_Aware_Document_Answer.md) | Revised-manual walkthrough, citations, runnable guards and deletion verification |

## Suggested study order

### 1. Prepare knowledge

Read **M06.1.1–M06.1.3** to understand ingestion, chunking and embeddings. Then read **M06.8.2** for the distinction between offline preparation and online serving.

At this stage, you should be able to explain how a source becomes searchable while retaining its document ID, version, location and permission metadata.

### 2. Build and compare the main retrieval pipeline

Read **M06.2.1–M06.2.3**, followed by **M06.8.1**.

Learn lexical search before comparing it with dense search. Compare both with hybrid retrieval on the same eligible corpus. Keep **basic reranking in the main pipeline** and retain an ablation to measure its contribution.

### 3. Assemble evidence and measure answers

Read **M06.4.1–M06.4.2**.

Separate candidate recall, ranking quality, context coverage, answer correctness and citation support. Work through the numerical examples rather than memorizing metric names alone.

### 4. Make access and maintenance reliable

Read **M06.5.1–M06.6.2**.

Understand failure diagnosis, tenant and document permissions, prompt injection, source updates, cache invalidation, deletion, reconciliation and production operations.

### 5. Understand advanced choices

Read **M06.3.1–M06.3.2**, then **M06.7.1** and **M06.9.1**.

The named advanced methods are explained thoroughly. Their implementation is elective and should be justified by measured improvement. Graph concepts are part of the knowledge foundation; choosing a graph-based production architecture still requires a workload-specific reason.

### 6. Complete the worked project

Read **M06.10.1** and perform the L3 exercise using your own permitted test corpus.

Follow a revised manual through ingestion, publication, authorized retrieval, citation generation, permission changes and deletion. Use the included runnable Python fixture to understand the access and version guards, then implement the actual retrieval experiment.

## Required foundation and optional extension

| Required foundation | Optional implementation extension |
|---|---|
| Faithful extraction and traceable source identity | HyDE |
| Appropriate chunking and compatible embeddings | RAPTOR |
| Lexical, dense and hybrid baseline comparison | Original trained Self-RAG approaches |
| Basic reranking | CRAG-style corrective orchestration |
| Evidence assembly and citation validation | Graph-based retrieval where justified |
| Defined retrieval and answer metrics | More elaborate multimodal or agentic workflows |
| Authorization-aware retrieval and tenant isolation | Enterprise Knowledge Platform expansion |
| Source updates, cache validity and deletion verification | Other measured refinements beyond the baseline |

An extension does not replace the foundation. A simple system with correct source, access and evidence behavior is a valid basis for further experiments.

## How to use the examples

The manuals, tenants, ships, equipment identifiers, policies and benchmark-style tables are fictional training fixtures unless explicitly identified otherwise. They illustrate software design and evaluation; they are not maritime procedures or external performance benchmarks.

Numerical demonstrations are labeled as synthetic. Executable teaching snippets are distinguished from conceptual pseudocode and schema-dependent examples. The standalone Python examples were checked during preparation.

The worked permission program demonstrates sequential in-memory behavior. Its notes explain what a real distributed service must additionally implement. This distinction prevents a teaching fixture from being mistaken for a complete production deployment.

## How to revise

For each guide:

1. Read the definition and mechanism sections.
2. Explain one worked example in your own words.
3. Use the comparison tables to choose between approaches.
4. Attempt recall and multiple-choice questions before reading the answers.
5. Practice the interview questions aloud.
6. Complete the exercise and retain the requested evidence.
7. Return to the revision sheet before moving to the next guide.

When an answer fails, identify the stage responsible. Avoid changing the model prompt before checking whether the correct, current, authorized evidence reached the model.

## Completion portfolio

A complete module portfolio should contain:

| Evidence | Minimum useful artifact |
|---|---|
| Baseline comparison | Corpus/query versions, lexical/dense/hybrid results and reranking comparison |
| Cited answer | Claim-to-source mapping with revision and precise locator |
| Permission correctness | Same-tenant and cross-tenant tests, including caches and citations |
| Revision correctness | Before/after answer using the approved active source |
| Source deletion | Serving denial, derived-artifact cleanup and stale-worker/rollback checks |
| Evaluation | Explicit metrics, labels, query slices and representative failures |
| Advanced-method decision | Measured adoption or a reasoned rejection |
| Operational reasoning | Latency/cost budget, freshness expectations and recovery approach |

L2 means you can explain the concepts and work through the examples. L3 requires demonstrated project work and reproducible evidence.

## Reading the Markdown files

Open a guide directly or extract the ZIP and keep all files in the same folder so relative links work. The files are editable Markdown. Diagrams use Mermaid syntax; the surrounding prose explains the same relationships if a viewer does not render Mermaid.

Primary papers and official documentation are linked within each guide and collected in its references section. Product APIs and settings should be checked against the version used in your own implementation.

