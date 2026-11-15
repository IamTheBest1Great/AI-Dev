# M06.1.3 — Embeddings: Full Question Bank with Interview-Style Answers

> **How to use this:** Read the question, try answering out loud, then check. Answers are written the way you'd actually say them in an interview — plain language first, technical detail second.

---

# PART 1 — Core Concepts & Boundaries

### 1.1 What Is an Embedding?

**Q1. What is an embedding?**
**A:** It's a way of turning a piece of text into a list of numbers — a vector — so a computer can compare texts mathematically. Texts the model considers related end up with vectors that are close together. The key point: it's a *representation*, not an answer.

**Q2. How do dense and traditional sparse representations differ?**
**A:** Dense vectors have a fixed size — say 768 numbers — and most of them are nonzero; the meaning is spread across all the coordinates. Traditional sparse vectors like BM25 have one slot per vocabulary word, and almost all slots are zero for any given document. Dense captures meaning; sparse captures exact words.

**Q3. Why are embeddings useful for retrieval?**
**A:** Because they let you find relevant passages even when the wording is different. If a user asks about "reducing equipment downtime" and the manual says "preventive maintenance improves availability," keyword search misses it but an embedding model can still place them near each other.

**Q4. Why are embeddings not an answer database?**
**A:** Because a vector doesn't store a readable answer — it's just coordinates. The system uses it to *find candidate passages*, then fetches the actual text and its source. You still need the original document for citations, verification, permissions, and updates.

**Q5. Why should embeddings not be treated as anonymized data?**
**A:** Because they're derived from the source content. Numeric doesn't mean harmless — the vector still carries information about the original text and inherits its confidentiality. Treat it with the same access rules as the source.

**Q6. What is the difference between a bi-encoder and a cross-encoder?**
**A:** A bi-encoder encodes the query and the document *separately* and compares the two vectors — that's fast because you can pre-compute document vectors. A cross-encoder feeds the query and document *together* through one model and outputs a single relevance score — much more accurate but far too slow to run over a whole corpus. In practice: bi-encoder for retrieval, cross-encoder for reranking the top candidates.

**Q7. Why does the guide say embeddings "may" capture semantic similarity rather than "will"?**
**A:** Because they're learned representations, not guarantees. They can miss domain-specific meanings, confuse near-identical part numbers, or rank a contradictory sentence as similar just because it discusses the same topic. "May" is the honest word.

**Q8. Can you reverse an embedding to recover the original text? What are the privacy implications?**
**A:** Not exactly — it's lossy, so you can't perfectly reconstruct. But research shows inversion attacks can recover a surprising amount of the original meaning from a vector, especially for short texts. So you should treat embeddings as sensitive derived data, not as a one-way anonymization step.

**Q9. What is the difference between an embedding and a cryptographic hash?**
**A:** A hash is designed to be irreversible and to change completely with any tiny input change — it's for integrity. An embedding is designed so that *similar* inputs produce *similar* outputs — it's for comparison. Opposite design goals.

**Q10. Explain embeddings without using machine-learning jargon.**
**A:** Imagine giving every sentence a set of map coordinates, where sentences with similar meanings get placed near each other. To find an answer, you drop the question onto the same map and look at what's nearby. You still have to go read those nearby documents to check they're right and that the user is allowed to see them.

---

### 1.2 The Three Boundaries

**Q11. What is the difference between similarity and relevance?**
**A:** Similarity is how close two vectors are under a chosen metric — a mathematical property. Relevance is whether that source actually helps answer the user's question — a task property. A passage can be highly similar and completely irrelevant, like an old version of a manual.

**Q12. Can two contradictory sentences receive a high similarity score? Give an example.**
**A:** Absolutely. "Start Unit A" and "Do NOT start Unit A" share almost every word and the same topic, so a dense model will place them very close. That's why retrieval needs reranking or contradiction checks downstream — similarity can't tell you which one is correct.

**Q13. A cosine score is 0.82 — does that mean 82% probability the passage answers the question?**
**A:** No. It means the vectors point in a similar direction. The number's meaning depends entirely on the model, the corpus, the query distribution, and how the model was trained. It's not a calibrated probability. If you want a threshold, you have to calibrate it on labeled data for your specific task.

**Q14. Why is a vector not a security boundary?**
**A:** Because the vector is just an index representation. It doesn't know who's allowed to see the source. Authorization has to be enforced separately — with tenant filters, ACLs, and row-level security — before you ever show the text to a user. A perfect similarity score on an unauthorized document is still unauthorized.

**Q15. A support bot cites a passage with cosine score 0.95 as definitive proof of a policy. Legal flags this. What is the core misunderstanding?**
**A:** They're treating a similarity score as a correctness or relevance probability. 0.95 just means "these vectors point the same way." It says nothing about whether the passage is current, authoritative, or actually supports the claim. Legal is right — you need evidence verification and source authority checks, not a score threshold.

**Q16. "Start Unit A" and "Do NOT start Unit A" — why might these rank close, and what must the system do?**
**A:** They share almost all vocabulary and topic, so the embedding sees them as similar. The system must not rely on similarity alone — it needs a reranker or a contradiction/negation check that actually reads the passage and the query, plus source authority rules.

---

### 1.3 Dense vs Sparse Representations

**Q17. Why is "learned sparse retrieval" not simply "old-style keyword search"?**
**A:** In BM25, the weights are fixed formulas based on term frequency. In learned sparse, a model *learns* which vocabulary dimensions to activate and how strongly — and some variants can even activate related terms that weren't literally in the text. So it's still sparse, but the weighting is learned, not hand-coded.

**Q18. Name one strength and one weakness of traditional BM25/TF-IDF retrieval.**
**A:** Strength: it's excellent at exact matches — part numbers, error codes, names, identifiers. Weakness: it misses paraphrases. If the query says "downtime" and the document says "unavailability," BM25 won't connect them.

**Q19. What is hybrid retrieval and when would you use it?**
**A:** It's running dense and sparse retrieval in parallel and combining the results. You'd use it when your corpus needs both paraphrase recall *and* precise identifier matching — which is most real corpora. But you should measure whether the fusion actually helps on your data; don't assume.

**Q20. Give a real example where dense would fail but sparse would succeed, and vice versa.**
**A:** Dense fails on "find error code E-4021" — the exact string doesn't carry much semantic meaning, so BM25 wins. Sparse fails on "how do I keep my machine running longer?" vs. a passage about "extending equipment lifespan through preventive maintenance" — different words, same meaning, so dense wins.

**Q21. What does "learned sparse" mean — how does a model decide which vocabulary dimensions to activate?**
**A:** The model is trained so that, for a given input, it outputs a weight for each vocabulary term — most weights come out zero or near-zero, but the terms that matter for meaning get nonzero weights. It's learned end-to-end from retrieval objectives, so it can learn to weight related terms the raw text never contained.

---

### 1.4 Similarity, Relevance & Scores

**Q22. What does similarity actually measure — and what does it not measure?**
**A:** It measures closeness under a chosen comparison function, based on what the model was trained to consider related. It does *not* measure truth, authority, currency, relevance to the user's actual need, or permission.

**Q23. Why do similarity thresholds need calibration per model and per task?**
**A:** Because score distributions differ by model, metric, corpus, and query type. A 0.75 threshold on model A might mean "clearly relevant," while on model B it might mean "barely related." Calibrate against labeled relevant and irrelevant examples for each configuration.

**Q24. What is the difference between a similarity score and a calibrated probability?**
**A:** A similarity score is just a distance-like number from the comparison function — it has no guaranteed statistical meaning. A calibrated probability means "out of 100 cases that scored like this, X were actually relevant," which requires deliberate calibration on labeled data. By default, an embedding score is the former, not the latter.

**Q25. Why might a historical manual be similar to a current-procedure question but still irrelevant?**
**A:** Because it discusses the same equipment and procedures — high semantic similarity — but it's an obsolete version. Similarity is blind to version and authority. You need metadata filters or reranking to enforce "current version only."

---

# PART 2 — End-to-End Embedding Lifecycle

### 2.1–2.2 Lifecycle Overview

**Q26. Which stage encodes the user's query at request time?**
**A:** Stage 3 — Online Query Encoding and Retrieval.

**Q27. Which stage is responsible for freezing the evaluation corpus before comparing models?**
**A:** Stage 4 — Evaluation, Re-Embedding and Migration. But really the freeze has to happen before any comparison starts; Stage 4 owns that discipline.

**Q28. What is the difference between Stage 2 and Stage 3, and why does it matter for caching?**
**A:** Stage 2 is the offline document-embedding path — you batch through your whole corpus. Stage 3 is the online query path — one query, low latency. They have different SLAs, different batch sizes, and different cache scopes. An embedding cache can serve both, but a cache of *retrieved results* or *generated answers* is a completely separate layer with its own invalidation and authorization concerns.

**Q29. What are the four stages of the embedding lifecycle?**
**A:** Stage 0 is foundations; Stage 1 is task analysis and the compatibility contract; Stage 2 is offline document encoding with caching and validation; Stage 3 is online query encoding and retrieval; Stage 4 is evaluation, re-embedding and migration. (Sometimes Stage 0 is folded into Stage 1.)

**Q30. Why is an embedding cache different from a retrieved-result cache or an answer cache?**
**A:** The embedding cache stores vectors for a specific text under a specific model configuration — it only invalidates when the *text or model* changes. A retrieved-result cache stores query→documents mappings, so it invalidates when *any* of those documents change. An answer cache stores generated prose, which invalidates when documents *or* the generation model change, and carries much higher correctness and privacy risk.

---

### 2.3 Lifecycle Stage Cards

**Q31. What are the key decisions in Stage 1?**
**A:** Symmetric vs. asymmetric retrieval; domain and language fit; token limit; pinned model revision; query/document prefixes; exact text-assembly recipe; tokenizer and truncation policy; pooling, normalization, output dimension, and metric; deployment boundary, throughput and cost.

**Q32. What is the 7-point validation firewall in Stage 2?**
**A:** Before accepting a vector: (1) it's associated with the correct input ID; (2) the expected count is present; (3) the dimension matches the index schema; (4) all coordinates are finite; (5) the norm is valid for the chosen metric; (6) the model/config identity is recorded; (7) the source is still active and eligible for publication.

**Q33. What are the common failure modes of Stage 3?**
**A:** Sorting distance descending instead of ascending; comparing incompatible encoders; zero-vector or NaN results; applying an uncalibrated universal threshold; and letting a semantically similar but unauthorized or outdated source into the answer context.

**Q34. What are the four steps of safe migration in Stage 4?**
**A:** Build a new generation from a checkpoint → evaluate it → reconcile all changes during the build (updates, deletions, ACL changes) → atomically switch query encoder and index together, with rollback ready.

**Q35. Why does Stage 0 come before model selection?**
**A:** Because you need to understand what embeddings can and can't tell you before you pick a model. If you don't know that similarity ≠ relevance and score ≠ probability, you'll pick a model and build a system on wrong assumptions.

**Q36. What happens if you skip Stage 1 and go straight to encoding?**
**A:** You end up with vectors that don't match your query pipeline — wrong prefix, wrong normalization, wrong text assembly — and you get silent correctness failures. No errors, just bad results. The contract has to be defined before you compute anything.

---

### 2.4 Stage Handoffs

**Q37. What's handed over from chunking to Stage 1, and what typically breaks?**
**A:** Approved chunk text, source ID, and source span. What breaks: unstable chunk boundaries, or inconsistent text — so the same paragraph comes out slightly differently on re-ingest, which then looks like a different input to the cache.

**Q38. What's handed over from Stage 2 to the vector index, and what failure mode occurs if corrupted?**
**A:** The validated vector, record ID, source version, and config ID. If corrupted — particularly if the ID association is wrong — you get syntactically valid vectors attached to the wrong source text. Search returns unrelated passages that look fine in isolation.

**Q39. Why must the query service assert the index's embedding configuration ID before serving?**
**A:** Because it's the only way to catch a silent mismatch. If someone swaps the query model but keeps the old index, both produce vectors of the same dimension, so nothing errors — the results just quietly stop meaning anything. An explicit config-ID check fails loudly instead.

---

### 2.5 Cross-Cutting Controls & Failure Classes

**Q40. Name the five failure classes.**
**A:** Compatibility failure, relevance failure, authorization failure, currency failure, and cache isolation failure.

**Q41. A retrieval system has no errors, but relevance quietly degrades after a deploy. Which control was most likely skipped?**
**A:** Identity and versioning — specifically, the query service didn't assert the index's embedding configuration ID. So a new query encoder got paired with an old index, or vice versa.

**Q42. How does identity/versioning differ across stages?**
**A:** Stage 1 pins the model and recipe. Stage 2 stores the config ID with each cache entry and each vector. Stage 3 asserts that the query config matches the index config. Stage 4 tracks index generations and source checkpoints.

**Q43. Why is "permissions" a cross-cutting concern rather than Stage 3-only?**
**A:** Because it has to hold at every stage. Stage 1 defines scope; Stage 2 must recheck eligibility and scope private caches per tenant; Stage 3 must enforce current ACLs at retrieval time; Stage 4 must replay ACL changes and deletions during migration. If it's only checked at one stage, there's a hole.

---

### 2.6 Model Selection Decision Flow

**Q44. Why should you change only one variable at a time when evaluating model improvements?**
**A:** So you can attribute the result. If you swap the model, change chunking, and add a reranker all at once, and quality goes up, you have no idea which change caused it — or whether one of them actually hurt and another helped more.

**Q45. If exact model codes are being missed, what should you try before fine-tuning?**
**A:** Add or improve lexical retrieval and test hybrid fusion. BM25 is already excellent at exact identifiers, and combining it with dense often solves the problem without any retraining.

**Q46. If negation/contradiction errors dominate, what architectural addition is recommended?**
**A:** A cross-encoder reranker plus explicit contradiction checks. Embeddings fundamentally struggle with negation, so you need something that reads the query and passage jointly.

---

### 2.7 Failure-to-Stage Diagnostic Table

**Q47. During an incident: latency up, scores shifted, some content missing, some over-exposed. What's the safe triage order?**
**A:** First, contain access exposure — block the leaking path immediately, don't wait for metrics. Then check whether the deployment changed the active index generation or query encoder (compatibility). Then verify the Stage 1 contract. Then Stage 3 ranking and authorization filters. Then Stage 2 cache scope and validation for cross-tenant leakage. Then Stage 4 reconciliation and rollout state.

**Q48. Results appear shuffled or unrelated — which stage do you check first?**
**A:** Stage 2. Shuffled results almost always mean vectors got mapped to the wrong record IDs — usually an async completion-order assumption in the batch job.

**Q49. Retrieval breaks after a model swap but returns no errors — what's the root cause?**
**A:** Silent compatibility failure. The query encoder and the document index are now in different learned spaces, but because the dimensions match, nothing throws. You need an explicit config-ID assertion to catch this.

**Q50. A "better model" claim comes from one benchmark run — what do you verify?**
**A:** Whether corpus, chunking, reranker, and evaluation set were held fixed. If multiple variables changed, the result is confounded. Also check whether the benchmark was run on the organization's own domain data or just a public leaderboard.

---

# PART 3 — Mathematical Foundations

### 3.1 Cosine Similarity

**Q51. What does cosine similarity compare?**
**A:** Direction. It's the dot product divided by the product of the vector lengths, so the lengths cancel out. Two vectors pointing the same way score 1 regardless of how long they are.

**Q52. Is cosine similarity defined for a zero vector? Why or why not?**
**A:** No. The formula divides by the product of the norms, and one of those norms is zero — division by zero. The system must explicitly reject or handle zero vectors rather than letting NaN propagate.

**Q53. For a unit query `[1,0]`, which vector has the same cosine similarity as `[0.8, 0.6]`?**
**A:** `[8, 6]`. Positive scaling doesn't change direction, so the cosine is identical — 0.8 in both cases. `[0,1]` is perpendicular (score 0), `[-0.8,-0.6]` is opposite (score -0.8), and `[0,0]` is undefined.

**Q54. What is the range of cosine similarity, and what do the extremes mean?**
**A:** Mathematically -1 to 1. 1 means identical direction; 0 means perpendicular/unrelated; -1 means opposite direction. In practice most models occupy a narrower band, often 0 to 1 for normalized embeddings.

**Q55. Can cosine similarity be negative? What does that indicate?**
**A:** Yes, mathematically. It means the vectors point in opposing directions — the model considers them dissimilar or even opposed. Many modern retrieval models rarely produce negatives because of how they're trained, but the formula allows it.

**Q56. What is L2 normalization, and why does it matter?**
**A:** Dividing a vector by its length so it becomes unit length — length 1. After that, cosine similarity equals the plain dot product, which makes indexing simpler and faster. But you have to apply it consistently to both queries and documents, or the rankings break.

**Q57. What is the relationship between cosine similarity and Pearson correlation?**
**A:** Pearson correlation is cosine similarity after *centering* both vectors — subtracting the mean from each. So cosine on raw vectors includes the mean offset; Pearson removes it. For embeddings, you almost always use cosine, not Pearson.

**Q58. Does cosine similarity satisfy the triangle inequality? Why does this matter for ANN indexes?**
**A:** Cosine similarity itself doesn't — it's a similarity, not a distance. But angular distance (arccos of cosine) does satisfy it. This matters because many ANN indexes assume metric properties to prune the search space; if the "distance" isn't a true metric, pruning can miss results or index structures can behave unexpectedly.

---

### 3.2 Dot Product

**Q59. Why does raw dot product differ from cosine?**
**A:** Because it doesn't divide by the lengths. So it's sensitive to both direction *and* magnitude. Two vectors pointing the same way but with different lengths get different dot products, while cosine would tie them.

**Q60. Vector C = `[8,6]` and A = `[0.8,0.6]` point the same direction. Do they rank the same under dot product?**
**A:** No. Against query `[1,0]`, A scores 0.8 and C scores 8.0 — C wins by 10×, purely because of magnitude. Under cosine they'd tie at 0.8.

**Q61. When would a model be trained for inner-product retrieval, and how does that change your indexing setup?**
**A:** Some models are trained with an inner-product objective, where magnitude is meaningful — often because magnitude encodes something like confidence or frequency. In that case you should *not* L2-normalize; you use raw dot product and configure the index accordingly. If you normalize anyway, you throw away a signal the model learned.

**Q62. What happens to dot-product rankings if one vector has 10× magnitude but the same direction?**
**A:** Its dot product is 10× larger, so it jumps to the top. That may be correct if the model intends magnitude to matter, or a bug if it doesn't — which is why you check the model's contract before choosing the metric.

---

### 3.3 Euclidean Distance

**Q63. When are cosine and Euclidean rankings equivalent?**
**A:** For unit-normalized vectors, under exact comparison. Because ‖a−b‖² = 2 − 2(a·b), minimizing Euclidean distance is the same as maximizing the dot product, which for unit vectors equals cosine. Same ranking.

**Q64. Index switches from cosine to Euclidean but the code still sorts descending. What breaks?**
**A:** The ranking inverts. Euclidean distance should be sorted *ascending* — smaller means closer. Sorting descending surfaces the farthest, least relevant vectors first. Classic "results are backwards" bug.

**Q65. What is the difference between Euclidean distance and squared Euclidean distance?**
**A:** Squared distance skips the square root. Since square root is monotonic for non-negative numbers, the *ranking* is identical — so you can sort by squared distance and save computation. But don't compare a squared distance to a threshold calibrated on plain distance.

**Q66. Some vector databases return "negative distance" — how do you interpret and sort this?**
**A:** It's a trick to turn a distance into a similarity, so the database can use "higher is better" everywhere. In that case you sort *descending* — the most negative distance becomes the highest score. Always check the documentation; don't assume the sign convention.

---

### 3.4 Equivalence for Unit Vectors

**Q67. Explain why cosine, dot product, and Euclidean distance give the same ranking for unit vectors.**
**A:** For unit vectors, ‖a−b‖² = ‖a‖² + ‖b‖² − 2(a·b) = 2 − 2(a·b). So as the dot product goes up, distance goes down — a perfect inverse relationship. And for unit vectors, the dot product *is* the cosine. So maximizing any one of them gives the same ordering.

**Q68. If exact rankings are equivalent, why do approximate indexes still return different results?**
**A:** Because ANN indexes don't compute exact rankings — they prune the search space using graph structure or clustering. Different metrics can be approximated differently, and quantization introduces its own error. So the *theoretical* ranking is identical, but the *approximate retrieved set* can differ.

---

### 3.5 Safe Math Rules

**Q69. What should happen when the pipeline produces a zero vector?**
**A:** The system should reject it or handle it under an explicitly declared policy. Silently computing cosine on it gives NaN, which then propagates into rankings and can appear at the top of results. The 7-point validation firewall's "norm is valid" check exists exactly for this.

**Q70. Why must normalization be applied consistently to both query and document pipelines?**
**A:** Because mixing normalized and unnormalized vectors changes the comparison. If documents are normalized but queries aren't, the effective metric is no longer cosine for all pairs — it's a mix of cosine and dot product, and rankings become inconsistent and hard to reason about.

---

# PART 4 — Dimensionality & Storage

**Q71. What is a 384-dimensional embedding?**
**A:** A vector with 384 coordinates. Nothing more — it doesn't mean 384 words, 384 tokens, or 384 documents. It's just the size of the array the model outputs.

**Q72. Two models both output 1024-dimensional vectors. Can they be compared directly?**
**A:** Not necessarily. Same shape just means the arrays fit. The two models learned entirely different coordinate spaces — coordinate 5 might mean something completely different in each. There's no reason a distance between them would be meaningful.

**Q73. What determines storage required for a vector index?**
**A:** Number of vectors × dimension × bytes per coordinate, ignoring overhead. Plus you need to budget for index structures (graphs, metadata), IDs, replicas, and operational headroom.

**Q74. Calculate storage for 1 billion 1536-dimensional float32 vectors.**
**A:** 1,000,000,000 × 1536 × 4 bytes = 6.144 × 10¹² bytes ≈ 6.14 TB in decimal units, about 5.59 TiB in binary. And that's *before* index overhead — realistically you'd need several times that.

**Q75. How much does float32 vs float16 vs int8 quantization save?**
**A:** float32 is 4 bytes per coordinate. float16 is 2 bytes — a 50% saving. int8 is 1 byte — a 75% saving. In exchange you lose precision, so you have to measure the retrieval quality impact; sometimes it's negligible, sometimes it's significant.

**Q76. A team truncates every vector from 1536 to 512 dimensions without checking the model docs. What's the risk?**
**A:** Unless the model was specifically trained to support shortened representations — like Matryoshka training, where the first N dimensions are trained to be independently useful — slicing arbitrary dimensions destroys the learned structure. Quality degrades unpredictably. It should be treated as a migration with evaluation, not a free storage optimization.

**Q77. What is Matryoshka representation learning, and how does it relate to truncation?**
**A:** It's a training technique where the model is trained so that the first 64, 128, 256... dimensions each form a useful representation on their own — like nested dolls. That makes truncation safe for models trained this way. You verify by checking the model card and testing quality at each truncation level.

**Q78. Why is "larger dimension = better retrieval" not always true?**
**A:** Because retrieval quality depends on training data, architecture, and task fit — not just capacity. A 1024-dim model trained on general web text can perform worse on specialized abbreviations than a 384-dim model trained on your domain. Evaluate quality per unit of storage, latency, and cost.

**Q79. Can you mix different embedding dimensions in one vector index? What breaks?**
**A:** Generally no — the index schema expects a fixed dimension, and even if the database allows it, you can't compute a distance between vectors of different sizes. You need separate collections or a migration.

**Q80. What additional storage overhead must you budget for in production?**
**A:** ANN index structures (HNSW graphs can add substantial overhead), stored IDs, metadata and filterable fields, replicas for availability, backups, and free space for re-indexing. Realistically, plan for 2–4× the raw vector size.

---

# PART 5 — Model Selection

**Q81. What's the difference between symmetric and asymmetric retrieval?**
**A:** Symmetric means the query and the document are similar in form — question-to-question, or sentence-to-sentence. Asymmetric means they're different — a short question matched to a long passage. Different models are trained for each, and your evaluation set should match the actual task.

**Q82. A leaderboard shows Model X with the highest MTEB average. Is that sufficient for a legal-document corpus?**
**A:** No. Leaderboard averages reflect benchmark conditions, not your private corpus. Shortlist X, then evaluate it on your own domain slices, your hard negatives, and your actual query distribution. Then compare against your lexical and hybrid baselines.

**Q83. Why are two 768-dimensional models not automatically compatible?**
**A:** Because dimension is just array shape. The models learned independent coordinate spaces with independent meanings. Even if both output 768 numbers, comparing a vector from one with a vector from the other produces a number that has no retrieval meaning.

**Q84. Why should model choice include language-specific tests?**
**A:** Because multilingual capability isn't binary — a model can be great at English and French, mediocre at Hindi, and terrible at cross-lingual retrieval from English to Hindi. Aggregate benchmarks hide these per-language weaknesses. Test each language pair you actually need.

**Q85. A multilingual model matches English questions to Hindi passages well in testing, but production answers are wrong. What do you check before blaming the model?**
**A:** Upstream extraction and OCR quality, tokenization, and whether the Hindi source text is actually correct. Better embeddings can't fix missing or garbled OCR — if the passage never made it into the index correctly, no model will retrieve it well.

**Q86. What are hard negatives, and why do they matter?**
**A:** Hard negatives are passages that look superficially relevant but aren't — right equipment, wrong revision; similar part numbers; opposite conditions. They matter because easy negatives make any model look good. Hard negatives expose the differences that actually matter in production.

**Q87. When would you fine-tune an embedding model vs. use a pre-trained one?**
**A:** First exhaust simpler options — hybrid retrieval, reranking, better chunking. Fine-tune when you have a real domain gap that those can't close, *and* you have labeled query-document pairs to train on, *and* you can justify the operational cost of maintaining a custom model. It's a last resort, not a first move.

**Q88. How would you build a domain-specific evaluation set?**
**A:** Collect real queries from logs or users, label the source spans that answer each one, include hard negatives, include "no authorized evidence exists" cases, and cover your important slices separately — exact identifiers, paraphrases, languages, revision-sensitive queries. Freeze it and version it so results stay comparable.

**Q89. What is the difference between MTEB and BEIR?**
**A:** MTEB (Massive Text Embedding Benchmark) covers many task types — classification, clustering, retrieval, reranking. BEIR focuses specifically on heterogeneous retrieval across many domains. Use both to shortlist, then evaluate locally.

**Q90. A general model confuses "pump" (equipment) with "pump" (verb). What's the first step before fine-tuning?**
**A:** Evaluate on domain queries and compare against a lexical baseline and hybrid retrieval. Often the combination of BM25 plus dense fixes the ambiguity without any retraining. Only fine-tune if the measured gap is still there after that.

**Q91. What is ColBERT, and how does its late-interaction approach differ from standard bi-encoder retrieval?**
**A:** Standard bi-encoders compress each document into a single vector. ColBERT keeps one vector *per token* and computes relevance as a "MaxSim" — for each query token, the maximum similarity to any document token. That's more expressive than single-vector retrieval but costs much more storage, since you're storing many vectors per document.

---

# PART 6 — Query–Document Compatibility

**Q92. Name three things that must match between query and document encoders.**
**A:** Any three of: model family and pinned revision, query vs. document mode, required prefix, tokenization/truncation policy, pooling and normalization, output dimension, similarity function.

**Q93. Why is "same dimension" an insufficient reason to swap the query model while keeping the old index?**
**A:** Because dimension says nothing about whether the learned spaces are compatible. You'll get results — they'll just be meaningless, with no error to warn you. It's a silent correctness failure.

**Q94. A team changes text assembly from "title + section path + body" to "body only" for updated records, but leaves old records untouched. What's the problem?**
**A:** The index now contains two different representation conventions under one label. Old records and new records were embedded from differently structured input, so they're not comparable on equal footing — you'll get inconsistent ranking behavior between old and new content.

**Q95. Documents use model A, queries use unrelated model B with the same dimension. What's the problem?**
**A:** The learned spaces may be incompatible. The system returns numerical scores, but those scores don't represent meaningful retrieval relevance — it's comparing apples to a different kind of apple that looks the same shape.

**Q96. What is E5's prefix convention, and why not copy it to every model?**
**A:** E5 expects `"query: "` before queries and `"passage: "` before documents. It's a model-specific training convention. Applying it to a different model that wasn't trained that way will likely hurt performance, not help — always check the model's own card.

**Q97. Can query and document encoders be different networks? If so, what must be true?**
**A:** Yes — DPR uses two separately parameterized encoders trained to work together. What matters is that they were *jointly trained* to produce compatible representations. The compatibility is the joint contract, not network identity.

**Q98. How do you detect a compatibility failure in production before users report bad results?**
**A:** Assert the index's embedding configuration ID in the query service at startup and on every request, and fail loudly on mismatch. Add observability around config IDs, and run a small "canary" query set whose expected results you know — if the canary degrades, you've caught it before users do.

**Q99. What happens if you apply L2 normalization to documents but not to queries?**
**A:** The effective comparison becomes a mix of cosine (for document-document relationships) and dot product (for query-document). Rankings become inconsistent and magnitude-dependent in unpredictable ways. Normalize both sides or neither.

**Q100. Why is permission enforcement separate from embedding compatibility?**
**A:** Because compatibility is about whether vectors can be meaningfully compared — a purely mathematical property. Permission is about who's allowed to see what — a policy property. A perfectly compatible, high-scoring passage can still be one the user is not authorized to see.

---

# PART 7 — Batch Generation & Validation

**Q101. Why can't you assume async batch completion order equals submission order?**
**A:** Because async processing reorders completions — different items take different amounts of time. If you zip results back to inputs by position, a vector can be silently attached to the wrong source text. You must use explicit response indices or ID mapping.

**Q102. List three of the seven validation-firewall checks.**
**A:** Any three of: correct input-ID association, expected count present, dimension matches schema, all coordinates finite, valid norm for the metric, model/config identity recorded, source still eligible for publication.

**Q103. A batch job partially fails; retry resubmits the whole batch. What goes wrong and how do you fix it?**
**A:** You can write duplicate active vectors for the items that already succeeded. Fix: make writes idempotent keyed on source ID plus config identity, and retry only the unresolved items. Distinguish transient errors from permanent ones so you're not retrying invalid input forever.

**Q104. What makes batch embedding reliable?**
**A:** Stable input IDs, explicit output-to-input mapping, validation of dimension and finiteness, recorded config identity, idempotent writes, partial-failure handling that doesn't duplicate successes, and a source-state recheck before publication.

**Q105. What is idempotent writing, and why does it matter?**
**A:** Writing the same record twice produces the same end state as writing it once — for example, an upsert keyed on (source ID, config ID). It matters because retries are inevitable; without idempotency, every retry risks creating duplicate or conflicting vectors.

**Q106. Difference between retryable and permanent input errors? Give examples.**
**A:** Retryable: timeouts, rate limits, transient 5xx from the embedding service — waiting and retrying may succeed. Permanent: input exceeds the model's token limit, malformed text, unsupported characters — retrying will fail every time, and you should route it to a data-quality queue instead of burning resources.

**Q107. A source is deleted during a long embedding job. What should publication do?**
**A:** Recheck source state before publishing and refuse to publish. A successful computation doesn't override the current source lifecycle — otherwise the embedding job resurrects a deleted document.

**Q108. How would you handle rate limiting when embedding 10 million documents via an API?**
**A:** Batch by token budget as well as item count; implement exponential backoff with jitter on 429s; use a work queue with concurrency limits; persist progress checkpoints so you can resume; and track per-request token counts so you can predict cost and throughput.

**Q109. Why batch by tokens as well as by item count?**
**A:** Because 100 short labels and 100 long passages impose completely different demands on the model — very different total token counts, memory usage, and latency. Batching by item count alone can create oversized batches that time out or OOM.

**Q110. How do you validate that a returned vector actually corresponds to the correct input text?**
**A:** Use explicit response indices or IDs from the API rather than positional zipping; store a hash of the input text alongside the vector; and on ingestion, re-derive the hash and compare. Mismatches fail the validation firewall.

---

# PART 8 — Embedding Caching

**Q111. What must an embedding cache key contain?**
**A:** The exact encoding input plus everything that affects the output: approved cache scope (tenant), model revision, input mode (query vs. document), preprocessing/tokenizer version, pooling/normalization, and output dimension. Serialize unambiguously before hashing.

**Q112. Why is text alone an insufficient cache key?**
**A:** Because the same text under a different model, mode, or preprocessing version produces a different vector. Reusing a cached vector after a model change silently mixes representation spaces — no error, just bad results.

**Q113. What does "verify after hit" mean?**
**A:** A cache hit is a shortcut to data, not a permission slip. Once the cache returns a document ID or vector, you still run that ID through the primary access-control layer — row-level security, tenant filters — before showing the source text to the user.

**Q114. Shared cache: 10ms hit, 500ms miss. A user is probing response times. What risk and how do you mitigate?**
**A:** A timing side-channel. The user can infer whether another tenant has previously cached — and therefore likely searched for — a specific private string. Mitigate with strict per-tenant cache isolation so hits only ever occur within the user's own scope.

**Q115. A heading is in the embedding input and changes while the body stays the same. Should the cache reuse the old vector?**
**A:** No. The actual encoding input changed, so the correct vector may differ. Cache identity must reflect the exact input, including any included metadata — not an approximate notion of "the document."

**Q116. Versioned-key lazy eviction vs. tag-based purging — when do you use each?**
**A:** Versioned keys (e.g., `doc12:v2`) are for *updates* — bump the version and the old entry is instantly bypassed, cleaned up later in the background. Tag-based purging is for *deletions or cascading invalidation* — you need to find all cache entries referencing a document and drop them atomically. Use both: versioned keys for the fast path, tag sets for the purge path.

**Q117. How does Redis Sets + Lua enable atomic tag-based invalidation?**
**A:** On write, you `SET` the cache entry and `SADD` its key into a tag set named for the document. On delete, you `SMEMBERS` that tag set to get all dependent keys, `DEL` them, then `DEL` the tag set. Wrapping `SMEMBERS` + `DEL` in a Lua script makes it atomic, so a new key can't be added mid-purge and orphaned.

**Q118. What is hierarchical cascading, and why is it needed?**
**A:** Tying child chunks to their parent nodes. If a parent section is modified or deleted, the cascade automatically wipes all child chunks, summaries, and linked precomputed windows. Without it, you'd purge only the top-level entry and leave orphaned derived data that still serves stale content.

**Q119. Why is a hard TTL necessary even with active invalidation?**
**A:** As a fail-safe. Active invalidation depends on every code path firing correctly; if one silently fails, stale entries live forever. A hard TTL guarantees any missed stale data eventually purges itself. But it's a backstop, not the primary correctness mechanism.

**Q120. TTL raised from 24h to 30 days; a week later stale results appear. What happened and what's the fix?**
**A:** The TTL was masking a broken active-invalidation path — probably a tag purge silently failing, or an edit path not bumping the version suffix. Extending the TTL simply let already-stale entries survive far longer before the fail-safe caught them. Fix: restore proper versioned keys on every content change and atomic tag-based purges on edit/delete, then keep the TTL short as a backstop.

**Q121. Why can query embedding caches contain sensitive data even though they store vectors, not prose?**
**A:** Because the vector is derived from the user's actual question — and inversion attacks can recover meaning from vectors. Plus the cache key itself may encode query text. Give query caches a retention policy, access controls, and avoid logging raw queries.

**Q122. How would you measure cache hit rate, and what does a sudden drop indicate?**
**A:** Hits / (hits + misses), tracked per cache layer and per tenant. A sudden drop without a deploy usually means something in the cache key changed — input text, model revision, tokenizer, preprocessing, cache scope, or TTL. It's a signal that the representation contract shifted.

**Q123. What is a cache stampede, and how do you prevent it?**
**A:** When a popular cache entry expires and many concurrent requests all miss at once, hammering the backend. Prevent with request coalescing ("single-flight" — only one request recomputes, others wait), or probabilistic early expiration (refresh slightly before expiry for a fraction of requests).

**Q124. How do you design a cache so one tenant's purge doesn't affect others?**
**A:** Namespace segregation — dedicated key prefixes or logical databases per tenant — plus per-tenant tag sets. Then purging tenant A's tag set only touches tenant A's keys. This also gives you instant full-purge capability for a single tenant without a global cache flush.

---

# PART 9 — Re-Embedding & Migration

**Q125. Does an ACL-only change always require re-embedding?**
**A:** No. If the embedded text is unchanged, just update the control metadata and invalidate affected serving caches. Authorization is enforced as a separate policy check, not baked into the vector.

**Q126. Why build a separate index generation for a model migration?**
**A:** So you can evaluate the new one against the old on the same questions, keep serving from the old one in the meantime, and roll back cleanly if something goes wrong. Mixing old and new vectors in one search is never safe.

**Q127. What is a "zombie document," and how does it arise?**
**A:** A deleted source that reappears after migration. It happens when the new index is built from a snapshot taken *before* the deletion, and the deletion is never reconciled or replayed before cutover. Switching to the new generation resurrects it.

**Q128. During a 48-hour build, a document's body is edited twice and its ACL changed once. How should the new generation treat it?**
**A:** Diff by change type. The body edits require re-embedding the *final current text* under the new contract. The ACL change is a metadata update — apply current policy, don't re-embed for it alone. If both happened, regenerate the vector for the new body and publish it with the current ACL. Reconcile before cutover so no stale body or revoked permission goes live.

**Q129. Migration works in staging but brings back deleted documents in production. What likely failed?**
**A:** The new index was built from an old source snapshot without replaying deletions or reconciling current source state before cutover. Immediately block those sources from serving, apply lifecycle changes to the new generation, verify caches and citations, then re-cut over. Future migrations need a checkpoint-and-reconcile step.

**Q130. What is an atomic cutover, and why must query encoder and index switch together?**
**A:** Switching both components in a single coordinated operation so no request ever pairs an old encoder with a new index, or vice versa. If they're switched separately — even briefly — some traffic sees incompatible spaces and produces garbage.

**Q131. How do you roll back a failed migration without serving incompatible pairs?**
**A:** Roll back both components together as a unit — restore the previous (encoder, index) pair atomically, not just one of them. Keep the old generation serving or warm until the new one is validated, and have a tested rollback runbook.

**Q132. What is a "generation" in vector indexes, and how do you track it?**
**A:** A separately identifiable version of the whole indexed population — built from a specific source checkpoint under a specific embedding contract. Track it with an explicit generation ID recorded in the index, the query service config, and your observability.

**Q133. How would you estimate cost and time of re-embedding 50 million documents?**
**A:** Measure tokens per document, multiply by document count to get total tokens, then divide by the embedding service's throughput (tokens/sec) and multiply by its price per token. Add index rebuild time, storage, and reconciliation overhead. Then pad for retries and failures.

**Q134. What is blue-green deployment for embeddings, and how does it differ from rolling update?**
**A:** Blue-green means you build the entire new generation (green) alongside the old (blue), validate it, then switch traffic over atomically. A rolling update gradually replaces instances, which is exactly what you *can't* do for embeddings — during the roll, some instances would serve incompatible pairs. Blue-green makes the switch atomic; rolling does not.

**Q135. How do you A/B test during migration without exposing users to incompatible results?**
**A:** Keep both generations fully self-consistent — each request is routed entirely to one generation or the other (by user, session, or flag), never mixing them. Segment by traffic, measure both, and switch fully only when the new one wins.

---

# PART 10 — Worked Example (M-104)

**Q136. Why freeze corpus and chunking before comparing embedding models?**
**A:** So the comparison isolates the embedding model. If you change chunking at the same time, you can't tell whether a quality difference came from the model or the chunks.

**Q137. Why use exact (non-approximate) vector search on a sample during baseline evaluation?**
**A:** To separate embedding-model quality from ANN approximation error. If you evaluate with an approximate index, poor results could be the index's fault, not the model's — you'd blame the wrong thing.

**Q138. For L3 depth, what must be demonstrated beyond building a new index?**
**A:** A migration carried out while live updates happen — including edits, deletions, and ACL changes during the build — with evidence that query and document spaces stayed compatible throughout publication. Not just a static before/after.

**Q139. What metrics would you report when comparing models in M-104?**
**A:** Recall@K, a rank-sensitive metric like MRR or nDCG, query latency, total vector storage, and per-slice breakdowns — exact identifiers, paraphrases, languages, revision-sensitive queries. Plus qualitative failure examples.

**Q140. Difference between Recall@K and MRR?**
**A:** Recall@K asks "of all the relevant items, how many appeared in the top K?" — it measures coverage. MRR (Mean Reciprocal Rank) asks "how high did the *first* relevant result rank?" — it measures how quickly the user finds something useful. Recall@K for candidate generation; MRR/nDCG for ranking quality.

**Q141. How would you design hard negatives for M-104?**
**A:** The v3 manual (wrong revision), the tenant B document (wrong tenant), passages about the same equipment but a different unit, and near-identical part numbers. Cases where a naive system would likely be fooled.

**Q142. What is an evidence label, and why isn't "similar paragraph" good enough?**
**A:** An evidence label marks the exact source span that actually *answers* the query, under the correct revision and authorization. A merely similar paragraph might discuss the same topic but not answer the question — labeling it relevant teaches the evaluation to reward the wrong behavior.

---

# PART 11 — Concept Map & Revision

**Q143. Define "compatible space" in your own words.**
**A:** Two representations are in a compatible space if distances between them are meaningful — because they were produced by the same model, or by encoders explicitly trained to work together. Same dimension is not enough; the coordinate meanings have to align.

**Q144. Difference between an "index generation" and a "model revision"?**
**A:** A model revision is the version of the embedding model itself — the thing that defines the representation space. An index generation is a version of the *indexed data* built under some model revision. You can have multiple generations under the same model (e.g., rebuilt for source updates), and new generations under a new model revision (a migration).

**Q145. Explain "Same contract, valid vectors, measured retrieval, traceable updates."**
**A:** Same contract — query and document encoders must follow one compatibility contract. Valid vectors — reject zero, NaN, wrong-dimension, and wrong-ID vectors via the validation firewall. Measured retrieval — evaluate on your own frozen dataset with lexical, dense, and hybrid baselines; don't trust leaderboards. Traceable updates — version everything: model configs, cache keys, index generations, migrations. If you can't trace it, you can't debug it.

---

# PART 12 — Interview Questions

**Q146. How would you choose a similarity function?**
**A:** Check the model's intended setup first — its card will say whether it expects cosine, dot product, or normalized vectors. Then configure the index to match, verify whether returned values are similarities or distances, and test the ranking on known examples. Don't switch between raw dot product and cosine casually, because magnitude can silently change the order.

**Q147. How would you evaluate two embedding models?**
**A:** Hold corpus, extraction, chunking, permissions, and the query set fixed. For each model, use its own correct query/document encoding. Compare retrieval metrics, hard query slices, latency, cost, and storage. Use exact search on a sample to remove ANN noise. And always include lexical and hybrid baselines.

**Q148. A team proposes one cosine threshold for every model. How do you respond?**
**A:** I'd explain that score distributions differ by model and task, so the same number means different things. We'd calibrate acceptance or abstention per configuration using labeled relevant and irrelevant examples. And we wouldn't rely on the threshold for authority, authorization, or factual correctness — those are separate checks.

---

# PART 13 — Senior & Staff-Level Production Scenarios

**Q149. False answer despite strong retrieval scores — where do you begin?**
**A:** Start by separating similarity, relevance, and truth. Trace the answer back to the exact source span. Check whether a high retrieval or reranker score is being treated as a correctness guarantee — it isn't. Verify the passage actually supports the claim, inspect negation and conditionals, and confirm revision and authority rules were applied. Strong recall is not evidence verification.

**Q150. Fast and accurate modes sharing one index with two models of the same dimension — evaluate.**
**A:** Reject it. Equal dimension doesn't make the spaces compatible — a query from model A against documents from model B is meaningless. Safe options: separate model-specific index generations with matching query encoders, or one model with different candidate-pool sizes and reranker settings for fast vs. accurate behavior. The latter is usually simpler to operate.

**Q151. Zero-vector rejects from page footers — how do you fix it?**
**A:** Keep the firewall — it's doing its job. Fix the root cause upstream: filter boilerplate and near-empty retrieval units during chunking, before encoding. Don't weaken the zero-norm check just to make the batch pass. Add a regression test so extraction changes don't reintroduce the boilerplate.

**Q152. One confidence threshold across three pipelines — recommendation?**
**A:** Don't apply one raw score threshold across different models and metrics. Ranges and semantics differ, and distances may run opposite to similarities. Calibrate per pipeline against labeled relevant and hard-negative examples. Validate that any common downstream stage produces comparable outputs before sharing a threshold. Keep authorization, currency, and factual support as separate checks.

**Q153. Reconciliation finds body and ACL changes — how should the new generation handle them?**
**A:** Diff by change type. Body-text change → re-embed under the new contract. ACL-only change → update metadata, no re-embedding needed. Both → regenerate for the new body and publish with current ACL. Pure deletion → absent from the new generation. Recheck source state before cutover.

**Q154. Latency spike, shifted scores, missing content, over-exposure — safe triage order?**
**A:** Contain the access exposure first — block the leaking path immediately. Then check whether the deploy changed the active index generation or query encoder. Then verify the Stage 1 contract. Then Stage 3 ranking (sort direction) and authorization filters. Then Stage 2 cache scope and validation for cross-tenant leakage. Then Stage 4 reconciliation and rollout state. Use config IDs in logs to find where the deployed system first diverged from known-good.

**Q155. Cache TTL raised 24h→30 days, stale results a week later — what happened, fix?**
**A:** The TTL was masking broken active invalidation — a silently failing tag purge or an edit path that didn't bump the version. Extending the TTL let already-stale entries survive far longer before the fail-safe caught them. Fix: restore versioned keys on every content change and atomic tag-based purges on edit/delete, then keep TTL short as a backstop so failures surface quickly instead of silently for weeks.

---

# NEWLY ADDED — Additional Interview & Real-Life Questions

## A. Embedding Fundamentals (Deep Dive)

**Q156. What is the "curse of dimensionality," and how does it affect nearest-neighbor search?**
**A:** In very high dimensions, distances between random points concentrate — the farthest and nearest neighbors become almost equally far. That makes "nearest" less meaningful and makes brute-force search expensive. It's a big part of why ANN indexes and dimensionality reduction techniques exist.

**Q157. Why do embeddings sometimes fail on numerical reasoning?**
**A:** Because they encode *semantic similarity*, not arithmetic. "5 minutes" and "300 seconds" describe the same duration but the model has no built-in mechanism to equate them — it's matching meaning patterns from training data, not computing values. Anything requiring exact numeric comparison should be handled by structured logic, not embeddings.

**Q158. Difference between semantic and lexical similarity? Give a case where they diverge.**
**A:** Lexical similarity is about shared words; semantic is about shared meaning. "Car" vs. "automobile" — low lexical, high semantic. "Bank" (river) vs. "bank" (financial) — high lexical, low semantic. That's why hybrid retrieval is often better than either alone.

**Q159. Bi-encoder latency vs. cross-encoder latency — when to use each?**
**A:** Bi-encoder: both sides encoded independently, so document vectors are pre-computed — milliseconds. Cross-encoder: query and document go through the model together — tens to hundreds of milliseconds per pair. So bi-encoder for first-stage retrieval over millions of docs; cross-encoder for reranking the top 50–100.

**Q160. What is hard negative mining in training?**
**A:** Deliberately selecting training negatives that the model currently ranks highly but which are actually irrelevant — forcing it to learn the fine distinctions that easy negatives never teach. It's the training-time counterpart of hard negatives in evaluation.

**Q161. How does Sentence-BERT differ from vanilla BERT for sentence similarity?**
**A:** Vanilla BERT has no efficient way to compare two sentences — you'd have to feed the pair through the model together every time, which doesn't scale. Sentence-BERT trains BERT with a siamese/triplet objective so each sentence maps to a single vector, enabling fast cosine comparison of pre-computed embeddings.

## B. Production Engineering & Operations

**Q162. What metrics would you alert on for an embedding service?**
**A:** Query latency (p50/p95/p99), embedding API error rate, cache hit rate, index query latency, zero-vector/NaN rejection rate, retrieval quality on a canary query set, config-ID mismatches, and per-tenant access anomalies. Quality alerts are as important as uptime alerts.

**Q163. Your embedding provider deprecates your model version — walk through the migration.**
**A:** Freeze the evaluation set, build a new index generation with the replacement model under a new contract, evaluate against the current one on the same queries, reconcile all source changes during the build, then atomic cutover of query encoder and index together with rollback ready. Never swap one side without the other.

**Q164. Embedding service outage — what's your fallback?**
**A:** For query embedding: a cached response for repeated queries, a graceful degradation message, or a lexical-only fallback path if you have hybrid retrieval. For document ingestion: queue and retry — it's offline, so you can tolerate delay. The key is that retrieval should degrade, not fail hard, if you've designed for it.

**Q165. Online vs. offline evaluation — when do you use each?**
**A:** Offline for model selection and regression testing — frozen dataset, reproducible metrics, no user exposure. Online for validating real-world impact — click-through, user satisfaction, task completion — because offline metrics don't always correlate with what users actually experience.

**Q166. Debug a sudden retrieval quality drop with no code deploy.**
**A:** Check what changed *outside* the code: source documents updated, ACLs changed, model provider pushed a silent update, cache TTL expired, an index rebuild happened, or query distribution shifted. Compare against the canary set, check config IDs and generation, and diff the source corpus.

**Q167. What's the SLA for an embedding service, and how do you design for it?**
**A:** Typically low-latency on the query path (tens of milliseconds) and high-throughput on the ingestion path. Design with pre-computed document embeddings, caching for queries, async batch ingestion with retries, and a fallback path so query-time embedding failures don't take down retrieval.

**Q168. How would you monitor embedding drift over time?**
**A:** Track distributional statistics of query embeddings (mean, variance, cluster density) against a baseline, monitor retrieval quality on a fixed canary query set, and alert when quality degrades even if no code changed. Corpus drift and query drift are both real.

**Q169. "Can we just use ChatGPT embeddings for everything?" — response?**
**A:** We could shortlist it, but "for everything" is the problem. It depends on the task, the domain, the language mix, the cost, and the deployment constraints. Let's evaluate it against our alternatives on our own data — including lexical and hybrid baselines — and decide with numbers, not vendor marketing.

**Q170. GDPR "right to be forgotten" when embeddings derive from personal data?**
**A:** If the vector is derived from personal data, it's in scope. You need to be able to identify all derived entries — vectors, caches, replicas — tied to a data subject and purge them, with cascade through tag-based invalidation. Keep source-to-derived-data mappings so deletion is complete, and document it in your retention policy.

**Q171. Environmental/cost impact of embedding a large corpus — how do you justify it?**
**A:** Estimate token count × provider price for the initial run, plus ongoing costs for updates and queries. Compare against the business value of the retrieval capability. Optimize with batching, caching, smaller models where quality allows, and incremental re-embedding rather than full rebuilds.

## C. Architecture & Indexing

**Q172. What is HNSW, and how does it affect latency and storage?**
**A:** Hierarchical Navigable Small World — a multi-layer graph where each vector connects to nearby neighbors, and search walks the graph greedily. It gives very low query latency but adds substantial memory/storage for the graph edges on top of the raw vectors. It's the most common ANN index for vector databases.

**Q173. IVF vs. HNSW — when would you choose IVF?**
**A:** IVF (Inverted File) clusters vectors and searches only the nearest clusters. It's cheaper to build and uses less memory than HNSW, but usually has lower recall and needs tuning of the number of probes. Choose it when memory is constrained or the dataset is huge and you can accept slightly lower recall.

**Q174. What is product quantization, and how does it trade storage for recall?**
**A:** PQ splits each vector into sub-vectors and quantizes each sub-vector to a codebook entry, storing the code instead of the raw floats. That dramatically cuts storage — often 10–100× — but introduces approximation error, so recall drops. Tune the number of sub-quantizers and codebook size to balance.

**Q175. How do you choose between exact and ANN search?**
**A:** Exact when the dataset is small enough, or when recall matters more than latency (e.g., evaluation), or when you need to separate model quality from index quality. ANN when the dataset is too large for brute-force and you can accept approximate recall in exchange for millisecond latency.

**Q176. What's the recall-latency trade-off in ANN, and how do you tune it?**
**A:** More search effort — more graph hops, more probes, larger candidate lists — gives higher recall but higher latency. You tune by adjusting HNSW's `ef_search`, IVF's `nprobe`, or the quantization settings, and you measure recall against an exact baseline at each setting to find the knee of the curve.

**Q177. Can you update a vector in-place in an ANN index, or must you rebuild?**
**A:** Most ANN indexes support incremental inserts and deletes, but deletes often leave tombstones and performance degrades over time, so periodic rebuilds are common. Dimension changes or metric changes require a new index. For model migrations, a new generation is the safe path rather than in-place updates.

## D. Security, Privacy & Compliance

**Q178. Can embeddings leak sensitive info through inversion attacks? What mitigations exist?**
**A:** Yes — research (e.g., vec2text-style attacks) shows text can be partially reconstructed from embeddings, especially for short inputs. Mitigations: treat embeddings as sensitive derived data, encrypt at rest, restrict access to vector stores, scope caches per tenant, and never expose raw embeddings via public APIs.

**Q179. How do you ensure tenant isolation in a shared vector database?**
**A:** Tenant ID as a mandatory filter on every query, namespace segregation in caches, row-level security or equivalent in the database, per-tenant encryption keys where possible, and audit logging. Defense in depth — don't rely on a single check.

**Q180. What is row-level security, and how does it apply to vector retrieval?**
**A:** A database feature where policies restrict which rows a given user can see, enforced by the database itself rather than application code. For vector retrieval, it means the ANN search runs within the user's permitted rows — so unauthorized documents never even become candidates, not just get filtered after.

**Q181. How do you audit which documents were used to answer a query?**
**A:** Log the final candidate IDs, their source versions and config IDs, the retrieval and reranking scores, and the generation of the index used. That trace lets you reconstruct exactly what evidence was in context for any answer.

**Q182. Risk of logging raw query embeddings?**
**A:** They're derived from user questions and can be partially inverted, so they're sensitive. Logging them creates a privacy exposure and a retention obligation. Prefer logging query *hashes* or aggregated statistics unless you have a specific, controlled need.

## E. Advanced Retrieval Patterns

**Q183. What is Reciprocal Rank Fusion, and how does it combine sparse and dense results?**
**A:** RRF scores each document as the sum of 1/(k + rank) across the result lists it appears in, where k is a smoothing constant (often 60). A document that ranks highly in *both* lists gets a high fused score. It's simple, requires no score normalization across incomparable systems, and works well as a default fusion method.

**Q184. What is a reranker, and why is it needed after embedding retrieval?**
**A:** A reranker — usually a cross-encoder — reads the query and each candidate passage together and produces a more accurate relevance score. It's needed because bi-encoder similarity is approximate and can't reason about negation, conditionals, or fine distinctions. Retrieval gets good candidates; the reranker picks the best ones.

**Q185. Retrieval vs. reranking latency budget?**
**A:** Retrieval (ANN) is typically single-digit to low-tens of milliseconds. Reranking a top-50 candidate list with a cross-encoder can be tens to hundreds of milliseconds depending on model size and batching. So you keep the retrieval pool large enough for recall but small enough that reranking stays within the overall latency SLA.

**Q186. How would you implement query expansion before embedding?**
**A:** Use an LLM or a thesaurus to generate synonyms or reformulations of the query, embed each variant, and merge the result lists (often with RRF). It helps with vocabulary mismatch but adds latency and can introduce drift — so measure whether it actually helps on your query distribution.

**Q187. What is HyDE, and when might it help?**
**A:** Hypothetical Document Embeddings — you ask an LLM to write a *hypothetical answer* to the query, embed that answer, and search with it instead of the raw query. It helps when queries are short and questions but documents are long and answer-like, closing the asymmetry. Downside: extra LLM latency and cost, and the hypothetical answer can hallucinate and mislead.

**Q188. Multi-vector (ColBERT-style) vs. single-vector retrieval?**
**A:** Single-vector compresses each document to one embedding — compact and fast, but loses token-level detail. Multi-vector keeps per-token embeddings and scores with late interaction (MaxSim) — more accurate, especially for fine-grained matching, but storage grows with document length. Choose based on whether you can afford the storage.

## F. Data Quality & Chunking Interaction

**Q189. Why do low-information chunks produce poor embeddings?**
**A:** Because there's almost no semantic content to encode. "Page 3 of 45" or a repeated footer has no distinguishing meaning, so the model produces a near-degenerate or zero vector. These end up clustering together and polluting results — which is why they should be filtered before encoding.

**Q190. How does chunk size interact with the model's token limit?**
**A:** If the chunk exceeds the model's max input tokens, it gets truncated — and everything past the cutoff is invisible to the embedding. That's why the chunk token budget must be set below the model's limit, with headroom for any prefix or title metadata you include.

**Q191. A chunk is truncated before embedding — how do you detect it?**
**A:** Tokenize the input and compare against the model's limit at ingestion. The 7-point validation firewall should include an input-length check; log and flag any chunk that exceeded it. Truncation is silent otherwise — the vector exists, it's just missing information.

**Q192. How do you handle tables and structured data?**
**A:** Flatten them into text in a way that preserves meaning — include column headers with each row, or convert to a sentence form. Naive extraction scrambles tables into unreadable sequences, which embed poorly. This is an extraction-stage problem, not a model problem.

**Q193. Why might embedding a whole document be a bad idea?**
**A:** Because the vector averages over everything in the document, so a query about one specific section gets diluted by all the unrelated content. Retrieval needs granularity — chunk-level embeddings let you find the exact passage. Whole-document embeddings are better for coarse filtering than for answering questions.

## G. Cost & Performance

**Q194. Estimate the cost of embedding a 100-million-document corpus.**
**A:** Estimate average tokens per document (say 500), multiply to get 50 billion tokens, then multiply by the provider's price per million tokens. Add index storage (100M × dimension × bytes) and compute for the rebuild. Then pad 20–30% for retries, re-embedding after fixes, and ongoing updates.

**Q195. Trade-off between embedding dimension and query latency?**
**A:** Higher dimension means more compute per distance calculation and more memory bandwidth, so brute-force search slows roughly linearly with dimension. ANN indexes mitigate this but still pay some cost. Lower dimension is faster and cheaper but may lose quality — evaluate both.

**Q196. How does batching affect throughput and cost?**
**A:** Larger batches improve throughput by amortizing per-request overhead and using hardware more efficiently, which usually lowers cost per token. But batches have a maximum size, and oversized batches increase tail latency. For offline ingestion, maximize batch size within limits; for online queries, batch size is usually 1.

**Q197. Larger model with fewer dimensions vs. smaller model with more dimensions — which?**
**A:** It's not a rule — measure both on your data. What matters is retrieval quality per unit of cost. A larger model with Matryoshka-style truncation can give you flexibility: use the first 512 dimensions for a fast index and the full 1536 for a precise rerank. Don't assume; benchmark.

## H. Evaluation & Benchmarking

**Q198. What is nDCG, and when would you use it instead of Recall@K?**
**A:** Normalized Discounted Cumulative Gain — it accounts for *graded* relevance (a perfect answer counts more than a partial one) and *position* (higher results are worth more). Use it when relevance isn't binary and when the ordering within the top K matters, not just whether relevant items appear somewhere in the list.

**Q199. How do you build a golden evaluation set that won't leak into training?**
**A:** Draw queries from real usage, label the answer spans carefully, version and freeze it, restrict access so it can't be used for training, and rotate or refresh it periodically. If you fine-tune on your evaluation set, you're measuring memorization, not generalization.

**Q200. Intrinsic vs. extrinsic evaluation of embeddings?**
**A:** Intrinsic evaluates the embedding directly on a proxy task — similarity benchmarks, clustering, classification. Extrinsic evaluates the end-to-end system — retrieval quality, answer correctness, user satisfaction. Intrinsic is faster and cheaper for model selection; extrinsic is what actually matters in production.

**Q201. How do you evaluate retrieval when there are multiple valid answers?**
**A:** Label *all* valid answer spans, then use coverage-based metrics (did the retrieved set include at least one valid answer?) plus rank-aware metrics over the set of valid answers. Don't penalize retrieving a different valid answer than the one you had in mind.

**Q202. What is the "no authorized evidence exists" case, and why must it be in your evaluation set?**
**A:** It's a query where the correct behavior is to return nothing — because the only relevant sources are unauthorized, deleted, or don't exist. If your evaluation only has answerable queries, you'll never test whether the system correctly abstains. A system that always returns *something* looks great on answerable queries and fails badly in production.

---

## Final Memory Aid

> **Same contract, valid vectors, measured retrieval, traceable updates.**

If you can explain each phrase — with a concrete example and a counter-example — you can handle almost any interview question on this topic.
