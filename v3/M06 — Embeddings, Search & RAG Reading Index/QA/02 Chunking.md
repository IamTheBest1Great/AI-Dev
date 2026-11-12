Here is the same Q&A content in **simple, easy-to-understand language**, organized section by section.

---

# Section 1 — Foundations (The Basics)

## Syllabus Questions

**SQ1. What is chunking for?**  
Chunking cuts documents into smaller pieces so the computer can find the right information quickly, and the AI can read it correctly. Each piece must still know where it came from and who is allowed to see it.

**SQ2. Why can very small chunks fail?**  
Because a tiny piece may miss important words like “what equipment,” “which unit,” “under what condition,” or “except when…”. It matches the question but does not give a full answer.

**SQ18. How do you choose a starting chunking method?**  
Look at your documents and the questions people ask. Start with a simple method that respects headings and paragraphs, and measure size in tokens (not words). Keep headings, source location, and permissions. Make test cases for tables, steps, exceptions, and restricted sections.

**SQ19. What is the difference between a retrieval unit and a context unit?**  
- **Retrieval unit** = the small piece the search engine scores (like one paragraph or one sentence).  
- **Context unit** = the bigger piece the AI actually reads (like the small piece plus its heading, nearby sentences, or parent section).  
Small for finding, bigger for understanding.

## Interview Questions

**Q1. Why not always use 500 tokens with 20% overlap?**  
Because every document is different. A fixed cut can split a heading from its explanation or a rule from its exception. Choose size and overlap by testing what actually works for your documents and questions.

**Q3. What is lost if a chunk only says “Inspect every 30 days”?**  
You lose: which equipment, which unit (Unit A or B), the section heading, the conditions, and any exceptions. That is **context loss** — not old data.

**Q4. How do you balance chunks that are too big and too small?**  
Use three levels:  
1. **Source unit** — the original section.  
2. **Retrieval unit** — a small child piece for precise search.  
3. **Answer-context unit** — the parent section or nearby sentences for the AI to read.  
Search small, then expand carefully.

**Q7. Explain the precision-versus-context trade-off.**  
Big chunks give more context but include junk and waste space. Small chunks are precise but may lose conditions or exceptions. The fix: search small, then expand to the parent or sentence window.

**Q21. What factors matter before choosing a chunking method?**  
Document type (prose, tables, code, PDFs), the kinds of questions users ask, token limits of the embedding model, reranker, and AI model, citation needs, security needs, and cost of re-indexing.

**Q31. How are chunking, embedding, and retrieval connected?**  
Chunking makes the pieces. Embedding turns each piece into a vector (a list of numbers). Retrieval compares the question vector to chunk vectors to find matches. Bad chunks = bad search results even with a good embedding model.

**Q32. Does bigger chunk size always improve retrieval?**  
No. Bigger chunks can mix unrelated topics, lower search precision, cost more tokens, slow things down, and bury the answer in the middle of a long passage.

**Q33. Can a chunk be grammatically correct but still bad for retrieval?**  
Yes. “Inspect every 30 days” is a complete sentence, but if the equipment name is only in the heading above, the chunk is ambiguous. Always keep or link the heading.

## Clarified Doubts

**Doubt 1. Is context loss the same as stale data?**  
No.  
- **Context loss** = the information is current but separated from what explains it (bad chunk boundary).  
- **Stale data** = the information is old because the source was updated or deleted but old chunks remain.

**Doubt 2. How do we balance too-large and too-small chunks in practice?**  
Keep three separate things: the original source unit, a small retrieval unit for search, and a larger answer-context unit for the AI. Retrieve small, then expand with parent section or sentence window.

---

# Section 2 — Stage A: Token and Model Budgets

## Syllabus Question

**SQ5. Why measure in tokens instead of words or characters?**  
Because AI models have token limits. Words and characters are rough guesses. Headings, prefixes, and formatting also use up tokens. Always use the exact tokenizer of the model that will read the chunk.

## Interview Questions

**Q18. Why is token-aware chunking important?**  
Models count in tokens, not words. One word, number, code, or punctuation mark can become several tokens. Two chunks with the same word count can have different token counts. Measure with the model’s tokenizer to avoid overflow.

**Q19. Why separate budgets for embedding, reranking, and answer generation?**  
Because each stage uses a different model and a different input:  
- **Embedding:** chunk + heading + formatting.  
- **Reranker:** user question + candidate chunk + formatting.  
- **Answer AI:** system prompt + chat history + question + all evidence + reserved output.  
A chunk that fits the embedder may fail the reranker or the answer AI.

**Q20. Chunk has 30 heading tokens + 470 content tokens + 20 formatting tokens. Limit is 512. Does it fit?**  
No. Total = 520. It is over by 8 tokens. Trim the body or split the chunk; do not let the API silently cut the end.

## Scenario

**Scenario 5. Chunk fits embedding but fails reranking.**  
Treat them as separate budgets. The reranker sees question + chunk + formatting together. Calculate the combined tokens and shorten or prepare the chunk without removing the useful information.

## Clarified Doubts

**Doubt 7. Why not rely on word count?**  
Words and tokens are different. A tokenizer can split one word into many tokens. Use the target model’s tokenizer.

**Doubt 8. Why separate token budgets?**  
Because embedding, reranking, and answer generation process different combinations of text with different model limits.

---

# Section 3 — Stage B: Cutting Boundaries, Strategies, and Overlap

## Syllabus Questions and MCQs

**SQ3. What is recursive chunking?**  
It tries big natural breaks first (sections), then paragraphs, then sentences, then words, until the piece fits the size limit. It needs a clear order of separators and a fallback rule.

**SQ4. How is semantic chunking different from sentence splitting?**  
Sentence splitting uses grammar (periods, question marks). Semantic chunking uses meaning: it compares neighboring sentences and cuts where the topic changes.

**SQ8. What does overlap fix?**  
It copies a little text from the end of one chunk to the start of the next, so information right at the cut is not lost. It does not fix missing headings, long conditions, table headers, or permission problems.

**Syl-MCQ11. Which statement is correct?**  
A. 500 characters = 500 tokens. B. Every sentence is a complete evidence unit. C. Chunk size units must be stated explicitly. D. Bigger chunks always improve retrieval.  
**Answer: C.**

**Syl-MCQ12. Which strategy uses headings and tables as boundaries?**  
A. Structure-aware chunking. B. Random chunking. C. Unrestricted parent expansion. D. Score normalization.  
**Answer: A.**

**Syl-MCQ13. Chunk says “Repeat this every 30 days” but misses the definition above. What is wrong?**  
A. Too much evidence. B. Missing context at the boundary. C. Perfect design. D. Wrong vector size.  
**Answer: B.**

**Syl-MCQ14. Chunk size 400, overlap 80. What is the stride?**  
A. 80. B. 320. C. 400. D. 480.  
**Answer: B.** Stride = 400 − 80 = 320.

**SQ20. When should you add overlap?**  
When testing shows that useful facts are being cut at chunk boundaries. Measure the effect on search quality, duplicates, index size, and token cost. If the real problem is a missing heading or prerequisite, use structure-aware chunking or parent-child expansion instead.

**SQ21. How do you chunk a maintenance table?**  
Keep the table structure. Each row chunk should include the table title, column headers, row labels, units, and footnotes. Cite the original table and row. Use structured data for exact calculations. Test similar equipment and different conditions.

## Interview Questions

**Q2. What goes wrong when you split without considering meaning?**  
You can separate an instruction from its equipment name or its safety exception. The AI then gets an incomplete piece and may answer wrongly.

**Q6. How should tables and safety exceptions be handled?**  
- **Tables:** keep title, column headers, row labels, units, and footnotes with every row chunk. Repeat headers if you split.  
- **Safety exceptions:** keep them with the rule they change, or link them through the same parent section.

**Q10. How do you chunk a maintenance table safely?**  
Use row-oriented chunks. Each chunk should say, for example: “Table: Equipment Inspection Schedule | Equipment: Generator B | Inspection interval: 15 days | Note: complete checklist first.” Keep the source location.

**Q13. Fixed-size vs. recursive chunking?**  
Fixed-size cuts mechanically at a set number of characters, words, or tokens. Recursive tries natural breaks first (section → paragraph → sentence → word) and only cuts smaller when needed.

**Q14. What does “implementation complexity” mean in recursive chunking?**  
It means the setup work: choosing the separator order, using a token-accurate length function, handling oversized sentences, keeping headings, and tracking exact source positions. It is not heavy compute.

**Q15. Sentence-based vs. semantic chunking?**  
Sentence-based uses grammar. Semantic uses meaning similarity between neighboring pieces and cuts where the topic shifts.

**Q16. What is one big limitation of semantic chunking?**  
It can split a rule from its exception because the exception uses different words. It also costs more (embeds every sentence) and needs threshold tuning.

**Q22. Chunk size vs. overlap size?**  
Chunk size = maximum size of one piece. Overlap = how much text is repeated between neighboring pieces. Stride = chunk size − overlap.

**Q23. How do you choose overlap percentage?**  
Do not pick a number blindly. Test small overlaps on real questions. Measure whether boundary errors drop enough to justify extra index size, duplicate results, and token cost.

**Q24. Why preserve document hierarchy?**  
Because headings often say which equipment, section, or department the text belongs to. Keep the section path in the chunk text or metadata, and link children to parents.

**Q27. What if a paragraph is bigger than the chunk size?**  
Split it at sentence boundaries first. If one sentence is still too big, split at clauses or words, and log that a hard split happened. Keep the parent heading and source location.

**Q28. Can one document need different chunking methods?**  
Yes. Prose can use recursive/header-aware splitting. Tables need row-plus-header chunks. Procedures need step-preserving chunks. Code needs syntax-aware chunks.

**Q37. Why might semantic chunking perform worse than recursive chunking?**  
Because semantic similarity can separate logically connected text that uses different vocabulary, like a rule and its exception. Structured documents often work better with recursive or structure-aware methods. Compare on the same questions and documents.

**Q39. Why do chunks still exceed the target after recursive splitting?**  
Three common reasons: the length function counts characters or words instead of tokens; there is no fallback for a long paragraph or OCR line with no separators; headings or prefixes were added after splitting and pushed the total over the limit.

**Q40. Paragraph boundary or sentence boundary?**  
Keep the paragraph if it is one coherent idea and fits the token budget. Split at sentences if the paragraph is too big, mixes topics, or you are building a sentence-window index. Always keep dependent conditions and exceptions together.

## Scenarios

**Scenario 1. Heading and content separated.**  
Problem: “Emergency Generator” is in one chunk, and the inspection instructions are in another. Fix: use header-aware chunking; put the heading in the chunk text or in inherited metadata.

**Scenario 2. Safety exception retrieved separately.**  
Keep the general rule and the exception together if they fit. If not, link both children to the same parent section so expansion brings both. Check the condition before answering.

**Scenario 7. Table row without headers.**  
A chunk says only “Generator B / 15 days.” Fix: repeat the table title and column headers in every row chunk, e.g., “Equipment: Generator B | Inspection interval: 15 days.”

**Scenario 8. Long procedure split in the middle of a step.**  
Keep complete steps together if possible. If a step must split, keep the step number, procedure heading, condition, and warning with each part, and link the parts.

**Scenario 10. PDF procedure crosses pages.**  
Page breaks are not meaning breaks. Stitch the procedure across pages, keep page numbers in metadata, and chunk by the document’s logical structure.

## Clarified Doubts

**Doubt 4. Does recursive chunking always make the biggest possible chunks?**  
No. It follows the separator order under a maximum size. A 336-token section stays one chunk if the cap is 400. If the cap is 100, it splits at natural boundaries.

**Doubt 5. What does complexity in recursive chunking mean?**  
It means configuration work: separator order, token counting, fallback handling, keeping headings, and tracking source positions. Not high CPU cost.

**Doubt 6. What is the limitation of semantic chunking?**  
Meaning boundaries are not always clear. A safety exception may use different words from the rule and get split away. Semantic similarity must be guided by structure and logic.

---

# Section 4 — Stage C: Lineage, Metadata, and Security

## Syllabus Questions

**SQ9. Why include source version in chunk metadata?**  
So the system can tell old from current information, produce correct citations, stop a new child from expanding to an old parent, and delete the right pieces when the source changes.

**SQ22. Chunking update causes citations to point to the wrong revision. Why?**  
Likely chunk IDs or parent links were reused without version or chunking-config information, or old chunks stayed active. Fix: inspect lineage, remove stale mappings, rebuild versioned links, test current and historical questions, and publish indexes atomically.

## Interview Questions

**Q25. What metadata should each chunk have?**  
- **Text/context metadata:** document title, section path, equipment/unit, table headers.  
- **Control metadata:** chunk_id, document_id, parent_id, previous/next IDs, version_id, chunking_config, source_span, tenant_id, ACL reference/version, current/deleted flags.  
Authorization metadata must be checked by the backend, never trusted from the chunk text.

**Q26. Why is source traceability important?**  
It lets you show where an answer came from, deduplicate by original location, compare chunking methods fairly, delete the right data, and debug bad answers.

## Scenario

**Scenario 12. Two companies have identical documents.**  
This is a tenant-isolation bug. Give every document, chunk, and parent a backend-verified tenant_id. Before search, expansion, secondary search, or citation, enforce tenant_id = logged-in user’s tenant.

---

# Section 5 — Stage D: Query-Time Retrieval and Expansion

## Syllabus Questions and MCQ

**SQ6. What is parent-child retrieval?**  
Search small child chunks for precision, then expand to the larger parent section for context. Always re-check permissions and version on the parent.

**SQ7. What is a sentence window of radius two?**  
The focus sentence plus up to two sentences before and two after — at most five sentences. Clip the window at section, version, and permission boundaries.

**Syl-MCQ16. User can access one paragraph in a mixed-permission section. What should parent expansion do?**  
A. Return whole section. B. Ask model to hide restricted parts. C. Expand only to authorized spans. D. Remove metadata.  
**Answer: C.** Access to one child is not access to all siblings.

## Interview Questions

**Q5. Retrieved child chunk lacks enough context. What do you do?**  
Do not let the AI guess. If the missing context is in the same section (heading, scope, exception), use parent-child expansion or a sentence window. Check version and permissions. If the missing context is in another section or document (“see Section 8”), run a new authorized search.

**Q8. Retrieval returns Unit A and Unit B together. What do you do?**  
Check metadata and structure. Separate the evidence by unit. If the question is about Unit B, prioritize Unit B chunks and verify the parent context so Unit A’s rule is not applied to Unit B.

**Q11. Risk of isolating “30 days” from the equipment name?**  
The interval becomes ambiguous and may be applied to the wrong equipment — for example, Unit A’s 30-day rule applied to Unit B, which has a 14-day rule. Keep the heading and equipment name, and link to the parent.

**Q12. Unit B has a 14-day schedule. What should retrieval return?**  
The 14-day schedule with clear Unit B context, any conditions or exceptions, and a source citation. It must not include or confuse Unit A’s 30-day schedule.

**Q29. Retrieval context vs. answer-generation context?**  
Retrieval context is the small unit the search engine scores. Answer-generation context is the expanded, deduplicated, token-budgeted text the AI actually reads.

## Scenarios

**Scenario 6. Parent section too big for the answer budget.**  
Do not insert the whole parent. Expand to a smaller subsection or a sentence window that includes the subject, conditions, exceptions, and explanation. Stay within the token budget and remove duplicates.

**Scenario 9. Correct topic, wrong unit.**  
Put the unit name in the chunk text and metadata. Filter by unit when possible. Use hybrid search for exact terms. Before answering, verify the assembled context clearly says Unit B.

**Scenario 11. “See Section 8” cross-reference.**  
Parent-child expansion only works inside the same section hierarchy. It will not fetch Section 8. Detect the cross-reference and run a second authorized search for Section 8. Then verify it is the referenced procedure.

## Clarified Doubt

**Doubt 3. What can parent expansion recover, and what can it not?**  
It can recover the heading, subject/equipment, scope, conditions, exceptions, definitions, nearby explanations, and parent-child relationships. It cannot recover information in a different document or distant cross-referenced section. That needs a secondary search.

---

# Section 6 — Stage E: Deduplication, Evaluation, and Deleting Old Data

## Syllabus Questions and MCQs

**SQ10. How do you compare chunking methods fairly?**  
Use the same documents, same questions, and same retrieval settings. Judge by stable source locations or facts, not by chunk IDs. Measure answer quality and token cost, not just search hit rate.

**Syl-MCQ15. Three chunks contain the same sentence because of overlap. How much independent support is that?**  
A. Three sources. B. One statement repeated three times. C. No support. D. Proof it is universally true.  
**Answer: B.**

**Syl-MCQ17. Method B has higher chunk Recall@10 but chunks are 10× bigger. Best interpretation?**  
A. B is definitely better. B. A is definitely better. C. Compare stable source coverage, irrelevant context, and token cost before deciding. D. Stop checking citations.  
**Answer: C.**

**SQ23. Team wants semantic chunking because it sounds advanced. How do you assess it?**  
Ask what measured problem it solves. Compare it with fixed and structure-aware recursive methods on the same questions and source facts. Include ingestion cost, boundary stability, reprocessing effort, retrieval quality, and answer support. Adopt only if the improvement is worth the extra complexity.

## Interview Questions

**Q9. Disadvantages of too much overlap?**  
More chunks, more storage and embedding cost, duplicate results fill Top-K, one fact looks like several independent proofs, and chunk-level Recall@K looks artificially high. Deduplicate by source location.

**Q17. How do you evaluate whether one chunking method is better?**  
Keep documents, questions, embedding model, and retrieval settings the same. Change one variable at a time. Measure against stable source facts: coverage, context relevance, answer correctness, faithfulness, citation accuracy, token use, latency, duplicate rate. Also read failure cases manually.

**Q30. Why evaluate with real questions instead of just looking at chunks?**  
Because a chunk can look clean but still fail at query time — missing heading, parent too big, vocabulary mismatch, or wrong ranking. Real questions test the whole path from search to expansion to cited answer.

**Q34. What if the source document has repeated content?**  
Separate intentional repetition from accidental duplication. If the same warning appears in two different sections with different scope, keep both. If it is boilerplate, overlap, or duplicate uploads, strip or deduplicate by source location or content hash.

**Q35. How should document updates be handled after chunking and embedding?**  
Treat it as a migration. Mark new chunks with the new version and current=true. Atomically deactivate or purge old child chunks, vectors, keyword entries, parents, sentence windows, summaries, and caches. At query time, enforce current=true and parent.version_id = child.version_id.

## Scenarios

**Scenario 3. Same fact in multiple retrieved chunks.**  
Too much overlap causes redundancy, token waste, and false confidence. Deduplicate by source location, not chunk ID. Reduce overlap if structure-aware boundaries already work.

**Scenario 4. Two versions have different instructions.**  
Use version metadata and current flags. Filter or prioritize the current authorized version. Deactivate or remove old chunks. Do not combine intervals or let the AI guess. If version authority is unclear, report the conflict.

---

# Section 7 — Cross-Stage Troubleshooting and Core Doubts

## Cross-Stage Interview Questions

**Q36. Chunks look correct but answers are still wrong. What do you check?**  
Check each handoff:  
1. **Stage A:** Was the chunk silently truncated during embedding?  
2. **Stage C/D1:** Are stale versions or wrong-unit chunks passing the filters?  
3. **Stage D2:** Are the right chunks failing to rank in Top-K?  
4. **Stage D3–D5:** Is a cross-reference being handled by parent expansion instead of a second search?  
5. **Stage D6/E1:** Is expansion flooding the prompt, duplicating evidence, or burying the answer in the middle?

**Q38. Is the problem chunking or retrieval?**  
Check in order:  
1. **Look in the index:** If the chunk is broken, severed, or truncated, the problem is chunking/ingestion.  
2. **Look at retrieved hits:** If good chunks exist but are not in Top-K, the problem is retrieval/filtering/ranking.  
3. **Look at the assembled prompt:** If the right child was retrieved but the answer is wrong, the problem is expansion, token budget, conflicting siblings, or the AI prompt.

## Core Doubts D1–D8 (Simple Language)

**D1. Four failure types.**  
- **Context loss:** current fact separated from what explains it. Fix with structure-aware chunks and expansion.  
- **Stale data:** old chunks remain after the source changes. Fix with version metadata, hard filters, and purging.  
- **Duplicate evidence:** same source statement repeated. Fix by deduplicating by source location and inserting shared parents once.  
- **Retrieval failure:** good chunks exist but search does not find them. Fix with hybrid search, query expansion, reranking, and better chunk size.

**D2. How to balance big and small chunks?**  
Keep the original source unit, use a small child for search, and expand to a parent or window for the AI. Enforce token budgets, permissions, and source-location deduplication.

**D3. What can parent expansion recover, and what can it not?**  
It recovers local structure: heading, subject/equipment, scope, conditions, exceptions, definitions, table headers. It cannot recover distant cross-references or other documents. Use a second authorized search for those.

**D4. Does recursive chunking always make the largest chunks?**  
No. It respects the size cap and natural boundaries. It does not mechanically make equal-sized pieces.

**D5. What does “implementation complexity” mean in recursive chunking?**  
It means setup choices: separator order, token counting, fallback for oversized text, keeping headings, and tracking source positions. It is not heavy compute.

**D6. Core limitation of semantic chunking?**  
It measures word/topic similarity, not logical dependency. It can split a rule from a differently worded exception.

**D7. Why not use word or character counts?**  
Models count tokens. Tokenizers split words, numbers, identifiers, punctuation, and non-Latin text in different ways. Use the exact tokenizer of the model that will read the chunk.

**D8. Why separate token budgets for embedding, reranking, and answering?**  
Because each stage uses a different model and a different mix of text: chunk+prefix for embedding, question+chunk for reranking, and system+history+question+all evidence+output for answering.

---

# Section 8 — Rapid-Fire Interview Challenge (Simple Answers)

| # | Question | Simple Answer |
|---|---|---|
| 1 | Is smaller chunk size always better? | No. Small chunks find precise matches but can lose equipment name, scope, and exceptions. Search small, then expand. |
| 2 | Why can overlap hurt? | It bloats the index, wastes tokens, makes one fact look like several proofs, and inflates chunk-level Recall@K. |
| 3 | Why is a heading important in a chunk? | Headings often contain the only mention of the subject, equipment variant, or operating scope. |
| 4 | When prefer recursive chunking? | For normal text with paragraphs and sentences and a strict token limit. It is the safe default. |
| 5 | When prefer semantic chunking? | For unstructured prose with no headings, and only if testing shows it beats simpler methods enough to justify the cost. |
| 6 | Why repeat table headers? | A row like “Generator B / 15 days” means nothing without the column header and units. |
| 7 | How handle a very long paragraph? | Split at sentences, then clauses/words if needed. Log hard splits. Keep headings and metadata. |
| 8 | What does parent-child retrieval solve? | It gives precise search with small children and enough context with the parent. |
| 9 | Why include version in metadata? | To filter old rules, stop new children expanding to old parents, and delete the right data on updates. |
| 10 | Why is source traceability necessary? | For citations, deduplication, fair evaluation, and targeted deletion/debugging. |
| 11 | Why different token limits per model? | Embedder, reranker, and answer AI see different inputs and have different maximum sizes. |
| 12 | What if information is in another section? | Parent-child only works locally. Use a second authorized search for cross-references. |
| 13 | How prevent duplicate evidence? | Deduplicate by original source location, not chunk ID. Insert a shared parent only once. |
| 14 | How investigate wrong RAG answers? | Check chunking, then retrieval, then expansion/assembly/AI, stage by stage. |
| 15 | How know a chunking method is better? | Keep everything else fixed, change one thing, and test on stable source facts plus real questions. Read failure cases. |

---

# Quick Coverage Check

This simple-language version covers:
- Syllabus Q&A: SQ1–SQ10, SQ18–SQ23
- Syllabus MCQs: Syl-MCQ11–Syl-MCQ17
- Interview Q&A: Q1–Q40
- Scenarios: 1–12
- Clarified doubts: Doubt 1–Doubt 8 and cross-stage D1–D8
- Rapid-fire challenge: 15 Q&A

All answers are shortened and simplified while keeping the technical meaning.
