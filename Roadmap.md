Here is the complete union of the documents you provided. All overlapping information has been combined, and every distinct detail from the files has been preserved.

---

# 🗺️ The Complete Production AI Engineer Roadmap

**Who this is for:** MERN stack developers who want to go beyond calling APIs and build AI systems that real businesses pay for. The roadmap adds missing layers like MCP, evals, multimodal pipelines, observability, security, and business-layer thinking.

### Your Unfair Advantages as a MERN Developer

Most pure ML engineers cannot do these, which gives you an edge:

* **React:** Build polished AI frontends, streaming UIs, and voice interfaces.
* **Node.js:** Run AI backends, queues, and webhooks at scale.
* **MongoDB:** Store conversation history, user profiles, and memory documents.
* **REST APIs:** Wire AI to any external tool instantly.
* **Authentication:** Ship multi-tenant AI SaaS from day one.

---

## ⚙️ Layer 0 — Engineering Foundations (Months 1-2)

*Big companies require these non-AI engineering foundations for AI systems.*

* **TypeScript (Strict Mode):** Required because LLM outputs are unpredictable JSON.
* Generics, conditional types, mapped types, template literal types, and recursive types for nested JSON.
* Zod for runtime validation of LLM JSON output (type-safe at parse time) and type inference.
* Discriminated unions for LLM responses and branded types for ID safety (e.g., preventing mixing up `TenantId` and `UserId`).
* Strict `tsconfig` configuration with no `any` types.


* **System Design:** Designing architectures that handle scale.
* CAP theorem for AI: consistency, availability, partition tolerance (e.g., RAG vector stores can be eventually consistent, billing cannot).
* Rate limiting patterns: Token bucket, leaky bucket, sliding window.
* Database sharding (100M+ rows, CPU > 70%), hash-based sharding for multi-tenant data, and query routing logic.
* Event-driven architecture: Kafka/Redis Streams for AI pipelines, producer-consumer patterns.
* Idempotency: Preventing duplicate processing in AI job queues.
* Back pressure: Queue-based buffering, load shedding, autoscaling, priority queues.


* **Databases (Beyond CRUD):**
* **PostgreSQL:** Index types (B-Tree, GIN, BRIN, partial, covering), reading `EXPLAIN ANALYZE` plans, connection pooling (PgBouncer), table partitioning by date.
* **Redis:** Pub/Sub for real-time events, Streams for queues, sorted sets for leaderboards/rate limiting, distributed locks (`SET NX EX`), TTL strategies.
* **MongoDB:** Aggregation pipelines (`$match`, `$lookup`, `$unwind`, `$group`) for complex conversation analytics.
* **pgvector:** HNSW vs IVFFlat indexes, hybrid search (vector + BM25), RRF scoring.
* **Time-series patterns:** Storing AI quality scores and querying week-over-week metrics.


* **Distributed Systems Basics:**
* Exactly-once vs at-least-once delivery (BullMQ defaults), idempotent consumers, deduplication.
* Saga pattern for multi-step workflows, compensating transactions, rollbacks.
* Outbox pattern for atomic DB writes and event publishing.


* **Observability:**
* Structured logging with JSON logging (Pino), requiring fields like event, model, tokens, latency, userId, tenantId.
* Metrics with Prometheus: Counters (total calls), histograms (latency p50/p95/p99), gauges (cache hit rate).
* Distributed tracing: Langfuse for AI, OpenTelemetry for infra, connecting AI traces to infra traces.


* **CI/CD for AI Systems:**
* Complete pipeline: Lint → Type check → Unit tests → Integration tests → Eval suite runs → Cost estimate checks → Security scans.
* Staging deployment (Docker build → deploy → smoke tests) and Canary deployment (5% traffic → monitor 30 min → full rollout or auto-rollback).
* Prompt-only changes: Shadow mode, gradual rollout, A/B quality tracking.


* **Layer 0 Project:** Type-safe LLM client wrapper with Zod validation, retry logic, and discriminated union responses.

---

## 🟢 Phase 1 — AI Foundations (Weeks 1-3)

*Goal: Talk fluently to AI models, control costs, and ship streaming UIs fast.*

**Basic Skills:**

* API setup (OpenAI, Anthropic, Google Gemini, Mistral) with API keys, orgs, projects, and rate limits/headers.
* Chat completions, Vercel AI SDK (`useChat`, `useCompletion`, `useObject`), and getting JSON output (`json_mode`, `response_format`).
* System prompts vs. user prompts, and conversation history management (message arrays).
* Token counting (`tiktoken`), cost management, streaming responses (SSE), and server-sent events architecture.

**Advanced Skills & Prompt Engineering:**

* Zero-shot, one-shot, and few-shot prompting (optimal example count 3-5).
* Chain-of-thought (CoT) and Tree-of-thought (ToT) reasoning.
* Role prompting, personas, and negative prompting ("DO NOT" patterns).
* Delimiter techniques (XML tags, triple backticks) and prompt injection defense (sanitization, escaping, role separation).
* Prompt compression (LLMLingua) and prompt caching (Anthropic/OpenAI) to save costs.
* Model tiering (e.g., GPT-4o Mini vs Claude Sonnet vs o1 reasoning series) and LiteLLM proxy for switching models.
* Fine-tuning GPT/Gemini on custom JSONL data.

**What Most Developers Miss:**

* Dynamic system prompts (injecting user data at runtime).
* Output validation before sending to users (hallucination defense, factual checks, PII, toxicity).
* Prompt versioning in databases from Day 1, A/B testing prompts for quality, and model fallback chains.

**Phase 1 Projects:**

* **AI Chatbot:** Basic chat in MERN, streaming token-by-token UI, typing indicators, conversation history.
* **AI Writing Assistant:** Multiple personas, prompt engineering, JSON output, tone selector.
* **Personal AI Tutor:** Multi-turn conversations, tracks weak topics, adapts difficulty.
* **AI Recipe Generator:** Structured JSON, macro calculator, shopping list export.
* **Multilingual Translator App:** Detects language automatically, token optimization, shows confidence score.
* **Prompt Playground:** Version prompts, cost estimator, compare outputs side-by-side.
* **AI Email Responder / Changelog Generator:** Paste emails to get drafts; feed git diffs for human-readable logs.

---

## 🟡 Phase 2 — Core AI Patterns (Weeks 4-8)

*Goal: Build the backbone of real AI applications — search, memory, retrieval, and structured output.*

**Chronological Learning Path & Core Skills:**

1. **Embeddings:** Text-to-dense-vector transformation, similarity properties, vector arithmetic.
2. **Generating Embeddings:** Using OpenAI (`text-embedding-3-small`), Cohere, batch embedding, rate limits.
3. **Vector Databases:** Pinecone (namespaces, metadata filtering), Supabase pgvector (hybrid search, row-level security), Weaviate, Qdrant, Chroma.
4. **Storing and Querying Vectors:** Cosine similarity, dot product, Euclidean distance, filtering metadata.
5. **RAG Basics:** Upload → chunk → embed → store → retrieve → generate.
6. **Frameworks:** LangChain.js (LCEL, chains, document loaders) and LlamaIndex.ts (data ingestion, indexing engines).
7. **Tool/Function Calling:** LLM → tool schema → execute → result → LLM. Designing schemas, parameter typing, parallel/sequential execution, error handling.
8. **Advanced RAG (Chunking & Re-ranking):** Chunking strategies (fixed-size, recursive, semantic, token-based) with 10-20% overlap. Re-ranking with cross-encoders.
9. **Hybrid Search:** Combining dense vectors with BM25 keyword search using RRF scoring.
10. **Multi-Query & Self-Querying:** LLM generates query variants or metadata filters automatically.
11. **Guardrails & AI Safety:** NeMo Guardrails, input/output moderation, validation pipelines.
12. **Structured Outputs:** Zod schemas, type inference, `Instructor.js` for reliable extraction, JSON mode vs function calling, exponential backoff retries.
13. **Knowledge Graphs (Neo4j):** Entity-relationship modeling, Cypher queries, automated entity extraction.
14. **Graph-Augmented RAG:** Vector search + graph traversal, multi-hop reasoning, HyDE (Hypothetical Document Embeddings), contextual compression, parent-child chunking.

**What Most Developers Miss:**

* Chunking experimentation (A/B testing chunk strategies) is the #1 RAG failure point.
* RAG evaluation using the RAGAS framework (Faithfulness, Answer relevancy, Context precision, Context recall).
* Document pre-processing (OCR, table extraction, image captioning before embedding).
* Incremental indexing (diff-based updates) to avoid full re-embedding, and index freshness strategies.

**Phase 2 Projects:**

* **PDF Q&A Bot:** Multi-PDF upload, source highlighting, per-doc namespace isolation.
* **AI Knowledge Base:** Advanced RAG, chunking strategies, real-time index updates.
* **Semantic Search Engine:** Hybrid search, shows relevance scores and match highlights.
* **Legal Document Analyzer:** Guardrails, strict schema validation, clause-by-clause risk scoring.
* **AI Study Buddy:** Knowledge graphs, multi-query RAG, auto-generate flashcards and quizzes.
* **Personal Finance Advisor:** Tool calling (calculator), document processing, budget suggestions.
* **RAG Quality Dashboard / Competitive Intelligence Engine:** RAGAS metrics, A/B testing, competitor tracking via web fetch.

---

## 🟠 Phase 3 — Bots & Messaging Platforms (Weeks 9-12)

*Goal: Deploy AI where users already live.*

**Basic Skills & Platform APIs:**

* Telegram Bot API, Discord.js (slash commands, embeds, threads), WhatsApp (Twilio/Meta API), Slack Bolt SDK, Instagram DMs, SMS (Twilio).
* Bot Architecture: Webhook setup, signature validation, bot commands (`/start`, `/help`), inline keyboards, storing conversation history (MongoDB), stateful flows, rate limiting.

**Advanced Bot Patterns:**

* AI phone call bots (Twilio Voice + Whisper + ElevenLabs text-to-speech).
* Email AI agents (Gmail API integration for auto-reply).
* Payments and subscriptions inside chat (Stripe checkout in Telegram/WhatsApp).
* Proactive bots (scheduled messages, price alerts), multi-platform bots, and background jobs using BullMQ + Redis.

**What Most Developers Miss:**

* Bot onboarding UX (progressive disclosure, interactive tutorials).
* Message queuing to avoid platform rate limits.
* Human handoff workflows based on confidence threshold detection.
* Bot analytics (command tracking, drop-off points).

**Phase 3 Projects:**

* **Telegram AI Assistant:** Cross-session memory, remembers user preferences.
* **Discord Study Server Bot:** Explains topics, weekly quiz tournaments, leaderboards.
* **WhatsApp Customer Support Bot:** Auto-replies, human escalation below 70% confidence.
* **AI News Digest Bot:** Web scraping, BullMQ scheduling, personalized topics.
* **Phone Receptionist Bot:** Twilio Voice, books calendar appointments.
* **Slack Standup Bot / E-commerce Bot with Payments / Instagram Lead Qualifier:** Workflow summarizations, Stripe cart checkouts, AI lead scoring.

---

## 🔴 Phase 4 — Agents & Automation (Weeks 13-17)

*Goal: Build AI systems that reason, take actions, use tools, remember information, interact with software, and complete real-world tasks autonomously.*

**Module Breakdown & Core Skills:**

1. **Agent Foundations:** Reactive vs proactive, ReAct pattern (Reason → Act → Observe loop).
2. **Tool Use & Execution:** Tool schemas, lifecycle, tool categories (Retrieval, Action, Computation). *Note: Tool descriptions are prompts.*.
3. **Agent Frameworks:** LangChain agents, LangGraph (stateful workflows, cycles, branching, checkpoints), OpenAI Assistants API (Threads, File Search), CrewAI.js (Roles, Goals, Delegation).
4. **Planning Systems:** Task decomposition, Planner-Executor pattern, Plan-and-Execute agents, Reflexion (self-reflection loops).
5. **Browser & Web Automation:** Puppeteer/Playwright, Browserbase (cloud sessions), Stagehand (AI-native browser control), Firecrawl/Jina Reader/Exa/Tavily for AI-optimized search.
6. **Data Extraction Pipelines:** Scrape → Extract → Store pipelines, n8n workflow automation, self-healing pipelines, change detection algorithms.
7. **MCP (Model Context Protocol):** Architecture (hosts, clients, servers, tools), building and publishing Node.js MCP servers to standardizing integrations.
8. **Advanced Agent Systems:** Computer Use API (Anthropic), Coding agents (Cursor/Devin models), Parallel agents, Multi-agent systems (collaboration, debate).
9. **Reliability & Production:** Circuit breakers (Max Steps, Max Cost), retries, observability (LangSmith/Langfuse logging), agent security (sandboxing, allowlists).

**What Most Developers Miss:**

* Agents need timeouts and circuit breakers; infinite loops are a common failure.
* Agent ≠ LLM: Most bugs are tool or state problems, not model problems.
* Trace logging from Day 1 and testing with adversarial inputs.
* Memory is often overused; good state and retrieval are better than massive long-term memory.

**Phase 4 Projects:**

* **AI Research Agent:** ReAct, web search, structured reports with citations.
* **Job Application Agent:** Puppeteer, form automation, follows up.
* **Price Tracker Agent / Competitor Monitor:** Monitors sites, alerts on changes, MCP integration.
* **LinkedIn Content Agent:** n8n, adapts tone based on engagement.
* **AI Code Reviewer:** GitHub API, suggests tests, links docs.
* **Data Extraction Pipeline / Autonomous Invoice Agent:** Scrapes layouts, reads Gmail/Stripe PDFs.

---

## 🔵 Phase 5 — Advanced AI Systems (Weeks 18-23)

*Goal: Build multimodal, voice, and multi-agent AI products.*

**Core Skills:**

* **Voice AI:** Whisper API (STT), OpenAI TTS/ElevenLabs/Cartesia, OpenAI Realtime API (WebSockets), Voice Activity Detection (VAD), speaker diarization.
* **Vision & Multimodal:** GPT-4o Vision/Claude/Gemini image understanding, multi-image reasoning, document vision (tables/handwriting), video frame extraction, image generation (DALL-E 3, Stable Diffusion, Flux), and image editing (inpainting).
* **Multi-Agent Systems:** Orchestrator + worker pattern, LangGraph stateful DAG workflows, CrewAI roles, shared state vs message passing, hierarchical delegation, agent checkpointing.
* **Memory Systems:** Mem0 (persistent cross-session memory), structured memory (user facts), memory compression, and forgetting mechanisms (GDPR compliance, staleness).
* **Edge & On-Device AI:** Transformers.js (browser inference), WebLLM (WebGPU), ONNX Runtime Web, Ollama (local serving, 4-bit quantization), React Native on-device (phi-3-mini).

**What Most Developers Miss:**

* Voice UI requires interruption handling (barge-in support).
* Multi-agent systems need shared context stores.
* Image resolution impacts AI understanding heavily.
* Realtime API is billed per minute, not per token.

**Phase 5 Projects:**

* **AI Content Agency:** LangGraph/CrewAI with a critic agent rejecting low quality.
* **Voice AI Doctor Assistant:** Real-time transcription, symptom extraction.
* **AI Interior Designer:** Vision models, DALL-E, before/after sliders.
* **AI Meeting Summarizer / Video Summarizer:** Diarization, action items, multimodal pipelines.
* **Real Estate AI Agent / Perplexity Clone:** Semantic search, agentic browsers, Exa API, citations.
* **AI Therapist / Podcast Generator:** Mem0 memory tracking, multi-voice debates.

---

## ⚫ Phase 6 — Production & Scale (Weeks 24-28)

*Goal: Ship reliable AI products that survive real users and real load.*

**Core Skills:**

* **Observability:** LangSmith, Langfuse, Helicone (cost analytics), OpenTelemetry, and custom dashboards (cost per feature, quality scores, cache hit rates).
* **Evals:** LLM-as-judge (rubric design, calibration), RAGAS framework, PromptFoo YAML eval suites, regression testing, red-teaming (jailbreaks), and building baseline eval datasets.
* **Cost Optimization:** Prompt compression (LLMLingua), response/semantic caching (GPTCache), model routing (LiteLLM proxy), Batch API.
* **Reliability:** Exponential backoff retries, fallback chains, circuit breakers, queue-based processing (BullMQ, SQS), graceful degradation.
* **Security:** Prompt injection defense, indirect prompt injection, PII detection (Presidio, GLiNER), data residency, moderation APIs, secret management, OWASP LLM Top 10.
* **Multi-Tenancy:** Namespace isolation (Pinecone, row-level security), per-tenant model configs, usage metering, tenant-level rate limiting.

**What Most Developers Miss:**

* Running evals and maintaining baseline datasets before shipping.
* Prompt versioning in databases, not in code.
* Setting up AI cost alerts and utilizing async pipelines (not everything needs to be real-time).

**Phase 6 Projects:**

* **AI SaaS Starter / Multi-Tenant Chatbot Platform:** Auth, Stripe billing, AI credits, white-labeling, per-client config.
* **AI API Gateway:** LiteLLM proxy, custom routing, cost tracking limits.
* **Prompt Management Dashboard:** PromptFoo, visual A/B testing.
* **AI Usage Analytics Platform / Eval Suite Builder:** Helicone tracking, LLM-as-judge CI/CD gates.

---

## 🟣 Phase 7 — Cutting Edge & Niche Moats (Weeks 29+)

*Goal: Stay ahead of 99% of developers with rare capabilities.*

**Core Skills:**

* **MCP (Model Context Protocol):** Architecture (hosts/clients/servers), Node.js server building, security models, npm publishing, Claude Desktop integration.
* **Fine-Tuning & Custom Models:** Decision frameworks (when NOT to fine-tune), OpenAI JSONL prep, LoRA/QLoRA (parameter-efficient tuning), Unsloth, PEFT patterns, dataset curation, evaluation against base models, and serving (Ollama, vLLM).
* **Autonomous Agent Loops:** Human-in-the-loop checkpoints, long-horizon tasks, self-correcting agents.
* **Niche Moats:** Document intelligence (AWS Textract/Azure), AI for code (AST-aware), Multimodal RAG (CLIP embeddings + pgvector), Audio AI, Geospatial AI, Accessibility AI.

**What Most Developers Miss:**

* Fine-tuning is expensive; exhaust prompt engineering and RAG first.
* Published MCP servers get high visibility in ecosystem.
* Niche domain expertise + AI beats generic AI.
* Local hosting offers zero API costs and full privacy.

**Phase 7 Projects:**

* **Autonomous Business Agent:** Mem0, email agents, Computer Use to manage CRM and invoices.
* **Personal AI OS:** Long-term memory, unified interface, controls apps.
* **Mini Cursor Clone / AI Dev Tool:** Reads codebases, writes features, fixes bugs.
* **Custom Fine-tuned Model API:** Fine-tune on niche data (legal/medical), host via Ollama.
* **Custom MCP Server / Multimodal RAG System:** Connect to internal APIs, search scanned docs.
* **AI Startup in a Box / AI SaaS in 72 Hours:** Fully autonomous company or rapid SaaS deployment.

---

## 💼 Phase 8 — The Business Layer

*Goal: Turn technical skills into income through positioning, pricing, and selling.*

* **Productizing AI:** Identify high-value use cases by industry, AI ROI framing (hours saved vs "uses GPT-4"), MVP scoping, pain point identification.
* **Freelancing & Consulting:** Upwork/Toptal positioning, proposal writing, discovery calls, retainer models, pricing strategies ($3k–$30k builds).
* **Building AI SaaS:** Niche selection ("AI for dentists"), pricing models (per-seat, usage-based), tech stack (Next.js + Clerk + Stripe + Supabase + OpenAI), launch strategies (Product Hunt, Indie Hackers), churn prevention.
* **Building in Public:** Content strategy (weekly build logs, failure posts), LinkedIn/Twitter/YouTube, Open source strategy, Networking (LangChain/Vercel contributions).
* **Phase 8 Actions:** Build/charge for a niche SaaS, write 10 LinkedIn posts, open-source a tool, land a $3k+ project, host a live build.

---

## 🛣️ Path B: ML Engineer / Model Customizer

*Focus: Work closer to the metal by training and deploying models yourself.*

* **Core Skills:** Neural network fundamentals (backpropagation, layers), Transformer architecture (attention, positional encoding), Tokenization (BPE), Fine-tuning (LoRA, full fine-tune), Quantization (GGUF, GPTQ), Model serving (vLLM, TGI), Running local models, Evaluation (evals), ML pipelines.
* **Capabilities:** Custom models outperforming APIs, self-hosted services, offline-first apps, high-performance inference, multi-model routers.

---

## 🧠 The 99th Percentile AI Engineer Edge (Production Capability)

*The layer that separates tutorial builders from engineers who think in systems.*

**Layer 1 — Mental Models**

1. **Think in Failure Modes:** Ship systems that degrade gracefully. Write failure mode docs before starting.
2. **Eval-First Mindset:** Define success before prompting. Build labeled datasets. Use `PromptFoo`, `RAGAS`, `LLM-as-judge`, `Braintrust`.
3. **Prompt as Code:** Store prompts in DBs with versioning, metadata, and A/B testing registries.
4. **Context Window Budget:** Treat context like RAM. System prompt (500 tokens), context (4000), history (500), tools (trimmed), user message.

**Layer 2 — Rare Technical Skills**
5.  **LangGraph Mastery:** Conditional branching, human-in-the-loop, resumable agents, parallel branches.
6.  **MCP Early Advantage:** Build and publish real MCP servers on npm.
7.  **AI Security:** OWASP LLM Top 10 implementation, prompt injection defense, PII scrubbing (`presidio`, `GLiNER`), jailbreak red-teaming.
8.  **Semantic Caching:** Cache by meaning (`GPTCache`), 40-60% cost reduction.
9.  **Structured Extraction at Scale:** Multi-page PDF extraction, table extraction (`pdfplumber`), schema auto-detection.
10. **Observability:** Custom metadata, cost attribution per feature, quality scoring per trace, anomaly detection, cohort analysis.

**Layer 3 — Architecture Thinking**
11. **Multi-Tenant Architecture:** Namespace isolation, tenant-level routing/quotas, prevent data leakage.
12. **Async AI Pipelines:** BullMQ job queues, webhooks, SSE streams, dead letter queues.
13. **Model Routers:** Build routers that classify complexity and fallback based on cost caps and latency.

**Layer 4 — Positioning**
14. **Niche Expert:** Pick a vertical (Legal, Healthcare, Finance, E-commerce).
15. **Proof-of-Work:** Have documented impact (numbers and metrics), not just GitHub repos.
16. **Build in Public:** Share failures, myth-bust, run evals publicly. Priority: LinkedIn > X > YouTube.

**Infrastructure & Process Basics:**

* **Infrastructure:** Docker multi-stage builds, sidecar containers. AWS standard services (ECS, S3, RDS, SQS, IAM), Modal/Runpod for GPUs, Kubernetes basics, Feature flags (LaunchDarkly, Growthbook).
* **Quality Data:** 4 levels of testing strategy (Deterministic, Mocked LLM, Evals, Shadow mode). Data pipeline quality (pre-processing, chunk validation, deduplication).
* **Process:** Read/write Technical Design Documents (TDDs), strict AI code reviews, Incident response runbooks.

---

## 🗓️ Sprints & Timelines

**The 30-Day Edge Sprint:**

* Week 1: Add evals/LangSmith to an existing project.
* Week 2: Publish an MCP server.
* Week 3: Write a public failure post.
* Week 4: Build a tiny AI product for a niche and charge $1.

**The 90-Day Sprint Plan:**

* Month 1 (Foundation): APIs, Prompting, Optimization, Embeddings, Tool Calling.
* Month 2 (Build): Advanced RAG, Telegram memory, LangGraph agents, Voice/Realtime API.
* Month 3 (Ship): Evals/Tracing, Multi-tenant SaaS, Fine-tuning, Portfolio polish.

**6-Month Production Capability Plan:**

* Month 1-2: TS Strict mode, Structured logging, CI pipelines, PostgreSQL EXPLAIN ANALYZE.
* Month 3-4: Docker, BullMQ async pipelines, Redis mastery, Feature flags.
* Month 5-6: Eval datasets, TDDs, Observability dashboards, Runbooks.

---

## 🏆 Top Projects & Tech Stack Summaries

**Top 15 Portfolio Projects:**

1. AI Content Agency
2. Perplexity Clone
3. Multi-Tenant Chatbot Platform
4. Phone Receptionist Bot
5. Autonomous Business Agent
6. Custom Fine-tuned Model API
7. Voice AI Doctor Assistant
8. AI Research Agent
9. Legal Document Analyzer
10. Eval Suite Builder
11. AI Code Reviewer
12. RAG Quality Dashboard
13. AI Meeting Summarizer
14. Custom MCP Server
15. Personal AI OS

**The Stack of a Senior AI Engineer:**

* TypeScript (strict) + Node.js 20+ with Pino logging.
* LangGraph for stateful agents, raw APIs for simple tasks.
* Anthropic + OpenAI via LiteLLM proxy.
* pgvector + Pinecone, PostgreSQL, Redis, BullMQ / SQS.
* Langfuse + OpenTelemetry + Grafana.
* PromptFoo + RAGAS + LLM-as-judge.
* Growthbook / LaunchDarkly, Vercel/Railway/Modal, Docker, Doppler.

**Red Flags / What Not To Do:**

* Don't build basic tutorials ("Clones"). Solve real pain.
* Don't skip evals or tune on vibes.
* Don't hardcode prompts.
* Don't build 20 half-finished projects; 3 polished ones are better.
* Don't ignore security (OWASP LLM Top 10).

**Resources:**

* Channels: AI Jason, Sam Witteveen, Leon van Zyl, Pixegami, Nick Dobos, Matt Williams, Fireship, Traversy Media, NetworkChuck.
* Docs: Anthropic Guide, OpenAI Cookbook, LangChain/Vercel AI SDK docs, LangGraph conceptual guide.
* Papers: RAG paper, ReAct paper, HyDE paper, Lost in the Middle paper.
