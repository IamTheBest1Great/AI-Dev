# 📘 What Most Developers Miss — Production-Ready AI Practices

Beyond the basics of API calls, prompt engineering, and optimization, there are critical practices that separate experimental prototypes from production‑grade AI systems. This section covers five areas that many developers overlook until they cause real problems.

---

## 1.4.1 Dynamic System Prompts

A static system prompt written once and never changed is a red flag. In production, system prompts must adapt to user context, roles, access levels, and real‑time data.

### Injecting User Data at Runtime

Instead of hardcoding user‑specific information, inject it dynamically. This keeps the prompt template clean and the logic maintainable.

```javascript
// ❌ Bad: Hardcoded user info in prompt
const systemPrompt = "You are helping John, a premium user from the US. John likes short answers.";

// ✅ Good: Dynamic injection
function buildSystemPrompt(user) {
  return `
    You are an AI assistant for our platform.
    Current user:
    - Name: ${user.name}
    - Plan: ${user.plan}   // free, pro, enterprise
    - Locale: ${user.locale}
    - Preferences: ${JSON.stringify(user.preferences)}
    
    Adapt your response accordingly.
  `;
}
```

### Role‑Based Prompt Variation

Different user roles require different behavior. A support agent needs precise data access and audit logging; an end user needs friendly, non‑technical answers.

| Role | Prompt Focus |
|------|--------------|
| **End user** | Friendly, concise, avoid jargon |
| **Admin** | Detailed, include system status, allow destructive actions |
| **Support agent** | Show internal IDs, suggest next steps, log all actions |
| **Anonymous visitor** | Restrict data access, no persistence |

**Implementation pattern**:
```javascript
const rolePrompts = {
  admin: "You have full access to system data. Be precise and include metrics where relevant.",
  support: "You are assisting a customer support agent. Include internal ticket IDs and suggest troubleshooting steps.",
  user: "You are a helpful assistant. Keep answers clear and actionable."
};

const systemPrompt = `${rolePrompts[user.role]}\nUser data: ${JSON.stringify(userContext)}`;
```

### Tier‑Based Constraints

Different subscription tiers get different capabilities. Enforce these through the system prompt, not just in your business logic.

```javascript
// Free tier: no file uploads, shorter context, slower model
// Pro tier: file uploads, longer responses, standard model
// Enterprise: unlimited, custom fine‑tuned model, no content filtering

function getTierConstraints(tier) {
  switch(tier) {
    case 'free':
      return "You can only answer text‑based questions. Maximum response length is 150 words. Do not process any uploaded files.";
    case 'pro':
      return "You can process uploaded PDFs and images. Responses limited to 500 words.";
    case 'enterprise':
      return "Full capabilities. No length restrictions. Use internal knowledge base.";
  }
}
```

**Why this matters**: Tiers enforced only at the API routing level can be bypassed by clever prompt injection. Baking constraints into the system prompt adds a second layer of defense.

---

## 1.4.2 Output Validation Before Sending to User

Never trust the LLM’s output blindly. Malicious or simply incorrect outputs can damage user trust and violate compliance. Implement a validation layer that checks every response before it reaches the user.

### Hallucination Detection

For RAG applications, you can check if the generated answer is supported by the retrieved context.

```javascript
async function detectHallucination(answer, retrievedChunks) {
  // Simple: check for unsupported factual claims
  const prompt = `
    Context: ${retrievedChunks.join('\n')}
    Answer: ${answer}
    
    Does the answer contain any information not present in the context? 
    Respond with YES or NO and a brief explanation.
  `;
  const response = await cheapModel(prompt);
  return response.includes('YES');
}
```

More advanced: use a **cross‑encoder** or a **fact‑checking LLM** to compute an entailment score between each claim in the answer and the context.

### Factual Consistency Checks

For tasks like summarization or data extraction, compare the output against the source using:
- **ROUGE/BLEU** for overlap (basic)
- **LLM‑based entailment** (more accurate but slower)
- **NER overlap** – named entities should match source

### PII Detection

Never output personal identifiable information unless explicitly authorized. Use regex + a PII detection library.

```javascript
import { PII } from 'pii-detection'; // or Microsoft Presidio

function redactPII(text) {
  return text
    .replace(/\\b[\\w.-]+@[\\w.-]+\\.\\w+\\b/g, '[EMAIL]')
    .replace(/\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b/g, '[PHONE]')
    .replace(/\\b\\d{3}-\\d{2}-\\d{4}\\b/g, '[SSN]');
}

// Call after LLM generation
const safeOutput = redactPII(llmOutput);
```

### Toxicity Scoring

Use OpenAI’s Moderation API or a local classifier to score content for hate, harassment, violence, etc. Reject outputs that exceed a threshold.

```javascript
async function isToxic(text) {
  const moderation = await openai.moderations.create({ input: text });
  const results = moderation.results[0];
  // Flag if any category score > 0.7 (adjust threshold)
  return Object.values(results.category_scores).some(score => score > 0.7);
}
```

**Validation pipeline** (run in order of increasing cost):
1. PII redaction (cheap regex)
2. Toxicity check (moderation API)
3. Hallucination detection (Llama 3.2 or small classifier)
4. Factual consistency (only for critical outputs, using GPT‑4o)

---

## 1.4.3 Prompt Versioning from Day 1

Prompts evolve constantly. Without versioning, you lose the ability to roll back, debug performance regressions, or audit which prompt produced which output.

### Database Storage vs Hardcoded

| Storage Method | Pros | Cons |
|----------------|------|------|
| **Hardcoded** | Simple, no external dependency | No version history, cannot change without redeploy, A/B testing impossible |
| **Database** | Versioning, A/B testing, dynamic updates | Requires migration, adds latency |
| **Config file + Git** | Version control, easy diff | Still requires redeploy for changes |

**Recommendation**: Store prompts in a database with a version field, but **cache them** aggressively to avoid DB latency. For small teams, a JSON file committed to Git with a simple versioning scheme is a good start.

### Version Numbering

Use semantic versioning for prompts: `MAJOR.MINOR.PATCH`
- **MAJOR**: Incompatible change (new output format, removed capability)
- **MINOR**: New behavior that is backward compatible (better examples, additional instructions)
- **PATCH**: Fixes (typos, clarifications)

Store prompt versions in a table:

| prompt_id | version | system_prompt | user_prompt_template | created_at | is_active |
|-----------|---------|---------------|----------------------|------------|-----------|
| summarize | 1.0.0 | "You are a summarizer..." | "Summarize: {{text}}" | 2025-01-01 | false |
| summarize | 1.1.0 | "You are a concise summarizer..." | "Summarize in 2-3 sentences: {{text}}" | 2025-02-15 | true |

### Change History

Maintain a changelog for each prompt version:
```json
{
  "version": "1.1.0",
  "date": "2025-02-15",
  "author": "alice",
  "changes": [
    "Added 'concise' instruction to system prompt",
    "Changed user template to enforce 2‑3 sentences"
  ],
  "performance_delta": {
    "accuracy": "+5%",
    "cost_per_call": "-10%"
  }
}
```

This allows you to correlate prompt changes with production metrics.

---

## 1.4.4 A/B Testing Prompts

Even experienced prompt engineers cannot predict which wording works best. Run controlled experiments to compare prompt variants.

### Traffic Splitting

Direct a percentage of traffic to variant prompts. Use a deterministic hash of user ID or session ID to ensure consistency for a given user.

```javascript
function getPromptVariant(userId, experimentName) {
  const hash = crypto.createHash('md5').update(`${userId}:${experimentName}`).digest('hex');
  const bucket = parseInt(hash.slice(0, 8), 16) % 100;
  
  if (bucket < 10) return 'variant_A';
  if (bucket < 20) return 'variant_B';
  return 'control';
}
```

### Statistical Significance

Before declaring a winner, ensure you have enough samples and that the difference is not due to random chance.

| Metric | Minimum Sample Size (per variant) | Notes |
|--------|----------------------------------|-------|
| Accuracy (binary) | 1,000 | Enough to detect 5% difference at 95% confidence |
| User satisfaction (1‑5 scale) | 500 | Need at least 50 responses per point |
| Cost per query | 10,000 | Cost has low variance; smaller samples suffice |

Use a simple t‑test or chi‑square test (many online calculators available) to check significance.

### Quality Score Comparison

Define a quality score for each response. This can be:
- **Human rating** (gold standard but expensive) – asked to rate 1‑5.
- **LLM‑as‑judge** (using a powerful model like GPT‑5 to rate outputs).
- **Task‑specific metrics** (exact match, F1, entailment).

**LLM‑as‑judge prompt**:
```text
You are an impartial evaluator. Compare the two answers to the user query.
Query: {{query}}
Answer A: {{answerA}}
Answer B: {{answerB}}

Which answer is better? Choose A, B, or Tie. Explain briefly.
```

Then track win rates across variants.

---

## 1.4.5 Model Fallback Chains

Models fail. Rate limits, timeouts, server errors, and content filtering can all block a response. A robust system degrades gracefully through a chain of fallbacks.

### Primary → Secondary → Tertiary Chain

Define ordered fallback models. Each step should be cheaper or faster than the previous, or at least provide a working alternative.

| Tier | Model | Use Case |
|------|-------|----------|
| Primary | `gpt-5` | Highest quality, but expensive and may ratelimit |
| Secondary | `claude-sonnet-4-5` | Good quality, different provider, lower cost |
| Tertiary | `gpt-4o-mini` | Cheap, reliable, always available |
| Final | Rule‑based response | "I'm sorry, our AI is currently unavailable. Please try again later." |

### Trigger Conditions

Activate fallback on:

| Condition | HTTP Code / Behavior | Fallback Action |
|-----------|----------------------|------------------|
| Rate limit | 429 | Try next provider immediately |
| Timeout | Request > 30s without response | Cancel and fallback |
| Server error | 500, 502, 503 | Retry once, then fallback |
| Content filter | Response flagged as unsafe | Use stricter model or block entirely |
| Malformed output | JSON parse error | Retry with lower temperature, then fallback |

### Quality Threshold for Fallback Acceptance

Not all fallback responses are equal. If the secondary model produces a very low‑quality answer, you might want to try the tertiary or even a cached response.

```javascript
async function callWithFallback(query, models) {
  for (const model of models) {
    try {
      const response = await callModel(model, query);
      const quality = await assessQuality(response, query); // quick LLM scoring
      if (quality > 0.7) { // acceptance threshold
        return { response, model };
      }
      // else continue to next model
    } catch (err) {
      logError(err);
      continue;
    }
  }
  return { response: DEFAULT_RESPONSE, model: 'fallback_default' };
}
```

**Implementation tip**: Cache the fallback decision for a particular query pattern. If a query type consistently fails with the primary model, pre‑emptively route it to the secondary.

---

## Summary Table: What Most Developers Miss

| Area | Common Mistake | Production Practice |
|------|----------------|---------------------|
| Dynamic prompts | Hardcoded system prompts | Inject user data, role, tier at runtime |
| Output validation | Showing raw LLM output | Detect hallucination, redact PII, score toxicity |
| Prompt versioning | Editing prompts in code without history | Store versions in DB, maintain changelog |
| A/B testing | Shipping the first prompt that “works” | Split traffic, measure significance |
| Fallback chains | Letting a single API error crash the user experience | Chain models by priority, trigger on failure conditions |

---

## 📝 Hands‑On Exercise

1. **Dynamic system prompts** – Build a simple chat app that changes its system prompt based on a user’s plan (free vs paid). Free users get short answers, paid users get detailed explanations.
2. **Output validation** – Write a function that takes LLM output and redacts emails and phone numbers. Then, add a toxicity check using the OpenAI Moderation API.
3. **Prompt versioning** – Create a SQLite table to store prompt templates with version numbers. Write a function to retrieve the active version of a prompt.
4. **A/B test simulation** – Define two prompts for the same task. Simulate 200 requests with a random split, collect a synthetic quality score (random but with a bias), and determine if the difference is statistically significant.
5. **Fallback chain** – Implement a function that tries OpenAI, then Anthropic, then a local fallback response, with proper error handling and retries.

---

## 🔗 Next Steps

- **Observability** – Add telemetry to track which prompt versions are used and how often fallbacks are triggered.
- **Automated prompt optimization** – Use tools like DSPy to programmatically optimize prompts.
- **Cost‑aware routing** – Choose between models based on both estimated quality and real‑time cost (e.g., spot instances of models).

These “missing” practices are what separate toys from tools. Implement them from day one, and your AI system will be production‑ready, scalable, and maintainable.
