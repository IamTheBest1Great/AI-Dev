# 🔴 Phase 4 — Agents & Automation
> **Goal:** Build AI that takes actions in the real world, not just responds
> **Timeline:** Weeks 13–17
> **Outcome:** You can build production agents with LangGraph, browser automation, MCP servers, and data pipelines.

---

## 📚 Table of Contents

1. [The Agent Mental Model](#the-agent-mental-model)
2. [4.1 — Agent Fundamentals](#41--agent-fundamentals)
3. [4.2 — Frameworks](#42--frameworks)
4. [4.3 — Browser & Web Automation](#43--browser--web-automation)
5. [4.4 — Data Pipelines](#44--data-pipelines)
6. [4.5 — MCP (Model Context Protocol)](#45--mcp-model-context-protocol)
7. [4.6 — What Most Developers Miss](#46--what-most-developers-miss)
8. [Phase 4 Projects](#-phase-4-projects)
9. [Master Checklist](#-master-checklist)

---

## The Agent Mental Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                 CHATBOT vs AGENT — KEY DIFFERENCE                  │
│                                                                     │
│  CHATBOT:                                                           │
│  User: "What's the weather in Mumbai?"                             │
│  Bot: "I don't have real-time data." (just text response)          │
│                                                                     │
│  AGENT:                                                             │
│  User: "What's the weather in Mumbai?"                             │
│  Agent: [THINKS: I should call the weather tool]                   │
│         [CALLS: get_weather({ city: "Mumbai" })]                   │
│         [RECEIVES: { temp: 32, humidity: 80, condition: "sunny" }] │
│         [RESPONDS: "It's 32°C and sunny in Mumbai right now!"]     │
│                                                                     │
│  The difference: AGENCY = ability to DECIDE what actions to take  │
│  and then EXECUTE those actions in the real world                  │
│                                                                     │
│  Agents can: browse web, run code, send emails, read files,        │
│  book appointments, create tickets, query databases, etc.          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4.1 — Agent Fundamentals

### 4.1.1 — The ReAct Pattern (Reason + Act)

The foundational pattern for all agents. Build this from scratch at least once.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE REACT LOOP                                  │
│                                                                     │
│  User: "Research the top 3 AI companies and their latest products" │
│                                                                     │
│  ITERATION 1:                                                       │
│  THOUGHT: I need to search for top AI companies first              │
│  ACTION: search_web({ query: "top AI companies 2025" })           │
│  OBSERVATION: [OpenAI, Anthropic, Google DeepMind, Meta AI...]    │
│                                                                     │
│  ITERATION 2:                                                       │
│  THOUGHT: Found top 3: OpenAI, Anthropic, Google. Need their      │
│           latest products. Let me search each.                     │
│  ACTION: search_web({ query: "OpenAI latest products 2025" })     │
│  OBSERVATION: [GPT-4o, o3, Sora, Operator...]                     │
│                                                                     │
│  ITERATION 3:                                                       │
│  THOUGHT: Have OpenAI products. Need Anthropic's.                 │
│  ACTION: search_web({ query: "Anthropic latest products 2025" })  │
│  OBSERVATION: [Claude 3.5, Claude 4, Claude Code...]              │
│                                                                     │
│  ITERATION 4:                                                       │
│  THOUGHT: Have all data. Can now write the final report.          │
│  FINAL ANSWER: "Here are the top 3 AI companies..."               │
│                                                                     │
│  KEY: The agent DECIDES what to do next at each step              │
└─────────────────────────────────────────────────────────────────────┘
```

**Build ReAct from scratch:**

```typescript
async function reactAgent(
  goal: string,
  tools: Tool[],
  maxSteps = 15
): Promise<string> {
  const thoughts: string[] = [];
  const observations: string[] = [];
  const toolCalls: string[] = [];

  for (let step = 0; step < maxSteps; step++) {
    // Build the prompt with full history
    const prompt = `
You are an intelligent agent completing tasks using available tools.

GOAL: ${goal}

AVAILABLE TOOLS:
${tools.map((t) => `- ${t.name}: ${t.description}`).join("\n")}

PREVIOUS STEPS:
${thoughts
  .map(
    (thought, i) => `
Step ${i + 1}:
THOUGHT: ${thought}
ACTION: ${toolCalls[i] || "N/A"}
OBSERVATION: ${observations[i] || "N/A"}`
  )
  .join("\n")}

Now decide what to do next. Either:
1. Use a tool: respond with EXACTLY this format:
   THOUGHT: [your reasoning]
   ACTION: tool_name | {"param": "value"}

2. Finish: respond with EXACTLY this format:
   THOUGHT: [why you're done]
   FINAL ANSWER: [your complete answer]

Response:`;

    const response = await callLLM(prompt);

    // Parse FINAL ANSWER
    if (response.includes("FINAL ANSWER:")) {
      const finalAnswer = response.split("FINAL ANSWER:")[1].trim();
      return finalAnswer;
    }

    // Parse THOUGHT + ACTION
    const thoughtMatch = response.match(/THOUGHT:\s*(.+?)(?=ACTION:|$)/s);
    const actionMatch = response.match(/ACTION:\s*(\w+)\s*\|\s*(\{.+\})/s);

    if (!thoughtMatch || !actionMatch) {
      console.warn("Could not parse agent response, retrying...");
      continue;
    }

    const thought = thoughtMatch[1].trim();
    const toolName = actionMatch[1].trim();
    const toolArgs = JSON.parse(actionMatch[2]);

    thoughts.push(thought);
    toolCalls.push(`${toolName}(${JSON.stringify(toolArgs)})`);

    // Execute the tool
    try {
      const result = await executeTool(toolName, toolArgs);
      observations.push(JSON.stringify(result).slice(0, 2000)); // Limit size
    } catch (error: any) {
      observations.push(`ERROR: ${error.message}`);
    }
  }

  throw new Error(`Agent exceeded max steps (${maxSteps})`);
}
```

### 4.1.2 — Agent Memory Types

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT MEMORY TYPES                              │
│                                                                     │
│  1. IN-CONTEXT MEMORY (conversation history)                       │
│     └── Stored in message array                                    │
│     └── Limited by context window (8k-200k tokens)                │
│     └── Lost when conversation ends                                 │
│     └── Best for: within-session context                           │
│                                                                     │
│  2. EXTERNAL MEMORY (database)                                     │
│     └── Stored in MongoDB/PostgreSQL                               │
│     └── Unlimited storage                                          │
│     └── Persists across sessions                                   │
│     └── Best for: user facts, preferences, history                 │
│                                                                     │
│  3. Mem0 (semantic memory)                                         │
│     └── Extracts facts from conversations                          │
│     └── Retrieves relevant facts before each response              │
│     └── Handles forgetting, updating contradictions                │
│     └── Best for: personalized AI assistants                       │
│                                                                     │
│  4. KNOWLEDGE GRAPHS (relational memory)                           │
│     └── Entities and relationships in Neo4j                        │
│     └── Multi-hop queries ("who manages the team that owns X?")    │
│     └── Best for: complex domain knowledge                         │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
import { MemoryClient } from "mem0ai";

const memory = new MemoryClient({ apiKey: process.env.MEM0_API_KEY! });

// Add memories after each conversation
async function updateAgentMemory(
  userId: string,
  conversation: { role: string; content: string }[]
) {
  await memory.add(conversation, {
    userId,
    metadata: { timestamp: new Date().toISOString() },
  });
}

// Retrieve relevant memories before responding
async function getRelevantMemories(
  userId: string,
  currentQuery: string
): Promise<string> {
  const memories = await memory.search(currentQuery, {
    userId,
    limit: 5,
  });

  if (memories.results.length === 0) return "";

  return `
RELEVANT MEMORIES ABOUT THIS USER:
${memories.results.map((m: any) => `- ${m.memory}`).join("\n")}
`;
}

// Use memories in system prompt
async function buildAgentSystemPrompt(
  userId: string,
  currentQuery: string
): Promise<string> {
  const relevantMemories = await getRelevantMemories(userId, currentQuery);

  return `
You are a helpful AI assistant with memory of past interactions.

${relevantMemories}

Use the memories above to personalize your response.
If a memory is outdated or contradicted by the current conversation, prefer the current conversation.
  `.trim();
}
```

### 4.1.3 — Agent Failure Modes & Defenses

```typescript
class SafeAgent {
  private iterationCount = 0;
  private readonly maxIterations: number;
  private readonly tokenBudget: number;
  private tokensUsed = 0;
  private lastActions: string[] = [];

  constructor(config: { maxIterations: number; tokenBudget: number }) {
    this.maxIterations = config.maxIterations;
    this.tokenBudget = config.tokenBudget;
  }

  // FAILURE MODE 1: Infinite loops
  private detectLoop(currentAction: string): boolean {
    // Check if we're doing the same action repeatedly
    const recentActions = this.lastActions.slice(-3);
    const isLooping = recentActions.filter((a) => a === currentAction).length >= 2;

    if (isLooping) {
      console.warn(`Detected potential loop: ${currentAction} repeated 3+ times`);
      return true;
    }

    this.lastActions.push(currentAction);
    if (this.lastActions.length > 10) this.lastActions.shift();
    return false;
  }

  // FAILURE MODE 2: Token budget exceeded
  private checkTokenBudget(tokens: number): void {
    this.tokensUsed += tokens;
    if (this.tokensUsed > this.tokenBudget) {
      throw new Error(
        `Token budget exceeded: ${this.tokensUsed}/${this.tokenBudget}`
      );
    }
  }

  // FAILURE MODE 3: Hallucinated tool calls
  private validateToolCall(
    toolName: string,
    args: unknown,
    availableTools: Tool[]
  ): void {
    const tool = availableTools.find((t) => t.name === toolName);
    if (!tool) {
      throw new Error(
        `Tool "${toolName}" does not exist. Available: ${availableTools.map((t) => t.name).join(", ")}`
      );
    }

    // Validate args match schema
    const validation = validateAgainstSchema(args, tool.input_schema);
    if (!validation.valid) {
      throw new Error(`Invalid args for tool ${toolName}: ${validation.error}`);
    }
  }

  // FAILURE MODE 4: Stuck states (no progress)
  private detectStuckState(
    currentObservation: string,
    previousObservations: string[]
  ): boolean {
    if (previousObservations.length < 3) return false;

    // Check if last 3 observations are essentially the same (no progress)
    const recentObs = previousObservations.slice(-3);
    const similarity = recentObs.filter((obs) =>
      obs.includes(currentObservation.slice(0, 50))
    ).length;

    return similarity >= 2;
  }
}
```

### 4.1.4 — Human-in-the-Loop

```typescript
// Pattern: Pause agent execution and wait for human approval
class HumanInLoopAgent {
  private pendingApprovals = new Map<string, {
    action: string;
    args: unknown;
    resolve: (approved: boolean) => void;
  }>();

  async executeWithApproval(
    toolName: string,
    args: unknown,
    approvalRequired = false
  ): Promise<unknown> {
    // High-stakes actions that need human approval
    const HIGH_STAKES_TOOLS = [
      "send_email",
      "create_payment",
      "delete_data",
      "post_to_social",
      "book_appointment",
    ];

    const needsApproval =
      approvalRequired || HIGH_STAKES_TOOLS.includes(toolName);

    if (needsApproval) {
      const approved = await this.requestApproval(toolName, args);

      if (!approved) {
        return {
          status: "rejected",
          message: "Human rejected this action",
        };
      }
    }

    return await executeTool(toolName, args);
  }

  private async requestApproval(
    toolName: string,
    args: unknown
  ): Promise<boolean> {
    const approvalId = generateId();

    // Send to human review interface
    await notifyHuman({
      approvalId,
      toolName,
      args,
      message: `Agent wants to execute: ${toolName}(${JSON.stringify(args)})`,
    });

    // Wait for human decision (timeout after 5 minutes)
    return new Promise((resolve) => {
      this.pendingApprovals.set(approvalId, {
        action: toolName,
        args,
        resolve,
      });

      // Timeout
      setTimeout(() => {
        if (this.pendingApprovals.has(approvalId)) {
          this.pendingApprovals.delete(approvalId);
          resolve(false); // Reject if no response
        }
      }, 5 * 60 * 1000);
    });
  }

  // Called when human approves/rejects
  respondToApproval(approvalId: string, approved: boolean) {
    const pending = this.pendingApprovals.get(approvalId);
    if (pending) {
      pending.resolve(approved);
      this.pendingApprovals.delete(approvalId);
    }
  }
}
```

---

## 4.2 — Frameworks

### 4.2.1 — LangGraph

LangGraph is for stateful agent workflows. This is the most important framework in the roadmap.

```
┌─────────────────────────────────────────────────────────────────────┐
│              LANGGRAPH CONCEPTS                                    │
│                                                                     │
│  NODES: Functions that transform state                             │
│  EDGES: Connections between nodes (can be conditional)            │
│  STATE: Shared data that persists across nodes                     │
│                                                                     │
│  Example: Research Agent Graph                                     │
│                                                                     │
│  [START]                                                            │
│     │                                                               │
│     ▼                                                               │
│  [plan_research]  ← Creates research plan                         │
│     │                                                               │
│     ▼                                                               │
│  [search_web]     ← Searches for info (can loop back!)            │
│     │                                                               │
│     ├── "more_needed" → back to [search_web]                      │
│     │                                                               │
│     └── "enough_data" → [synthesize]                              │
│                            │                                        │
│                            ▼                                        │
│                         [review]                                    │
│                            │                                        │
│                  ┌─────────┴─────────┐                             │
│              "approve"           "revise"                          │
│                  │                   │                              │
│                [END]          back to [synthesize]                 │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
import { StateGraph, END, START } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";

// 1. DEFINE STATE using Annotation
const ResearchState = Annotation.Root({
  query: Annotation<string>(),
  plan: Annotation<string[]>({
    reducer: (prev, curr) => curr ?? prev ?? [],
    default: () => [],
  }),
  searchResults: Annotation<string[]>({
    reducer: (prev, curr) => [...(prev ?? []), ...(curr ?? [])], // Append, not replace
    default: () => [],
  }),
  draftReport: Annotation<string>({
    default: () => "",
  }),
  finalReport: Annotation<string>({
    default: () => "",
  }),
  iterationCount: Annotation<number>({
    reducer: (prev, curr) => (curr ?? 0) + (prev ?? 0),
    default: () => 0,
  }),
  needsRevision: Annotation<boolean>({
    default: () => false,
  }),
});

type ResearchStateType = typeof ResearchState.State;

// 2. DEFINE NODES

async function planResearch(
  state: ResearchStateType
): Promise<Partial<ResearchStateType>> {
  const plan = await callLLM(`
Create a research plan for: "${state.query}"
Return a JSON array of 3-5 specific search queries.
["query1", "query2", "query3"]`);

  return { plan: JSON.parse(plan) };
}

async function searchWeb(
  state: ResearchStateType
): Promise<Partial<ResearchStateType>> {
  // Take the next unprocessed plan item
  const processedCount = state.searchResults.length;
  const nextQuery = state.plan[processedCount];

  if (!nextQuery) return { iterationCount: 1 };

  const results = await tavilySearch(nextQuery);
  const resultText = results.map((r: any) => `${r.title}: ${r.content}`).join("\n\n");

  return {
    searchResults: [resultText], // This APPENDS due to our reducer
    iterationCount: 1,
  };
}

async function synthesizeReport(
  state: ResearchStateType
): Promise<Partial<ResearchStateType>> {
  const draft = await callLLM(`
Based on these research findings, write a comprehensive report about: "${state.query}"

Research Findings:
${state.searchResults.join("\n\n---\n\n")}

Write a well-structured report with:
1. Executive Summary
2. Key Findings (3-5 points)
3. Details & Analysis
4. Conclusion`);

  return { draftReport: draft };
}

async function reviewReport(
  state: ResearchStateType
): Promise<Partial<ResearchStateType>> {
  const review = await callLLM(`
Review this research report for quality. 
Respond with EXACTLY one of:
- "APPROVED" if the report is complete and accurate
- "NEEDS_REVISION: [specific issues]" if it needs work

Report:
${state.draftReport}`);

  if (review.startsWith("APPROVED")) {
    return { finalReport: state.draftReport, needsRevision: false };
  } else {
    return { needsRevision: true };
  }
}

// 3. DEFINE ROUTING FUNCTIONS

function shouldContinueSearching(state: ResearchStateType): string {
  const hasMorePlanItems = state.searchResults.length < state.plan.length;
  const withinLimit = state.iterationCount < 10;

  if (hasMorePlanItems && withinLimit) return "search";
  return "synthesize";
}

function shouldRevise(state: ResearchStateType): string {
  if (state.needsRevision && state.iterationCount < 3) return "revise";
  return "end";
}

// 4. BUILD THE GRAPH
const workflow = new StateGraph(ResearchState)
  .addNode("plan", planResearch)
  .addNode("search", searchWeb)
  .addNode("synthesize", synthesizeReport)
  .addNode("review", reviewReport)

  // Edges (connections)
  .addEdge(START, "plan")
  .addEdge("plan", "search")

  // Conditional edge: search → more searching OR synthesize
  .addConditionalEdges("search", shouldContinueSearching, {
    search: "search",       // Loop back to search
    synthesize: "synthesize",
  })

  .addEdge("synthesize", "review")

  // Conditional edge: review → revise OR end
  .addConditionalEdges("review", shouldRevise, {
    revise: "synthesize",  // Go back to synthesize with same data
    end: END,
  });

// Compile the graph
const researchAgent = workflow.compile();

// RUN IT
const result = await researchAgent.invoke({
  query: "What are the implications of GPT-5 for software development?",
});

console.log(result.finalReport);
```

**LangGraph with persistence (production):**

```typescript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

// Use PostgreSQL to save checkpoints (survives server restarts!)
const checkpointer = new PostgresSaver({
  connectionString: process.env.DATABASE_URL!,
});

const researchAgent = workflow.compile({ checkpointer });

// Thread IDs allow multiple concurrent conversations
const config = { configurable: { thread_id: `research-${userId}-${Date.now()}` } };

// Start the research
const result = await researchAgent.invoke(
  { query: "Latest developments in quantum computing" },
  config
);

// Later, resume an interrupted task:
const resumed = await researchAgent.invoke(null, config); // null = resume from checkpoint
```

**Parallel branches in LangGraph:**

```typescript
// Run multiple agents in parallel, merge results
import { Send } from "@langchain/langgraph";

// Fan-out: create parallel research tasks
function createParallelTasks(
  state: ResearchStateType
): Send[] {
  // Returns an array of Send objects — each runs in parallel!
  return state.plan.map((searchQuery, i) =>
    new Send("search", { ...state, currentQuery: searchQuery, taskIndex: i })
  );
}

workflow
  .addEdge("plan", "distribute")
  .addConditionalEdges("distribute", createParallelTasks) // PARALLEL EXECUTION
  .addEdge("search", "aggregate") // All parallel tasks converge here
  .addEdge("aggregate", "synthesize");
```

### 4.2.2 — OpenAI Assistants API

```typescript
// OpenAI manages conversation history (threads) for you
const assistant = await openai.beta.assistants.create({
  name: "Research Assistant",
  instructions: `
You are a research assistant. Use file_search to find information in uploaded files.
Always cite your sources with [source] notation.
  `,
  model: "gpt-4o",
  tools: [
    { type: "file_search" },     // Built-in RAG over uploaded files
    { type: "code_interpreter" }, // Python code execution
  ],
});

// Create a thread (conversation)
const thread = await openai.beta.threads.create();

// Add a message
await openai.beta.threads.messages.create(thread.id, {
  role: "user",
  content: "What does the uploaded financial report say about Q3 revenue?",
});

// Run the assistant
const run = await openai.beta.threads.runs.createAndPoll(thread.id, {
  assistant_id: assistant.id,
});

// Get the response
const messages = await openai.beta.threads.messages.list(thread.id);
const answer = messages.data[0].content[0];
if (answer.type === "text") {
  console.log(answer.text.value);
}

// Upload a file for the assistant to analyze
const file = await openai.files.create({
  file: fs.createReadStream("financial_report.pdf"),
  purpose: "assistants",
});

// Attach file to the thread
await openai.beta.threads.messages.create(thread.id, {
  role: "user",
  content: "Analyze this report",
  attachments: [{ file_id: file.id, tools: [{ type: "file_search" }] }],
});
```

---

## 4.3 — Browser & Web Automation

### 4.3.1 — Playwright for Web Automation

```typescript
import { chromium, Browser, Page } from "playwright";

// Reusable browser manager
class BrowserManager {
  private browser: Browser | null = null;

  async getBrowser(): Promise<Browser> {
    if (!this.browser) {
      this.browser = await chromium.launch({
        headless: true,              // No visible window
        args: [
          "--no-sandbox",
          "--disable-setuid-sandbox",
          "--disable-dev-shm-usage",
          "--disable-accelerated-2d-canvas",
          "--no-first-run",
          "--no-zygote",
        ],
      });
    }
    return this.browser;
  }

  async getPage(): Promise<Page> {
    const browser = await this.getBrowser();
    const context = await browser.newContext({
      userAgent:
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      viewport: { width: 1280, height: 720 },
    });
    const page = await context.newPage();
    page.setDefaultTimeout(30000);
    return page;
  }

  async close() {
    await this.browser?.close();
    this.browser = null;
  }
}

const browserManager = new BrowserManager();

// Scrape a page and extract structured data with AI
async function scrapeAndExtract<T>(
  url: string,
  schema: z.ZodType<T>,
  instructions: string
): Promise<T> {
  const page = await browserManager.getPage();

  try {
    await page.goto(url, { waitUntil: "networkidle" });

    // Extract readable text (remove scripts, styles, etc.)
    const text = await page.evaluate(() => {
      // Remove noise
      const elements = document.querySelectorAll(
        "script, style, nav, footer, header, iframe"
      );
      elements.forEach((el) => el.remove());
      return document.body.innerText;
    });

    // Take screenshot for visual verification
    const screenshot = await page.screenshot({ type: "jpeg", quality: 50 });

    // Extract structured data with LLM
    return await extractWithRetry(
      `${instructions}\n\nPage content:\n${text.slice(0, 10000)}`,
      schema
    );
  } finally {
    await page.close();
  }
}

// Fill and submit a form (for job applications, signups, etc.)
async function fillAndSubmitForm(
  url: string,
  formData: Record<string, string>
): Promise<boolean> {
  const page = await browserManager.getPage();

  try {
    await page.goto(url, { waitUntil: "networkidle" });

    // Fill each form field
    for (const [fieldId, value] of Object.entries(formData)) {
      // Try different selectors
      const selectors = [
        `#${fieldId}`,
        `[name="${fieldId}"]`,
        `[placeholder*="${fieldId}" i]`,
        `label:has-text("${fieldId}") + input`,
      ];

      for (const selector of selectors) {
        try {
          await page.fill(selector, value);
          break;
        } catch {
          continue;
        }
      }
    }

    // Submit the form
    await page.keyboard.press("Enter");
    // OR: await page.click('[type="submit"]');

    // Wait for confirmation
    await page.waitForLoadState("networkidle");

    // Check for success indicators
    const pageText = await page.textContent("body");
    return (
      pageText?.includes("success") ||
      pageText?.includes("submitted") ||
      pageText?.includes("thank you") ||
      false
    );
  } finally {
    await page.close();
  }
}
```

### 4.3.2 — Stagehand — AI-Native Browser Automation

```typescript
import { Stagehand } from "@browserbasehq/stagehand";

// Stagehand lets you describe actions in natural language!
const stagehand = new Stagehand({
  env: "LOCAL", // or "BROWSERBASE" for cloud
  verbose: true,
});

await stagehand.init();
const page = stagehand.page;

// Natural language actions — no CSS selectors needed!
await page.goto("https://app.example.com/login");
await stagehand.act("Click the login button");
await stagehand.act("Fill in the email field with user@example.com");
await stagehand.act("Fill in the password field with mypassword");
await stagehand.act("Click Submit");

// Extract structured data with natural language
const products = await stagehand.extract({
  instruction: "Extract all product names and prices from this page",
  schema: z.array(
    z.object({
      name: z.string(),
      price: z.number(),
      inStock: z.boolean(),
    })
  ),
});

// Observe — understand what's on the page
const actions = await stagehand.observe();
console.log(actions); // ["click login button", "fill email field", ...]

await stagehand.close();
```

### 4.3.3 — Firecrawl for Clean Content Extraction

```typescript
import FirecrawlApp from "@mendable/firecrawl-js";

const firecrawl = new FirecrawlApp({ apiKey: process.env.FIRECRAWL_API_KEY! });

// Scrape a single page — clean markdown output
async function scrapeClean(url: string): Promise<string> {
  const result = await firecrawl.scrapeUrl(url, {
    formats: ["markdown"],
    onlyMainContent: true,  // Remove navigation, ads, etc.
    excludeTags: ["footer", "header", "nav", "aside"],
  });

  return result.markdown || "";
}

// Crawl an entire site
async function crawlSite(url: string, maxPages = 50): Promise<string[]> {
  const result = await firecrawl.crawlUrl(url, {
    limit: maxPages,
    scrapeOptions: {
      formats: ["markdown"],
      onlyMainContent: true,
    },
  });

  return result.data?.map((page: any) => page.markdown) || [];
}

// Alternative: Jina Reader (free!)
async function jinaRead(url: string): Promise<string> {
  const response = await fetch(`https://r.jina.ai/${url}`, {
    headers: {
      Authorization: `Bearer ${process.env.JINA_API_KEY}`,
      Accept: "text/markdown",
    },
  });
  return response.text();
}
```

### 4.3.4 — AI-Optimized Search APIs

```typescript
import Tavily from "@tavily/core";
import Exa from "exa-js";

const tavily = new Tavily({ apiKey: process.env.TAVILY_API_KEY! });
const exa = new Exa(process.env.EXA_API_KEY!);

// Tavily — great for general research, returns full page content
async function searchWithTavily(query: string): Promise<SearchResult[]> {
  const response = await tavily.search(query, {
    searchDepth: "advanced",  // More thorough
    maxResults: 5,
    includeAnswer: true,      // Gets AI-synthesized answer too
    includeRawContent: true,  // Full page content
    includeImages: false,
  });

  return response.results.map((r) => ({
    title: r.title,
    url: r.url,
    content: r.content,
    score: r.score,
  }));
}

// Exa — great for finding specific content types
async function searchWithExa(query: string): Promise<SearchResult[]> {
  const response = await exa.searchAndContents(query, {
    numResults: 5,
    text: true,               // Get full text content
    highlights: true,         // Get highlighted relevant snippets
    useAutoprompt: true,      // Exa rewrites query for better results
    type: "neural",           // Semantic search (vs keyword)
  });

  return response.results.map((r) => ({
    title: r.title,
    url: r.url,
    content: r.text || "",
    highlights: r.highlights,
  }));
}

// Agent-ready search tool
const searchTool: Tool = {
  name: "search_web",
  description: `
Search the web for current information.
Use this when you need:
- Current news or recent events
- Factual information you're not certain about
- Specific data, statistics, or examples
- Information about companies, people, or products

Returns: title, URL, and relevant content from top results.`,
  input_schema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description:
          "Specific search query. Be precise. Include relevant context.",
      },
      recency: {
        type: "string",
        enum: ["any", "day", "week", "month"],
        description: "How recent the results should be. Default: 'any'",
        default: "any",
      },
    },
    required: ["query"],
  },
  execute: async ({ query, recency }) => {
    const results = await searchWithTavily(query);
    return results.slice(0, 3); // Return top 3
  },
};
```

---

## 4.4 — Data Pipelines

### 4.4.1 — Web Scraping + AI Cleaning

```typescript
// Complete data extraction pipeline
async function buildDataPipeline(
  urls: string[],
  outputSchema: z.ZodType,
  extractionInstructions: string
) {
  const results = [];

  for (const url of urls) {
    try {
      // Step 1: Fetch clean content
      const content = await scrapeClean(url);

      // Step 2: AI extraction
      const extracted = await extractWithRetry(
        `${extractionInstructions}\n\nContent:\n${content}`,
        outputSchema
      );

      // Step 3: Validate
      const validated = outputSchema.safeParse(extracted);
      if (!validated.success) {
        console.error(`Validation failed for ${url}:`, validated.error.issues);
        continue;
      }

      results.push({ url, data: validated.data, extractedAt: new Date() });
    } catch (error: any) {
      console.error(`Failed for ${url}:`, error.message);
    }
  }

  return results;
}

// Competitor monitoring pipeline
const CompetitorDataSchema = z.object({
  companyName: z.string(),
  productName: z.string(),
  currentPrice: z.number().nullable(),
  keyFeatures: z.array(z.string()),
  recentChanges: z.string().nullable(),
  lastUpdated: z.string(),
});

// Run daily
const competitorData = await buildDataPipeline(
  ["https://competitor1.com/pricing", "https://competitor2.com/features"],
  CompetitorDataSchema,
  "Extract pricing and feature information from this competitor's page."
);
```

### 4.4.2 — n8n Workflow Automation

n8n is a self-hosted workflow automation tool. Think Zapier but with code nodes and full control.

```
┌─────────────────────────────────────────────────────────────────────┐
│              n8n WORKFLOW EXAMPLE: Lead Qualification              │
│                                                                     │
│  [Webhook Trigger]  ← New form submission                          │
│       │                                                             │
│       ▼                                                             │
│  [HTTP Request]  ← Enrich lead with Clearbit                       │
│       │                                                             │
│       ▼                                                             │
│  [OpenAI Node]   ← Score lead 1-10 with GPT                        │
│       │                                                             │
│       ├── score >= 8  → [Slack Node] "Hot lead!" + [CRM Node]      │
│       │                                                             │
│       └── score < 8   → [Email Node] "Thanks for interest"         │
└─────────────────────────────────────────────────────────────────────┘
```

```javascript
// n8n code node example (runs in n8n's built-in code editor)
const items = $input.all();

for (const item of items) {
  const leadData = item.json;
  
  // Call OpenAI to score the lead
  const response = await $http.request({
    method: "POST",
    url: "https://api.openai.com/v1/chat/completions",
    headers: {
      Authorization: `Bearer ${$env.OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: {
      model: "gpt-4o-mini",
      messages: [{
        role: "user",
        content: `
Score this sales lead 1-10 for fit with our B2B SaaS product.
Return ONLY a number.

Lead:
Company: ${leadData.company}
Title: ${leadData.jobTitle}
Email: ${leadData.email}
Message: ${leadData.message}

Score:`,
      }],
      max_tokens: 5,
    },
  });
  
  item.json.aiScore = parseInt(response.choices[0].message.content);
}

return items;
```

### 4.4.3 — Self-Healing Pipelines

```typescript
class SelfHealingPipeline {
  async run<T>(
    input: unknown,
    stages: PipelineStage<T>[],
    options: { maxRetries: number; fallbackValue?: T }
  ): Promise<T> {
    let currentInput = input;

    for (const stage of stages) {
      let lastError: Error | null = null;

      for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
        try {
          currentInput = await stage.execute(currentInput, {
            attempt,
            previousError: lastError?.message,
          });
          break; // Success, move to next stage
        } catch (error: any) {
          lastError = error;
          console.warn(
            `Stage "${stage.name}" failed (attempt ${attempt + 1}):`,
            error.message
          );

          if (attempt < options.maxRetries) {
            // Try alternative strategy
            if (stage.fallback) {
              try {
                currentInput = await stage.fallback(currentInput);
                break;
              } catch {
                // Fallback also failed, retry main
              }
            }
            await sleep(1000 * attempt); // Backoff
          }
        }
      }

      if (lastError && !currentInput) {
        if (options.fallbackValue !== undefined) {
          return options.fallbackValue;
        }
        throw new Error(`Pipeline failed at stage "${stage.name}": ${lastError.message}`);
      }
    }

    return currentInput as T;
  }
}

// Example: Competitor price scraping pipeline
const pricePipeline = new SelfHealingPipeline();

const price = await pricePipeline.run(
  { url: "https://competitor.com/pricing" },
  [
    {
      name: "fetch-page",
      execute: async ({ url }) => ({ content: await scrapeClean(url), url }),
      fallback: async ({ url }) => ({
        content: await jinaRead(url), // Alternative scraper
        url,
      }),
    },
    {
      name: "extract-price",
      execute: async ({ content }) => {
        const data = await extractWithRetry(
          `Extract the current pricing for the "Pro" plan. Return JSON: { "price": number, "currency": string, "period": "monthly"|"annual" }`,
          PriceSchema,
          content
        );
        return data;
      },
      fallback: async ({ content }) => {
        // Try different extraction prompt
        return await extractWithRetry(
          `What does this page charge for their main plan? JSON: { "price": number, "currency": string, "period": string }`,
          PriceSchema,
          content
        );
      },
    },
  ],
  { maxRetries: 2 }
);
```

### 4.4.4 — Change Detection

```typescript
import { createHash } from "crypto";
import { diffWords } from "diff";

class ChangeDetector {
  async detectChanges(
    source: string,
    currentContent: string,
    options: {
      checksum: boolean;
      semanticDiff: boolean;
      threshold: number; // 0-1, percentage of change to report
    }
  ): Promise<ChangeReport | null> {
    // Get previous snapshot
    const previous = await db.snapshots.findFirst({
      where: { source },
      orderBy: { capturedAt: "desc" },
    });

    if (!previous) {
      // First snapshot — just save it
      await db.snapshots.create({
        data: {
          source,
          content: currentContent,
          checksum: md5(currentContent),
          capturedAt: new Date(),
        },
      });
      return null;
    }

    // Quick checksum comparison
    if (options.checksum) {
      const currentChecksum = md5(currentContent);
      if (currentChecksum === previous.checksum) return null; // No change
    }

    // Detailed diff
    const wordDiff = diffWords(previous.content, currentContent);

    const addedText = wordDiff
      .filter((d) => d.added)
      .map((d) => d.value)
      .join(" ");

    const removedText = wordDiff
      .filter((d) => d.removed)
      .map((d) => d.value)
      .join(" ");

    const changeRatio =
      (addedText.length + removedText.length) /
      (previous.content.length + currentContent.length);

    // Only report if change exceeds threshold
    if (changeRatio < options.threshold) return null;

    // Use LLM to summarize what changed (semantic diff)
    let semanticSummary = "";
    if (options.semanticDiff && (addedText || removedText)) {
      semanticSummary = await callLLM(`
Summarize what changed between two versions of this content.
Be specific about what was added and removed.

Added: ${addedText.slice(0, 2000)}
Removed: ${removedText.slice(0, 2000)}

Concise summary of changes:`);
    }

    // Save new snapshot
    await db.snapshots.create({
      data: {
        source,
        content: currentContent,
        checksum: md5(currentContent),
        capturedAt: new Date(),
        changeFromPrevious: semanticSummary,
      },
    });

    return {
      source,
      changeRatio,
      addedText,
      removedText,
      semanticSummary,
      detectedAt: new Date(),
    };
  }
}
```

---

## 4.5 — MCP (Model Context Protocol)

### 4.5.1 — MCP Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MCP ARCHITECTURE                                │
│                                                                     │
│  HOST                         SERVER                               │
│  (Claude Desktop /            (Your custom server)                 │
│   Claude Code /                                                     │
│   Your App)                                                         │
│                                                                     │
│  ┌─────────────────────┐     ┌──────────────────────────────────┐ │
│  │ Claude AI           │     │ Your MCP Server                   │ │
│  │                     │     │                                   │ │
│  │ ← User asks question│     │ Tools:                            │ │
│  │                     │◄───►│  - search_docs()                  │ │
│  │ → Decides to call   │     │  - create_jira_ticket()           │ │
│  │   MCP tool          │     │  - query_database()               │ │
│  │                     │     │                                   │ │
│  │ ← Receives result   │     │ Resources:                        │ │
│  │                     │     │  - db://customers                 │ │
│  │ → Answers user      │     │  - file://reports/                │ │
│  └─────────────────────┘     └──────────────────────────────────┘ │
│                                                                     │
│  Protocol: JSON-RPC over stdio (Claude Desktop)                    │
│            or HTTP/SSE (web apps)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.5.2 — Building a Complete MCP Server

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";

// Create the server
const server = new Server(
  {
    name: "company-knowledge-base",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},      // We support tool calling
      resources: {},  // We support resource reading
    },
  }
);

// Define available tools
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "search_documentation",
      description:
        "Search company documentation, policies, and knowledge base. " +
        "Use this for questions about internal processes, products, or policies.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Natural language search query",
          },
          category: {
            type: "string",
            enum: ["engineering", "hr", "finance", "legal", "all"],
            description: "Filter by document category",
            default: "all",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "create_jira_ticket",
      description:
        "Create a new Jira ticket for bug reports or feature requests. " +
        "Use when the user reports a bug or requests a new feature.",
      inputSchema: {
        type: "object",
        properties: {
          title: { type: "string", description: "Short ticket title" },
          description: { type: "string", description: "Detailed description" },
          type: {
            type: "string",
            enum: ["bug", "feature", "task", "improvement"],
          },
          priority: {
            type: "string",
            enum: ["low", "medium", "high", "critical"],
            default: "medium",
          },
          labels: {
            type: "array",
            items: { type: "string" },
            description: "Optional labels",
          },
        },
        required: ["title", "description", "type"],
      },
    },
    {
      name: "query_analytics",
      description:
        "Query the analytics database for metrics and reports. " +
        "Use for questions about user numbers, revenue, or usage statistics.",
      inputSchema: {
        type: "object",
        properties: {
          metric: {
            type: "string",
            enum: ["dau", "mau", "revenue", "churn", "signups"],
            description: "The metric to query",
          },
          dateRange: {
            type: "string",
            enum: ["today", "this_week", "this_month", "last_30_days", "this_year"],
            default: "this_month",
          },
        },
        required: ["metric"],
      },
    },
  ],
}));

// Execute tools when Claude calls them
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "search_documentation": {
        const { query, category = "all" } = args as {
          query: string;
          category?: string;
        };

        const results = await searchDocumentation(query, category);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                {
                  results: results.map((r) => ({
                    title: r.title,
                    content: r.content.slice(0, 500), // Truncate for context
                    url: r.url,
                    relevanceScore: r.score,
                  })),
                  totalFound: results.length,
                },
                null,
                2
              ),
            },
          ],
        };
      }

      case "create_jira_ticket": {
        const { title, description, type, priority, labels } = args as {
          title: string;
          description: string;
          type: string;
          priority?: string;
          labels?: string[];
        };

        const ticket = await jiraClient.createIssue({
          project: { key: "ENG" },
          summary: title,
          description,
          issuetype: { name: type },
          priority: { name: priority || "Medium" },
          labels,
        });

        return {
          content: [
            {
              type: "text",
              text: `Ticket created successfully!\nKey: ${ticket.key}\nURL: ${ticket.self}\nEstimated review: ${estimateReviewTime(type, priority)}`,
            },
          ],
        };
      }

      case "query_analytics": {
        const { metric, dateRange } = args as {
          metric: string;
          dateRange?: string;
        };

        const data = await analyticsDB.query(metric, dateRange || "this_month");

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(data, null, 2),
            },
          ],
        };
      }

      default:
        throw new McpError(ErrorCode.MethodNotFound, `Tool not found: ${name}`);
    }
  } catch (error: any) {
    if (error instanceof McpError) throw error;
    throw new McpError(ErrorCode.InternalError, `Tool execution failed: ${error.message}`);
  }
});

// Define available resources (data sources Claude can read)
server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: [
    {
      uri: "db://customers/recent",
      name: "Recent Customers",
      description: "List of customers who signed up in the last 30 days",
      mimeType: "application/json",
    },
    {
      uri: "file://runbooks/",
      name: "Operations Runbooks",
      description: "Operational runbooks and incident response guides",
      mimeType: "text/markdown",
    },
  ],
}));

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  if (uri === "db://customers/recent") {
    const customers = await db.customers.findMany({
      where: {
        createdAt: { gte: new Date(Date.now() - 30 * 86400000) },
      },
      select: { id: true, name: true, email: true, plan: true, createdAt: true },
      take: 100,
    });

    return {
      contents: [
        {
          uri,
          mimeType: "application/json",
          text: JSON.stringify(customers, null, 2),
        },
      ],
    };
  }

  throw new McpError(ErrorCode.InvalidRequest, `Unknown resource: ${uri}`);
});

// Start the server (stdio transport for Claude Desktop)
const transport = new StdioServerTransport();
await server.connect(transport);

// Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):
// {
//   "mcpServers": {
//     "company-kb": {
//       "command": "node",
//       "args": ["/path/to/your/mcp-server.js"],
//       "env": {
//         "DATABASE_URL": "...",
//         "JIRA_API_KEY": "..."
//       }
//     }
//   }
// }
```

### 4.5.3 — Publishing MCP Servers

```bash
# Package as an npm package for easy installation
# package.json
{
  "name": "@yourname/mcp-notion",
  "version": "1.0.0",
  "description": "MCP server for Notion - gives Claude access to your workspace",
  "main": "dist/index.js",
  "bin": {
    "mcp-notion": "dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js"
  }
}

# Users can run it with:
npx @yourname/mcp-notion

# Or in Claude Desktop config:
# {
#   "mcpServers": {
#     "notion": {
#       "command": "npx",
#       "args": ["-y", "@yourname/mcp-notion"],
#       "env": { "NOTION_API_KEY": "..." }
#     }
#   }
# }
```

---

## 4.6 — What Most Developers Miss

### 4.6.1 — Timeouts and Circuit Breakers for Agents

```typescript
// Agents without limits can run for hours and cost hundreds of dollars!

class BoundedAgent {
  private readonly config: {
    maxIterations: number;
    maxTimeMs: number;
    maxCostUsd: number;
    maxTokens: number;
  };

  async run(task: string): Promise<AgentResult> {
    const startTime = Date.now();
    let totalCost = 0;
    let totalTokens = 0;

    // CIRCUIT BREAKER: Stop if any limit is hit
    const checkLimits = () => {
      if (Date.now() - startTime > this.config.maxTimeMs) {
        throw new Error(
          `Time limit exceeded: ${this.config.maxTimeMs}ms`
        );
      }
      if (totalCost > this.config.maxCostUsd) {
        throw new Error(
          `Cost limit exceeded: $${totalCost.toFixed(4)} > $${this.config.maxCostUsd}`
        );
      }
      if (totalTokens > this.config.maxTokens) {
        throw new Error(
          `Token limit exceeded: ${totalTokens} > ${this.config.maxTokens}`
        );
      }
    };

    for (let iteration = 0; iteration < this.config.maxIterations; iteration++) {
      checkLimits();

      const response = await this.callLLM(task);
      totalCost += response.cost;
      totalTokens += response.tokens;

      if (response.isDone) {
        return { result: response.answer, cost: totalCost, tokens: totalTokens };
      }

      await this.executeTools(response.toolCalls);
    }

    throw new Error(`Max iterations exceeded (${this.config.maxIterations})`);
  }
}
```

### 4.6.2 — Langfuse Tracing for Agents

```typescript
import Langfuse from "langfuse";

const langfuse = new Langfuse({
  secretKey: process.env.LANGFUSE_SECRET_KEY!,
  publicKey: process.env.LANGFUSE_PUBLIC_KEY!,
});

async function runTracedAgent(task: string, userId: string): Promise<string> {
  // Create a trace for this entire agent run
  const trace = langfuse.trace({
    name: "research-agent",
    userId,
    input: task,
    metadata: { agentVersion: "v2.1", model: "claude-sonnet-4-20250514" },
  });

  let iteration = 0;

  try {
    const result = await runAgentWithTracking(task, {
      onIteration: async (thought, toolName, toolArgs, observation) => {
        iteration++;

        // Create a span for each iteration
        const span = trace.span({
          name: `iteration-${iteration}`,
          input: { thought, toolName, toolArgs },
        });

        span.end({
          output: { observation: observation.slice(0, 500) },
          metadata: { iteration },
        });
      },

      onLLMCall: async (messages, response) => {
        // Track each LLM call cost
        const generation = trace.generation({
          name: "llm-call",
          model: "claude-sonnet-4-20250514",
          input: messages,
        });

        generation.end({
          output: response.content,
          usage: {
            input: response.usage.inputTokens,
            output: response.usage.outputTokens,
          },
        });
      },
    });

    trace.update({ output: result, status: "success" });
    await langfuse.flushAsync();

    return result;
  } catch (error: any) {
    trace.update({ status: "error", metadata: { error: error.message } });
    await langfuse.flushAsync();
    throw error;
  }
}
// Now you can see every step of every agent run in Langfuse dashboard!
// → Debug why an agent made a wrong decision
// → See exactly how much each agent run costs
// → Identify which tool calls fail most often
```

---

## 🔨 Phase 4 Projects

### Project 1: AI Research Agent with Citations

**Architecture:**
```
User Query
    ↓
[LangGraph Plan Node] → Creates 4-5 specific search queries
    ↓
[LangGraph Search Nodes] → Runs in parallel via Send
    ↓
[Aggregate Node] → Merges all search results with RRF
    ↓
[Synthesize Node] → Writes structured report with citations
    ↓
[Review Node] → Scores quality, revises if < 0.8
    ↓
[Output Node] → Formats final report with [1][2][3] citations
```

**Stand-out features:**
- Source credibility scoring (domain authority, publication date)
- "Conflicting sources" alert when sources disagree
- Export as PDF with clickable citations
- Stream progress to frontend (which search is running, current step)

### Project 2: Competitor Monitor

**Architecture:**
```
BullMQ Cron (daily) 
    ↓
[Fetch Competitor Pages] → Firecrawl/Playwright
    ↓
[Change Detection] → Compare with last snapshot
    ↓  (if changes found)
[AI Analysis] → Summarize what changed and why it matters
    ↓
[Slack Notification] → Alert with diff summary
    ↓
[Save to DB] → Historical tracking
```

**Features:**
- Tracks pricing pages, feature announcements, blog posts
- Semantic diff (understands WHAT changed, not just character diff)
- Priority alerts (pricing changes > blog posts)
- Weekly summary report

### Project 3: AI Code Reviewer

**Architecture:**
```
GitHub Webhook (PR opened/updated)
    ↓
[Fetch PR Diff] → GitHub API
    ↓
[Analyze Changes] → LangGraph multi-step review
    ├── [Security Scan Node] → OWASP checks
    ├── [Performance Node] → N+1 queries, inefficiencies
    ├── [Test Coverage Node] → Missing tests
    └── [Code Style Node] → Consistency checks
    ↓
[Synthesize Review] → Combine all findings
    ↓
[Post Comment] → GitHub PR comment with actionable feedback
```

---

## ✅ Master Checklist

Before moving to Phase 5, verify you can:

**Agent Fundamentals**
- [ ] Build a ReAct agent from scratch (no framework)
- [ ] Implement all 4 agent failure mode defenses
- [ ] Build a human-in-the-loop checkpoint
- [ ] Use Mem0 for persistent cross-session memory

**LangGraph**
- [ ] Build a graph with at least 3 nodes and conditional edges
- [ ] Implement a graph with a cycle (loop back to a previous node)
- [ ] Add PostgreSQL checkpointing for resumability
- [ ] Create parallel branches with Send API
- [ ] Stream graph updates to a frontend

**Browser Automation**
- [ ] Scrape a page with Playwright and extract structured data with LLM
- [ ] Use Stagehand for natural language browser control
- [ ] Implement Firecrawl for clean content extraction
- [ ] Build a self-healing scraping pipeline

**MCP**
- [ ] Build a complete MCP server with 3+ tools
- [ ] Add resources to your MCP server
- [ ] Connect it to Claude Desktop and test it
- [ ] Package it as an npm package

**Production Readiness**
- [ ] Add timeouts and cost limits to all agents
- [ ] Trace every agent run with Langfuse
- [ ] Implement n8n workflow for a real automation task
- [ ] Build a change detection system

---

*Phase 4 complete. Agent development is the highest-value skill in this entire roadmap. Companies pay $5k-$50k for well-built agent systems.*
