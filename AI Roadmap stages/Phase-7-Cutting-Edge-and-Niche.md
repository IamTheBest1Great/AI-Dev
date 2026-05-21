# 🟣 Phase 7 — Cutting Edge & Niche Moats
> **Goal:** Stay ahead of 99% of developers with capabilities few have mastered
> **Timeline:** Weeks 29+
> **Outcome:** You can build and publish MCP servers, fine-tune open-source models, run autonomous agent loops, and develop niche AI skills that command premium rates.

---

## 📚 Table of Contents

1. [7.1 — MCP (Model Context Protocol)](#71--mcp-model-context-protocol)
2. [7.2 — Fine-Tuning & Custom Models](#72--fine-tuning--custom-models)
3. [7.3 — Autonomous Agent Loops](#73--autonomous-agent-loops)
4. [7.4 — Niche Skills That Pay Well](#74--niche-skills-that-pay-well)
5. [7.5 — What Most Developers Miss](#75--what-most-developers-miss)
6. [Phase 7 Projects](#-phase-7-projects)
7. [Master Checklist](#-master-checklist)

---

## Why Phase 7 Matters

```
┌─────────────────────────────────────────────────────────────────────┐
│              THE COMPETITIVE LANDSCAPE IN 2025                     │
│                                                                     │
│  "AI developer" on LinkedIn: 100,000+ people                       │
│  All using: Next.js + OpenAI API + Pinecone                        │
│  All building: chatbots and RAG systems                            │
│                                                                     │
│  "Published MCP server on npm": < 500 people                      │
│  "Fine-tuned Llama for legal domain": < 200 people                │
│  "Built autonomous invoice processing agent": < 100 people        │
│  "Built multimodal RAG with CLIP": < 50 people                    │
│                                                                     │
│  The rarer the skill, the higher the pay.                          │
│  These are the moats that take months to build.                    │
│  Start them NOW before they become crowded.                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7.1 — MCP (Model Context Protocol)

### 7.1.1 — MCP Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────────────┐
│              COMPLETE MCP ARCHITECTURE                             │
│                                                                     │
│  HOST APPLICATION                                                   │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │  Claude Desktop / Claude Code / Your Custom App           │    │
│  │                                                           │    │
│  │  ┌─────────────┐    ┌──────────────────────────────────┐ │    │
│  │  │ Claude AI   │◄──►│ MCP CLIENT                       │ │    │
│  │  │             │    │ (manages connections to servers)  │ │    │
│  │  └─────────────┘    └──────────────┬───────────────────┘ │    │
│  └──────────────────────────────────── │ ────────────────────┘    │
│                                         │ JSON-RPC                 │
│                    ┌────────────────────┼────────────────────┐    │
│                    ▼                    ▼                    ▼    │
│              ┌──────────┐        ┌──────────┐        ┌──────────┐ │
│              │ MCP      │        │ MCP      │        │ MCP      │ │
│              │ SERVER 1 │        │ SERVER 2 │        │ SERVER 3 │ │
│              │ (Notion) │        │ (GitHub) │        │ (Your DB)│ │
│              └──────────┘        └──────────┘        └──────────┘ │
│                   │                    │                    │      │
│              [Notion API]        [GitHub API]        [PostgreSQL]  │
│                                                                     │
│  WHAT EACH COMPONENT DOES:                                         │
│  HOST: The AI application that uses Claude                        │
│  CLIENT: Built into the host, manages server connections          │
│  SERVER: Your Node.js server exposing tools/resources             │
│                                                                     │
│  THREE THINGS A SERVER CAN EXPOSE:                                 │
│  TOOLS: Functions Claude can call (search, create, update)        │
│  RESOURCES: Data Claude can read (db://users, file://docs/)       │
│  PROMPTS: Reusable prompt templates Claude can use                │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.1.2 — Building Production MCP Servers

Complete, production-ready MCP server with tools, resources, and prompts:

```typescript
#!/usr/bin/env node
// mcp-server/index.ts

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

// Initialize server
const server = new Server(
  {
    name: "acme-internal-tools",
    version: "2.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: { subscribe: true },
      prompts: {},
    },
  }
);

// ================================================================
// TOOLS — Functions Claude can call
// ================================================================
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "search_internal_docs",
      description: `Search Acme's internal documentation, wikis, and knowledge base.
      
USE THIS WHEN:
- User asks about internal processes, policies, or procedures
- User needs to find information about products or features
- User wants to understand how something works internally

DO NOT USE when the question is general knowledge (not Acme-specific).`,
      inputSchema: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description: "Specific search query. Be precise. Include context.",
            maxLength: 500,
          },
          category: {
            type: "string",
            enum: ["engineering", "product", "design", "hr", "finance", "legal", "all"],
            description: "Filter by document category. Use 'all' if unsure.",
            default: "all",
          },
          dateRange: {
            type: "string",
            enum: ["any", "last_month", "last_quarter", "last_year"],
            description: "Filter by recency",
            default: "any",
          },
        },
        required: ["query"],
      },
    },

    {
      name: "create_linear_issue",
      description: `Create a new issue in Linear (project management).
      
USE THIS WHEN:
- User reports a bug that needs tracking
- User requests a feature to be documented
- User wants to create a task for the team
- User explicitly asks to create an issue/ticket

ALWAYS confirm with the user before creating (unless explicitly told to proceed).`,
      inputSchema: {
        type: "object" as const,
        properties: {
          title: {
            type: "string",
            description: "Concise issue title (max 100 chars)",
            maxLength: 100,
          },
          description: {
            type: "string",
            description: "Detailed description of the issue",
          },
          type: {
            type: "string",
            enum: ["bug", "feature", "improvement", "task", "question"],
          },
          priority: {
            type: "string",
            enum: ["urgent", "high", "medium", "low", "no_priority"],
            default: "medium",
          },
          teamId: {
            type: "string",
            description: "Team to assign the issue to",
            enum: ["engineering", "product", "design", "support"],
          },
          assigneeEmail: {
            type: "string",
            description: "Email of person to assign to (optional)",
          },
          labels: {
            type: "array",
            items: { type: "string" },
            description: "Labels to add to the issue",
          },
        },
        required: ["title", "description", "type"],
      },
    },

    {
      name: "query_metrics",
      description: `Query business and product metrics from our analytics database.
      
USE THIS WHEN:
- User asks about user numbers, growth, or engagement
- User wants revenue or billing data
- User needs performance metrics
- User asks "how many", "what percentage", "compare" type questions

You can only read data, not modify it.`,
      inputSchema: {
        type: "object" as const,
        properties: {
          metric: {
            type: "string",
            enum: [
              "dau", "mau", "wau",           // User metrics
              "revenue_mrr", "revenue_arr",    // Revenue metrics
              "churn_rate", "retention_rate",  // Health metrics
              "signups", "activations",        // Growth metrics
              "api_calls", "error_rate",       // Technical metrics
            ],
            description: "The metric to query",
          },
          period: {
            type: "string",
            enum: ["today", "yesterday", "this_week", "last_week", "this_month", "last_month", "this_quarter", "this_year"],
            default: "this_month",
          },
          breakdown: {
            type: "string",
            enum: ["total", "by_plan", "by_country", "by_channel"],
            description: "How to break down the metric",
            default: "total",
          },
        },
        required: ["metric"],
      },
    },

    {
      name: "send_notification",
      description: `Send a notification to a team or channel.
      
USE THIS WHEN:
- User explicitly asks to notify someone or a team
- User wants to send an alert

REQUIRES user confirmation before sending.
Can send to: Slack channels, email lists, or PagerDuty.`,
      inputSchema: {
        type: "object" as const,
        properties: {
          channel: {
            type: "string",
            description: "Slack channel name (without #) or email address",
          },
          message: {
            type: "string",
            description: "The notification message",
            maxLength: 2000,
          },
          priority: {
            type: "string",
            enum: ["info", "warning", "critical"],
            default: "info",
          },
        },
        required: ["channel", "message"],
      },
    },
  ],
}));

// Execute tools
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "search_internal_docs": {
        const { query, category = "all", dateRange = "any" } = args as {
          query: string;
          category?: string;
          dateRange?: string;
        };

        // Validate input
        if (!query || query.trim().length < 3) {
          throw new McpError(ErrorCode.InvalidParams, "Query must be at least 3 characters");
        }

        const results = await searchInternalDocs(query, category, dateRange);

        if (results.length === 0) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                results: [],
                message: "No results found. Try broader search terms or different category.",
              }),
            }],
          };
        }

        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              results: results.slice(0, 5).map(r => ({
                title: r.title,
                excerpt: r.content.slice(0, 300) + "...",
                url: r.url,
                category: r.category,
                relevanceScore: r.score,
                lastUpdated: r.updatedAt,
              })),
              totalFound: results.length,
              searchQuery: query,
            }, null, 2),
          }],
        };
      }

      case "create_linear_issue": {
        const { title, description, type, priority, teamId, assigneeEmail, labels } = args as {
          title: string;
          description: string;
          type: string;
          priority?: string;
          teamId?: string;
          assigneeEmail?: string;
          labels?: string[];
        };

        // Create the issue
        const issue = await linearClient.issueCreate({
          teamId: await resolveTeamId(teamId ?? "engineering"),
          title,
          description,
          priority: linearPriorityMap[priority ?? "medium"],
          labelIds: await resolveLabels(labels ?? []),
          assigneeId: assigneeEmail ? await resolveUserId(assigneeEmail) : undefined,
        });

        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              success: true,
              issueKey: issue.identifier,
              issueUrl: issue.url,
              message: `Issue created: ${issue.identifier} - "${title}"`,
            }, null, 2),
          }],
        };
      }

      case "query_metrics": {
        const { metric, period = "this_month", breakdown = "total" } = args as {
          metric: string;
          period?: string;
          breakdown?: string;
        };

        const data = await queryAnalytics(metric, period, breakdown);

        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              metric,
              period,
              breakdown,
              data,
              queriedAt: new Date().toISOString(),
            }, null, 2),
          }],
        };
      }

      case "send_notification": {
        const { channel, message, priority = "info" } = args as {
          channel: string;
          message: string;
          priority?: string;
        };

        // Validate channel
        const allowedChannels = await getAllowedChannels();
        if (!allowedChannels.includes(channel) && !channel.includes("@")) {
          throw new McpError(
            ErrorCode.InvalidParams,
            `Channel "${channel}" not in allowed list: ${allowedChannels.join(", ")}`
          );
        }

        await sendSlackMessage(channel, message, priority as "info" | "warning" | "critical");

        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              success: true,
              sentTo: channel,
              priority,
              message: "Notification sent successfully",
            }),
          }],
        };
      }

      default:
        throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
    }
  } catch (error: any) {
    if (error instanceof McpError) throw error;
    throw new McpError(
      ErrorCode.InternalError,
      `Tool execution failed: ${error.message}`
    );
  }
});

// ================================================================
// RESOURCES — Data sources Claude can read
// ================================================================
server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: [
    {
      uri: "acme://customers/list",
      name: "Customer List",
      description: "Current customer list with plan and status (read-only)",
      mimeType: "application/json",
    },
    {
      uri: "acme://runbooks/list",
      name: "Available Runbooks",
      description: "List of available operational runbooks",
      mimeType: "application/json",
    },
  ],
}));

// Resource templates (dynamic resources)
server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({
  resourceTemplates: [
    {
      uriTemplate: "acme://runbooks/{name}",
      name: "Runbook by Name",
      description: "Get a specific operational runbook",
      mimeType: "text/markdown",
    },
    {
      uriTemplate: "acme://customers/{customerId}",
      name: "Customer Details",
      description: "Get details for a specific customer",
      mimeType: "application/json",
    },
  ],
}));

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  // Static resources
  if (uri === "acme://customers/list") {
    const customers = await db.customers.findMany({
      select: { id: true, name: true, plan: true, status: true, createdAt: true },
      where: { status: "active" },
      take: 100,
    });
    return {
      contents: [{
        uri,
        mimeType: "application/json",
        text: JSON.stringify({ customers, fetchedAt: new Date().toISOString() }, null, 2),
      }],
    };
  }

  // Dynamic: runbook by name
  const runbookMatch = uri.match(/^acme:\/\/runbooks\/(.+)$/);
  if (runbookMatch) {
    const runbookName = decodeURIComponent(runbookMatch[1]);
    const runbook = await getRunbook(runbookName);
    if (!runbook) {
      throw new McpError(ErrorCode.InvalidRequest, `Runbook not found: ${runbookName}`);
    }
    return {
      contents: [{
        uri,
        mimeType: "text/markdown",
        text: runbook.content,
      }],
    };
  }

  throw new McpError(ErrorCode.InvalidRequest, `Unknown resource URI: ${uri}`);
});

// ================================================================
// PROMPTS — Reusable prompt templates
// ================================================================
server.setRequestHandler(ListPromptsRequestSchema, async () => ({
  prompts: [
    {
      name: "incident-response",
      description: "Template for responding to production incidents",
      arguments: [
        { name: "severity", description: "Incident severity (P0-P3)", required: true },
        { name: "description", description: "Brief description of the incident", required: true },
      ],
    },
    {
      name: "customer-email",
      description: "Generate a customer communication email",
      arguments: [
        { name: "customerName", description: "Customer's name", required: true },
        { name: "issue", description: "The issue or topic", required: true },
        { name: "tone", description: "Email tone: formal/casual/apologetic", required: false },
      ],
    },
  ],
}));

server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "incident-response":
      return {
        description: "Incident response template",
        messages: [{
          role: "user",
          content: {
            type: "text",
            text: `You are an incident response coordinator.

Incident Details:
- Severity: ${args?.severity}
- Description: ${args?.description}

Please:
1. Create an incident timeline template
2. List immediate actions to take
3. Draft a status page message
4. Identify stakeholders to notify
5. Suggest post-incident review questions`,
          },
        }],
      };

    case "customer-email":
      return {
        description: "Customer email template",
        messages: [{
          role: "user",
          content: {
            type: "text",
            text: `Draft a professional email to ${args?.customerName} regarding: ${args?.issue}

Tone: ${args?.tone ?? "professional and helpful"}
Company: Acme Corp
Your role: Customer Success Manager

Requirements:
- Acknowledge the issue clearly
- Provide specific next steps
- Set realistic expectations
- Offer additional support
- Professional signature`,
          },
        }],
      };

    default:
      throw new McpError(ErrorCode.InvalidRequest, `Unknown prompt: ${name}`);
  }
});

// Start server
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("MCP Server running..."); // stderr for logs (stdout reserved for JSON-RPC)
```

### 7.1.3 — Publishing MCP Servers

```json
// package.json — structure for publishable MCP server
{
  "name": "@yourname/mcp-acme-tools",
  "version": "1.0.0",
  "description": "MCP server for Acme internal tools — gives Claude access to docs, Linear, and metrics",
  "main": "dist/index.js",
  "type": "module",
  "bin": {
    "mcp-acme-tools": "dist/index.js"
  },
  "scripts": {
    "build": "tsc && chmod +x dist/index.js",
    "dev": "tsx src/index.ts",
    "prepublishOnly": "npm run build"
  },
  "files": ["dist/", "README.md"],
  "keywords": ["mcp", "model-context-protocol", "claude", "anthropic"],
  "engines": { "node": ">=18" }
}
```

```markdown
# README.md — this sells your MCP server

## @yourname/mcp-acme-tools

Give Claude access to your Acme internal tools — search docs, create Linear issues, and query metrics directly in Claude Desktop or Claude Code.

## Quick Start

Add to your Claude Desktop config (~/.config/claude/claude_desktop_config.json):

```json
{
  "mcpServers": {
    "acme-tools": {
      "command": "npx",
      "args": ["-y", "@yourname/mcp-acme-tools"],
      "env": {
        "DATABASE_URL": "your-database-url",
        "LINEAR_API_KEY": "your-linear-key",
        "ANALYTICS_API_KEY": "your-analytics-key"
      }
    }
  }
}
```

Restart Claude Desktop. You'll see a 🔌 icon indicating the MCP connection.

## Available Tools
- **search_internal_docs** — Search company documentation
- **create_linear_issue** — Create bugs/features in Linear
- **query_metrics** — Get DAU, MRR, churn, and other metrics

## Available Resources
- `acme://customers/list` — Current customer list
- `acme://runbooks/{name}` — Operational runbooks
```

---

## 7.2 — Fine-Tuning & Custom Models

### 7.2.1 — When to Fine-Tune Decision Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│              FINE-TUNING DECISION TREE                             │
│                                                                     │
│  STEP 1: Have you tried few-shot prompting with 10-20 examples?   │
│  NO → Try this first (free, instant)                               │
│  YES → Continue                                                     │
│                                                                     │
│  STEP 2: Have you tried RAG with relevant documents?              │
│  NO → Try RAG (often solves "model doesn't know" problems)        │
│  YES → Continue                                                     │
│                                                                     │
│  STEP 3: Have you optimized system prompt + CoT?                   │
│  NO → Spend 1-2 days on prompt engineering first                   │
│  YES → Continue                                                     │
│                                                                     │
│  STEP 4: Do you STILL need fine-tuning?                           │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ GOOD reasons:                                         │         │
│  │ ✓ Consistent output format prompting can't achieve   │         │
│  │ ✓ Domain tone/style (medical, legal, brand voice)    │         │
│  │ ✓ Task where smaller model = faster + cheaper        │         │
│  │ ✓ Data privacy (can't send to external API)          │         │
│  └──────────────────────────────────────────────────────┘         │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ BAD reasons:                                          │         │
│  │ ✗ "Model doesn't know current information" → use RAG │         │
│  │ ✗ "I want it smarter" → use bigger base model        │         │
│  │ ✗ "It hallucinates" → use RAG + output validation    │         │
│  └──────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2.2 — OpenAI Fine-Tuning (Fastest Path)

```typescript
import OpenAI from "openai";
import fs from "fs";
import readline from "readline";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// STEP 1: Prepare training data (JSONL format)
interface TrainingExample {
  messages: {
    role: "system" | "user" | "assistant";
    content: string;
  }[];
}

// Example: Fine-tune for legal contract analysis
const trainingData: TrainingExample[] = [
  {
    messages: [
      {
        role: "system",
        content: "You are a legal contract analyzer. Extract key clauses and assess risk levels. Always output structured JSON.",
      },
      {
        role: "user",
        content: `Analyze this contract clause: "Either party may terminate this Agreement upon 30 days written notice."`,
      },
      {
        role: "assistant",
        content: JSON.stringify({
          clauseType: "termination",
          riskLevel: "low",
          keyTerms: ["30 days", "written notice", "either party"],
          assessment: "Standard termination clause. Balanced for both parties.",
          redFlags: [],
          recommendations: ["Consider adding cure period for breach before termination"],
        }, null, 2),
      },
    ],
  },
  // ... 49-499 more high-quality examples
];

// Validate examples
function validateTrainingData(data: TrainingExample[]): string[] {
  const errors: string[] = [];

  for (let i = 0; i < data.length; i++) {
    const example = data[i];

    if (!example.messages || example.messages.length < 2) {
      errors.push(`Example ${i}: Must have at least 2 messages`);
    }

    const hasUser = example.messages.some(m => m.role === "user");
    const hasAssistant = example.messages.some(m => m.role === "assistant");

    if (!hasUser) errors.push(`Example ${i}: Missing user message`);
    if (!hasAssistant) errors.push(`Example ${i}: Missing assistant message`);

    const lastMsg = example.messages[example.messages.length - 1];
    if (lastMsg.role !== "assistant") {
      errors.push(`Example ${i}: Last message must be assistant`);
    }

    // Check assistant response is valid JSON (for our use case)
    try {
      JSON.parse(lastMsg.content);
    } catch {
      errors.push(`Example ${i}: Assistant response is not valid JSON`);
    }
  }

  return errors;
}

// Write to JSONL
function writeJSONL(data: TrainingExample[], outputPath: string): void {
  const lines = data.map(example => JSON.stringify(example));
  fs.writeFileSync(outputPath, lines.join("\n"));
  console.log(`✓ Wrote ${data.length} examples to ${outputPath}`);
}

// STEP 2: Upload file
async function uploadTrainingFile(filePath: string): Promise<string> {
  console.log("Uploading training file...");

  const file = await openai.files.create({
    file: fs.createReadStream(filePath),
    purpose: "fine-tune",
  });

  console.log(`✓ Uploaded: ${file.id} (${file.size} bytes)`);
  return file.id;
}

// STEP 3: Start fine-tuning job
async function startFineTuning(
  fileId: string,
  options: {
    baseModel?: string;
    nEpochs?: number;
    suffix?: string;
  } = {}
): Promise<string> {
  const {
    baseModel = "gpt-4o-mini-2024-07-18", // Cheapest fine-tunable model
    nEpochs = "auto",                       // Let OpenAI decide
    suffix = "legal-analyzer",
  } = options;

  console.log(`Starting fine-tune of ${baseModel}...`);

  const job = await openai.fineTuning.jobs.create({
    training_file: fileId,
    model: baseModel,
    hyperparameters: {
      n_epochs: nEpochs,
      batch_size: "auto",
      learning_rate_multiplier: "auto",
    },
    suffix,
  });

  console.log(`✓ Job started: ${job.id}`);
  return job.id;
}

// STEP 4: Monitor job progress
async function waitForFineTuning(jobId: string): Promise<string> {
  console.log("Monitoring fine-tuning progress...");

  while (true) {
    const job = await openai.fineTuning.jobs.retrieve(jobId);

    console.log(`Status: ${job.status} | ${new Date().toLocaleTimeString()}`);

    if (job.status === "succeeded") {
      console.log(`✓ Fine-tuning complete!`);
      console.log(`Model: ${job.fine_tuned_model}`);
      console.log(`Trained tokens: ${job.trained_tokens}`);
      return job.fine_tuned_model!;
    }

    if (job.status === "failed" || job.status === "cancelled") {
      throw new Error(`Fine-tuning ${job.status}: ${JSON.stringify(job.error)}`);
    }

    // Print recent events
    const events = await openai.fineTuning.jobs.listEvents(jobId, { limit: 5 });
    for (const event of events.data.reverse()) {
      console.log(`  [${event.level}] ${event.message}`);
    }

    await new Promise(resolve => setTimeout(resolve, 60000)); // Check every minute
  }
}

// STEP 5: Evaluate fine-tuned model vs base model
async function evaluateModels(
  fineTunedModel: string,
  evalExamples: TrainingExample[]
): Promise<EvaluationReport> {
  const baseModel = "gpt-4o-mini";

  let ftCorrect = 0;
  let baseCorrect = 0;
  const results: any[] = [];

  for (const example of evalExamples) {
    const userMsg = example.messages.find(m => m.role === "user")!;
    const expectedResponse = example.messages.find(m => m.role === "assistant")!.content;
    const systemMsg = example.messages.find(m => m.role === "system");

    const messages = [
      ...(systemMsg ? [systemMsg] : []),
      userMsg,
    ];

    // Test fine-tuned model
    const ftResponse = await openai.chat.completions.create({
      model: fineTunedModel,
      messages,
      max_tokens: 500,
    });
    const ftOutput = ftResponse.choices[0].message.content!;
    const ftCorrectResult = isCorrectResponse(ftOutput, expectedResponse);

    // Test base model
    const baseResponse = await openai.chat.completions.create({
      model: baseModel,
      messages,
      max_tokens: 500,
    });
    const baseOutput = baseResponse.choices[0].message.content!;
    const baseCorrectResult = isCorrectResponse(baseOutput, expectedResponse);

    if (ftCorrectResult) ftCorrect++;
    if (baseCorrectResult) baseCorrect++;

    results.push({
      input: userMsg.content.slice(0, 100),
      ftCorrect: ftCorrectResult,
      baseCorrect: baseCorrectResult,
    });
  }

  const n = evalExamples.length;
  const report = {
    fineTunedModel,
    baseModel,
    fineTunedAccuracy: ftCorrect / n,
    baseModelAccuracy: baseCorrect / n,
    improvement: (ftCorrect - baseCorrect) / n,
    recommendation: ftCorrect >= baseCorrect
      ? "DEPLOY: Fine-tuned model is better or equal"
      : "DO NOT DEPLOY: Base model performs better",
    results,
  };

  console.log(`\n📊 EVALUATION RESULTS:`);
  console.log(`Fine-tuned: ${(report.fineTunedAccuracy * 100).toFixed(1)}%`);
  console.log(`Base model: ${(report.baseModelAccuracy * 100).toFixed(1)}%`);
  console.log(`Improvement: ${(report.improvement * 100).toFixed(1)}pp`);
  console.log(`\nRecommendation: ${report.recommendation}`);

  return report;
}

function isCorrectResponse(actual: string, expected: string): boolean {
  try {
    const actualParsed = JSON.parse(actual);
    const expectedParsed = JSON.parse(expected);

    // Check key fields match
    return (
      actualParsed.clauseType === expectedParsed.clauseType &&
      actualParsed.riskLevel === expectedParsed.riskLevel
    );
  } catch {
    return false;
  }
}

// Complete pipeline
async function runFineTuningPipeline() {
  const TRAINING_FILE = "training.jsonl";
  const EVAL_FILE = "eval.jsonl";

  // Validate data
  const errors = validateTrainingData(trainingData);
  if (errors.length > 0) {
    console.error("Validation errors:", errors);
    process.exit(1);
  }

  // Write files
  writeJSONL(trainingData, TRAINING_FILE);

  // Upload and train
  const fileId = await uploadTrainingFile(TRAINING_FILE);
  const jobId = await startFineTuning(fileId, { suffix: "legal-v1" });
  const fineTunedModel = await waitForFineTuning(jobId);

  // Evaluate
  const evalData = trainingData.slice(0, 20); // Use held-out eval set
  const report = await evaluateModels(fineTunedModel, evalData);

  if (report.fineTunedAccuracy >= report.baseModelAccuracy) {
    console.log(`\n✓ Deploy: ${fineTunedModel}`);
    // Save to config
    fs.writeFileSync(".env.production", `LEGAL_MODEL=${fineTunedModel}\n`, { flag: "a" });
  }
}
```

### 7.2.3 — LoRA Fine-Tuning with Unsloth

```python
# fine_tune.py — Run on GPU (Modal, Runpod, or local)
# pip install unsloth transformers datasets trl

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import json

# Load base model with 4-bit quantization (runs on consumer GPU!)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-3b-instruct",
    max_seq_length=2048,
    dtype=None,           # Auto-detect
    load_in_4bit=True,   # 4-bit quantization (saves 75% VRAM)
)

# Add LoRA adapters (only train these ~1% of parameters)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # LoRA rank (8-64, higher = more params, better quality)
    target_modules=[                # Which layers to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,                  # Usually = r
    lora_dropout=0,                 # 0 is fine for most cases
    bias="none",
    use_gradient_checkpointing="unsloth",  # Saves VRAM
    random_state=42,
)

# Prepare training data (Alpaca format)
def format_training_example(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    prompt = f"""Below is an instruction that describes a task.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""

    return {"text": prompt}

# Load your dataset
with open("training_data.jsonl", "r") as f:
    raw_data = [json.loads(line) for line in f]

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_training_example)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,             # 3 full passes through dataset
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",            # Memory-efficient optimizer
    ),
)

trainer.train()

# Save the LoRA adapter (small file, not full model)
model.save_pretrained("lora-adapter")
tokenizer.save_pretrained("lora-adapter")

# Export to GGUF for Ollama (merge adapter + base model)
model.save_pretrained_gguf(
    "model-gguf",
    tokenizer,
    quantization_method="q4_k_m"    # Best quality/size ratio
)
```

```bash
# Import fine-tuned model into Ollama
# Create Modelfile
cat > Modelfile << EOF
FROM ./model-gguf/unsloth.Q4_K_M.gguf

TEMPLATE """Below is an instruction.

### Instruction:
{{ .Prompt }}

### Response:
"""

PARAMETER temperature 0.1
PARAMETER stop "### Instruction:"
EOF

ollama create legal-analyzer -f Modelfile
ollama run legal-analyzer "Analyze this contract clause: 'Governing law: New York'"
```

### 7.2.4 — Dataset Curation Best Practices

```typescript
// Quality guidelines for training data

class DatasetCurator {
  async curate(rawExamples: TrainingExample[]): Promise<TrainingExample[]> {
    const curated: TrainingExample[] = [];

    for (const example of rawExamples) {
      // Quality checks
      if (!this.meetsQualityBar(example)) continue;

      curated.push(example);
    }

    // Deduplication
    const deduplicated = await this.deduplicate(curated);

    // Balance categories
    const balanced = this.balance(deduplicated);

    console.log(`
Dataset curation complete:
  Original: ${rawExamples.length}
  After quality filter: ${curated.length}
  After deduplication: ${deduplicated.length}
  After balancing: ${balanced.length}
    `);

    return balanced;
  }

  private meetsQualityBar(example: TrainingExample): boolean {
    const assistant = example.messages.find(m => m.role === "assistant");
    if (!assistant) return false;

    // Minimum response length
    if (assistant.content.length < 50) {
      console.warn("Skipping: Response too short");
      return false;
    }

    // Maximum response length
    if (assistant.content.length > 2000) {
      console.warn("Skipping: Response too long");
      return false;
    }

    // Check for placeholder text
    const badPhrases = ["TODO", "PLACEHOLDER", "EXAMPLE OUTPUT", "[INSERT]"];
    if (badPhrases.some(p => assistant.content.includes(p))) {
      console.warn("Skipping: Contains placeholder text");
      return false;
    }

    return true;
  }

  private async deduplicate(
    examples: TrainingExample[]
  ): Promise<TrainingExample[]> {
    const seen = new Set<string>();
    const unique: TrainingExample[] = [];

    for (const example of examples) {
      // Hash the user message (normalize whitespace)
      const userMsg = example.messages
        .find(m => m.role === "user")?.content
        .toLowerCase()
        .replace(/\s+/g, " ")
        .trim() ?? "";

      const hash = md5(userMsg);

      if (!seen.has(hash)) {
        seen.add(hash);
        unique.push(example);
      }
    }

    return unique;
  }

  private balance(examples: TrainingExample[]): TrainingExample[] {
    // For classification tasks, ensure balanced class distribution
    // For generation tasks, ensure topic diversity

    // Count categories
    const categoryCounts = new Map<string, number>();
    for (const example of examples) {
      try {
        const response = JSON.parse(
          example.messages.find(m => m.role === "assistant")!.content
        );
        const category = response.category ?? response.type ?? "unknown";
        categoryCounts.set(category, (categoryCounts.get(category) ?? 0) + 1);
      } catch {
        categoryCounts.set("unknown", (categoryCounts.get("unknown") ?? 0) + 1);
      }
    }

    const minCount = Math.min(...categoryCounts.values());
    const maxPerCategory = Math.min(minCount * 2, 200); // Allow up to 2x imbalance

    // Undersample overrepresented categories
    const categoryUsed = new Map<string, number>();

    return examples.filter(example => {
      try {
        const response = JSON.parse(
          example.messages.find(m => m.role === "assistant")!.content
        );
        const category = response.category ?? response.type ?? "unknown";
        const used = categoryUsed.get(category) ?? 0;

        if (used >= maxPerCategory) return false;
        categoryUsed.set(category, used + 1);
        return true;
      } catch {
        return true;
      }
    });
  }
}
```

---

## 7.3 — Autonomous Agent Loops

### 7.3.1 — Long-Horizon Tasks with Checkpointing

```typescript
// For tasks that take hours or days
class LongRunningAgent {
  private taskId: string;
  private checkpointer: TaskCheckpointer;

  async runTask(
    taskDefinition: TaskDefinition
  ): Promise<TaskResult> {
    // Try to resume from checkpoint
    const checkpoint = await this.checkpointer.load(taskDefinition.id);

    let state: TaskState = checkpoint ?? {
      taskId: taskDefinition.id,
      phase: "planning",
      completedSteps: [],
      pendingSteps: [],
      artifacts: {},
      startedAt: new Date().toISOString(),
      lastUpdatedAt: new Date().toISOString(),
    };

    // Planning phase
    if (state.phase === "planning") {
      state.pendingSteps = await this.createExecutionPlan(taskDefinition);
      state.phase = "executing";
      await this.checkpointer.save(state); // Save after planning
    }

    // Execution phase
    while (state.pendingSteps.length > 0) {
      const nextStep = state.pendingSteps[0];

      console.log(`Executing step: ${nextStep.id} (${nextStep.type})`);

      try {
        const stepResult = await this.executeStep(nextStep, state);

        // Move step from pending to completed
        state.completedSteps.push({
          ...nextStep,
          result: stepResult,
          completedAt: new Date().toISOString(),
        });
        state.pendingSteps.shift();
        state.artifacts[nextStep.id] = stepResult;
        state.lastUpdatedAt = new Date().toISOString();

        // Save checkpoint after each successful step
        await this.checkpointer.save(state);

      } catch (error: any) {
        console.error(`Step ${nextStep.id} failed: ${error.message}`);

        if (nextStep.retryCount < 3) {
          // Mark for retry
          state.pendingSteps[0] = {
            ...nextStep,
            retryCount: (nextStep.retryCount ?? 0) + 1,
          };
          await this.checkpointer.save(state);
          await sleep(5000 * nextStep.retryCount); // Exponential backoff
        } else {
          // Step permanently failed — try to adapt or fail gracefully
          const adaptedPlan = await this.adaptPlanAfterFailure(state, nextStep, error);
          if (adaptedPlan) {
            state.pendingSteps = adaptedPlan;
            await this.checkpointer.save(state);
          } else {
            throw new Error(`Task failed at step ${nextStep.id}: ${error.message}`);
          }
        }
      }
    }

    // Synthesis phase
    state.phase = "synthesizing";
    await this.checkpointer.save(state);

    const finalResult = await this.synthesizeResults(state);

    // Mark complete
    state.phase = "completed";
    state.result = finalResult;
    await this.checkpointer.save(state);

    return finalResult;
  }

  private async adaptPlanAfterFailure(
    state: TaskState,
    failedStep: TaskStep,
    error: Error
  ): Promise<TaskStep[] | null> {
    // Ask LLM to create an alternative plan
    const adaptation = await callLLM(`
A step in the task plan failed. Create an alternative approach.

Task: ${JSON.stringify(state.artifacts)}
Failed step: ${JSON.stringify(failedStep)}
Error: ${error.message}

Completed steps: ${state.completedSteps.map(s => s.id).join(", ")}

Provide an alternative plan for the remaining steps.
If the task cannot be completed, respond with "CANNOT_COMPLETE: reason"

Alternative plan (JSON array of steps) or CANNOT_COMPLETE:`);

    if (adaptation.includes("CANNOT_COMPLETE")) return null;

    try {
      return JSON.parse(adaptation);
    } catch {
      return null;
    }
  }
}

class TaskCheckpointer {
  constructor(private redis: Redis) {}

  async save(state: TaskState): Promise<void> {
    const key = `task:checkpoint:${state.taskId}`;
    await this.redis.set(key, JSON.stringify(state), "EX", 7 * 24 * 3600); // 7 days
    console.log(`Checkpoint saved: ${state.phase} (${state.completedSteps.length} steps done)`);
  }

  async load(taskId: string): Promise<TaskState | null> {
    const key = `task:checkpoint:${taskId}`;
    const saved = await this.redis.get(key);
    if (!saved) return null;

    const state = JSON.parse(saved) as TaskState;
    console.log(`Resuming from checkpoint: ${state.phase} (${state.completedSteps.length} steps done)`);
    return state;
  }
}
```

### 7.3.2 — Self-Correcting Agents

```typescript
class SelfCorrectingAgent {
  async run(task: string, maxCorrectionAttempts = 3): Promise<string> {
    let currentResult = await this.initialAttempt(task);
    let correctionCount = 0;

    while (correctionCount < maxCorrectionAttempts) {
      // Self-evaluation
      const evaluation = await this.evaluateResult(task, currentResult);

      if (evaluation.quality >= 0.8) {
        console.log(`Quality threshold met: ${evaluation.quality}`);
        return currentResult;
      }

      console.log(`Quality ${evaluation.quality} below threshold. Issues: ${evaluation.issues.join(", ")}`);

      // Self-correction
      currentResult = await this.correctResult(
        task,
        currentResult,
        evaluation.issues
      );
      correctionCount++;
    }

    // Return best result even if below threshold
    return currentResult;
  }

  private async initialAttempt(task: string): Promise<string> {
    return await callLLM(`Complete this task thoroughly:
${task}`);
  }

  private async evaluateResult(
    task: string,
    result: string
  ): Promise<{ quality: number; issues: string[] }> {
    const evaluation = await callLLM(`
Evaluate this task completion on a scale of 0-10.

ORIGINAL TASK: ${task}
RESULT: ${result}

Check for:
1. Completeness (does it fully address the task?)
2. Accuracy (is the information correct?)
3. Format (is it well-structured?)
4. Clarity (is it easy to understand?)

Return JSON: {
  "score": 0-10,
  "issues": ["issue1", "issue2"] (empty if perfect)
}`, EvaluationSchema);

    return {
      quality: evaluation.score / 10,
      issues: evaluation.issues,
    };
  }

  private async correctResult(
    task: string,
    previousResult: string,
    issues: string[]
  ): Promise<string> {
    return await callLLM(`Improve this result based on the identified issues.

ORIGINAL TASK: ${task}

PREVIOUS RESULT (with issues):
${previousResult}

ISSUES TO FIX:
${issues.map((issue, i) => `${i+1}. ${issue}`).join("\n")}

Provide an improved result that addresses all issues:`);
  }
}
```

---

## 7.4 — Niche Skills That Pay Well

### 7.4.1 — Document Intelligence

```typescript
import { TextractClient, AnalyzeDocumentCommand } from "@aws-sdk/client-textract";

const textract = new TextractClient({ region: "ap-south-1" });

// AWS Textract — handles scanned documents, tables, forms
async function extractWithTextract(imageBuffer: Buffer): Promise<TextractResult> {
  const command = new AnalyzeDocumentCommand({
    Document: { Bytes: imageBuffer },
    FeatureTypes: ["TABLES", "FORMS", "SIGNATURES"],
  });

  const response = await textract.send(command);

  // Parse blocks into structured data
  const lines: string[] = [];
  const tables: string[][][] = [];
  const formFields: Record<string, string> = {};

  const blockMap = new Map(response.Blocks?.map(b => [b.Id, b]));

  for (const block of response.Blocks ?? []) {
    if (block.BlockType === "LINE") {
      lines.push(block.Text ?? "");
    }

    if (block.BlockType === "TABLE") {
      const table = extractTable(block, blockMap);
      tables.push(table);
    }

    if (block.BlockType === "KEY_VALUE_SET" && block.EntityTypes?.includes("KEY")) {
      const { key, value } = extractKeyValue(block, blockMap);
      if (key && value) formFields[key] = value;
    }
  }

  return { lines, tables, formFields };
}

// Post-process with AI for accuracy and structure
async function intelligentDocumentExtraction(
  imageBuffer: Buffer,
  documentType: "invoice" | "contract" | "form" | "id_document"
): Promise<DocumentExtractionResult> {
  // Step 1: AWS Textract for raw extraction
  const rawExtraction = await extractWithTextract(imageBuffer);

  // Step 2: Vision LLM for context and correction
  const visionExtraction = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    messages: [{
      role: "user",
      content: [
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/jpeg",
            data: imageBuffer.toString("base64"),
          },
        },
        {
          type: "text",
          text: `This is a ${documentType}. 
Raw extraction from OCR:
${JSON.stringify(rawExtraction, null, 2)}

Please:
1. Correct any OCR errors you can identify from the image
2. Structure the data appropriately for a ${documentType}
3. Extract any fields the OCR might have missed
4. Validate the extracted data makes sense

Return structured JSON for a ${documentType}.`,
        },
      ],
    }],
  });

  const aiResult = JSON.parse(visionExtraction.content[0].text);

  return {
    rawOCR: rawExtraction,
    aiEnhanced: aiResult,
    confidence: calculateConfidence(rawExtraction, aiResult),
  };
}
```

### 7.4.2 — AI for Code

```typescript
import Parser from "tree-sitter";
import JavaScript from "tree-sitter-javascript";
import TypeScript from "tree-sitter-typescript";

// AST-aware code analysis
async function analyzeCodeWithAST(
  code: string,
  language: "javascript" | "typescript"
): Promise<CodeAnalysis> {
  const parser = new Parser();
  parser.setLanguage(language === "typescript" ? TypeScript.typescript : JavaScript);

  const tree = parser.parse(code);

  // Extract functions, classes, imports
  const functions = extractFunctions(tree.rootNode);
  const classes = extractClasses(tree.rootNode);
  const imports = extractImports(tree.rootNode);
  const complexity = calculateCyclomaticComplexity(tree.rootNode);

  // AI-powered analysis with AST context
  const analysis = await callLLM(`
Analyze this code with the following AST context.

CODE:
${code}

AST ANALYSIS:
- Functions: ${functions.map(f => f.name).join(", ")}
- Classes: ${classes.map(c => c.name).join(", ")}
- Imports: ${imports.join(", ")}
- Cyclomatic Complexity: ${complexity}

Provide:
1. Security vulnerabilities (with line numbers)
2. Performance issues (with specific patterns)
3. Code quality issues (anti-patterns)
4. Missing test cases (specific scenarios)
5. Suggested improvements (with code examples)

Format as JSON.`);

  return JSON.parse(analysis);
}

// Semantic code search — find similar code by meaning
async function semanticCodeSearch(
  query: string,
  codebase: { path: string; code: string }[]
): Promise<SearchResult[]> {
  // Embed the search query
  const queryEmbedding = await embedText(query);

  // Embed all code files (or chunks)
  const codeEmbeddings = await embedBatch(
    codebase.map(file => `
File: ${file.path}
${file.code.slice(0, 500)}
    `)
  );

  // Find most similar
  const similarities = codeEmbeddings.map((emb, i) => ({
    file: codebase[i],
    similarity: cosineSimilarity(queryEmbedding, emb),
  }));

  return similarities
    .sort((a, b) => b.similarity - a.similarity)
    .filter(r => r.similarity > 0.7)
    .slice(0, 5);
}

// Auto-generate comprehensive tests
async function generateTests(
  functionCode: string,
  language: "typescript" | "javascript",
  framework: "jest" | "vitest" | "mocha"
): Promise<string> {
  return await callLLM(`
Generate comprehensive ${framework} tests for this function.

FUNCTION:
${functionCode}

Requirements:
1. Happy path tests (expected inputs)
2. Edge cases (empty, null, undefined, extreme values)
3. Error cases (invalid inputs, expected throws)
4. Async behavior (if applicable)
5. Boundary conditions

Use ${language} and ${framework}.
Include descriptive test names.
Add comments explaining what each test verifies.
Aim for >90% code coverage.

Return ONLY the test code, no explanation:`);
}
```

### 7.4.3 — Multimodal RAG with CLIP

```typescript
// CLIP enables TRUE multimodal search — same embedding space for text AND images
// "a man in a red jacket" can find images of men in red jackets!

// Using Hugging Face API for CLIP embeddings
async function clipEmbedText(text: string): Promise<number[]> {
  const response = await fetch(
    "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.HF_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: { text },
        options: { wait_for_model: true },
      }),
    }
  );
  const result = await response.json();
  return result.embeddings[0];
}

async function clipEmbedImage(imageBuffer: Buffer): Promise<number[]> {
  const base64 = imageBuffer.toString("base64");

  const response = await fetch(
    "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.HF_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: { image: base64 },
        options: { wait_for_model: true },
      }),
    }
  );
  const result = await response.json();
  return result.embeddings[0];
}

// Index images with CLIP
async function indexImagesWithCLIP(
  images: { buffer: Buffer; path: string; metadata: Record<string, unknown> }[]
) {
  for (const image of images) {
    // Generate CLIP embedding for the image
    const clipEmbedding = await clipEmbedImage(image.buffer);

    // Generate text description for additional context
    const description = await generateImageCaption(image.buffer);
    const textEmbedding = await clipEmbedText(description);

    // Average the two embeddings for better retrieval
    const combinedEmbedding = clipEmbedding.map(
      (v, i) => (v + textEmbedding[i]) / 2
    );

    await vectorDB.upsert({
      id: generateId(),
      content: description,
      embedding: combinedEmbedding, // 512-dim for CLIP
      metadata: {
        type: "image",
        path: image.path,
        ...image.metadata,
      },
    });
  }
}

// Search images with text queries
async function searchImages(textQuery: string): Promise<SearchResult[]> {
  // Embed the text query with CLIP (same space as image embeddings!)
  const queryEmbedding = await clipEmbedText(textQuery);

  // Search finds matching images
  return await vectorDB.search(queryEmbedding, "images", 10, {
    type: "image",
  });
}

// Example: "Find all charts showing revenue growth"
// This works because CLIP understands visual concepts from text
```

### 7.4.4 — Geospatial AI

```typescript
// Embed location context for geographic AI applications
async function embedLocationContext(location: {
  latitude: number;
  longitude: number;
  address?: string;
  city?: string;
  country?: string;
}): Promise<number[]> {
  // Create rich textual description of location
  const locationText = `
Location: ${location.address ?? `${location.latitude}, ${location.longitude}`}
City: ${location.city ?? "Unknown"}
Country: ${location.country ?? "Unknown"}
Region: ${await getRegionDescription(location.latitude, location.longitude)}
Nearby landmarks: ${await getNearbyLandmarks(location.latitude, location.longitude)}
  `.trim();

  return await embedText(locationText);
}

// Natural language to geospatial query
async function naturalLanguageGeoSearch(
  query: string,
  userLocation: { latitude: number; longitude: number }
): Promise<GeoSearchResult[]> {
  // Parse intent from natural language
  const intent = await callLLM(`
Parse this location-based search query.
User's current location: ${userLocation.latitude}, ${userLocation.longitude}

Query: "${query}"

Return JSON:
{
  "category": "restaurant|hotel|hospital|atm|...",
  "filters": { "openNow": bool, "rating": number, "priceRange": "low|medium|high" },
  "maxDistance": number (km),
  "sortBy": "distance|rating|price"
}`, GeoIntentSchema);

  // Use Google Maps API or custom location DB
  return await searchNearby(userLocation, intent);
}
```

---

## 7.5 — What Most Developers Miss

### 7.5.1 — Fine-Tuning Cost Analysis

```typescript
// Calculate ROI before fine-tuning
function calculateFineTuningROI(config: {
  baseModelCostPerToken: number;   // e.g., $0.15/1M for gpt-4o-mini
  fineTunedCostPerToken: number;   // e.g., same or slightly higher
  trainingTokens: number;          // Total training tokens
  trainingCostPerToken: number;    // e.g., $8/1M for gpt-4o-mini fine-tuning
  dailyTokenUsage: number;         // Tokens used per day in production
  qualityImprovementPercent: number; // How much better is fine-tuned?
}): ROIAnalysis {
  const trainingCost =
    (config.trainingTokens / 1_000_000) * config.trainingCostPerToken;

  // Fine-tuned model might be more efficient (shorter prompts needed)
  // Let's say 20% fewer tokens needed in production
  const productionSavings =
    (config.dailyTokenUsage * 0.2 * config.baseModelCostPerToken) / 1_000_000;

  const daysToROI = trainingCost / productionSavings;

  return {
    trainingCost,
    productionSavingsPerDay: productionSavings,
    breakEvenDays: Math.ceil(daysToROI),
    recommendation: daysToROI < 30
      ? "RECOMMENDED: Breaks even in < 30 days"
      : daysToROI < 90
      ? "CONSIDER: Breaks even in 30-90 days"
      : "SKIP: Takes > 90 days to break even — optimize prompts instead",
  };
}

// Example:
const roi = calculateFineTuningROI({
  baseModelCostPerToken: 0.15,     // gpt-4o-mini
  fineTunedCostPerToken: 0.15,     // same
  trainingTokens: 500_000,         // 500 training examples × 1000 tokens
  trainingCostPerToken: 8,         // $8/1M for fine-tuning
  dailyTokenUsage: 10_000_000,     // 10M tokens/day production
  qualityImprovementPercent: 20,
});
// Training cost: $4
// Daily savings: ~$0.30
// Break-even: 14 days
// → RECOMMENDED!
```

### 7.5.2 — Published MCP Server Strategy

```markdown
# How to get visibility for your MCP server

## Step 1: Build something genuinely useful
- Target a tool that many developers use daily
- GitHub, Notion, Jira, Linear, Slack, Airtable, Supabase
- Or a niche tool with passionate users

## Step 2: Make it dead simple to install
- npx @yourname/mcp-toolname (no local setup)
- Single env var setup (not 10 variables)
- Working demo video in README (30 seconds)

## Step 3: Publish to the right places
- npm (primary distribution)
- GitHub with good README
- smithery.ai (official MCP registry — gets you in-app visibility)
- Awesome-MCP list on GitHub

## Step 4: Content marketing
- Write a technical blog post: "How I built an MCP server for X"
- Post on LinkedIn (tag @Anthropic)
- Share in relevant Discord servers (LangChain, Anthropic)
- Post to Hacker News "Show HN" if it's impressive

## Step 5: Maintain it
- Respond to GitHub issues within 24 hours
- Add new features based on user requests
- Keep up with MCP spec updates

This gives you:
- Visibility in the Claude ecosystem
- Open source credibility
- Leads for consulting work
- Portfolio project with real users
```

---

## 🔨 Phase 7 Projects

### Project 1: Autonomous Business Agent

**What it does:** Completely automates invoice processing and follow-up for a freelancer or small business.

**Capabilities:**
- Reads invoice emails from Gmail
- Extracts invoice data (amount, client, due date) using vision + Textract
- Logs to Google Sheets accounting system
- Creates Linear/Jira task for follow-up
- Sends reminder emails when invoice is overdue (7, 14, 30 days)
- Updates CRM when payment received

**Tech stack:** LangGraph + Gmail API + Google Sheets + Computer Use + Mem0

**Why impressive:** This agent runs autonomously 24/7 and saves 5+ hours/week for any small business. Charge $2,000-5,000 to build + $300/month maintenance.

### Project 2: Custom Fine-Tuned Model API

**What it does:** A private API running a fine-tuned model for legal document analysis.

**Build process:**
1. Collect 200-500 legal contract examples (publicly available contracts)
2. Create training data: clause → analysis JSON
3. Fine-tune gpt-4o-mini or Llama 3.2 3B
4. Evaluate: does it match GPT-4o quality at 1/10th the cost?
5. Deploy via Ollama (local) or vLLM (cloud)
6. Build REST API wrapper
7. Benchmark vs GPT-4o and Claude Sonnet

**Stand-out:** Provide a benchmark PDF showing accuracy vs cost vs latency for your model vs the big models.

### Project 3: Published MCP Server

**Build and publish a production MCP server for one of:**
- GitHub (PR analysis, issue creation from conversation)
- Notion (read/write pages, databases)
- Linear (issue creation, project management)
- Your company's internal tools
- Supabase (read/write database tables)

**Requirements for this project:**
- Minimum 3 tools, 2 resources, 1 prompt template
- Published to npm as `@yourname/mcp-[service]`
- Listed on smithery.ai
- README with installation video
- 10+ GitHub stars (promote it!)

---

## ✅ Master Checklist

Before declaring Phase 7 complete:

**MCP**
- [ ] Built MCP server with 3+ tools, 2+ resources, 1+ prompt template
- [ ] Server handles errors gracefully (never crashes)
- [ ] Published to npm and works with `npx`
- [ ] Tested with Claude Desktop — all tools work
- [ ] Listed on smithery.ai registry
- [ ] At least 5 real users (not just yourself)

**Fine-Tuning**
- [ ] Created training dataset with 50+ high-quality examples
- [ ] Validated dataset quality (no duplicates, proper format)
- [ ] Ran OpenAI fine-tuning pipeline end-to-end
- [ ] Evaluated fine-tuned vs base model with metrics (not vibes)
- [ ] Can explain when fine-tuning IS and IS NOT the right approach
- [ ] Set up Ollama serving for at least one local model

**Autonomous Agents**
- [ ] Built agent that runs for 30+ steps without human intervention
- [ ] Implemented checkpointing that survives server restart
- [ ] Self-correction loop that improves output quality
- [ ] Agent has meaningful timeouts and cost limits

**Niche Skills (pick 2)**
- [ ] Document intelligence: extract from scanned PDF + tables
- [ ] Semantic code search across a real codebase
- [ ] Test generation for an existing function
- [ ] Multimodal RAG with image + text search
- [ ] Location-based AI query

---

## Resources for Phase 7

**MCP:**
- [Official MCP Spec](https://modelcontextprotocol.io) — read the full spec
- [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers) — existing servers for inspiration
- [Smithery Registry](https://smithery.ai) — where to publish

**Fine-Tuning:**
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [Unsloth GitHub](https://github.com/unslothai/unsloth) — fastest LoRA training
- [Hugging Face PEFT Docs](https://huggingface.co/docs/peft) — comprehensive fine-tuning docs

**Papers:**
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — original paper
- [QLoRA](https://arxiv.org/abs/2305.14314) — 4-bit quantized fine-tuning
- [RLHF](https://arxiv.org/abs/2203.02155) — how ChatGPT was trained

---

*Phase 7 complete. You now have skills that place you in the top 1% of AI developers globally. The combination of MCP, fine-tuning, autonomous agents, and niche specialization creates moats that take months to build — and that's exactly why they're valuable.*
