# 🟠 Phase 3 — Bots & Messaging Platforms
> **Goal:** Deploy AI where users already live — not just your app
> **Timeline:** Weeks 9–12
> **Outcome:** You can build production bots on Telegram, WhatsApp, Discord, Slack, with stateful conversations, payments, voice calls, and background job processing.

---

## 📚 Table of Contents

1. [The Bot Architecture Mental Model](#the-bot-architecture-mental-model)
2. [3.1 — Platform APIs](#31--platform-apis)
3. [3.2 — Bot Architecture](#32--bot-architecture)
4. [3.3 — Advanced Bot Patterns](#33--advanced-bot-patterns)
5. [3.4 — What Most Developers Miss](#34--what-most-developers-miss)
6. [Phase 3 Projects](#-phase-3-projects)
7. [Master Checklist](#-master-checklist)

---

## The Bot Architecture Mental Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOW BOT COMMUNICATION WORKS                     │
│                                                                     │
│  POLLING (simple, for development):                                │
│  ┌──────────┐  "any messages?"   ┌──────────────┐                 │
│  │ Your Bot │ ─────────────────► │ Telegram API │                 │
│  │  Server  │ ◄───────────────── │              │                 │
│  └──────────┘  "yes, here's one" └──────────────┘                 │
│  → Your server constantly polls every 1-2 seconds                 │
│  → Works without public URL                                        │
│  → Bad for production (wasteful, delays)                           │
│                                                                     │
│  WEBHOOKS (production):                                            │
│  ┌──────────┐                    ┌──────────────┐                 │
│  │ Your Bot │ ◄──────────────── │ Telegram API │                 │
│  │  Server  │  "new message NOW" │              │                 │
│  └──────────┘                    └──────────────┘                 │
│  → Telegram pushes to YOUR server instantly                       │
│  → Requires public HTTPS URL                                       │
│  → Production standard                                             │
│                                                                     │
│  The Flow for every platform:                                      │
│  User sends message                                                │
│       ↓                                                             │
│  Platform API receives it                                          │
│       ↓                                                             │
│  Platform POSTs to your webhook URL                                │
│       ↓                                                             │
│  Your server processes it                                          │
│       ↓                                                             │
│  Your server calls LLM                                             │
│       ↓                                                             │
│  Your server calls platform API to send response                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3.1 — Platform APIs

### 3.1.1 — Telegram Bot API

Telegram is the most developer-friendly platform. Start here.

```typescript
// Install: npm install node-telegram-bot-api grammy
// Use grammy — it's the modern, TypeScript-first Telegram framework

import { Bot, Context, session, SessionFlavor } from "grammy";

// Session data structure
interface SessionData {
  conversationHistory: { role: string; content: string }[];
  userPreferences: { language: string; responseStyle: string };
  messageCount: number;
}

type MyContext = Context & SessionFlavor<SessionData>;

const bot = new Bot<MyContext>(process.env.TELEGRAM_BOT_TOKEN!);

// Add session middleware (stores state per user)
bot.use(
  session({
    initial: (): SessionData => ({
      conversationHistory: [],
      userPreferences: { language: "en", responseStyle: "formal" },
      messageCount: 0,
    }),
    // Use Redis for session storage in production
    storage: redisStorage, // from @grammyjs/storage-redis
  })
);

// Handle /start command
bot.command("start", async (ctx) => {
  const user = ctx.from;
  await ctx.reply(
    `👋 Hello ${user?.first_name}! I'm your AI assistant.\n\n` +
    `I can help you with:\n` +
    `• /ask [question] — Ask me anything\n` +
    `• /clear — Clear conversation history\n` +
    `• /settings — Change preferences\n\n` +
    `Or just type your message to start chatting!`,
    {
      reply_markup: {
        keyboard: [
          [{ text: "🤖 Ask AI" }, { text: "⚙️ Settings" }],
          [{ text: "📚 Help" }, { text: "🗑️ Clear History" }],
        ],
        resize_keyboard: true,
      },
    }
  );
});

// Handle regular messages
bot.on("message:text", async (ctx) => {
  const userMessage = ctx.message.text;
  const session = ctx.session;

  // Add to conversation history
  session.conversationHistory.push({
    role: "user",
    content: userMessage,
  });
  session.messageCount++;

  // Show typing indicator (important for UX!)
  await ctx.replyWithChatAction("typing");

  try {
    // Call LLM
    const response = await callLLMWithHistory(
      session.conversationHistory,
      buildUserSystemPrompt(ctx.from, session.userPreferences)
    );

    // Add response to history
    session.conversationHistory.push({
      role: "assistant",
      content: response,
    });

    // Trim history if too long (keep last 20 messages)
    if (session.conversationHistory.length > 20) {
      session.conversationHistory = session.conversationHistory.slice(-20);
    }

    // Send response with Markdown formatting
    await ctx.reply(response, { parse_mode: "Markdown" });
  } catch (error) {
    await ctx.reply(
      "❌ Sorry, I encountered an error. Please try again.\n\n" +
      "If the problem persists, use /clear to reset."
    );
  }
});

// Handle /clear command
bot.command("clear", async (ctx) => {
  ctx.session.conversationHistory = [];
  await ctx.reply("✅ Conversation history cleared!");
});

// Inline keyboards for interactive UX
bot.on("callback_query:data", async (ctx) => {
  const data = ctx.callbackQuery.data;

  if (data.startsWith("action:")) {
    const action = data.split(":")[1];
    await ctx.answerCallbackQuery(); // Must acknowledge callback!

    switch (action) {
      case "summarize":
        await ctx.editMessageText("📝 Generating summary...");
        const summary = await generateSummary(ctx.session.conversationHistory);
        await ctx.editMessageText(summary);
        break;
      case "export":
        await exportConversation(ctx);
        break;
    }
  }
});

// Start the bot
// For development: use polling
bot.start();

// For production: use webhooks
// bot.api.setWebhook("https://yourdomain.com/telegram-webhook");
// Then handle POST requests at that URL
```

**Telegram-specific features:**

```typescript
// Inline keyboards (buttons attached to messages)
await ctx.reply("Choose an option:", {
  reply_markup: {
    inline_keyboard: [
      [
        { text: "✅ Yes", callback_data: "action:yes" },
        { text: "❌ No", callback_data: "action:no" },
      ],
      [{ text: "📊 Show Analysis", callback_data: "action:analyze" }],
    ],
  },
});

// Send rich content
await ctx.replyWithPhoto("https://example.com/chart.png", {
  caption: "Here's your data visualization",
  parse_mode: "Markdown",
});

// Send a document
await ctx.replyWithDocument(
  { source: pdfBuffer, filename: "report.pdf" },
  { caption: "Your generated report" }
);

// Handle file uploads from users
bot.on("message:document", async (ctx) => {
  const file = await ctx.getFile();
  const fileUrl = `https://api.telegram.org/file/bot${TOKEN}/${file.file_path}`;
  // Download and process the file
  const buffer = await downloadFile(fileUrl);
  await processUploadedDocument(ctx, buffer);
});
```

---

### 3.1.2 — Discord.js

```typescript
import {
  Client,
  Events,
  GatewayIntentBits,
  SlashCommandBuilder,
  REST,
  Routes,
  ChatInputCommandInteraction,
} from "discord.js";

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.DirectMessages,
  ],
});

// Define slash commands
const commands = [
  new SlashCommandBuilder()
    .setName("ask")
    .setDescription("Ask the AI a question")
    .addStringOption((option) =>
      option
        .setName("question")
        .setDescription("Your question")
        .setRequired(true)
    ),

  new SlashCommandBuilder()
    .setName("summarize")
    .setDescription("Summarize the last N messages")
    .addIntegerOption((option) =>
      option
        .setName("count")
        .setDescription("Number of messages to summarize (default: 20)")
        .setMinValue(5)
        .setMaxValue(100)
    ),

  new SlashCommandBuilder()
    .setName("quiz")
    .setDescription("Start a quiz on a topic")
    .addStringOption((option) =>
      option
        .setName("topic")
        .setDescription("Quiz topic (e.g., JavaScript, History, Science)")
        .setRequired(true)
    ),
].map((command) => command.toJSON());

// Register commands with Discord
async function registerCommands() {
  const rest = new REST().setToken(process.env.DISCORD_TOKEN!);
  await rest.put(
    Routes.applicationGuildCommands(
      process.env.CLIENT_ID!,
      process.env.GUILD_ID!
    ),
    { body: commands }
  );
  console.log("Slash commands registered!");
}

// Handle slash commands
client.on(Events.InteractionCreate, async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  const cmd = interaction as ChatInputCommandInteraction;

  switch (cmd.commandName) {
    case "ask":
      await handleAskCommand(cmd);
      break;
    case "summarize":
      await handleSummarizeCommand(cmd);
      break;
    case "quiz":
      await handleQuizCommand(cmd);
      break;
  }
});

async function handleAskCommand(interaction: ChatInputCommandInteraction) {
  const question = interaction.options.getString("question", true);

  // Discord requires response within 3 seconds or it fails
  // For LLM calls (can be slow), defer first!
  await interaction.deferReply();

  try {
    const answer = await callLLM(question);

    // Discord message limit is 2000 characters
    if (answer.length > 1900) {
      const chunks = splitIntoChunks(answer, 1900);
      await interaction.editReply(chunks[0]);
      for (const chunk of chunks.slice(1)) {
        await interaction.followUp(chunk);
      }
    } else {
      await interaction.editReply(answer);
    }
  } catch (error) {
    await interaction.editReply("❌ An error occurred. Please try again.");
  }
}

async function handleSummarizeCommand(interaction: ChatInputCommandInteraction) {
  const count = interaction.options.getInteger("count") || 20;

  await interaction.deferReply({ ephemeral: true }); // Only visible to requester

  // Fetch recent messages from the channel
  const messages = await interaction.channel?.messages.fetch({ limit: count });
  if (!messages) {
    await interaction.editReply("❌ Could not fetch messages.");
    return;
  }

  const messageText = [...messages.values()]
    .reverse()
    .map((m) => `${m.author.username}: ${m.content}`)
    .join("\n");

  const summary = await callLLM(`
Summarize these Discord messages in 3-5 bullet points.
Focus on key decisions, questions raised, and action items.

Messages:
${messageText}

Summary:`);

  await interaction.editReply(`📝 **Summary of last ${count} messages:**\n\n${summary}`);
}

// Handle regular messages (for AI context in specific channels)
client.on(Events.MessageCreate, async (message) => {
  // Ignore bot messages to prevent loops!
  if (message.author.bot) return;

  // Only respond in designated AI channels or DMs
  const isAIChannel = message.channelId === process.env.AI_CHANNEL_ID;
  const isDM = message.channel.isDMBased();

  if (!isAIChannel && !isDM) return;

  // Respond to mentions
  if (message.mentions.has(client.user!)) {
    const question = message.content
      .replace(/<@!?\d+>/g, "")  // Remove @mention
      .trim();

    await message.channel.sendTyping();

    const response = await callLLM(question);
    await message.reply(response);
  }
});

client.login(process.env.DISCORD_TOKEN);
```

---

### 3.1.3 — WhatsApp (Twilio / Meta Cloud API)

```typescript
import twilio from "twilio";
import express from "express";

const client = twilio(
  process.env.TWILIO_ACCOUNT_SID!,
  process.env.TWILIO_AUTH_TOKEN!
);

const app = express();
app.use(express.urlencoded({ extended: false }));

// WhatsApp webhook endpoint
app.post("/whatsapp-webhook", async (req, res) => {
  // ALWAYS validate signature first!
  const signature = req.headers["x-twilio-signature"] as string;
  const url = `https://yourdomain.com/whatsapp-webhook`;

  const isValid = twilio.validateRequest(
    process.env.TWILIO_AUTH_TOKEN!,
    signature,
    url,
    req.body
  );

  if (!isValid) {
    res.status(403).send("Forbidden");
    return;
  }

  const { Body, From, MediaUrl0, MediaContentType0 } = req.body;
  const userPhone = From.replace("whatsapp:", ""); // "+919876543210"

  // Handle different message types
  let userMessage = Body || "";

  // Handle media messages
  if (MediaUrl0) {
    if (MediaContentType0?.startsWith("image/")) {
      userMessage = await analyzeImageFromUrl(MediaUrl0);
    } else if (MediaContentType0 === "application/pdf") {
      userMessage = await extractTextFromPDFUrl(MediaUrl0);
    }
  }

  // Get or create session
  const session = await getOrCreateSession(userPhone);

  // Add typing indicator
  // Note: WhatsApp doesn't have native typing indicator via Twilio
  // But you can send "⏳ Processing..." and edit it later

  // Process message
  const response = await processMessage(userPhone, userMessage, session);

  // Send WhatsApp message using Twilio
  await client.messages.create({
    from: `whatsapp:${process.env.TWILIO_WHATSAPP_NUMBER}`,
    to: `whatsapp:${userPhone}`,
    body: response,
  });

  res.status(200).send("OK");
});

// Send WhatsApp message with buttons (WhatsApp template)
async function sendInteractiveMessage(to: string) {
  await client.messages.create({
    from: `whatsapp:${process.env.TWILIO_WHATSAPP_NUMBER}`,
    to: `whatsapp:${to}`,
    body: "What would you like help with?",
    // Interactive buttons require approved templates on WhatsApp Business
    persistentAction: [
      "uri:https://yourdomain.com|Visit Website",
    ],
  });
}
```

**Meta Cloud API (Direct, no Twilio):**

```typescript
// More control, required for high-volume
const META_API_URL = "https://graph.facebook.com/v18.0";
const PHONE_NUMBER_ID = process.env.META_PHONE_NUMBER_ID!;
const ACCESS_TOKEN = process.env.META_ACCESS_TOKEN!;

async function sendWhatsAppMessage(to: string, message: string) {
  await fetch(`${META_API_URL}/${PHONE_NUMBER_ID}/messages`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ACCESS_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messaging_product: "whatsapp",
      recipient_type: "individual",
      to,
      type: "text",
      text: { body: message, preview_url: false },
    }),
  });
}

// Send list message (interactive)
async function sendListMessage(to: string) {
  await fetch(`${META_API_URL}/${PHONE_NUMBER_ID}/messages`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ACCESS_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      messaging_product: "whatsapp",
      to,
      type: "interactive",
      interactive: {
        type: "list",
        header: { type: "text", text: "How can I help?" },
        body: { text: "Choose a topic:" },
        action: {
          button: "Choose",
          sections: [
            {
              title: "Support",
              rows: [
                { id: "billing", title: "Billing Issue" },
                { id: "technical", title: "Technical Problem" },
              ],
            },
            {
              title: "Information",
              rows: [
                { id: "pricing", title: "Pricing & Plans" },
                { id: "features", title: "Product Features" },
              ],
            },
          ],
        },
      },
    }),
  });
}
```

---

### 3.1.4 — Slack Bolt SDK

```typescript
import { App, ExpressReceiver } from "@slack/bolt";

const receiver = new ExpressReceiver({
  signingSecret: process.env.SLACK_SIGNING_SECRET!,
});

const app = new App({
  token: process.env.SLACK_BOT_TOKEN!,
  receiver,
});

// Respond to app mentions
app.event("app_mention", async ({ event, say, client }) => {
  const userMessage = event.text.replace(/<@[A-Z0-9]+>/g, "").trim();

  // Show "bot is typing" indicator
  await client.chat.postMessage({
    channel: event.channel,
    text: "⏳ Thinking...",
    thread_ts: event.ts,
  });

  const response = await callLLM(userMessage);

  // Update the "thinking" message
  await say({ text: response, thread_ts: event.ts });
});

// Slash command handler
app.command("/ask", async ({ command, ack, respond }) => {
  await ack(); // Acknowledge immediately (required within 3 seconds!)

  const question = command.text;

  if (!question) {
    await respond("❌ Please provide a question: `/ask your question here`");
    return;
  }

  await respond({
    text: "⏳ Processing your question...",
    response_type: "in_channel", // visible to all, or "ephemeral" for private
  });

  const answer = await callLLM(question);

  await respond({
    text: answer,
    replace_original: true, // Replace the "Processing..." message
  });
});

// Block Kit — rich interactive messages
app.command("/standup", async ({ ack, respond }) => {
  await ack();

  await respond({
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: "📋 *Daily Standup Check-in*",
        },
      },
      {
        type: "input",
        block_id: "yesterday",
        element: {
          type: "plain_text_input",
          multiline: true,
          placeholder: { type: "plain_text", text: "What did you work on yesterday?" },
        },
        label: { type: "plain_text", text: "Yesterday" },
      },
      {
        type: "input",
        block_id: "today",
        element: {
          type: "plain_text_input",
          multiline: true,
          placeholder: { type: "plain_text", text: "What will you work on today?" },
        },
        label: { type: "plain_text", text: "Today" },
      },
      {
        type: "input",
        block_id: "blockers",
        element: {
          type: "plain_text_input",
          placeholder: { type: "plain_text", text: "Any blockers? (type 'none' if not)" },
        },
        label: { type: "plain_text", text: "Blockers" },
      },
      {
        type: "actions",
        elements: [
          {
            type: "button",
            text: { type: "plain_text", text: "Submit Standup" },
            action_id: "submit_standup",
            style: "primary",
          },
        ],
      },
    ],
  });
});

// Handle Block Kit form submission
app.action("submit_standup", async ({ ack, body, client }) => {
  await ack();

  const values = body.state?.values;
  const yesterday = values?.yesterday?.["input"]?.value;
  const today = values?.today?.["input"]?.value;
  const blockers = values?.blockers?.["input"]?.value;

  // Analyze with AI
  const analysis = await callLLM(`
Analyze this standup update and identify if there are any notable concerns or highlights.

Yesterday: ${yesterday}
Today: ${today}
Blockers: ${blockers}

Brief analysis (2-3 sentences max):`);

  // Post to standup channel
  await client.chat.postMessage({
    channel: process.env.STANDUP_CHANNEL!,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `*${body.user.name}'s Standup:*\n\n*Yesterday:* ${yesterday}\n*Today:* ${today}\n*Blockers:* ${blockers}\n\n*AI Analysis:* ${analysis}`,
        },
      },
    ],
  });
});

(async () => {
  await app.start(3000);
  console.log("Slack bot running on port 3000");
})();
```

---

## 3.2 — Bot Architecture

### 3.2.1 — Webhook Setup & Security

```typescript
import crypto from "crypto";
import express from "express";

const app = express();

// CRITICAL: Verify webhook signatures to prevent fake requests
// Each platform has its own method:

// TELEGRAM verification
function verifyTelegramWebhook(body: string, secretToken: string): boolean {
  // Telegram sends X-Telegram-Bot-Api-Secret-Token header
  // Set this secret when registering the webhook
  return true; // Compare against your set secret token
}

// SLACK verification
function verifySlackWebhook(
  body: string,
  timestamp: string,
  signature: string
): boolean {
  const signingSecret = process.env.SLACK_SIGNING_SECRET!;
  const baseString = `v0:${timestamp}:${body}`;
  const hmac = crypto.createHmac("sha256", signingSecret);
  hmac.update(baseString);
  const computedSig = `v0=${hmac.digest("hex")}`;
  return crypto.timingSafeEqual(
    Buffer.from(computedSig),
    Buffer.from(signature)
  );
}

// GITHUB webhook verification (same pattern used for many services)
function verifyGithubWebhook(body: string, signature: string): boolean {
  const secret = process.env.GITHUB_WEBHOOK_SECRET!;
  const hmac = crypto.createHmac("sha256", secret);
  hmac.update(body);
  const expected = `sha256=${hmac.digest("hex")}`;
  return crypto.timingSafeEqual(
    Buffer.from(expected),
    Buffer.from(signature)
  );
}

// Replay attack prevention — check timestamp is recent
function isTimestampRecent(timestamp: string, maxAgeSeconds = 300): boolean {
  const requestTime = parseInt(timestamp);
  const currentTime = Math.floor(Date.now() / 1000);
  return Math.abs(currentTime - requestTime) <= maxAgeSeconds;
}

// Express middleware for webhook verification
function webhookVerificationMiddleware(platform: "slack" | "github") {
  return (req: express.Request, res: express.Response, next: express.NextFunction) => {
    const rawBody = (req as any).rawBody; // Need to capture raw body

    if (platform === "slack") {
      const timestamp = req.headers["x-slack-request-timestamp"] as string;
      const signature = req.headers["x-slack-signature"] as string;

      if (!isTimestampRecent(timestamp)) {
        return res.status(403).json({ error: "Request too old" });
      }

      if (!verifySlackWebhook(rawBody, timestamp, signature)) {
        return res.status(403).json({ error: "Invalid signature" });
      }
    }

    next();
  };
}
```

### 3.2.2 — Stateful Conversation Flows (State Machine)

```
┌─────────────────────────────────────────────────────────────────────┐
│              STATE MACHINE FOR MULTI-STEP BOT FLOWS                │
│                                                                     │
│  Example: Customer Support Ticket Flow                             │
│                                                                     │
│  [START]                                                            │
│     │                                                               │
│     ▼                                                               │
│  [GREET] → "What's your issue?"                                    │
│     │                                                               │
│     ▼                                                               │
│  [CLASSIFY_ISSUE] → AI classifies: billing/technical/feature       │
│     │                                                               │
│     ├── billing → [COLLECT_INVOICE] → "What's your invoice #?"    │
│     │                │                                              │
│     │                ▼                                              │
│     │            [PROCESS_BILLING] → Resolve or escalate           │
│     │                                                               │
│     ├── technical → [COLLECT_DETAILS] → "What error do you see?"  │
│     │                │                                              │
│     │                ▼                                              │
│     │            [TROUBLESHOOT] → AI provides steps                │
│     │                                                               │
│     └── other → [AI_RESPONSE] → Direct LLM response               │
│                                                                     │
│  Each state has: enter action, user input handling, exit condition │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// State machine implementation
type BotState =
  | "initial"
  | "collecting_issue"
  | "collecting_billing_info"
  | "collecting_technical_details"
  | "awaiting_confirmation"
  | "completed";

interface ConversationSession {
  userId: string;
  platform: string;
  state: BotState;
  collectedData: Record<string, string>;
  history: { role: string; content: string }[];
  createdAt: Date;
  lastActiveAt: Date;
}

class StatefulBotFlow {
  private sessions = new Map<string, ConversationSession>();

  async processMessage(
    userId: string,
    platform: string,
    message: string
  ): Promise<string> {
    const session = this.getOrCreateSession(userId, platform);
    session.lastActiveAt = new Date();

    // Route based on current state
    switch (session.state) {
      case "initial":
        return this.handleInitialState(session, message);
      case "collecting_issue":
        return this.handleCollectIssue(session, message);
      case "collecting_billing_info":
        return this.handleBillingInfo(session, message);
      case "collecting_technical_details":
        return this.handleTechnicalDetails(session, message);
      case "awaiting_confirmation":
        return this.handleConfirmation(session, message);
      default:
        return this.handleGeneralConversation(session, message);
    }
  }

  private async handleInitialState(
    session: ConversationSession,
    message: string
  ): Promise<string> {
    // Classify the initial message
    const classification = await callLLM(`
Classify this customer message into one category.
Return ONLY the category name, nothing else.

Categories: billing, technical, feature_request, general

Message: "${message}"
Category:`);

    session.collectedData.issueType = classification.trim();

    if (classification.includes("billing")) {
      session.state = "collecting_billing_info";
      return "I understand you have a billing issue. Could you please provide your invoice number or the email address on your account?";
    } else if (classification.includes("technical")) {
      session.state = "collecting_technical_details";
      return "I'll help you with the technical issue. Can you describe what error you're seeing and on which device/browser?";
    } else {
      // Handle directly with AI
      const response = await callLLMWithHistory(
        [...session.history, { role: "user", content: message }],
        SUPPORT_SYSTEM_PROMPT
      );
      session.history.push(
        { role: "user", content: message },
        { role: "assistant", content: response }
      );
      return response;
    }
  }

  private getOrCreateSession(
    userId: string,
    platform: string
  ): ConversationSession {
    const key = `${platform}:${userId}`;
    if (!this.sessions.has(key)) {
      this.sessions.set(key, {
        userId,
        platform,
        state: "initial",
        collectedData: {},
        history: [],
        createdAt: new Date(),
        lastActiveAt: new Date(),
      });
    }
    return this.sessions.get(key)!;
  }
}
```

### 3.2.3 — Conversation History in MongoDB

```typescript
import { MongoClient, Collection } from "mongodb";

interface BotSession {
  _id: string;               // userId:platform (unique key)
  userId: string;
  platform: "telegram" | "whatsapp" | "discord" | "slack";
  messages: {
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: Date;
    tokensUsed?: number;
  }[];
  metadata: {
    userName?: string;
    language?: string;
    preferences?: Record<string, string>;
    state?: string;
  };
  createdAt: Date;
  updatedAt: Date;
  messageCount: number;
}

class BotSessionManager {
  private collection: Collection<BotSession>;

  constructor(db: MongoClient) {
    this.collection = db.db("bot").collection<BotSession>("sessions");
    this.setupIndexes();
  }

  private async setupIndexes() {
    await this.collection.createIndex({ userId: 1, platform: 1 }, { unique: true });
    await this.collection.createIndex({ updatedAt: 1 }, { expireAfterSeconds: 7 * 24 * 3600 }); // 7 day TTL
  }

  async getOrCreateSession(
    userId: string,
    platform: BotSession["platform"],
    userName?: string
  ): Promise<BotSession> {
    const session = await this.collection.findOne({ userId, platform });

    if (session) {
      return session;
    }

    // Create new session
    const newSession: BotSession = {
      _id: `${platform}:${userId}`,
      userId,
      platform,
      messages: [],
      metadata: { userName },
      createdAt: new Date(),
      updatedAt: new Date(),
      messageCount: 0,
    };

    await this.collection.insertOne(newSession);
    return newSession;
  }

  async addMessage(
    userId: string,
    platform: BotSession["platform"],
    role: "user" | "assistant",
    content: string,
    tokensUsed?: number
  ): Promise<void> {
    const MAX_HISTORY = 50; // Keep last 50 messages in DB

    await this.collection.findOneAndUpdate(
      { userId, platform },
      {
        $push: {
          messages: {
            $each: [{ role, content, timestamp: new Date(), tokensUsed }],
            $slice: -MAX_HISTORY, // Keep only last 50 messages
          },
        },
        $set: { updatedAt: new Date() },
        $inc: { messageCount: 1 },
      },
      { upsert: true }
    );
  }

  async getRecentMessages(
    userId: string,
    platform: BotSession["platform"],
    limit = 10
  ): Promise<BotSession["messages"]> {
    const session = await this.collection.findOne(
      { userId, platform },
      { projection: { messages: { $slice: -limit } } }
    );

    return session?.messages || [];
  }

  // Analytics: most active users, common message patterns
  async getAnalytics(platform: string): Promise<{
    totalUsers: number;
    totalMessages: number;
    avgMessagesPerUser: number;
    activeUsersToday: number;
  }> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const [stats, activeToday] = await Promise.all([
      this.collection.aggregate([
        { $match: { platform } },
        {
          $group: {
            _id: null,
            totalUsers: { $sum: 1 },
            totalMessages: { $sum: "$messageCount" },
            avgMessages: { $avg: "$messageCount" },
          },
        },
      ]).toArray(),

      this.collection.countDocuments({
        platform,
        updatedAt: { $gte: today },
      }),
    ]);

    return {
      totalUsers: stats[0]?.totalUsers || 0,
      totalMessages: stats[0]?.totalMessages || 0,
      avgMessagesPerUser: Math.round(stats[0]?.avgMessages || 0),
      activeUsersToday: activeToday,
    };
  }
}
```

### 3.2.4 — Rate Limiting Per User

```typescript
import Redis from "ioredis";

const redis = new Redis(process.env.REDIS_URL!);

interface RateLimitConfig {
  requestsPerMinute: number;
  requestsPerDay: number;
  tokensPerDay: number;
}

const PLATFORM_LIMITS: Record<string, RateLimitConfig> = {
  free: { requestsPerMinute: 5, requestsPerDay: 100, tokensPerDay: 50000 },
  pro: { requestsPerMinute: 20, requestsPerDay: 1000, tokensPerDay: 500000 },
  unlimited: { requestsPerMinute: 100, requestsPerDay: 10000, tokensPerDay: 5000000 },
};

async function checkRateLimit(
  userId: string,
  plan: "free" | "pro" | "unlimited" = "free"
): Promise<{ allowed: boolean; reason?: string; retryAfter?: number }> {
  const limits = PLATFORM_LIMITS[plan];
  const now = Date.now();
  const minute = Math.floor(now / 60000);
  const today = new Date().toISOString().split("T")[0];

  const pipeline = redis.pipeline();

  // Per-minute counter
  const minuteKey = `rl:${userId}:min:${minute}`;
  pipeline.incr(minuteKey);
  pipeline.expire(minuteKey, 120); // 2 minute TTL

  // Per-day counter
  const dayKey = `rl:${userId}:day:${today}`;
  pipeline.incr(dayKey);
  pipeline.expire(dayKey, 86400 * 2); // 2 day TTL

  const results = await pipeline.exec();

  const minuteCount = results?.[0]?.[1] as number;
  const dayCount = results?.[2]?.[1] as number;

  if (minuteCount > limits.requestsPerMinute) {
    const secondsUntilReset = 60 - (Math.floor(now / 1000) % 60);
    return {
      allowed: false,
      reason: `Too many requests. Limit: ${limits.requestsPerMinute}/minute`,
      retryAfter: secondsUntilReset,
    };
  }

  if (dayCount > limits.requestsPerDay) {
    return {
      allowed: false,
      reason: `Daily limit reached. Limit: ${limits.requestsPerDay}/day. Resets midnight UTC.`,
      retryAfter: secondsUntilMidnightUTC(),
    };
  }

  return { allowed: true };
}

function secondsUntilMidnightUTC(): number {
  const now = new Date();
  const midnight = new Date(now);
  midnight.setUTCHours(24, 0, 0, 0);
  return Math.floor((midnight.getTime() - now.getTime()) / 1000);
}
```

---

## 3.3 — Advanced Bot Patterns

### 3.3.1 — AI Phone Call Bots

```
┌─────────────────────────────────────────────────────────────────────┐
│              PHONE BOT ARCHITECTURE                                │
│                                                                     │
│  Incoming Call                                                      │
│       ↓                                                             │
│  Twilio → your webhook (TwiML response)                            │
│       ↓                                                             │
│  Record audio → Twilio stores it                                   │
│       ↓                                                             │
│  Send audio URL → your server                                      │
│       ↓                                                             │
│  Whisper transcribes audio → text                                  │
│       ↓                                                             │
│  LLM processes text → generates response                           │
│       ↓                                                             │
│  ElevenLabs converts response → audio MP3                          │
│       ↓                                                             │
│  Twilio plays audio → to caller                                    │
│       ↓                                                             │
│  Repeat until call ends                                             │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
import twilio from "twilio";
import OpenAI from "openai";
import ElevenLabs from "elevenlabs";
import express from "express";

const twilioClient = twilio(
  process.env.TWILIO_ACCOUNT_SID!,
  process.env.TWILIO_AUTH_TOKEN!
);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });
const elevenlabs = new ElevenLabs({ apiKey: process.env.ELEVENLABS_API_KEY! });

const app = express();
app.use(express.urlencoded({ extended: false }));

// Initial TwiML response when call comes in
app.post("/voice-webhook", async (req, res) => {
  const twiml = new twilio.twiml.VoiceResponse();

  // Greet the caller
  twiml.say({
    voice: "Polly.Joanna", // AWS Polly voice via Twilio
  }, "Hello! Welcome to Acme Corp support. How can I help you today?");

  // Record their response
  twiml.record({
    action: "/voice-process",
    method: "POST",
    maxLength: 30, // 30 second recording limit
    transcribe: false, // We'll use Whisper ourselves
    trim: "trim-silence",
  });

  res.type("text/xml").send(twiml.toString());
});

// Process recording and generate AI response
app.post("/voice-process", async (req, res) => {
  const { RecordingUrl, CallSid } = req.body;

  const twiml = new twilio.twiml.VoiceResponse();

  try {
    // 1. Download and transcribe with Whisper
    const audioBuffer = await downloadAudio(RecordingUrl);
    const transcription = await openai.audio.transcriptions.create({
      file: new File([audioBuffer], "audio.mp3", { type: "audio/mp3" }),
      model: "whisper-1",
      language: "en",
    });
    const userText = transcription.text;
    console.log(`User said: ${userText}`);

    // 2. Get AI response
    const session = await getCallSession(CallSid);
    session.messages.push({ role: "user", content: userText });

    const aiResponse = await openai.chat.completions.create({
      model: "gpt-4o-mini",  // Faster model for voice (reduce latency!)
      messages: [
        { role: "system", content: PHONE_BOT_SYSTEM_PROMPT },
        ...session.messages,
      ],
      max_tokens: 150, // Keep responses SHORT for voice
    });

    const responseText = aiResponse.choices[0].message.content || "";
    session.messages.push({ role: "assistant", content: responseText });

    // 3. Convert to speech with ElevenLabs
    const audioStream = await elevenlabs.generate({
      voice: "Rachel", // Natural voice
      text: responseText,
      model_id: "eleven_turbo_v2", // Fast model
    });

    // Save audio to accessible URL
    const audioUrl = await saveAudioToS3(audioStream);

    // 4. Play the response
    twiml.play(audioUrl);

    // 5. Record the next response
    twiml.record({
      action: "/voice-process",
      method: "POST",
      maxLength: 30,
    });
  } catch (error) {
    twiml.say("I'm sorry, I encountered an error. Please call back later.");
    twiml.hangup();
  }

  res.type("text/xml").send(twiml.toString());
});

// Calendar booking during phone call
async function bookAppointment(
  date: string,
  time: string,
  callerName: string
): Promise<string> {
  const { google } = await import("googleapis");
  const calendar = google.calendar({ version: "v3", auth: getGoogleAuth() });

  await calendar.events.insert({
    calendarId: "primary",
    requestBody: {
      summary: `Appointment with ${callerName}`,
      start: { dateTime: `${date}T${time}:00`, timeZone: "Asia/Kolkata" },
      end: { dateTime: `${date}T${addHour(time)}:00`, timeZone: "Asia/Kolkata" },
    },
  });

  return `Appointment booked for ${date} at ${time}!`;
}
```

### 3.3.2 — Stripe Payments in Bots

```typescript
import Stripe from "stripe";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);

// Create payment link for Telegram
async function createPaymentLink(
  userId: string,
  planName: string,
  priceId: string
): Promise<string> {
  const paymentLink = await stripe.paymentLinks.create({
    line_items: [{ price: priceId, quantity: 1 }],
    metadata: {
      userId,
      platform: "telegram",
    },
    after_completion: {
      type: "redirect",
      redirect: {
        url: `https://t.me/${process.env.BOT_USERNAME}?start=payment_success`,
      },
    },
  });

  return paymentLink.url;
}

// Telegram payment flow
bot.on("message:text", async (ctx) => {
  if (ctx.message.text === "💳 Subscribe") {
    const paymentUrl = await createPaymentLink(
      ctx.from.id.toString(),
      "Pro Plan",
      process.env.STRIPE_PRO_PRICE_ID!
    );

    await ctx.reply("Choose your plan:", {
      reply_markup: {
        inline_keyboard: [
          [
            {
              text: "🚀 Pro Plan - $19/month",
              url: paymentUrl,
            },
          ],
          [
            {
              text: "💎 Enterprise - $99/month",
              callback_data: "plan:enterprise",
            },
          ],
        ],
      },
    });
  }
});

// Stripe webhook to activate subscription after payment
app.post("/stripe-webhook", express.raw({ type: "application/json" }), async (req, res) => {
  const sig = req.headers["stripe-signature"]!;
  const event = stripe.webhooks.constructEvent(
    req.body,
    sig,
    process.env.STRIPE_WEBHOOK_SECRET!
  );

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    const userId = session.metadata?.userId;
    const platform = session.metadata?.platform;

    if (userId && platform === "telegram") {
      // Activate subscription in DB
      await db.users.update({
        where: { telegramId: userId },
        data: { plan: "pro", subscriptionActive: true },
      });

      // Notify user in Telegram
      await bot.api.sendMessage(
        parseInt(userId),
        "🎉 Payment successful! Your Pro plan is now active. Enjoy unlimited messages!"
      );
    }
  }

  res.json({ received: true });
});
```

### 3.3.3 — Proactive Bots with BullMQ

```typescript
import { Queue, Worker } from "bullmq";
import Redis from "ioredis";

const connection = new Redis(process.env.REDIS_URL!);

// Job queues for different scheduled tasks
const scheduledMessagesQueue = new Queue("scheduled-messages", { connection });
const dailyDigestQueue = new Queue("daily-digest", { connection });
const priceAlertQueue = new Queue("price-alerts", { connection });

// Schedule a message for a specific time
async function scheduleMessage(
  userId: string,
  message: string,
  sendAt: Date
): Promise<void> {
  const delay = sendAt.getTime() - Date.now();

  await scheduledMessagesQueue.add(
    "send-message",
    { userId, message, platform: "telegram" },
    {
      delay: Math.max(0, delay),
      jobId: `msg-${userId}-${sendAt.getTime()}`, // Idempotent
    }
  );
}

// Worker for scheduled messages
const scheduledMessageWorker = new Worker(
  "scheduled-messages",
  async (job) => {
    const { userId, message, platform } = job.data;

    if (platform === "telegram") {
      await bot.api.sendMessage(userId, message);
    }
  },
  { connection }
);

// Daily digest job (runs at 8 AM)
async function setupDailyDigest() {
  // Add repeating job for 8 AM every day
  await dailyDigestQueue.add(
    "send-digest",
    {},
    {
      repeat: {
        pattern: "0 8 * * *", // Cron: 8 AM every day
        tz: "Asia/Kolkata",
      },
    }
  );
}

const digestWorker = new Worker(
  "daily-digest",
  async (job) => {
    // Get all users who opted in for daily digest
    const users = await db.users.findMany({
      where: { dailyDigestEnabled: true },
    });

    // Generate personalized digest for each user
    await Promise.all(
      users.map(async (user) => {
        const digest = await generatePersonalizedDigest(user);

        await bot.api.sendMessage(
          user.telegramId,
          `☀️ *Good morning, ${user.firstName}!*\n\n${digest}`,
          { parse_mode: "Markdown" }
        );
      })
    );
  },
  { connection, concurrency: 5 }
);

// Price alert system
interface PriceAlertJob {
  userId: string;
  productUrl: string;
  targetPrice: number;
  currentPrice: number;
}

const priceAlertWorker = new Worker<PriceAlertJob>(
  "price-alerts",
  async (job) => {
    const { productUrl, targetPrice, userId } = job.data;

    // Scrape current price
    const currentPrice = await scrapePriceFromUrl(productUrl);

    if (currentPrice <= targetPrice) {
      // Alert the user!
      await bot.api.sendMessage(
        userId,
        `🚨 *Price Alert!*\n\nProduct: ${productUrl}\nTarget: ₹${targetPrice}\nCurrent: ₹${currentPrice}\n\nPrice dropped!`,
        {
          parse_mode: "Markdown",
          reply_markup: {
            inline_keyboard: [[{ text: "🛒 Buy Now", url: productUrl }]],
          },
        }
      );

      // Remove this alert
      await db.priceAlerts.delete({ where: { userId, productUrl } });
    }
  },
  { connection }
);

// Schedule price check every hour for all active alerts
await priceAlertQueue.add(
  "check-prices",
  {},
  { repeat: { pattern: "0 * * * *" } } // Every hour
);
```

### 3.3.4 — Multi-Platform Bot (One Brain, Multiple Platforms)

```typescript
// Abstract interface for all platforms
interface BotPlatform {
  sendMessage(userId: string, message: string): Promise<void>;
  sendTypingIndicator(userId: string): Promise<void>;
  formatMessage(message: string): string; // Platform-specific formatting
}

class TelegramPlatform implements BotPlatform {
  async sendMessage(userId: string, message: string) {
    await bot.api.sendMessage(userId, this.formatMessage(message), {
      parse_mode: "Markdown",
    });
  }
  async sendTypingIndicator(userId: string) {
    await bot.api.sendChatAction(userId, "typing");
  }
  formatMessage(message: string): string {
    // Telegram supports markdown
    return message;
  }
}

class WhatsAppPlatform implements BotPlatform {
  async sendMessage(userId: string, message: string) {
    await sendWhatsAppMessage(userId, this.formatMessage(message));
  }
  async sendTypingIndicator(userId: string) {
    // WhatsApp doesn't support typing indicator via API
  }
  formatMessage(message: string): string {
    // Strip markdown, WhatsApp uses different formatting
    return message.replace(/\*\*/g, "*").replace(/`{3}[\s\S]*?`{3}/g, "");
  }
}

class SlackPlatform implements BotPlatform {
  async sendMessage(userId: string, message: string) {
    await app.client.chat.postMessage({
      channel: userId,
      text: this.formatMessage(message),
      mrkdwn: true,
    });
  }
  async sendTypingIndicator(userId: string) {
    // Slack doesn't support typing indicator for bots easily
  }
  formatMessage(message: string): string {
    // Convert markdown to Slack's mrkdwn format
    return message
      .replace(/\*\*(.*?)\*\*/g, "*$1*")   // Bold
      .replace(/__(.*?)__/g, "_$1_")         // Italic
      .replace(/`(.*?)`/g, "`$1`");          // Code
  }
}

// Universal message handler
class UniversalBot {
  private platforms: Record<string, BotPlatform> = {
    telegram: new TelegramPlatform(),
    whatsapp: new WhatsAppPlatform(),
    slack: new SlackPlatform(),
  };

  async handleMessage(
    platform: string,
    userId: string,
    message: string
  ): Promise<void> {
    const bot = this.platforms[platform];
    if (!bot) throw new Error(`Unknown platform: ${platform}`);

    // Show typing indicator
    await bot.sendTypingIndicator(userId);

    // Get session (same session DB regardless of platform)
    const session = await sessionManager.getSession(`${platform}:${userId}`);

    // Process with unified AI logic
    const response = await aiProcessor.process(message, session);

    // Send formatted for this specific platform
    await bot.sendMessage(userId, response);

    // Save session
    await sessionManager.saveSession(session);
  }
}
```

---

## 3.4 — What Most Developers Miss

### 3.4.1 — Bot Onboarding UX

```typescript
// BAD: Drop user into blank chat
async function badOnboarding(ctx: Context) {
  await ctx.reply("Hello!"); // What am I supposed to do??
}

// GOOD: Guide users from the first message
async function goodOnboarding(ctx: Context) {
  const user = ctx.from;

  // Progressive onboarding
  await ctx.reply(
    `👋 Welcome, ${user?.first_name}!\n\n` +
    `I'm your AI assistant powered by Claude. Here's what I can do:\n\n` +
    `📝 **Answer questions** — Just ask me anything\n` +
    `📄 **Analyze documents** — Send me a PDF\n` +
    `🔍 **Search knowledge base** — Ask about company policies\n` +
    `📅 **Book appointments** — "Book me a call for Friday"\n\n` +
    `*Try asking:* "What is your refund policy?"`,
    { parse_mode: "Markdown" }
  );

  // Show quick-start buttons
  await ctx.reply("Or pick a topic to get started:", {
    reply_markup: {
      inline_keyboard: [
        [
          { text: "💼 Business Help", callback_data: "topic:business" },
          { text: "🛠️ Technical Issue", callback_data: "topic:technical" },
        ],
        [
          { text: "💰 Billing", callback_data: "topic:billing" },
          { text: "📊 Analytics", callback_data: "topic:analytics" },
        ],
      ],
    },
  });

  // Track onboarding start
  await analytics.track("bot_onboarding_started", { userId: user?.id, platform: "telegram" });
}
```

### 3.4.2 — Human Handoff with Confidence Scoring

```typescript
async function respondWithHandoffCheck(
  userId: string,
  platform: string,
  userMessage: string,
  session: ConversationSession
): Promise<void> {
  // Get both the response AND a confidence score
  const { response, confidence, category } = await callLLMWithMetadata(`
Answer this customer query and rate your confidence.

Query: "${userMessage}"

Respond with JSON:
{
  "response": "your answer here",
  "confidence": 0.0 to 1.0 (how confident you are in this answer),
  "category": "billing|technical|feature|general",
  "requiresHuman": true/false (set true if you can't fully answer)
}

JSON:`, session.history);

  const platformBot = getBotForPlatform(platform);

  if (confidence < 0.70 || response.requiresHuman) {
    // Low confidence — offer human handoff
    await platformBot.sendMessage(
      userId,
      `I can provide some initial information, but this query might benefit from human review.\n\n` +
      `*What I know:* ${response.response}\n\n` +
      `Would you like me to connect you with a support agent?`
    );

    await platformBot.sendOptions(userId, [
      { label: "✅ Yes, connect me to support", value: "handoff:yes" },
      { label: "❌ No, this helps enough", value: "handoff:no" },
    ]);

    // Save context for human agent
    await saveHandoffContext(userId, {
      userQuery: userMessage,
      aiResponse: response.response,
      confidence,
      sessionHistory: session.history,
      category,
    });
  } else {
    // High confidence — respond directly
    await platformBot.sendMessage(userId, response.response);
  }

  // Log confidence for monitoring
  await metrics.record("bot_response_confidence", confidence, {
    platform,
    category,
    hadHandoff: confidence < 0.70,
  });
}
```

### 3.4.3 — Bot Analytics

```typescript
// Track every important event
class BotAnalytics {
  async trackEvent(event: {
    type: string;
    userId: string;
    platform: string;
    data?: Record<string, unknown>;
  }) {
    await db.botEvents.create({
      data: {
        eventType: event.type,
        userId: event.userId,
        platform: event.platform,
        eventData: event.data,
        createdAt: new Date(),
      },
    });
  }

  // Analytics queries
  async getDropOffPoints(): Promise<{ state: string; dropOffCount: number }[]> {
    return await db.botEvents.groupBy({
      by: ["eventData.state"],
      where: { eventType: "conversation_abandoned" },
      _count: true,
      orderBy: { _count: { id: "desc" } },
    });
  }

  async getTopCommands(platform: string, days = 30): Promise<{ command: string; count: number }[]> {
    return await db.botEvents.groupBy({
      by: ["eventData.command"],
      where: {
        eventType: "command_used",
        platform,
        createdAt: { gte: new Date(Date.now() - days * 86400000) },
      },
      _count: true,
      orderBy: { _count: { id: "desc" } },
    });
  }

  async getMostFailedIntents(): Promise<{ intent: string; failCount: number }[]> {
    return db.botEvents.groupBy({
      by: ["eventData.intent"],
      where: { eventType: "intent_failed" },
      _count: true,
      orderBy: { _count: { id: "desc" } },
    });
  }
}
```

---

## 🔨 Phase 3 Projects

### Project 1: Telegram AI Assistant with Cross-Session Memory

**What to build:** A Telegram bot that remembers users across sessions.

**Key features:**
- Conversation history stored in MongoDB
- User preferences (language, style) persisted in Redis
- `/settings` command to update preferences
- Automatic conversation summarization after 20 turns
- Usage tracking with daily limits
- Export conversation as PDF

**Stand-out twist:** The bot remembers: "Last time you asked about X, here's an update..."

### Project 2: WhatsApp Customer Support Bot with Human Escalation

**What to build:** A WhatsApp bot for a small business.

**Key features:**
- Receives customer queries via WhatsApp
- Classifies and attempts to answer with RAG
- Shows confidence score
- Below 70% confidence → offers to connect to human
- Stripe payment integration for quick checkout
- BullMQ for handling rate limits and message queues

**Stand-out twist:** When confidence < 70%, the bot says: "I'm 65% confident about this. Let me connect you with our team." and sends the full context to the human agent.

### Project 3: Phone Receptionist Bot with Calendar Booking

**What to build:** An AI phone receptionist for a clinic or restaurant.

**Key features:**
- Handles incoming Twilio calls
- Whisper transcription of caller audio
- ElevenLabs high-quality voice response
- Google Calendar integration for booking
- Sends SMS confirmation after booking
- Falls back gracefully when it can't handle a query

**Why this is impressive:** Businesses pay $500–2000/month for this. You can build it in a week.

---

## ✅ Master Checklist

Before moving to Phase 4, verify you can:

**Platform Setup**
- [ ] Build a Telegram bot with webhooks, commands, and inline keyboards
- [ ] Handle WhatsApp messages with proper signature validation
- [ ] Build a Discord bot with slash commands and deferred replies
- [ ] Set up Slack Bolt with events, commands, and Block Kit modals

**Bot Architecture**
- [ ] Implement webhook signature validation for at least 2 platforms
- [ ] Build a state machine for multi-step conversation flows
- [ ] Store and retrieve conversation history from MongoDB
- [ ] Implement 3-tier rate limiting (per-minute, per-day, per-token)

**Advanced Features**
- [ ] Build an AI phone call bot with Twilio + Whisper + ElevenLabs
- [ ] Embed Stripe payment links inside a bot
- [ ] Schedule messages with BullMQ cron jobs
- [ ] Build a multi-platform bot (same handler, different platforms)

**Production Readiness**
- [ ] Implement proper bot onboarding UX
- [ ] Add confidence-based human handoff
- [ ] Track analytics (commands, drop-off, failures)
- [ ] Test with mock webhook payloads before deploying

---

*Phase 3 complete. Bot development is one of the most commercially valuable skills in this roadmap — businesses pay monthly recurring revenue for these solutions.*
