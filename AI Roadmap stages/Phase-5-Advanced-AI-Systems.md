# 🔵 Phase 5 — Advanced AI Systems
> **Goal:** Build multimodal, voice, and multi-agent AI products
> **Timeline:** Weeks 18–23
> **Outcome:** You can build voice AI pipelines, analyze images and video, orchestrate multi-agent systems with shared memory, and deploy on-device AI.

---

## 📚 Table of Contents

1. [5.1 — Voice AI](#51--voice-ai)
2. [5.2 — Vision & Multimodal](#52--vision--multimodal)
3. [5.3 — Multi-Agent Systems](#53--multi-agent-systems)
4. [5.4 — Memory Systems](#54--memory-systems)
5. [5.5 — Edge & On-Device AI](#55--edge--on-device-ai)
6. [5.6 — What Most Developers Miss](#56--what-most-developers-miss)
7. [Phase 5 Projects](#-phase-5-projects)
8. [Master Checklist](#-master-checklist)

---

## 5.1 — Voice AI

### The Voice AI Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VOICE AI PIPELINE                               │
│                                                                     │
│  SPEECH → TEXT → AI → TEXT → SPEECH                               │
│                                                                     │
│  User speaks                                                        │
│       ↓                                                             │
│  [VAD] Voice Activity Detection                                    │
│  "Did user stop speaking?"                                         │
│       ↓ (user stopped)                                             │
│  [STT] Speech-to-Text (Whisper / Deepgram)                        │
│  Audio → "What is the capital of France?"                         │
│       ↓                                                             │
│  [LLM] AI Processing                                               │
│  Text → "The capital of France is Paris."                         │
│       ↓                                                             │
│  [TTS] Text-to-Speech (ElevenLabs / OpenAI TTS)                   │
│  Text → Audio file (MP3/PCM)                                       │
│       ↓                                                             │
│  User hears AI response                                             │
│                                                                     │
│  KEY METRICS:                                                       │
│  Time to first audio token: should be < 500ms                     │
│  Total response latency: should be < 2s                           │
│  (Users abandon voice apps with > 3s latency)                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.1.1 — Whisper API (Speech-to-Text)

```typescript
import OpenAI from "openai";
import { createReadStream } from "fs";
import { Readable } from "stream";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// BASIC: Transcribe an audio file
async function transcribeFile(audioPath: string): Promise<TranscriptionResult> {
  const transcription = await openai.audio.transcriptions.create({
    file: createReadStream(audioPath),
    model: "whisper-1",
    language: "en",              // omit for auto-detection
    response_format: "verbose_json", // includes word-level timestamps
    timestamp_granularities: ["word", "segment"], // word + segment timestamps
  });

  return {
    text: transcription.text,
    segments: transcription.segments, // Array of timed segments
    words: transcription.words,       // Array of timed words
    duration: transcription.duration,
    language: transcription.language,
  };
}

// ADVANCED: Transcribe from buffer (from WebSocket audio)
async function transcribeBuffer(
  audioBuffer: Buffer,
  mimeType: string = "audio/webm"
): Promise<string> {
  // Create a File-like object from buffer
  const audioFile = new File([audioBuffer], "audio.webm", { type: mimeType });

  const transcription = await openai.audio.transcriptions.create({
    file: audioFile,
    model: "whisper-1",
    response_format: "text",
  });

  return transcription;
}

// SUPPORTED FORMATS:
// mp3, mp4, mpeg, mpga, m4a, wav, webm
// Max file size: 25MB
// Recommended: convert to mp3 for smaller files

// LANGUAGE CODES for better accuracy:
// "en" English, "hi" Hindi, "ta" Tamil, "te" Telugu
// "mr" Marathi, "gu" Gujarati, "bn" Bengali
// Omit for auto-detection (slightly slower)

// COST: $0.006 per minute of audio
// 1 hour of audio = $0.36 (very cheap!)
```

### 5.1.2 — Real-Time Streaming Transcription

```typescript
// Whisper doesn't support streaming — use Deepgram or AssemblyAI for real-time

import { createClient, LiveTranscriptionEvents } from "@deepgram/sdk";

const deepgram = createClient(process.env.DEEPGRAM_API_KEY!);

function createRealtimeTranscription(
  onTranscript: (text: string, isFinal: boolean) => void,
  onError: (error: Error) => void
): DeepgramConnection {
  const connection = deepgram.listen.live({
    model: "nova-2",           // Best accuracy
    language: "en-IN",         // Indian English
    smart_format: true,        // Auto-punctuation
    interim_results: true,     // Get results as user speaks
    endpointing: 300,          // 300ms silence = end of utterance
    filler_words: false,       // Remove "um", "uh"
    diarize: false,            // Disable for single speaker
  });

  connection.on(LiveTranscriptionEvents.Open, () => {
    console.log("Deepgram connection opened");
  });

  connection.on(LiveTranscriptionEvents.Transcript, (data) => {
    const transcript = data.channel?.alternatives?.[0]?.transcript;
    const isFinal = data.is_final;

    if (transcript && transcript.trim()) {
      onTranscript(transcript, isFinal);
    }
  });

  connection.on(LiveTranscriptionEvents.Error, (error) => {
    onError(new Error(error.message));
  });

  return connection;
}

// Use in WebSocket server:
import { WebSocketServer } from "ws";

const wss = new WebSocketServer({ port: 8080 });

wss.on("connection", (ws) => {
  let deepgramConnection: DeepgramConnection | null = null;
  let finalTranscript = "";

  ws.on("message", (data: Buffer) => {
    const message = JSON.parse(data.toString());

    if (message.type === "start_recording") {
      // Start Deepgram connection
      deepgramConnection = createRealtimeTranscription(
        (text, isFinal) => {
          if (isFinal) {
            finalTranscript += " " + text;
            ws.send(JSON.stringify({ type: "final_transcript", text }));
          } else {
            ws.send(JSON.stringify({ type: "interim_transcript", text }));
          }
        },
        (error) => ws.send(JSON.stringify({ type: "error", message: error.message }))
      );
    } else if (message.type === "audio_chunk") {
      // Forward audio to Deepgram
      deepgramConnection?.send(Buffer.from(message.audio, "base64"));
    } else if (message.type === "stop_recording") {
      // Close Deepgram and process final transcript
      deepgramConnection?.requestClose();
      processVoiceInput(finalTranscript, ws);
      finalTranscript = "";
    }
  });
});
```

### 5.1.3 — Text-to-Speech

```typescript
import ElevenLabs from "elevenlabs";
import OpenAI from "openai";

const elevenlabs = new ElevenLabs({ apiKey: process.env.ELEVENLABS_API_KEY! });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// OPTION 1: OpenAI TTS (cheap, good quality)
async function openAITTS(
  text: string,
  voice: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer" = "nova"
): Promise<Buffer> {
  const mp3 = await openai.audio.speech.create({
    model: "tts-1",           // tts-1 = faster, tts-1-hd = higher quality
    voice,
    input: text,
    speed: 1.0,               // 0.25 to 4.0
  });

  return Buffer.from(await mp3.arrayBuffer());
}
// Cost: $0.015 per 1000 characters (very affordable)
// Latency: ~500ms for short texts

// OPTION 2: ElevenLabs (premium quality, voice cloning)
async function elevenLabsTTS(
  text: string,
  voiceId: string = "21m00Tcm4TlvDq8ikWAM" // Rachel - natural voice
): Promise<Buffer> {
  const audioStream = await elevenlabs.generate({
    voice: voiceId,
    text,
    model_id: "eleven_turbo_v2",    // Best speed/quality balance
    voice_settings: {
      stability: 0.5,               // 0-1: higher = more stable/monotone
      similarity_boost: 0.75,       // 0-1: higher = closer to original voice
      style: 0.0,                   // 0-1: style exaggeration
      use_speaker_boost: true,      // Enhanced clarity
    },
  });

  const chunks: Buffer[] = [];
  for await (const chunk of audioStream) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

// STREAMING TTS (lower latency — start playing before fully generated)
async function streamTTSToClient(text: string, res: Response) {
  res.setHeader("Content-Type", "audio/mpeg");
  res.setHeader("Transfer-Encoding", "chunked");

  // ElevenLabs streaming
  const audioStream = await elevenlabs.generate({
    voice: "21m00Tcm4TlvDq8ikWAM",
    text,
    model_id: "eleven_turbo_v2",
    stream: true, // Enable streaming!
  });

  // Pipe each chunk to response as it arrives
  for await (const chunk of audioStream) {
    res.write(chunk); // Browser starts playing while generating!
  }
  res.end();
}

// VOICE CLONING (ElevenLabs)
async function cloneVoice(audioSamples: Buffer[]): Promise<string> {
  const voiceId = await elevenlabs.voices.add({
    name: "Custom Voice",
    files: audioSamples.map((sample, i) =>
      new File([sample], `sample_${i}.mp3`, { type: "audio/mp3" })
    ),
    description: "Cloned voice for our AI assistant",
  });
  return voiceId.voice_id;
}
// Requires: 1-3 minutes of clean audio
// NOTE: Only clone voices you have rights to!
```

### 5.1.4 — OpenAI Realtime API (Full-Duplex Voice)

The most advanced voice AI available. No separate STT/TTS needed.

```typescript
import { RealtimeClient } from "@openai/realtime-api-beta";
import { WebSocket } from "ws";

// Server-side implementation
class RealtimeVoiceSession {
  private client: RealtimeClient;
  private ws: WebSocket; // Client WebSocket connection

  constructor(ws: WebSocket) {
    this.ws = ws;
    this.client = new RealtimeClient({
      apiKey: process.env.OPENAI_API_KEY!,
    });

    this.setupEventHandlers();
  }

  async start() {
    await this.client.connect();

    // Configure the session
    await this.client.updateSession({
      instructions: `
You are a helpful voice assistant for Acme Corp customer support.
Keep responses SHORT (1-3 sentences max).
Be conversational and natural.
If you don't know something, say so clearly.
      `.trim(),

      voice: "alloy",                     // alloy, echo, fable, onyx, nova, shimmer
      input_audio_format: "pcm16",         // Raw PCM audio from browser
      output_audio_format: "pcm16",        // Raw PCM audio to browser
      input_audio_transcription: {
        model: "whisper-1",               // Transcribe user speech too
      },
      turn_detection: {
        type: "server_vad",               // Server-side Voice Activity Detection
        threshold: 0.5,                   // 0-1: sensitivity
        prefix_padding_ms: 300,           // Audio before speech (ms)
        silence_duration_ms: 500,         // Silence before end-of-turn (ms)
      },
      tools: [
        {
          type: "function",
          name: "search_knowledge_base",
          description: "Search company knowledge base for answers",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string", description: "Search query" },
            },
            required: ["query"],
          },
        },
      ],
    });
  }

  private setupEventHandlers() {
    // Stream AI audio to client as it generates
    this.client.on("response.audio.delta", ({ delta }) => {
      const audioBuffer = Buffer.from(delta, "base64");
      this.ws.send(
        JSON.stringify({
          type: "audio_delta",
          audio: audioBuffer.toString("base64"),
        })
      );
    });

    // Send transcript of user speech to show in UI
    this.client.on("conversation.item.input_audio_transcription.completed", ({ transcript }) => {
      this.ws.send(
        JSON.stringify({ type: "user_transcript", text: transcript })
      );
    });

    // Send AI text for display
    this.client.on("response.audio_transcript.delta", ({ delta }) => {
      this.ws.send(
        JSON.stringify({ type: "assistant_transcript_delta", text: delta })
      );
    });

    // Handle tool calls
    this.client.on("response.function_call_arguments.done", async ({ name, arguments: args }) => {
      const result = await this.executeTool(name, JSON.parse(args));

      // Send result back to complete the response
      this.client.realtime.send("conversation.item.create", {
        item: {
          type: "function_call_output",
          call_id: name, // Match to the tool call
          output: JSON.stringify(result),
        },
      });

      this.client.realtime.send("response.create");
    });

    // Handle interruptions (user starts speaking while AI is talking)
    this.client.on("input_audio_buffer.speech_started", () => {
      this.ws.send(JSON.stringify({ type: "user_speaking" }));
      // The API automatically cancels the current response!
    });
  }

  // Forward audio from browser to OpenAI
  sendAudio(audioBuffer: Buffer) {
    this.client.appendInputAudio(audioBuffer);
  }

  async executeTool(name: string, args: Record<string, unknown>): Promise<unknown> {
    if (name === "search_knowledge_base") {
      return await searchKnowledgeBase(args.query as string);
    }
    throw new Error(`Unknown tool: ${name}`);
  }

  async end() {
    this.client.disconnect();
  }
}

// BILLING NOTE: Realtime API charges per MINUTE of audio
// Input audio: $0.06/min, Output audio: $0.24/min
// A 5-minute conversation = ~$1.50 (expensive! — use only when voice UX is essential)
```

**Frontend (React) for Realtime voice:**

```typescript
// React hook for voice UI
function useVoiceChat() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [aiSpeaking, setAISpeaking] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  const startSession = async () => {
    // Get microphone access
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,    // Remove echo
        noiseSuppression: true,    // Remove background noise
        sampleRate: 24000,         // 24kHz for Realtime API
        channelCount: 1,           // Mono
      },
    });
    mediaStreamRef.current = stream;

    // Create WebSocket to our server
    wsRef.current = new WebSocket("wss://yourapp.com/voice");

    // Audio context for playback
    audioContextRef.current = new AudioContext({ sampleRate: 24000 });

    // Process microphone audio
    const audioContext = new AudioContext({ sampleRate: 24000 });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (event) => {
      if (!isListening) return;

      const inputData = event.inputBuffer.getChannelData(0);
      // Convert float32 to PCM16
      const pcm16 = float32ToPCM16(inputData);
      wsRef.current?.send(
        JSON.stringify({
          type: "audio_chunk",
          audio: Buffer.from(pcm16.buffer).toString("base64"),
        })
      );
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    // Handle messages from server
    wsRef.current.onmessage = async (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "audio_delta":
          // Play AI audio immediately
          await playAudioChunk(Buffer.from(data.audio, "base64"));
          setAISpeaking(true);
          break;

        case "user_transcript":
          setTranscript((prev) => prev + "\nYou: " + data.text);
          break;

        case "assistant_transcript_delta":
          setTranscript((prev) => prev + data.text);
          break;

        case "user_speaking":
          // AI stopped — user is speaking
          setAISpeaking(false);
          break;
      }
    };

    setIsListening(true);
  };

  const stopSession = () => {
    wsRef.current?.close();
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    setIsListening(false);
  };

  return { isListening, transcript, aiSpeaking, startSession, stopSession };
}

function float32ToPCM16(float32Array: Float32Array): Int16Array {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return int16Array;
}
```

### 5.1.5 — Voice Activity Detection (VAD)

```typescript
// Client-side VAD with @ricky0123/vad-web
import { MicVAD } from "@ricky0123/vad-web";

const vad = await MicVAD.new({
  onSpeechStart: () => {
    console.log("User started speaking");
    setIsUserSpeaking(true);
    // If AI is speaking, interrupt it
    if (aiAudioPlayer.isPlaying()) {
      aiAudioPlayer.stop();
      notifyServerOfInterruption();
    }
  },

  onSpeechEnd: (audio: Float32Array) => {
    console.log("User stopped speaking");
    setIsUserSpeaking(false);
    // Send audio for processing
    sendAudioForProcessing(audio);
  },

  // Tune these for your use case:
  positiveSpeechThreshold: 0.6,   // Confidence to start "speaking"
  negativeSpeechThreshold: 0.35,  // Confidence to stop "speaking"
  minSpeechFrames: 3,             // Min frames before triggering
  preSpeechPadFrames: 1,          // Frames to include before speech
  redemptionFrames: 8,            // Frames of silence before "speech end"
});

await vad.start();
```

---

## 5.2 — Vision & Multimodal

### 5.2.1 — Image Understanding

```typescript
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });

// Analyze an image from URL
async function analyzeImageFromURL(imageUrl: string, instruction: string): Promise<string> {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "url", url: imageUrl },
          },
          { type: "text", text: instruction },
        ],
      },
    ],
  });

  return response.content[0].text;
}

// Analyze uploaded image (base64)
async function analyzeUploadedImage(
  imageBuffer: Buffer,
  mimeType: "image/jpeg" | "image/png" | "image/webp" | "image/gif",
  instruction: string
): Promise<string> {
  const base64 = imageBuffer.toString("base64");

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: mimeType, data: base64 },
          },
          { type: "text", text: instruction },
        ],
      },
    ],
  });

  return response.content[0].text;
}

// PRACTICAL EXAMPLES:

// 1. Invoice extraction from image
async function extractInvoiceFromImage(imageBuffer: Buffer): Promise<Invoice> {
  const extraction = await analyzeUploadedImage(
    imageBuffer,
    "image/jpeg",
    `Extract all invoice data from this image. Return ONLY valid JSON:
{
  "invoiceNumber": "string",
  "vendorName": "string",
  "totalAmount": number,
  "currency": "string",
  "dueDate": "YYYY-MM-DD or null",
  "lineItems": [{"description": "string", "amount": number}],
  "taxAmount": number or null
}

If any field is not visible, set it to null.`
  );

  return InvoiceSchema.parse(JSON.parse(extraction));
}

// 2. Product catalog generation from product photos
async function generateProductDescription(
  productImages: Buffer[]
): Promise<ProductDescription> {
  const content: (Anthropic.ImageBlockParam | Anthropic.TextBlockParam)[] = [
    ...productImages.map((img) => ({
      type: "image" as const,
      source: {
        type: "base64" as const,
        media_type: "image/jpeg" as const,
        data: img.toString("base64"),
      },
    })),
    {
      type: "text" as const,
      text: `Analyze these product images and generate:
1. A compelling product title (max 100 chars)
2. A detailed description (200-300 words)
3. Key features list (5-8 bullet points)
4. Suggested categories (2-3)
5. Target audience

Return as JSON with keys: title, description, features, categories, targetAudience`,
    },
  ];

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1500,
    messages: [{ role: "user", content }],
  });

  return JSON.parse(response.content[0].text);
}

// 3. Document vision — extract tables, forms, handwriting
async function extractFromDocument(pdfImageBuffer: Buffer): Promise<DocumentData> {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/jpeg",
              data: pdfImageBuffer.toString("base64"),
            },
          },
          {
            type: "text",
            text: `Extract ALL content from this document image:
1. All text (including handwritten text)
2. All tables (as JSON arrays)
3. All form fields and their values
4. Any signatures or stamps present

Return as structured JSON.`,
          },
        ],
      },
    ],
  });

  return JSON.parse(response.content[0].text);
}
```

### 5.2.2 — Multi-Image Reasoning

```typescript
// Before/After comparison
async function analyzeDifferences(
  beforeImage: Buffer,
  afterImage: Buffer
): Promise<string> {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: "image/jpeg", data: beforeImage.toString("base64") },
          },
          {
            type: "image",
            source: { type: "base64", media_type: "image/jpeg", data: afterImage.toString("base64") },
          },
          {
            type: "text",
            text: `The first image is BEFORE and the second is AFTER.
Describe all visible differences between these two images.
Be specific about what changed, what was added, and what was removed.`,
          },
        ],
      },
    ],
  });
  return response.content[0].text;
}

// Sequence analysis (e.g., security camera frames)
async function analyzeImageSequence(
  frames: { buffer: Buffer; timestamp: string }[]
): Promise<SequenceAnalysis> {
  const content = [
    ...frames.map((frame, i) => ({
      type: "image" as const,
      source: {
        type: "base64" as const,
        media_type: "image/jpeg" as const,
        data: frame.buffer.toString("base64"),
      },
    })),
    {
      type: "text" as const,
      text: `These ${frames.length} images are frames from a video sequence at timestamps: ${frames.map((f) => f.timestamp).join(", ")}.

Analyze the sequence and describe:
1. What is happening overall
2. Key events in chronological order
3. Any notable changes or movements
4. Any persons or objects of interest

Return as JSON: { "summary": string, "events": [{timestamp, description}], "persons": number, "anomalies": string[] }`,
    },
  ];

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    messages: [{ role: "user", content }],
  });

  return JSON.parse(response.content[0].text);
}
```

### 5.2.3 — Video Understanding

```typescript
import ffmpeg from "fluent-ffmpeg";

// Extract frames from video for analysis
async function extractVideoFrames(
  videoPath: string,
  options: {
    fps?: number;          // Frames per second to extract
    maxFrames?: number;    // Maximum number of frames
    quality?: number;      // JPEG quality (1-100)
  } = {}
): Promise<Buffer[]> {
  const { fps = 1, maxFrames = 50, quality = 50 } = options;
  const frames: Buffer[] = [];
  const outputDir = `/tmp/frames-${Date.now()}`;

  await fs.mkdir(outputDir, { recursive: true });

  await new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions([
        `-vf fps=${fps}`,             // Extract at specified FPS
        `-vframes ${maxFrames}`,       // Max frame count
        `-q:v ${Math.round(31 - quality * 0.3)}`, // Quality (inverted)
      ])
      .output(`${outputDir}/frame-%04d.jpg`)
      .on("end", resolve)
      .on("error", reject)
      .run();
  });

  // Read all frames
  const frameFiles = await fs.readdir(outputDir);
  for (const file of frameFiles.sort()) {
    frames.push(await fs.readFile(`${outputDir}/${file}`));
  }

  // Cleanup
  await fs.rm(outputDir, { recursive: true });

  return frames;
}

// Full video analysis pipeline
async function analyzeVideo(videoPath: string): Promise<VideoAnalysis> {
  // 1. Extract audio for transcription
  const audioPath = `/tmp/audio-${Date.now()}.mp3`;
  await extractAudioFromVideo(videoPath, audioPath);
  const transcript = await transcribeFile(audioPath);

  // 2. Extract key frames
  const frames = await extractVideoFrames(videoPath, { fps: 0.5, maxFrames: 20 });

  // 3. Analyze frames with vision
  const visualAnalysis = await analyzeImageSequence(
    frames.map((f, i) => ({
      buffer: f,
      timestamp: `${Math.round(i * 2)}s`, // Approximate timestamps
    }))
  );

  // 4. Combine audio + visual for comprehensive analysis
  const combinedAnalysis = await callLLM(`
Analyze this video content combining audio transcript and visual analysis.

TRANSCRIPT:
${transcript.text}

VISUAL ANALYSIS:
${JSON.stringify(visualAnalysis, null, 2)}

Provide:
1. Executive summary (2-3 sentences)
2. Key topics discussed
3. Action items mentioned
4. Sentiment and tone
5. Speaker count and roles (if determinable)

Return as JSON.`);

  return JSON.parse(combinedAnalysis);
}
```

### 5.2.4 — Image Generation

```typescript
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// DALL-E 3 generation
async function generateImage(
  prompt: string,
  options: {
    size?: "1024x1024" | "1024x1792" | "1792x1024";
    quality?: "standard" | "hd";
    style?: "vivid" | "natural";
  } = {}
): Promise<string> {
  const { size = "1024x1024", quality = "standard", style = "natural" } = options;

  const response = await openai.images.generate({
    model: "dall-e-3",
    prompt,
    n: 1,
    size,
    quality,
    style,
    response_format: "url",
  });

  return response.data[0].url!;
}

// IMAGE PROMPT ENGINEERING:
// Bad: "a cat"
// Good: "A professional product photo of a sleek black cat sitting on a white marble surface,
//        soft studio lighting, shallow depth of field, 50mm lens, commercial photography style"

// Stable Diffusion via Replicate
async function generateWithStableDiffusion(prompt: string): Promise<string> {
  const response = await fetch("https://api.replicate.com/v1/predictions", {
    method: "POST",
    headers: {
      Authorization: `Token ${process.env.REPLICATE_API_TOKEN}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      version: "stability-ai/sdxl",
      input: {
        prompt,
        negative_prompt: "blur, low quality, distorted, watermark",
        width: 1024,
        height: 1024,
        num_inference_steps: 25,
      },
    }),
  });

  const prediction = await response.json();
  // Poll for completion
  return await pollReplicateResult(prediction.id);
}

// Inpainting — edit specific region of image
async function inpaintImage(
  imageUrl: string,
  maskUrl: string,  // White = edit, Black = keep
  editInstruction: string
): Promise<string> {
  const response = await openai.images.edit({
    image: await urlToFile(imageUrl),
    mask: await urlToFile(maskUrl),
    prompt: editInstruction,
    n: 1,
    size: "1024x1024",
  });

  return response.data[0].url!;
}
```

### 5.2.5 — Multimodal RAG

```typescript
// Embed images alongside text for unified search
async function captionAndEmbedImage(
  imageBuffer: Buffer,
  documentId: string,
  pageNumber: number
): Promise<ImageChunk> {
  // Step 1: Generate detailed caption for search
  const caption = await anthropic.messages.create({
    model: "claude-haiku-4",  // Cheap model for captioning
    max_tokens: 300,
    messages: [{
      role: "user",
      content: [
        {
          type: "image",
          source: { type: "base64", media_type: "image/jpeg", data: imageBuffer.toString("base64") },
        },
        {
          type: "text",
          text: `Describe this image/chart/figure comprehensively for a search index.
Include:
- All visible text, numbers, and labels
- What type of visualization it is (chart, diagram, photo, etc.)
- What information it conveys
- Key data points if it's a chart/graph
Be specific and detailed.`,
        },
      ],
    }],
  });

  const captionText = caption.content[0].text;

  // Step 2: Embed the caption (text embedding captures meaning)
  const embedding = await embedText(captionText);

  return {
    id: `${documentId}-img-p${pageNumber}`,
    documentId,
    pageNumber,
    caption: captionText,
    imageData: imageBuffer.toString("base64"),
    embedding,
    contentType: "image",
  };
}

// Search across text AND images
async function multimodalSearch(
  query: string,
  tenantId: string
): Promise<(TextChunk | ImageChunk)[]> {
  const queryEmbedding = await embedText(query);

  // Search both text and image chunks simultaneously
  const [textResults, imageResults] = await Promise.all([
    vectorDB.search(queryEmbedding, tenantId, 5, { contentType: "text" }),
    vectorDB.search(queryEmbedding, tenantId, 3, { contentType: "image" }),
  ]);

  // Merge and re-rank
  const combined = [...textResults, ...imageResults].sort(
    (a, b) => b.similarity - a.similarity
  );

  return combined.slice(0, 5);
}

// Answer using both text and image context
async function multimodalRAGAnswer(
  query: string,
  results: (TextChunk | ImageChunk)[]
): Promise<string> {
  const textChunks = results.filter((r) => r.contentType === "text");
  const imageChunks = results.filter((r) => r.contentType === "image") as ImageChunk[];

  const content: (Anthropic.ImageBlockParam | Anthropic.TextBlockParam)[] = [
    // Include relevant images
    ...imageChunks.map((chunk) => ({
      type: "image" as const,
      source: {
        type: "base64" as const,
        media_type: "image/jpeg" as const,
        data: chunk.imageData,
      },
    })),
    // Include text context and question
    {
      type: "text" as const,
      text: `Answer this question using the provided context (text and images above).

Text context:
${textChunks.map((c, i) => `[Source ${i+1}]: ${c.content}`).join("\n\n")}

Question: ${query}

Answer concisely using evidence from both the text and images. Cite sources.`,
    },
  ];

  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [{ role: "user", content }],
  });

  return response.content[0].text;
}
```

---

## 5.3 — Multi-Agent Systems

### 5.3.1 — Orchestrator + Worker Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│              ORCHESTRATOR + WORKER PATTERN                         │
│                                                                     │
│                    [ORCHESTRATOR]                                   │
│                    "Manager Agent"                                  │
│                         │                                           │
│          ┌──────────────┼──────────────┐                           │
│          │              │              │                            │
│          ▼              ▼              ▼                            │
│     [RESEARCHER]   [DATA ANALYST]   [WRITER]                       │
│     Worker Agent   Worker Agent     Worker Agent                    │
│          │              │              │                            │
│          └──────────────┼──────────────┘                           │
│                         │                                           │
│                    [SYNTHESIZER]                                    │
│                    "Editor Agent"                                   │
│                         │                                           │
│                    [FINAL OUTPUT]                                   │
│                                                                     │
│  Key: Each agent has a specific ROLE and GOAL.                     │
│  Orchestrator decides which workers to call and when.              │
│  Workers focus on their specialty.                                  │
└─────────────────────────────────────────────────────────────────────┘
```

```typescript
// Full multi-agent content agency
class ContentAgency {
  private agents: Map<string, AgentRole>;

  constructor() {
    this.agents = new Map([
      ["researcher", {
        name: "Research Director",
        role: "Senior Research Analyst with 15 years experience",
        goal: "Find accurate, well-sourced information on the given topic",
        backstory: "You specialize in verifying facts and finding credible sources. You never make claims without evidence.",
        tools: [searchWebTool, fetchPageTool, searchAcademicTool],
      }],
      ["writer", {
        name: "Content Writer",
        role: "Expert Content Writer for technical audiences",
        goal: "Transform research into engaging, accurate content",
        backstory: "You've written for TechCrunch, Wired, and MIT Technology Review. You make complex topics accessible.",
        tools: [grammarCheckTool, readabilityTool],
      }],
      ["seo_specialist", {
        name: "SEO Specialist",
        role: "Senior SEO Strategist",
        goal: "Optimize content for search engines without compromising quality",
        backstory: "You've ranked thousands of articles on page 1. You understand both technical SEO and content strategy.",
        tools: [keywordResearchTool, seoAnalysisTool],
      }],
      ["critic", {
        name: "Quality Editor",
        role: "Demanding Senior Editor",
        goal: "Ensure content is accurate, clear, engaging, and meets high standards",
        backstory: "You've edited for top publications. You reject mediocre work and demand excellence.",
        tools: [factCheckTool, plagiarismCheckTool],
      }],
    ]);
  }

  async createContent(topic: string, targetAudience: string): Promise<ContentPiece> {
    // Step 1: Research
    const research = await this.runAgent("researcher", `
Research this topic thoroughly: "${topic}"
Target audience: ${targetAudience}

Gather:
1. Key facts and statistics (with sources)
2. Expert opinions
3. Recent developments (2024-2025)
4. Common misconceptions to address
5. Unique angles not commonly covered`);

    // Step 2: SEO Research (parallel with research is fine)
    const seoData = await this.runAgent("seo_specialist", `
Identify the best SEO strategy for: "${topic}"

Provide:
1. Primary keyword
2. 5-8 secondary keywords
3. Questions to answer (from People Also Ask)
4. Optimal content length
5. Meta description suggestion`);

    // Step 3: Write
    let draft = await this.runAgent("writer", `
Write a comprehensive article about: "${topic}"

Research provided:
${research}

SEO guidelines:
${seoData}

Requirements:
- Target audience: ${targetAudience}
- Length: 1500-2000 words
- Format: Introduction, 4-5 main sections, Conclusion
- Include the key questions identified in SEO research
- Cite all facts from research`);

    // Step 4: Critic review (may loop multiple times)
    let revisionCount = 0;
    while (revisionCount < 3) {
      const review = await this.runAgent("critic", `
Review this article critically:
${draft}

Original research: ${research}

Rate 1-10 on:
- Accuracy (are all facts correct and sourced?)
- Clarity (is it understandable for ${targetAudience}?)
- Engagement (will readers stay until the end?)
- Completeness (does it fully cover the topic?)

If overall score < 8, list specific issues to fix.
If overall score >= 8, respond with "APPROVED".`);

      if (review.includes("APPROVED")) break;

      // Revise based on feedback
      draft = await this.runAgent("writer", `
Revise this article based on editor feedback:

Current draft:
${draft}

Editor feedback:
${review}

Fix all issues while maintaining the original research accuracy.`);

      revisionCount++;
    }

    return { topic, draft, research, seoData, revisionCount };
  }

  private async runAgent(agentName: string, task: string): Promise<string> {
    const agent = this.agents.get(agentName);
    if (!agent) throw new Error(`Unknown agent: ${agentName}`);

    return await callLLMWithPersona(agent, task);
  }
}
```

### 5.3.2 — LangGraph for Multi-Agent

```typescript
// Multi-agent content creation with LangGraph
import { StateGraph, END, Send } from "@langchain/langgraph";

const ContentState = Annotation.Root({
  topic: Annotation<string>(),
  audience: Annotation<string>(),
  researchFindings: Annotation<string[]>({
    reducer: (prev, curr) => [...(prev ?? []), ...(curr ?? [])],
    default: () => [],
  }),
  draft: Annotation<string>({ default: () => "" }),
  editorFeedback: Annotation<string[]>({
    reducer: (prev, curr) => [...(prev ?? []), ...(curr ?? [])],
    default: () => [],
  }),
  revisionCount: Annotation<number>({
    reducer: (prev, curr) => (prev ?? 0) + (curr ?? 0),
    default: () => 0,
  }),
  approved: Annotation<boolean>({ default: () => false }),
  finalContent: Annotation<string>({ default: () => "" }),
});

// Parallel research phase — multiple agents search simultaneously
function distributeResearchTasks(state: typeof ContentState.State): Send[] {
  const researchTasks = [
    { query: `${state.topic} latest research 2025`, aspect: "recent_developments" },
    { query: `${state.topic} statistics data`, aspect: "statistics" },
    { query: `${state.topic} expert opinion`, aspect: "expert_views" },
    { query: `${state.topic} common misconceptions`, aspect: "misconceptions" },
  ];

  // Each task runs in a separate parallel branch!
  return researchTasks.map((task) =>
    new Send("research_subtask", { ...state, currentResearchTask: task })
  );
}

// Multi-agent graph
const multiAgentWorkflow = new StateGraph(ContentState)
  .addNode("orchestrate", orchestrateResearch)
  .addNode("research_subtask", executeResearchTask)  // Runs in parallel
  .addNode("aggregate_research", aggregateAllResearch)
  .addNode("write_draft", writeDraft)
  .addNode("review_draft", reviewDraft)
  .addNode("revise_draft", reviseDraft)

  .addEdge(START, "orchestrate")
  .addConditionalEdges("orchestrate", distributeResearchTasks)  // PARALLEL!
  .addEdge("research_subtask", "aggregate_research")
  .addEdge("aggregate_research", "write_draft")
  .addEdge("write_draft", "review_draft")
  .addConditionalEdges("review_draft", decideNextStep, {
    approve: END,
    revise: "revise_draft",
  })
  .addEdge("revise_draft", "review_draft");  // Loop until approved

const contentSystem = multiAgentWorkflow.compile();
```

---

## 5.4 — Memory Systems

### 5.4.1 — Mem0 — Production Memory Layer

```typescript
import { MemoryClient } from "mem0ai";

const memory = new MemoryClient({ apiKey: process.env.MEM0_API_KEY! });

class PersonalizedAIAssistant {
  // After each conversation, extract and store memories
  async saveConversationMemories(
    userId: string,
    conversation: { role: string; content: string }[]
  ): Promise<void> {
    await memory.add(conversation, {
      userId,
      metadata: {
        timestamp: new Date().toISOString(),
        platform: "web",
      },
    });
    // Mem0 automatically:
    // 1. Extracts facts ("User prefers Python over JavaScript")
    // 2. Stores them semantically
    // 3. Handles contradictions ("User now prefers TypeScript" overwrites old fact)
    // 4. Merges related facts
  }

  // Build personalized context before each response
  async buildPersonalizedPrompt(
    userId: string,
    currentQuery: string
  ): Promise<string> {
    const memories = await memory.search(currentQuery, {
      userId,
      limit: 10,
    });

    if (memories.results.length === 0) {
      return ""; // No memories yet
    }

    return `
WHAT I KNOW ABOUT THIS USER (from past interactions):
${memories.results.map((m: any, i: number) => `${i + 1}. ${m.memory}`).join("\n")}

Use this context to personalize your response.
Don't mention that you "remember" these things — just naturally incorporate them.
    `.trim();
  }

  // Get all memories for a user (for settings page)
  async getAllMemories(userId: string): Promise<Memory[]> {
    const result = await memory.getAll({ userId });
    return result.results;
  }

  // Delete a specific memory (GDPR, user request)
  async deleteMemory(memoryId: string): Promise<void> {
    await memory.delete(memoryId);
  }

  // Delete ALL memories for a user (account deletion)
  async deleteAllUserMemories(userId: string): Promise<void> {
    await memory.deleteAll({ userId });
  }
}
```

### 5.4.2 — Custom Memory Implementation

```typescript
// Build your own memory system when Mem0 isn't suitable

interface Memory {
  id: string;
  userId: string;
  content: string;        // The fact or preference
  category: "preference" | "fact" | "goal" | "context";
  confidence: number;     // 0-1
  source: string;         // "conversation" | "profile" | "behavior"
  embedding: number[];    // For semantic search
  createdAt: Date;
  lastReinforcedAt: Date;
  reinforcementCount: number;
  isActive: boolean;
}

class CustomMemorySystem {
  // Extract memories from conversation using LLM
  async extractMemories(
    userId: string,
    conversation: { role: string; content: string }[]
  ): Promise<void> {
    const conversationText = conversation
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");

    const extraction = await callLLM(`
Extract factual information about the user from this conversation.
Only extract CONCRETE facts, preferences, or goals — not opinions or temporary states.

Conversation:
${conversationText}

Return JSON array:
[
  {
    "content": "User prefers Python for backend development",
    "category": "preference",
    "confidence": 0.9
  },
  ...
]

Return empty array if no notable facts found.`);

    const extracted: { content: string; category: string; confidence: number }[] =
      JSON.parse(extraction);

    for (const item of extracted) {
      await this.storeMemory(userId, item.content, item.category as any, item.confidence);
    }
  }

  async storeMemory(
    userId: string,
    content: string,
    category: Memory["category"],
    confidence: number
  ): Promise<void> {
    const embedding = await embedText(content);

    // Check if similar memory exists
    const similar = await vectorDB.search(embedding, userId, 1, {
      contentType: "memory",
    });

    if (similar.length > 0 && similar[0].similarity > 0.9) {
      // Update existing memory
      await db.memories.update({
        where: { id: similar[0].id },
        data: {
          reinforcementCount: { increment: 1 },
          lastReinforcedAt: new Date(),
          confidence: Math.min(1, confidence + 0.1), // Increase confidence
        },
      });
    } else {
      // Create new memory
      await db.memories.create({
        data: {
          userId,
          content,
          category,
          confidence,
          source: "conversation",
          embedding,
          createdAt: new Date(),
          lastReinforcedAt: new Date(),
          reinforcementCount: 1,
          isActive: true,
        },
      });

      // Also store in vector DB for retrieval
      await vectorDB.upsert({
        id: generateId(),
        tenantId: userId,
        content,
        embedding,
        metadata: { category, confidence, contentType: "memory" },
      });
    }
  }

  // Semantic retrieval of relevant memories
  async retrieveRelevantMemories(
    userId: string,
    currentContext: string,
    limit = 5
  ): Promise<Memory[]> {
    const contextEmbedding = await embedText(currentContext);

    const results = await vectorDB.search(contextEmbedding, userId, limit, {
      contentType: "memory",
    });

    // Only return active, high-confidence memories
    return results
      .filter((r) => r.metadata.confidence >= 0.6)
      .map((r) => ({ ...r, content: r.content }));
  }

  // Memory compression — summarize many memories into fewer
  async compressMemories(userId: string): Promise<void> {
    const allMemories = await db.memories.findMany({
      where: { userId, isActive: true },
      orderBy: { category: "asc" },
    });

    if (allMemories.length < 20) return; // Don't compress if few memories

    const grouped = groupBy(allMemories, "category");

    for (const [category, memories] of Object.entries(grouped)) {
      if (memories.length < 5) continue;

      // Compress into summary
      const summary = await callLLM(`
These are memories about a user in the "${category}" category.
Merge them into 2-3 comprehensive statements, keeping the most important information.

Memories:
${memories.map((m) => m.content).join("\n")}

Merged statements (JSON array):`);

      const merged: string[] = JSON.parse(summary);

      // Delete old memories
      await db.memories.updateMany({
        where: { id: { in: memories.map((m) => m.id) } },
        data: { isActive: false },
      });

      // Create compressed memories
      for (const content of merged) {
        await this.storeMemory(userId, content, category as any, 0.8);
      }
    }
  }
}
```

---

## 5.5 — Edge & On-Device AI

### 5.5.1 — Transformers.js (Browser AI)

```typescript
// Run ML models entirely in the browser — no server, no API key, full privacy!
import { pipeline, env } from "@xenova/transformers";

// Configure Transformers.js
env.useBrowserCache = true;       // Cache models in browser
env.allowLocalModels = false;     // Force remote model loading

// Sentiment analysis in browser
async function analyzeSentimentInBrowser(text: string): Promise<SentimentResult> {
  const classifier = await pipeline(
    "sentiment-analysis",
    "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
  );

  const result = await classifier(text);
  return result[0] as SentimentResult;
}

// Named Entity Recognition in browser
async function extractEntitiesInBrowser(text: string): Promise<Entity[]> {
  const ner = await pipeline("ner", "Xenova/bert-base-NER");
  return await ner(text) as Entity[];
}

// Text embeddings in browser (for client-side semantic search)
async function embedInBrowser(text: string): Promise<number[]> {
  const embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );

  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

// Image classification in browser
async function classifyImageInBrowser(imageUrl: string): Promise<ClassificationResult[]> {
  const classifier = await pipeline(
    "image-classification",
    "Xenova/vit-base-patch16-224"
  );

  return await classifier(imageUrl) as ClassificationResult[];
}

// React component with browser AI
function PrivacyFirstSentimentAnalyzer() {
  const [model, setModel] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load model when component mounts
    pipeline("sentiment-analysis").then((pipe) => {
      setModel(pipe);
      setLoading(false);
    });
  }, []);

  const analyze = async (text: string) => {
    if (!model) return;
    const result = await model(text);
    return result;
    // No API call! No data leaves the browser!
  };

  if (loading) return <div>Loading AI model... (one-time download)</div>;

  return <AnalysisUI onAnalyze={analyze} />;
}
```

### 5.5.2 — Ollama (Local Model Serving)

```typescript
import OpenAI from "openai";

// Use Ollama with OpenAI-compatible API
const ollama = new OpenAI({
  baseURL: "http://localhost:11434/v1",
  apiKey: "ollama", // Required but ignored by Ollama
});

// Available models (pull with: ollama pull <model>)
const OLLAMA_MODELS = {
  "llama3.2:3b": {
    size: "2GB",
    ramRequired: "4GB",
    speed: "fast",
    use: "simple tasks, development",
  },
  "llama3.2:8b": {
    size: "5GB",
    ramRequired: "8GB",
    speed: "medium",
    use: "balanced quality/speed",
  },
  "qwen2.5:14b": {
    size: "9GB",
    ramRequired: "16GB",
    speed: "slow",
    use: "complex tasks, best quality",
  },
  "deepseek-coder:6.7b": {
    size: "4GB",
    ramRequired: "8GB",
    speed: "fast",
    use: "code generation",
  },
  "nomic-embed-text": {
    size: "274MB",
    ramRequired: "1GB",
    speed: "very fast",
    use: "embeddings (free!)",
  },
};

// Call local model
async function callLocalModel(
  prompt: string,
  model = "llama3.2:3b"
): Promise<string> {
  const response = await ollama.chat.completions.create({
    model,
    messages: [{ role: "user", content: prompt }],
    stream: false,
  });

  return response.choices[0].message.content || "";
}

// Local embeddings — completely free!
async function embedLocally(text: string): Promise<number[]> {
  const response = await ollama.embeddings.create({
    model: "nomic-embed-text",
    input: text,
  });
  return response.data[0].embedding;
}

// Smart routing: local for sensitive, cloud for complex
async function smartModelRouter(
  query: string,
  hasSensitiveData: boolean
): Promise<string> {
  if (hasSensitiveData) {
    // Never send sensitive data to cloud APIs
    return await callLocalModel(query, "qwen2.5:14b");
  }

  if (isSimpleQuery(query)) {
    // Use cheap cloud model
    return await callCloudLLM(query, "claude-haiku-4");
  }

  // Complex query → best cloud model
  return await callCloudLLM(query, "claude-sonnet-4-20250514");
}
```

### 5.5.3 — Model Quantization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION LEVELS                             │
│                                                                     │
│  FP32 (32-bit float)                                               │
│  ├── Full precision, best quality                                  │
│  ├── Llama 3.2 8B = ~32GB VRAM                                    │
│  └── Use for: training, research                                   │
│                                                                     │
│  FP16 (16-bit float)                                               │
│  ├── Half precision, minimal quality loss                          │
│  ├── Llama 3.2 8B = ~16GB VRAM                                    │
│  └── Use for: production GPU inference                             │
│                                                                     │
│  INT8 (8-bit integer)                                              │
│  ├── Quarter precision, slight quality loss                        │
│  ├── Llama 3.2 8B = ~8GB VRAM                                     │
│  └── Use for: production, limited VRAM                            │
│                                                                     │
│  INT4 / Q4 (4-bit integer) ← THE SWEET SPOT                       │
│  ├── 4x compression from FP32                                      │
│  ├── Llama 3.2 8B = ~4GB VRAM (runs on consumer GPU!)            │
│  ├── ~5% quality loss (barely noticeable)                         │
│  └── Use for: edge deployment, local models                       │
│                                                                     │
│  Naming convention: Q4_K_M                                         │
│  Q4 = 4-bit, K = K-quant method, M = Medium size variant          │
└─────────────────────────────────────────────────────────────────────┘
```

```bash
# Pull quantized models with Ollama
ollama pull llama3.2:3b-instruct-q4_K_M    # 4-bit quantized (recommended)
ollama pull llama3.2:3b-instruct-q8_0      # 8-bit quantized (better quality)
ollama pull llama3.2:3b-instruct-fp16      # Full precision (needs GPU)

# Check model info
ollama show llama3.2:3b --modelfile
```

---

## 5.6 — What Most Developers Miss

### 5.6.1 — Voice UI Interruption Handling

```typescript
// The most common voice UI failure: AI keeps talking after user interrupts

class VoiceSessionManager {
  private isAIGenerating = false;
  private currentAudioBuffer: Buffer[] = [];
  private audioPlayer: AudioPlayer;

  onUserSpeechDetected() {
    if (this.isAIGenerating) {
      // IMMEDIATELY stop AI audio
      this.audioPlayer.stop();
      this.currentAudioBuffer = [];
      this.isAIGenerating = false;

      // Cancel the generation on server
      this.ws.send(JSON.stringify({ type: "interrupt_generation" }));

      console.log("User interrupted AI — stopped generation");
    }
  }

  onAIAudioChunk(chunk: Buffer) {
    if (!this.isAIGenerating) return; // Dropped if user already interrupted

    this.currentAudioBuffer.push(chunk);
    this.audioPlayer.play(chunk); // Play each chunk as it arrives
  }

  // Add small buffer to prevent accidental interruptions
  private interruptionThrottleMs = 500;
  private lastInterruptionTime = 0;

  onUserSpeechStart() {
    const now = Date.now();
    if (now - this.lastInterruptionTime < this.interruptionThrottleMs) return;

    this.lastInterruptionTime = now;
    this.onUserSpeechDetected();
  }
}
```

### 5.6.2 — Multi-Agent Shared Context Store

```typescript
// WRONG: Agents pass messages and lose context
const result1 = await agent1.run(task);
const result2 = await agent2.run(result1); // Only gets agent1's output, not shared state

// RIGHT: Shared state store accessible by all agents
class SharedAgentContext {
  private store: Map<string, unknown> = new Map();
  private events: AgentEvent[] = [];

  // Typed get/set for shared state
  set<T>(key: string, value: T): void {
    this.store.set(key, value);
    this.recordEvent({ type: "state_update", key, value });
  }

  get<T>(key: string): T | undefined {
    return this.store.get(key) as T | undefined;
  }

  // Event log — each agent can see what others did
  recordEvent(event: AgentEvent): void {
    this.events.push({
      ...event,
      timestamp: new Date(),
      agentId: getCurrentAgentId(),
    });
  }

  getRecentEvents(limit = 10): AgentEvent[] {
    return this.events.slice(-limit);
  }

  // Build context summary for agents
  buildContextSummary(): string {
    const recentEvents = this.getRecentEvents();
    const storeSnapshot = Object.fromEntries(this.store);

    return `
SHARED CONTEXT:
${JSON.stringify(storeSnapshot, null, 2)}

RECENT AGENT ACTIONS:
${recentEvents.map((e) => `[${e.agentId}] ${e.type}: ${JSON.stringify(e)}`).join("\n")}
    `.trim();
  }
}

// Each agent gets the shared context
async function runResearcher(context: SharedAgentContext): Promise<void> {
  const results = await performResearch(context.get("topic"));
  context.set("research_results", results);
  context.recordEvent({ type: "research_complete", resultCount: results.length });
}

async function runWriter(context: SharedAgentContext): Promise<void> {
  const research = context.get<ResearchResult[]>("research_results");
  if (!research) throw new Error("Research not found — run researcher first");

  const draft = await writeDraft(research, context.buildContextSummary());
  context.set("draft", draft);
}
```

---

## 🔨 Phase 5 Projects

### Project 1: Voice AI Doctor Assistant

**Architecture:**
```
Browser mic → WebSocket → VAD detection
    ↓
Deepgram real-time transcription
    ↓
[LangGraph Agent]
    ├── [Symptom Extraction Node] → Zod schema
    ├── [Knowledge Base Node] → RAG over medical docs
    └── [Safety Check Node] → Is this an emergency?
    ↓
ElevenLabs TTS → Audio back to browser
```

**Key features:**
- Real-time transcription shown in UI
- Structured symptom extraction
- Emergency detection (chest pain → immediately refer to emergency)
- Session summary exported as PDF

### Project 2: AI Content Agency

**Architecture:**
```
Topic input
    ↓
[LangGraph Orchestrator]
    ↓ (parallel via Send)
[Research Agent]  [SEO Agent]  [Competitor Analysis]
    ↓ (all complete)
[Aggregation Node]
    ↓
[Writer Agent]
    ↓
[Critic Agent] ← loops until quality >= 8/10
    ↓
[Final Publisher] → Blog post + metadata
```

**Stand-out twist:** Critic agent uses ACTUAL quality metrics (readability score, fact density, SEO score) not just LLM opinion.

### Project 3: AI Meeting Summarizer

**Architecture:**
```
Upload audio file (or real-time meeting)
    ↓
Deepgram diarization (who said what)
    ↓
Full transcript with speaker labels
    ↓
[LangGraph Processing]
    ├── [Summary Node] → Executive summary
    ├── [Action Items Node] → Assigned to specific people
    ├── [Decisions Node] → Key decisions made
    └── [Follow-ups Node] → Questions to answer
    ↓
Export to Notion / Google Docs / Slack
```

---

## ✅ Master Checklist

Before moving to Phase 6, verify you can:

**Voice AI**
- [ ] Build end-to-end STT → LLM → TTS pipeline
- [ ] Implement real-time streaming transcription with Deepgram
- [ ] Stream TTS audio to browser as it generates
- [ ] Set up OpenAI Realtime API with VAD and tool calling
- [ ] Handle interruptions properly (stop AI audio when user speaks)

**Vision**
- [ ] Analyze images with GPT-4o Vision and Claude
- [ ] Build multi-image comparison (before/after)
- [ ] Extract structured data from document images
- [ ] Implement multimodal RAG (search images and text together)
- [ ] Extract frames from video and analyze sequences

**Multi-Agent**
- [ ] Build an orchestrator + worker multi-agent system
- [ ] Implement parallel agent execution with LangGraph Send
- [ ] Use a shared context store accessible to all agents
- [ ] Add a critic/review agent that rejects low-quality output

**Memory**
- [ ] Integrate Mem0 for persistent cross-session memory
- [ ] Build a custom memory extraction pipeline
- [ ] Implement memory compression for long-term users
- [ ] Add GDPR-compliant memory deletion

**Edge AI**
- [ ] Run inference with Transformers.js in the browser
- [ ] Set up Ollama with at least 3 different models
- [ ] Implement smart routing (local vs cloud based on query type)
- [ ] Explain quantization and when to use Q4 vs Q8

---

*Phase 5 complete. Multimodal AI is the next wave. Developers who can combine voice, vision, multi-agent orchestration, and persistent memory are extremely rare and highly valued.*
