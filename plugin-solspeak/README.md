# @elizaos/plugin-solspeak

Give your ElizaOS agents a voice with SolSpeak TTS.

Uses the **turbo model** optimized for real-time voice agents with paralinguistic tag support.

## Features

- **Fast Generation** - Turbo model for low-latency voice agent responses
- **Voice Cloning** - Clone any voice from audio samples or YouTube
- **Emotion Tags** - Add `[laugh]`, `[sigh]`, `[cough]` and more to make speech expressive
- **Simple Setup** - Just configure your voice and go

## Installation

```bash
pnpm add @elizaos/plugin-solspeak
```

## Quick Start

### 1. Configure Your Character

```typescript
import { solspeakPlugin } from "@elizaos/plugin-solspeak";

const character = {
  name: "MyAgent",
  plugins: [solspeakPlugin],
  settings: {
    SOLSPEAK_URL: "https://mcp.thethirdroom.xyz",  // Your SolSpeak server
    SOLSPEAK_VOICE: "agent-voice",                  // Your cloned voice name
    SOLSPEAK_EXAGGERATION: "0.5",                   // Expressiveness (0-1)
    SOLSPEAK_CFG_WEIGHT: "0.5",                     // Speech pace (0-1)
  },
};
```

### 2. Clone a Voice (One-Time Setup)

Before using the plugin, clone a voice on your SolSpeak server:

```bash
# From YouTube
curl -X POST "https://mcp.thethirdroom.xyz/api/voices/youtube" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "agent-voice",
    "youtube_url": "https://youtube.com/watch?v=...",
    "timestamp": "1:30",
    "duration": 15
  }'
```

### 3. Use in Conversations

The agent responds to TTS commands:

```
User: Say hello world
Agent: [Generates speech] "hello world"

User: TTS: That's hilarious! [laugh] I love it!
Agent: [Generates speech with laughter]

User: Generate TTS of Welcome to the future
Agent: [Generates speech] "Welcome to the future"
```

## Configuration

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `SOLSPEAK_URL` | No | `https://mcp.thethirdroom.xyz` | SolSpeak API endpoint |
| `SOLSPEAK_VOICE` | **Yes** | - | Voice name for cloning |
| `SOLSPEAK_EXAGGERATION` | No | `0.5` | Expressiveness (0.0-1.0) |
| `SOLSPEAK_CFG_WEIGHT` | No | `0.5` | Speech pace (0.0-1.0, lower=slower) |

## Paralinguistic Tags

Embed these tags in your text to add expressiveness:

| Tag | Effect |
|-----|--------|
| `[laugh]` | Laughter |
| `[chuckle]` | Light chuckle |
| `[sigh]` | Sigh |
| `[cough]` | Cough |
| `[gasp]` | Gasp |
| `[groan]` | Groan |
| `[yawn]` | Yawn |
| `[clearing throat]` | Throat clear |

**Example:**
```
"That's amazing! [laugh] I can't believe you did it. [sigh] What a relief."
```

## Action Triggers

The `SPEAK` action responds to these phrases:

- `say ...`
- `speak ...`
- `TTS: ...`
- `generate TTS of ...`
- `create TTS for ...`
- `read aloud ...`
- `text to speech ...`

### Inline Overrides

Override settings per-request:

```
Say "Hello world" voice=different-voice exaggeration=0.8
```

## Programmatic Usage

Use the service directly in your code:

```typescript
import { SolSpeakService } from "@elizaos/plugin-solspeak";

// In an action handler or evaluator
const tts = new SolSpeakService(runtime);

const result = await tts.speak({
  text: "Hello from my agent! [laugh]",
  voice_name: "my-voice",
  exaggeration: 0.6,
});

if (result.status === "success") {
  console.log("Audio URL:", result.download_url);
}
```

### List Available Voices

```typescript
const voices = await tts.listVoices();
console.log("Available voices:", Object.keys(voices.available_voices));
```

## Token Gating (SolSpeak Token Holders)

This plugin can be restricted to SolSpeak token holders only. The token mint address is hardcoded in the plugin source.

```typescript
const character = {
  name: "TokenGatedAgent",
  plugins: [solspeakPlugin],
  settings: {
    SOLSPEAK_VOICE: "agent-voice",

    // Enable token gating (uncomment to require SolSpeak tokens)
    // SOLSPEAK_TOKEN_GATE: "true",
    // SOLSPEAK_TOKEN_MIN_BALANCE: "100",  // Minimum tokens required
  },
};
```

### Token Gate Settings

| Setting | Required | Default | Description |
|---------|----------|---------|-------------|
| `SOLSPEAK_TOKEN_GATE` | No | `false` | Set to `"true"` to enable |
| `SOLSPEAK_TOKEN_MIN_BALANCE` | No | `1` | Minimum token balance |
| `SOLSPEAK_SOLANA_RPC` | No | mainnet | Solana RPC endpoint |

### Programmatic Token Check

```typescript
import { checkTokenAccess, SOLSPEAK_TOKEN_MINT } from "@elizaos/plugin-solspeak";

console.log("SolSpeak token:", SOLSPEAK_TOKEN_MINT);

// Check if user holds tokens
const access = await checkTokenAccess(runtime, "WalletAddress...");
if (!access.allowed) {
  console.log(access.message); // "Insufficient token balance. Have 50, need 100"
}
```

## Voice Cloning Tips

For best results:
- Use **10-15 seconds** of clear speech
- Minimize background noise
- Single speaker only
- Natural speaking pace

## API Endpoints

The plugin uses these SolSpeak REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tts` | POST | Generate speech |
| `/api/voices` | GET | List voices |
| `/api/voices/youtube` | POST | Clone voice from YouTube |
| `/download/{filename}` | GET | Download audio |

## Development

```bash
# Install dependencies
pnpm install

# Build
pnpm build

# Development mode
pnpm dev

# Lint
pnpm lint
```

## Requirements

- Node.js 18+
- ElizaOS core 0.1.0+
- SolSpeak TTS server with turbo model

## License

MIT
