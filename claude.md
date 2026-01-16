# Chatterbox TTS MCP Server

FastMCP server for text-to-speech generation with voice cloning.

## Project Structure

```
chatterbox-mcp/
├── server.py              # Entry point (run this)
├── chatterbox_server/     # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration (paths, ports, URLs)
│   ├── models.py          # Model loading, caching, pool management
│   ├── audio.py           # Audio processing utilities
│   ├── tts.py             # Core TTS generation logic
│   ├── voices.py          # Voice management (list, save, delete, youtube, transcripts)
│   ├── mcp_tools.py       # MCP tool definitions
│   ├── api.py             # REST API endpoints
│   └── server.py          # Server setup and startup
├── ui/                    # Web interface
├── voices/                # Voice reference files + voice_transcripts.json
├── checkpoints/           # Fish Speech model checkpoint
└── output/                # Generated audio files
```

## Hardware

- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- Handles 3+ concurrent model instances (~2GB each)

## Models

- `standard` - English, CFG/exaggeration controls, 500M params (Chatterbox)
- `turbo` - English, fast generation, paralinguistic tags, requires voice (Chatterbox)
- `fish` - Fish Speech (OpenAudio S1), multilingual (13 languages), 45+ emotional markers

### Fish Speech Model (OpenAudio S1) - RECOMMENDED

**Languages**: English, Chinese, Japanese, Korean, French, German, Spanish, Arabic, Russian, Dutch, Italian, Polish, Portuguese

**Performance**: ~275 tokens/sec with torch.compile on RTX 5090

#### Emotion Markers (place at START of sentences)

**Basic Emotions**:
`(happy)`, `(sad)`, `(angry)`, `(excited)`, `(calm)`, `(nervous)`, `(confident)`, `(surprised)`, `(satisfied)`, `(delighted)`, `(scared)`, `(worried)`, `(upset)`, `(frustrated)`, `(depressed)`, `(proud)`, `(relaxed)`, `(grateful)`, `(curious)`, `(confused)`, `(joyful)`

**Intense Emotions**:
`(hysterical)`, `(furious)`, `(panicked)`, `(astonished)`, `(disdainful)`, `(scornful)`, `(sarcastic)`, `(sneering)`

**Tone Modifiers**:
- `(whispering)` - soft, secretive
- `(soft tone)` - gentle, calm
- `(shouting)` - loud, energetic
- `(screaming)` - high intensity
- `(in a hurry tone)` - rushed speech

**Paralinguistic Effects**:
`(laughing)`, `(chuckling)`, `(sobbing)`, `(crying loudly)`, `(sighing)`, `(groaning)`, `(panting)`, `(gasping)`, `(yawning)`

**Audience Effects**:
`(crowd laughing)`, `(background laughter)`, `(audience laughing)`

**Manual Laughter**: Use "Ha, ha, ha" or "Hahaha" in text.

#### Examples

```python
# Single emotion
"(excited) We won the championship!"

# Stacked markers
"(happy)(laughing) That's hilarious! Ha ha ha!"

# Changing emotions
"(sad) I can't believe it. (angry) But I won't give up! (confident) I'll come back stronger."

# Whispering
"(whispering) Don't tell anyone, but I know the secret."

# Intense
"(furious)(shouting) How could you do this?!"
```

#### Voice Cloning

**Requires a transcript** of the reference audio. Transcripts are:
- Auto-loaded from `voices/voice_transcripts.json` if previously saved
- Saved via `set_voice_transcript(voice_name, transcript)` MCP tool
- Saved via `POST /api/voices/transcript` API endpoint
- Provided when cloning: `clone_voice_from_youtube(..., transcript="exact words in clip")`

**Streaming**: Fish supports streaming via `/api/tts/stream` (SSE). MCP returns complete audio only.

## Configuration

Edit `chatterbox_server/config.py`:
```python
PORT = 8765
PUBLIC_URL = "https://mcp.thethirdroom.xyz"  # or None for local
POOL_SIZE = 3  # Model instances
```

## Running

```bash
python server.py
```

## Code Patterns

### Adding MCP Tools

In `chatterbox_server/mcp_tools.py`:
```python
@mcp.tool
async def new_tool(param: str = Field(description="...")) -> dict:
    """Tool docstring."""
    return await asyncio.to_thread(implementation_func, param)
```

### Adding REST API Endpoints

In `chatterbox_server/api.py`:
```python
@router.post("/new", summary="New Endpoint")
async def api_new_endpoint(request: NewRequest):
    result = await asyncio.to_thread(implementation_func, **request.dict())
    return result
```

### Adding Core Logic

In `chatterbox_server/tts.py` or `chatterbox_server/voices.py`:
```python
def new_function(param: str) -> dict:
    """Synchronous implementation."""
    # Do work
    return {"status": "success"}
```

## Key Design Decisions

- **Async tools with thread pool**: All MCP/API handlers use `asyncio.to_thread()` for non-blocking I/O
- **Model pool**: Thread-safe queue of model instances for concurrent requests
- **Voice caching**: Conditionals cached by (path, mtime, exaggeration) to avoid librosa preprocessing
- **scipy for audio**: Use scipy.io.wavfile instead of torchaudio (FFmpeg issues)

## Testing

```bash
# Clone voice from YouTube (with transcript for fish support)
curl -X POST "http://localhost:8765/api/voices/youtube" \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "youtube_url": "https://youtube.com/...", "timestamp": "1:30", "transcript": "what they say in the clip"}'

# Set transcript for existing voice
curl -X POST "http://localhost:8765/api/voices/transcript" \
  -H "Content-Type: application/json" \
  -d '{"voice_name": "test", "transcript": "the transcription text"}'

# TTS with standard model
curl -X POST "http://localhost:8765/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "model": "standard"}'

# TTS with fish model (transcript auto-loaded)
curl -X POST "http://localhost:8765/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "(excited) Hello!", "model": "fish", "voice_name": "test"}'

# Streaming TTS (SSE)
curl -X POST "http://localhost:8765/api/tts/stream" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "model": "fish", "voice_name": "test"}'

# Download
curl "http://localhost:8765/download/tts_123456.wav" -o out.wav
```

## Dependencies

- Python 3.10+, PyTorch with CUDA
- fastmcp, scipy, numpy, uvicorn, starlette
- chatterbox-tts (Resemble AI)
- fish-speech (Fish Audio OpenAudio S1)
- yt-dlp, ffmpeg (YouTube cloning)

### Fish Speech Setup

1. Accept license at: https://huggingface.co/fishaudio/openaudio-s1-mini
2. Download model: `huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini`
