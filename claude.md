# Chatterbox TTS MCP Server

A FastMCP server exposing Chatterbox TTS capabilities for remote text-to-speech generation with voice cloning.

## Architecture

- **server.py** - Main MCP server with TTS tools, REST API, and Web UI
- **proxy.py** - STDIO proxy for Claude Desktop to connect to the HTTP server
- **ui/** - Web interface for manual TTS invocation

## Key Components

### Models
- `standard` - English, CFG/exaggeration controls, 500M params
- `multilingual` - 23 languages, 500M params
- `turbo` - Fastest, supports paralinguistic tags (not cached locally)

### Voice System
- Voices stored in `voices/` as WAV files
- Voice conditionals are cached to avoid repeated librosa preprocessing
- Clone from: saved voices, base64 audio, YouTube videos

### Concurrency
- Model pool (`_pool_size=2`) for parallel batch processing
- Thread-safe voice conditionals cache with `_voice_cache_lock`
- Uses `ThreadPoolExecutor` for concurrent TTS generation

## Code Patterns

### Adding New MCP Tools
```python
@mcp.tool
def new_tool(
    param: str = Field(description="Parameter description")
) -> dict:
    """Tool docstring shown to LLM."""
    return {"status": "success", ...}
```

### Adding REST API Endpoints
```python
async def api_new_endpoint(request):
    data = await request.json()
    result = new_tool(**data)
    return JSONResponse(result)

# Add route in __main__:
api_routes.append(Route("/api/new", api_new_endpoint, methods=["POST"]))
```

### Audio Processing
- Use `scipy.io.wavfile` for saving (not torchaudio - FFmpeg compatibility issues)
- Convert float32 [-1,1] to int16 for WAV output
- Auto-chunk long text via `split_text_into_chunks()` (~280 chars per chunk)

## Configuration

Edit these paths in `server.py`:
```python
OUTPUT_DIR = Path(r"C:\Users\tazzo\chatterbox-mcp\output")
VOICES_DIR = Path(r"C:\Users\tazzo\chatterbox-mcp\voices")
SERVER_PORT = 8765
```

## Running

```bash
# Start server (Windows)
python server.py
# or: run_server.bat

# Connect Claude Code
# Add to ~/.claude/settings.json:
{
  "mcpServers": {
    "chatterbox": {
      "type": "http",
      "url": "http://<server-ip>:8765/mcp"
    }
  }
}
```

## Testing Tools

```bash
# Test voice upload
curl -X POST "http://localhost:8765/upload_voice/testvoice" -F "file=@sample.wav"

# Test TTS via REST API
curl -X POST "http://localhost:8765/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "standard"}'

# Download generated audio
curl "http://localhost:8765/download/tts_123456.wav" -o output.wav
```

## Dependencies

- Python 3.10+
- PyTorch with CUDA
- fastmcp, scipy, numpy
- chatterbox-tts (from Resemble AI)
- yt-dlp, ffmpeg (for YouTube voice cloning)
