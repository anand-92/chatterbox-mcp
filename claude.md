# Chatterbox TTS MCP Server

FastMCP server for text-to-speech generation with voice cloning.

## Project Structure

```
chatterbox-mcp/
├── server.py              # Entry point (run this)
├── chatterbox/            # Main package
│   ├── __init__.py
│   ├── config.py          # Configuration (paths, ports, URLs)
│   ├── models.py          # Model loading, caching, pool management
│   ├── audio.py           # Audio processing utilities
│   ├── tts.py             # Core TTS generation logic
│   ├── voices.py          # Voice management (list, save, delete, youtube)
│   ├── mcp_tools.py       # MCP tool definitions
│   ├── api.py             # REST API endpoints
│   └── server.py          # Server setup and startup
├── ui/                    # Web interface
├── voices/                # Voice reference files
└── output/                # Generated audio files
```

## Hardware

- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- Handles 3+ concurrent model instances (~2GB each)

## Models

- `standard` - English, CFG/exaggeration controls, 500M params
- `multilingual` - 23 languages, 500M params

## Configuration

Edit `chatterbox/config.py`:
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

In `chatterbox/mcp_tools.py`:
```python
@mcp.tool
async def new_tool(param: str = Field(description="...")) -> dict:
    """Tool docstring."""
    return await asyncio.to_thread(implementation_func, param)
```

### Adding REST API Endpoints

In `chatterbox/api.py`:
```python
async def api_new_endpoint(request):
    data = await request.json()
    result = await asyncio.to_thread(implementation_func, **data)
    return JSONResponse(result)

# Add to get_api_routes():
Route("/api/new", api_new_endpoint, methods=["POST"]),
```

### Adding Core Logic

In `chatterbox/tts.py` or `chatterbox/voices.py`:
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
# Voice upload
curl -X POST "http://localhost:8765/upload_voice/test" -F "file=@audio.wav"

# TTS
curl -X POST "http://localhost:8765/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "model": "standard"}'

# Download
curl "http://localhost:8765/download/tts_123456.wav" -o out.wav
```

## Dependencies

- Python 3.10+, PyTorch with CUDA
- fastmcp, scipy, numpy, uvicorn, starlette
- chatterbox-tts (Resemble AI)
- yt-dlp, ffmpeg (YouTube cloning)
