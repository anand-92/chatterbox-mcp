# Chatterbox TTS MCP Server

A FastMCP server that exposes [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) capabilities over the network. Connect from Claude Code on your laptop to generate speech on your GPU-equipped desktop.

## Features

- **Text-to-Speech**: High-quality speech synthesis with multiple models
- **Voice Cloning**: Clone any voice with 5-15 seconds of reference audio
- **Voice Management**: Save, list, and reuse voice references by name
- **Long Text Support**: Auto-chunking for texts longer than 40 seconds
- **Remote Access**: Generate speech from any machine on your network
- **Download URLs**: Fetch generated audio via HTTP

## Models

| Model | Languages | Features |
|-------|-----------|----------|
| `standard` | English | CFG weight, exaggeration control, zero-shot cloning |
| `turbo` | English | Fast generation, paralinguistic tags, requires voice |

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/chatterbox-mcp.git
cd chatterbox-mcp

# Install dependencies
pip install -r requirements.txt

# Install wsproto for websocket support (no deprecation warnings)
pip install wsproto
```

## Configuration

Edit `server.py` to set your paths:

```python
OUTPUT_DIR = Path(r"C:\path\to\output")
VOICES_DIR = Path(r"C:\path\to\voices")
```

## Usage

### Start the server

```bash
python server.py
# or
run_server.bat
```

### Connect from Claude Code

Add to your MCP config (`~/.claude/settings.json` or `.claude/settings.json`):

```json
{
  "mcpServers": {
    "chatterbox": {
      "type": "url",
      "url": "http://<your-pc-ip>:8765/mcp"
    }
  }
}
```

## Tools

### `text_to_speech`
Generate speech from text.

```python
# Basic TTS
text_to_speech(text="Hello, world!")

# With voice cloning
text_to_speech(text="Hello", voice_name="david")

# Turbo with paralinguistic tags
text_to_speech(text="That's hilarious! [laugh]", model="turbo", voice_name="david")

# Expressive/dramatic
text_to_speech(text="...", exaggeration=0.7, cfg_weight=0.3)
```

### `list_voices`
List all saved voice references.

### `save_voice`
Save a voice reference for later use.

```python
save_voice(name="david", audio_url="http://example.com/voice.wav")
```

### `delete_voice`
Delete a saved voice reference.

## Tips

- **Default settings** (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts
- **Fast speakers**: Lower `cfg_weight` to ~0.3 for better pacing
- **Expressive speech**: Use `exaggeration=0.7`, `cfg_weight=0.3`
- **Voice cloning**: Use 5-15 seconds of clean speech, minimal background noise
- **Long text**: Automatically split into chunks - no manual splitting needed

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~4GB VRAM for standard model

## License

MIT
