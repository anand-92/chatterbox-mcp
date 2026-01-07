# Chatterbox TTS API Reference

Complete documentation for the Chatterbox TTS server's MCP tools and REST API endpoints.

## Table of Contents

- [Overview](#overview)
- [Server Configuration](#server-configuration)
- [MCP Tools](#mcp-tools)
  - [text_to_speech](#text_to_speech)
  - [generate_conversation](#generate_conversation)
  - [list_voices](#list_voices)
  - [delete_voice](#delete_voice)
  - [clone_voice_from_youtube](#clone_voice_from_youtube)
- [REST API Endpoints](#rest-api-endpoints)
  - [POST /api/tts](#post-apitts)
  - [POST /api/conversation](#post-apiconversation)
  - [GET /api/voices](#get-apivoices)
  - [POST /api/voices/save](#post-apivoicessave)
  - [POST /api/voices/delete](#post-apivoicesdelete)
  - [POST /api/voices/youtube](#post-apivoicesyoutube)
- [Utility Endpoints](#utility-endpoints)
  - [GET /download/{filename}](#get-downloadfilename)
- [Models](#models)
- [Error Handling](#error-handling)

---

## Overview

The Chatterbox TTS server provides two interfaces:

1. **MCP (Model Context Protocol)**: For integration with Claude Code and other MCP-compatible clients
2. **REST API**: For web applications and direct HTTP access

Base URLs:
- MCP Endpoint: `http://<server-ip>:8765/mcp`
- REST API: `http://<server-ip>:8765/api/`
- Web UI: `http://<server-ip>:8765/ui/`

---

## Server Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SERVER_PORT` | `8765` | HTTP server port |
| `OUTPUT_DIR` | `./output` | Directory for generated audio files |
| `VOICES_DIR` | `./voices` | Directory for voice reference files |
| `PUBLIC_URL` | `null` | Public URL when behind a tunnel (e.g., Cloudflare) |
| `_pool_size` | `3` | Number of concurrent model instances |

---

## MCP Tools

### text_to_speech

Generate speech from text using Chatterbox TTS.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to convert to speech. Supports paralinguistic tags with turbo model. |
| `model` | string | No | `"standard"` | Model to use: `"standard"` or `"turbo"` |
| `voice_name` | string | No | `null` | Name of a saved voice for cloning. Takes priority over `voice_audio_base64`. |
| `voice_audio_base64` | string | No | `null` | Base64-encoded WAV audio for one-off voice cloning (5-15 seconds recommended). |
| `exaggeration` | float | No | `0.5` | Expressiveness level (0.0-1.0). Higher = more expressive. |
| `cfg_weight` | float | No | `0.5` | CFG weight (0.0-1.0). Lower = slower, more deliberate speech. |

**Returns:**

```json
{
  "status": "success",
  "filename": "tts_1703123456.wav",
  "download_url": "http://server:8765/download/tts_1703123456.wav",
  "size_bytes": 245760,
  "message": "Use curl to download the file from download_url"
}
```

**Examples:**

```python
# Basic TTS
text_to_speech(text="Hello, world!")

# With voice cloning
text_to_speech(text="Hello from a cloned voice", voice_name="david")

# Turbo with paralinguistic tags
text_to_speech(text="That's hilarious! [laugh]", model="turbo", voice_name="david")

# Expressive speech
text_to_speech(text="This is amazing!", exaggeration=0.8, cfg_weight=0.3)
```

---

### generate_conversation

Generate a multi-voice conversation and return a single combined audio file.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `items` | array | Yes | - | List of dialogue items |
| `output_name` | string | No | Auto-generated | Output filename (without `.wav`) |
| `silence_between` | float | No | `0.4` | Seconds of silence between clips |

**Item Structure:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | - | Dialogue text |
| `voice_name` | string | No | `null` | Speaker's voice |
| `exaggeration` | float | No | `0.5` | Expressiveness (0.0-1.0) |
| `cfg_weight` | float | No | `0.5` | CFG weight (0.0-1.0) |

**Returns:**

```json
{
  "status": "success",
  "filename": "podcast_episode_1.wav",
  "download_url": "http://server:8765/download/podcast_episode_1.wav",
  "size_bytes": 1536000,
  "duration_seconds": 32.5,
  "clips_generated": 5
}
```

**Example:**

```python
generate_conversation(
    items=[
        {"text": "Welcome to the show!", "voice_name": "host", "exaggeration": 0.6},
        {"text": "Thanks for having me.", "voice_name": "guest", "exaggeration": 0.5},
        {"text": "Let's dive in.", "voice_name": "host", "exaggeration": 0.6}
    ],
    output_name="podcast_episode_1",
    silence_between=0.5
)
```

---

### list_voices

List all saved voices available for voice cloning.

**Parameters:** None

**Returns:**

```json
{
  "voices_directory": "C:\\Users\\tazzo\\chatterbox-mcp\\voices",
  "available_voices": {
    "david": {
      "filename": "david.wav",
      "size_bytes": 384000,
      "size_mb": 0.37
    },
    "sarah": {
      "filename": "sarah.wav",
      "size_bytes": 512000,
      "size_mb": 0.49
    }
  },
  "count": 2,
  "usage": "Use voice_name parameter in text_to_speech, e.g., text_to_speech(text='Hello', voice_name='david')"
}
```

---

### delete_voice

Delete a saved voice from the voices directory.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | Name of the voice to delete |

**Returns:**

```json
{
  "status": "success",
  "deleted": "david",
  "message": "Voice 'david' has been deleted"
}
```

---

### clone_voice_from_youtube

Clone a voice directly from a YouTube video.

**Parameters:**

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `name` | string | Yes | - | Name for the voice |
| `youtube_url` | string | Yes | - | YouTube video URL |
| `timestamp` | string | Yes | - | Start timestamp (`MM:SS` or `HH:MM:SS`) |
| `duration` | int | No | `15` | Duration in seconds to extract (10-15 recommended) |

**Returns:**

```json
{
  "status": "success",
  "voice_name": "celebrity",
  "file_path": "C:\\Users\\tazzo\\chatterbox-mcp\\voices\\celebrity.wav",
  "size_bytes": 720000,
  "timestamp": "5:30",
  "duration": 15,
  "usage": "text_to_speech(text='Your text', voice_name='celebrity')"
}
```

**Example:**

```python
clone_voice_from_youtube(
    name="celebrity",
    youtube_url="https://youtube.com/watch?v=abc123",
    timestamp="5:30",
    duration=12
)
```

---

## REST API Endpoints

All REST API endpoints are prefixed with `/api/`. They mirror the MCP tools for web application integration.

### POST /api/tts

Generate speech from text.

**Request Body:**

```json
{
  "text": "Hello, world!",
  "model": "standard",
  "voice_name": "david",
  "voice_audio_base64": null,
  "exaggeration": 0.5,
  "cfg_weight": 0.5
}
```

**Response:** Same as `text_to_speech` MCP tool.

**curl Example:**

```bash
curl -X POST "http://localhost:8765/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "model": "standard"}'
```

---

### POST /api/conversation

Generate a multi-voice conversation.

**Request Body:**

```json
{
  "items": [
    {"text": "Hello!", "voice_name": "alice"},
    {"text": "Hi there!", "voice_name": "bob"}
  ],
  "output_name": "my_conversation",
  "silence_between": 0.4
}
```

**Response:** Same as `generate_conversation` MCP tool.

---

### GET /api/voices

List all available voices.

**Response:** Same as `list_voices` MCP tool.

**curl Example:**

```bash
curl "http://localhost:8765/api/voices"
```

---

### POST /api/voices/save

Save a voice from a URL.

**Request Body:**

```json
{
  "name": "david",
  "audio_url": "https://example.com/voice.wav"
}
```

**Response:**

```json
{
  "status": "success",
  "voice_name": "david",
  "file_path": "/path/to/voices/david.wav",
  "size_bytes": 384000,
  "usage": "text_to_speech(text='Your text', voice_name='david')"
}
```

---

### POST /api/voices/delete

Delete a saved voice.

**Request Body:**

```json
{
  "name": "david"
}
```

**Response:** Same as `delete_voice` MCP tool.

---

### POST /api/voices/youtube

Clone a voice from YouTube.

**Request Body:**

```json
{
  "name": "celebrity",
  "youtube_url": "https://youtube.com/watch?v=abc123",
  "timestamp": "5:30",
  "duration": 15
}
```

**Response:** Same as `clone_voice_from_youtube` MCP tool.

---

## Utility Endpoints

### GET /download/{filename}

Download a generated audio file.

**Parameters:**

| Name | Location | Type | Description |
|------|----------|------|-------------|
| `filename` | path | string | Name of the WAV file to download |

**Response:** WAV audio file (`audio/wav`)

**Example:**

```bash
curl "http://localhost:8765/download/tts_1703123456.wav" -o output.wav
```

---

## Models

### Standard Model

- **Parameters:** 500M
- **Languages:** English only
- **Features:** CFG weight control, exaggeration control, zero-shot voice cloning
- **Best for:** General English TTS with fine-grained control

### Turbo Model

- **Parameters:** 350M
- **Languages:** English only
- **Features:** Paralinguistic tags, lower VRAM usage, fastest generation
- **Requires:** Voice reference (voice cloning mandatory)
- **Best for:** Real-time voice agents, expressive speech with tags

---

## Error Handling

All endpoints return errors in a consistent format:

```json
{
  "status": "error",
  "message": "Detailed error message"
}
```

**Common Error Codes:**

| HTTP Status | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - File or voice not found |
| 500 | Internal Server Error - Generation failed |

**Common Error Messages:**

| Error | Cause | Solution |
|-------|-------|----------|
| `Voice 'X' not found` | Voice name doesn't exist | Use `list_voices` to see available voices |
| `yt-dlp not found` | Missing dependency | Install with `pip install yt-dlp` |
| `ffmpeg not found` | Missing dependency | Install ffmpeg |
| `No items provided` | Empty conversation | Provide at least one item |
| `Must provide audio_url` | Missing voice source | Provide URL or use upload endpoint |

---

## Best Practices

### Voice Cloning

1. Use 10-15 seconds of clean speech for reference audio
2. Minimize background noise in reference clips
3. Cache voice files locally for faster generation

### Performance

1. Use `generate_conversation` for multi-speaker dialogues (server-side stitching is faster)
2. Lower `exaggeration` and higher `cfg_weight` for faster generation
3. The model pool supports 3 concurrent requests by default

### Text Formatting

1. Use proper punctuation for natural pacing
2. Commas create short pauses, periods create longer stops
3. Long text is automatically chunked (~280 chars per chunk)
4. Paralinguistic tags only work with the turbo model
