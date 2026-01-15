"""REST API endpoints with Swagger documentation."""

import asyncio
import base64
import json
from typing import AsyncGenerator, List, Literal, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from . import tts, voices
from .config import config
from .models import get_status


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class TTSRequest(BaseModel):
    """Request body for text-to-speech generation."""
    text: str = Field(..., description="The text to convert to speech")
    model: Literal["standard", "turbo", "f5", "fish"] = Field(
        default="standard",
        description="Model: 'standard' (default), 'turbo' (fast, requires voice), 'f5' (F5-TTS), or 'fish' (Fish Speech, multilingual)"
    )
    voice_name: Optional[str] = Field(
        default=None,
        description="Name of a saved voice to clone"
    )
    voice_audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded WAV audio for voice cloning (5-15 seconds)"
    )
    exaggeration: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Exaggeration level (0.0-1.0). Higher = more expressive (standard/turbo only)"
    )
    cfg_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="CFG weight (0.0-1.0). Lower = slower, more deliberate speech (standard/turbo only)"
    )
    voice_text: Optional[str] = Field(
        default=None,
        description="Transcription of the reference audio. REQUIRED for fish model voice cloning."
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed (0.5-2.0). Only for F5-TTS model."
    )
    # Fish Speech specific parameters
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Sampling temperature (0.1-1.0). Higher = more varied. (fish only)"
    )
    top_p: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Top-p sampling (0.1-1.0). (fish only)"
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=0.9,
        le=2.0,
        description="Repetition penalty (0.9-2.0). (fish only)"
    )


class TTSResponse(BaseModel):
    """Response from TTS generation."""
    status: str
    filename: str
    download_url: str
    size_bytes: int
    message: Optional[str] = None


class ConversationItem(BaseModel):
    """Single item in a conversation."""
    text: str = Field(..., description="The text for this speaker")
    voice_name: Optional[str] = Field(default=None, description="Voice to use")
    model: Literal["standard", "turbo", "fish"] = Field(default="standard", description="Model to use")
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Fish Speech specific
    voice_text: Optional[str] = Field(default=None, description="Transcription for fish voice cloning")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    top_p: float = Field(default=0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)


class ConversationRequest(BaseModel):
    """Request body for conversation generation."""
    items: List[ConversationItem] = Field(..., description="List of dialogue items")
    output_name: Optional[str] = Field(
        default=None,
        description="Output filename (without .wav)"
    )
    silence_between: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Seconds of silence between clips"
    )


class ConversationResponse(BaseModel):
    """Response from conversation generation."""
    status: str
    filename: str
    download_url: str
    size_bytes: int
    duration_seconds: float
    clips_generated: int


class VoiceSaveRequest(BaseModel):
    """Request body for saving a voice."""
    name: str = Field(..., description="Name for the voice (saved as name.wav)")
    audio_url: Optional[str] = Field(
        default=None,
        description="URL to download the voice audio from"
    )


class VoiceDeleteRequest(BaseModel):
    """Request body for deleting a voice."""
    name: str = Field(..., description="Name of the voice to delete")
    password: str = Field(..., description="Password required for deletion")


class YouTubeCloneRequest(BaseModel):
    """Request body for cloning voice from YouTube."""
    name: str = Field(..., description="Name for the voice")
    youtube_url: str = Field(..., description="YouTube video URL")
    timestamp: str = Field(default="0:00", description="Start timestamp (MM:SS or HH:MM:SS)")
    duration: int = Field(default=15, ge=5, le=30, description="Duration in seconds (10-15 recommended)")
    transcript: Optional[str] = Field(
        default=None,
        description="Transcript of the audio clip. Required for Fish model voice cloning. Saved for future use."
    )


class VoiceTranscriptRequest(BaseModel):
    """Request body for setting a voice transcript."""
    voice_name: str = Field(..., description="Name of the voice")
    transcript: str = Field(..., description="Transcript of the reference audio")


class TTSStreamRequest(BaseModel):
    """Request body for streaming TTS."""
    text: str = Field(..., description="The text to convert to speech")
    model: Literal["standard", "turbo", "fish"] = Field(
        default="standard",
        description="Model: 'standard', 'turbo', or 'fish'"
    )
    voice_name: Optional[str] = Field(
        default=None,
        description="Name of a saved voice to clone"
    )
    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Fish Speech specific
    voice_text: Optional[str] = Field(default=None, description="Transcription for fish voice cloning")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0)
    top_p: float = Field(default=0.8, ge=0.1, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    cuda_available: bool
    cuda_device: Optional[str]
    models_loaded: List[str]
    pool_initialized: bool
    voice_cache_size: int


class ErrorResponse(BaseModel):
    """Error response."""
    status: str = "error"
    message: str


# ============================================================================
# API Router
# ============================================================================

router = APIRouter(prefix="/api", tags=["TTS API"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check server status, CUDA availability, and loaded models"
)
async def api_health():
    """Health check endpoint."""
    try:
        status = await asyncio.to_thread(get_status)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/cleanup",
    summary="Cleanup Old Files",
    description="Remove output files older than the configured max age"
)
async def api_cleanup():
    """Cleanup old output files."""
    try:
        result = await asyncio.to_thread(tts.cleanup_old_outputs)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tts",
    response_model=TTSResponse,
    summary="Text to Speech",
    description="""
Generate speech from text using SolSpeak TTS.

**Models:**
- `standard` - English, best quality, supports voice cloning
- `turbo` - Faster, requires voice_name, supports paralinguistic tags: [laugh], [chuckle], [sigh], [cough], [gasp], [groan], [yawn], [clearing throat]
- `fish` - Fish Speech (OpenAudio S1), multilingual, 45+ emotional markers like (angry), (excited), (laughing)

**Tips:**
- Higher exaggeration = more expressive speech (standard/turbo)
- Lower cfg_weight = slower, more deliberate pacing (standard/turbo)
- For fish model: use emotional markers like (excited), (sad), (laughing) in text
- Fish voice cloning requires both voice_name and voice_text (transcription)
"""
)
async def api_text_to_speech(request: TTSRequest):
    """Generate speech from text."""
    try:
        result = await asyncio.to_thread(
            tts.generate_tts,
            text=request.text,
            model=request.model,
            voice_name=request.voice_name,
            voice_audio_base64=request.voice_audio_base64,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            voice_text=request.voice_text,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/tts/stream",
    summary="Text to Speech (Streaming)",
    description="""
Stream audio chunks as they're generated via Server-Sent Events (SSE).

Each chunk is sent as an SSE event with base64-encoded WAV audio.
Client receives audio progressively instead of waiting for full generation.

**Event format:**
```
event: chunk
data: {"index": 0, "total": 3, "audio_base64": "UklGRi..."}

event: chunk
data: {"index": 1, "total": 3, "audio_base64": "UklGRi..."}

event: done
data: {"total_chunks": 3}
```

**Example client (JavaScript):**
```javascript
const eventSource = new EventSource('/api/tts/stream', {method: 'POST', body: JSON.stringify({text: "..."})});
eventSource.addEventListener('chunk', (e) => {
    const data = JSON.parse(e.data);
    playAudioChunk(atob(data.audio_base64));
});
```
"""
)
async def api_text_to_speech_stream(request: TTSStreamRequest):
    """Stream TTS audio chunks via SSE."""
    event_queue: asyncio.Queue = asyncio.Queue()

    def run_generator_sync(queue_put) -> None:
        """Run the sync generator and push events via callback."""
        try:
            for chunk_data in tts.generate_tts_streaming(
                text=request.text,
                model=request.model,
                voice_name=request.voice_name,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
                voice_text=request.voice_text,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            ):
                queue_put(("chunk", chunk_data))
            queue_put(("done", None))
        except Exception as e:
            queue_put(("error", str(e)))

    async def generate_sse() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()

        def queue_put_threadsafe(item):
            loop.call_soon_threadsafe(event_queue.put_nowait, item)

        loop.run_in_executor(None, run_generator_sync, queue_put_threadsafe)
        total_chunks = 0

        while True:
            event_type, data = await event_queue.get()

            if event_type == "chunk":
                chunk_index, num_chunks, audio_bytes = data
                total_chunks = num_chunks
                payload = json.dumps({
                    "index": chunk_index,
                    "total": num_chunks,
                    "audio_base64": base64.b64encode(audio_bytes).decode('utf-8')
                })
                yield f"event: chunk\ndata: {payload}\n\n"

            elif event_type == "done":
                yield f"event: done\ndata: {json.dumps({'total_chunks': total_chunks})}\n\n"
                break

            elif event_type == "error":
                yield f"event: error\ndata: {json.dumps({'error': data})}\n\n"
                break

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post(
    "/conversation",
    response_model=ConversationResponse,
    summary="Generate Conversation",
    description="""
Generate a multi-voice conversation as a single audio file.

Each item can specify its own voice and model. Perfect for podcasts, dialogues, or multi-speaker content.

**Example:**
```json
{
  "items": [
    {"text": "Welcome!", "voice_name": "host", "model": "turbo"},
    {"text": "Thanks for having me.", "voice_name": "guest", "model": "turbo"}
  ],
  "output_name": "podcast_ep1",
  "silence_between": 0.4
}
```
"""
)
async def api_generate_conversation(request: ConversationRequest):
    """Generate multi-voice conversation."""
    try:
        items = [item.model_dump() for item in request.items]
        result = await asyncio.to_thread(
            tts.generate_conversation,
            items=items,
            output_name=request.output_name,
            silence_between=request.silence_between
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/voices",
    summary="List Voices",
    description="List all saved voices available for voice cloning"
)
async def api_list_voices():
    """List available voices."""
    try:
        result = await asyncio.to_thread(voices.list_voices)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/voices/save",
    summary="Save Voice from URL",
    description="Download and save a voice reference audio from a URL"
)
async def api_save_voice(request: VoiceSaveRequest):
    """Save a voice from URL."""
    try:
        result = await asyncio.to_thread(
            voices.save_voice,
            name=request.name,
            audio_url=request.audio_url
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/voices/delete",
    summary="Delete Voice",
    description="Delete a saved voice from the voices directory. Requires password."
)
async def api_delete_voice(request: VoiceDeleteRequest):
    """Delete a voice."""
    try:
        result = await asyncio.to_thread(voices.delete_voice, name=request.name, password=request.password)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/voices/youtube",
    summary="Clone Voice from YouTube",
    description="""
Clone a voice directly from a YouTube video.

Downloads the audio, extracts a clip at the specified timestamp, and saves it as a voice.

**Tip:** Include `transcript` for Fish model voice cloning - it will be saved for future use!

**Example:**
```json
{
  "name": "celebrity",
  "youtube_url": "https://youtube.com/watch?v=xxx",
  "timestamp": "5:10",
  "duration": 15,
  "transcript": "What the person is saying in the clip"
}
```
"""
)
async def api_clone_voice_from_youtube(request: YouTubeCloneRequest):
    """Clone voice from YouTube."""
    try:
        result = await asyncio.to_thread(
            voices.clone_voice_from_youtube,
            name=request.name,
            youtube_url=request.youtube_url,
            timestamp=request.timestamp,
            duration=request.duration,
            transcript=request.transcript
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/voices/transcript",
    summary="Set Voice Transcript",
    description="""
Set or update the transcript for an existing voice.

The transcript is used for Fish model voice cloning and is automatically loaded when using the voice.

**Example:**
```json
{
  "voice_name": "eminem",
  "transcript": "dj, writer, a producer, does something..."
}
```
"""
)
async def api_set_voice_transcript(request: VoiceTranscriptRequest):
    """Set transcript for a voice."""
    try:
        result = await asyncio.to_thread(
            voices.set_voice_transcript,
            voice_name=request.voice_name,
            transcript=request.transcript
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/voices/{voice_name}/transcript",
    summary="Get Voice Transcript",
    description="Get the saved transcript for a voice"
)
async def api_get_voice_transcript(voice_name: str):
    """Get transcript for a voice."""
    transcript = voices.get_voice_transcript(voice_name)
    if transcript:
        return {"voice_name": voice_name, "transcript": transcript}
    raise HTTPException(status_code=404, detail=f"No transcript found for voice '{voice_name}'")


# ============================================================================
# Non-API Routes (download, upload) - these stay as regular routes
# ============================================================================

def create_api_app() -> FastAPI:
    """Create the FastAPI application with Swagger docs."""
    app = FastAPI(
        title="SolSpeak TTS API",
        description="""
## Text-to-Speech API with Voice Cloning

Generate high-quality speech from text using SolSpeak TTS.

### Features
- **Voice Cloning**: Clone any voice from a 10-15 second audio sample
- **Multiple Models**: Standard (best quality), Turbo (fast + paralinguistic tags), Fish (multilingual + emotions)
- **Conversations**: Generate multi-speaker dialogues as single audio files
- **YouTube Cloning**: Extract voice samples directly from YouTube videos

### Quick Start
1. Clone a voice: `POST /api/voices/youtube` with YouTube URL and timestamp
2. Generate speech: `POST /api/tts` with text and voice_name
3. Download audio from the returned `download_url`

### Models
- **standard**: English, best quality, voice cloning with exaggeration/cfg_weight
- **turbo**: Fast, requires voice, supports paralinguistic tags like [laugh], [sigh]
- **fish**: Fish Speech (OpenAudio S1), multilingual (13 languages), 45+ emotional markers

### Emotional Markers (Fish model)
Embed these markers in parentheses for expressive speech:
`(angry)`, `(sad)`, `(excited)`, `(laughing)`, `(crying)`, `(whispering)`, `(shouting)`, etc.

Example: `"(excited) This is amazing! (laughing) I can't believe it!"`

### Paralinguistic Tags (Turbo model)
Embed these tags in brackets for expressive speech:
`[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, `[gasp]`, `[groan]`, `[yawn]`, `[clearing throat]`

Example: `"That's hilarious! [laugh] I can't believe it."`
""",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    app.include_router(router)

    # Add download endpoint
    @app.get(
        "/download/{filename}",
        tags=["Files"],
        summary="Download Audio",
        description="Download a generated audio file by filename"
    )
    async def download_audio(filename: str = Path(..., description="The audio filename to download")):
        """Serve generated audio files for download."""
        file_path = config.OUTPUT_DIR / filename
        if file_path.exists() and file_path.suffix == ".wav":
            return FileResponse(file_path, media_type="audio/wav", filename=filename)
        raise HTTPException(status_code=404, detail="File not found")

    return app
