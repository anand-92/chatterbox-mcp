"""REST API endpoints for web UI."""

import asyncio

from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from . import tts, voices
from .config import config
from .models import get_status


async def api_health(request):
    """Health check endpoint."""
    try:
        status = await asyncio.to_thread(get_status)
        return JSONResponse(status)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_cleanup(request):
    """Cleanup old output files."""
    try:
        result = await asyncio.to_thread(tts.cleanup_old_outputs)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_text_to_speech(request):
    """REST API: Generate speech from text."""
    try:
        data = await request.json()
        result = await asyncio.to_thread(
            tts.generate_tts,
            text=data.get("text", ""),
            model=data.get("model", "standard"),
            voice_name=data.get("voice_name"),
            voice_audio_base64=data.get("voice_audio_base64"),
            language=data.get("language"),
            exaggeration=float(data.get("exaggeration", 0.5)),
            cfg_weight=float(data.get("cfg_weight", 0.5))
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_generate_conversation(request):
    """REST API: Generate multi-voice conversation."""
    try:
        data = await request.json()
        result = await asyncio.to_thread(
            tts.generate_conversation,
            items=data.get("items", []),
            output_name=data.get("output_name"),
            silence_between=float(data.get("silence_between", 0.4))
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_list_voices(request):
    """REST API: List available voices."""
    try:
        result = await asyncio.to_thread(voices.list_voices)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_save_voice(request):
    """REST API: Save a voice from URL."""
    try:
        data = await request.json()
        result = await asyncio.to_thread(
            voices.save_voice,
            name=data.get("name", ""),
            audio_url=data.get("audio_url")
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_delete_voice(request):
    """REST API: Delete a voice."""
    try:
        data = await request.json()
        result = await asyncio.to_thread(voices.delete_voice, name=data.get("name", ""))
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def api_clone_voice_from_youtube(request):
    """REST API: Clone voice from YouTube."""
    try:
        data = await request.json()
        result = await asyncio.to_thread(
            voices.clone_voice_from_youtube,
            name=data.get("name", ""),
            youtube_url=data.get("youtube_url", ""),
            timestamp=data.get("timestamp", "0:00"),
            duration=int(data.get("duration", 15))
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


async def download_audio(request):
    """Serve generated audio files for download."""
    filename = request.path_params["filename"]
    file_path = config.OUTPUT_DIR / filename
    if file_path.exists() and file_path.suffix == ".wav":
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    return JSONResponse({"error": "File not found"}, status_code=404)


async def upload_voice(request):
    """Upload a voice audio file."""
    voice_name = request.path_params["voice_name"]

    safe_name = "".join(c for c in voice_name if c.isalnum() or c in "-_").lower()
    if not safe_name:
        return JSONResponse({"error": "Invalid voice name"}, status_code=400)

    form = await request.form()
    uploaded_file = form.get("file")
    if not uploaded_file:
        return JSONResponse(
            {"error": "No file uploaded. Use: curl -X POST 'url' -F 'file=@audio.wav'"},
            status_code=400
        )

    output_path = config.VOICES_DIR / f"{safe_name}.wav"
    contents = await uploaded_file.read()
    with open(output_path, "wb") as f:
        f.write(contents)

    print(f"Voice '{safe_name}' uploaded: {len(contents)} bytes")
    return JSONResponse({
        "status": "success",
        "voice_name": safe_name,
        "size_bytes": len(contents),
        "usage": f"text_to_speech(text='Your text', voice_name='{safe_name}')"
    })


def get_api_routes():
    """Get all REST API routes."""
    return [
        Route("/api/health", api_health, methods=["GET"]),
        Route("/api/cleanup", api_cleanup, methods=["POST"]),
        Route("/api/tts", api_text_to_speech, methods=["POST"]),
        Route("/api/conversation", api_generate_conversation, methods=["POST"]),
        Route("/api/voices", api_list_voices, methods=["GET"]),
        Route("/api/voices/save", api_save_voice, methods=["POST"]),
        Route("/api/voices/delete", api_delete_voice, methods=["POST"]),
        Route("/api/voices/youtube", api_clone_voice_from_youtube, methods=["POST"]),
        Route("/download/{filename}", download_audio),
        Route("/upload_voice/{voice_name}", upload_voice, methods=["POST"]),
    ]
