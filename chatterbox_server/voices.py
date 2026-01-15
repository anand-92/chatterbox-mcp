"""Voice management functions."""

import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from .config import config


# ============================================================================
# Voice Transcript Management
# ============================================================================

def _get_transcripts_file() -> Path:
    """Get path to the voice transcripts JSON file."""
    return config.VOICES_DIR / "voice_transcripts.json"


def _load_transcripts() -> dict:
    """Load voice transcripts from disk."""
    transcripts_file = _get_transcripts_file()
    if transcripts_file.exists():
        try:
            with open(transcripts_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_transcripts(transcripts: dict) -> None:
    """Save voice transcripts to disk."""
    transcripts_file = _get_transcripts_file()
    with open(transcripts_file, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)


def get_voice_transcript(voice_name: str) -> Optional[str]:
    """Get the transcript for a voice."""
    transcripts = _load_transcripts()
    return transcripts.get(voice_name)


def set_voice_transcript(voice_name: str, transcript: str) -> dict:
    """Set or update the transcript for a voice."""
    # Check if voice exists
    voice_file = config.VOICES_DIR / f"{voice_name}.wav"
    if not voice_file.exists():
        available = [f.stem for f in config.VOICES_DIR.glob("*.wav")]
        raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")

    transcripts = _load_transcripts()
    transcripts[voice_name] = transcript
    _save_transcripts(transcripts)

    return {
        "status": "success",
        "voice_name": voice_name,
        "transcript": transcript,
        "message": f"Transcript saved for voice '{voice_name}'"
    }


def delete_voice_transcript(voice_name: str) -> dict:
    """Delete the transcript for a voice."""
    transcripts = _load_transcripts()
    if voice_name in transcripts:
        del transcripts[voice_name]
        _save_transcripts(transcripts)
        return {
            "status": "success",
            "voice_name": voice_name,
            "message": f"Transcript deleted for voice '{voice_name}'"
        }
    return {
        "status": "not_found",
        "voice_name": voice_name,
        "message": f"No transcript found for voice '{voice_name}'"
    }


# ============================================================================
# Voice Management
# ============================================================================


def list_voices() -> dict:
    """List all saved voices available for voice cloning."""
    transcripts = _load_transcripts()
    voices = {}
    for voice_file in config.VOICES_DIR.glob("*.wav"):
        voice_name = voice_file.stem
        stat = voice_file.stat()
        voices[voice_name] = {
            "filename": voice_file.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "has_transcript": voice_name in transcripts,
            "transcript": transcripts.get(voice_name),
        }
    return {
        "voices_directory": str(config.VOICES_DIR),
        "available_voices": voices,
        "count": len(voices),
        "voices_with_transcripts": sum(1 for v in voices.values() if v["has_transcript"]),
        "usage": "Use voice_name parameter in text_to_speech. For fish model, transcript is auto-loaded if saved."
    }


def save_voice(name: str, audio_url: Optional[str] = None) -> dict:
    """Save a voice reference audio for later use."""
    if not audio_url:
        raise ValueError("Must provide audio_url")

    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()
    if not safe_name:
        raise ValueError("Name must contain at least one alphanumeric character")

    output_path = config.VOICES_DIR / f"{safe_name}.wav"

    print(f"Downloading voice from {audio_url}...")
    urllib.request.urlretrieve(audio_url, output_path)

    stat = output_path.stat()
    return {
        "status": "success",
        "voice_name": safe_name,
        "file_path": str(output_path),
        "size_bytes": stat.st_size,
        "usage": f"text_to_speech(text='Your text', voice_name='{safe_name}')"
    }


DELETE_PASSWORD = "7447"


def delete_voice(name: str, password: str) -> dict:
    """Delete a saved voice from the voices directory."""
    if password != DELETE_PASSWORD:
        raise ValueError("Invalid password")

    voice_file = config.VOICES_DIR / f"{name}.wav"
    if not voice_file.exists():
        available = [f.stem for f in config.VOICES_DIR.glob("*.wav")]
        raise ValueError(f"Voice '{name}' not found. Available voices: {available}")

    voice_file.unlink()
    return {
        "status": "success",
        "deleted": name,
        "message": f"Voice '{name}' has been deleted"
    }


def clone_voice_from_youtube(
    name: str,
    youtube_url: str,
    timestamp: str,
    duration: int = 15,
    transcript: Optional[str] = None
) -> dict:
    """Clone a voice directly from a YouTube video."""
    # Check for required tools
    yt_dlp = shutil.which("yt-dlp")
    ffmpeg = shutil.which("ffmpeg")

    # Windows fallback paths
    if not yt_dlp:
        win_yt_dlp = Path(os.path.expanduser(
            "~/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/"
            "LocalCache/local-packages/Python312/Scripts/yt-dlp.exe"
        ))
        if win_yt_dlp.exists():
            yt_dlp = str(win_yt_dlp)

    if not ffmpeg:
        win_ffmpeg = Path(os.path.expanduser(
            "~/AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
        ))
        if win_ffmpeg.exists():
            ffmpeg = str(win_ffmpeg)

    if not yt_dlp:
        raise ValueError("yt-dlp not found. Install with: pip install yt-dlp")
    if not ffmpeg:
        raise ValueError("ffmpeg not found. Please install ffmpeg.")

    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()
    if not safe_name:
        raise ValueError("Name must contain at least one alphanumeric character")

    temp_dir = Path(tempfile.mkdtemp())
    temp_audio = temp_dir / "audio.wav"
    output_path = config.VOICES_DIR / f"{safe_name}.wav"

    try:
        print(f"Downloading audio from {youtube_url}...")
        result = subprocess.run([
            yt_dlp,
            "-x",
            "--audio-format", "wav",
            "-o", str(temp_audio.with_suffix(".%(ext)s")),
            youtube_url
        ], capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise ValueError(f"yt-dlp failed: {result.stderr}")

        downloaded_files = list(temp_dir.glob("audio.*"))
        if not downloaded_files:
            raise ValueError("No audio file downloaded")
        downloaded_audio = downloaded_files[0]

        # Parse timestamp
        parts = timestamp.split(":")
        if len(parts) == 2:
            timestamp_formatted = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
        elif len(parts) == 3:
            timestamp_formatted = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
        else:
            timestamp_formatted = timestamp

        print(f"Extracting {duration}s clip at {timestamp}...")
        result = subprocess.run([
            ffmpeg,
            "-i", str(downloaded_audio),
            "-ss", timestamp_formatted,
            "-t", str(duration),
            "-ar", "24000",
            "-ac", "1",
            "-y",
            str(output_path)
        ], capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise ValueError(f"ffmpeg failed: {result.stderr}")

        stat = output_path.stat()
        print(f"Voice '{safe_name}' created: {stat.st_size} bytes")

        # Save transcript if provided
        if transcript:
            set_voice_transcript(safe_name, transcript)
            print(f"Transcript saved for voice '{safe_name}'")

        has_transcript = transcript is not None
        usage = (
            f"text_to_speech(text='Your text', model='fish', voice_name='{safe_name}')  # transcript auto-loaded"
            if has_transcript
            else f"text_to_speech(text='Your text', voice_name='{safe_name}')"
        )

        return {
            "status": "success",
            "voice_name": safe_name,
            "file_path": str(output_path),
            "size_bytes": stat.st_size,
            "timestamp": timestamp,
            "duration": duration,
            "has_transcript": has_transcript,
            "transcript": transcript,
            "usage": usage,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
