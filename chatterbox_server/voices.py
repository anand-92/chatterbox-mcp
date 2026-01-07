"""Voice management functions."""

import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from .config import config


def list_voices() -> dict:
    """List all saved voices available for voice cloning."""
    voices = {}
    for voice_file in config.VOICES_DIR.glob("*.wav"):
        stat = voice_file.stat()
        voices[voice_file.stem] = {
            "filename": voice_file.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
        }
    return {
        "voices_directory": str(config.VOICES_DIR),
        "available_voices": voices,
        "count": len(voices),
        "usage": "Use voice_name parameter in text_to_speech"
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
    duration: int = 15
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

        return {
            "status": "success",
            "voice_name": safe_name,
            "file_path": str(output_path),
            "size_bytes": stat.st_size,
            "timestamp": timestamp,
            "duration": duration,
            "usage": f"text_to_speech(text='Your text', voice_name='{safe_name}')"
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
