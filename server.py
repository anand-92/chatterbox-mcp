"""
Chatterbox TTS MCP Server

A FastMCP server that exposes Chatterbox TTS capabilities over the network.
Supports text-to-speech generation and voice cloning.

Usage:
    python server.py

Connect from Claude Code on your laptop:
    Add to your MCP config with URL: http://<your-pc-ip>:8765/mcp
"""

import io
import base64
import tempfile
import os
import time
import re
from pathlib import Path
from typing import Optional, Literal, List

import torch
import torchaudio as ta
import numpy as np
from scipy.io import wavfile
from fastmcp import FastMCP

# CUDA optimizations for RTX 5090
if torch.cuda.is_available():
    # Enable TF32 for faster matrix multiplies (minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Use faster cuDNN algorithms
    torch.backends.cudnn.benchmark = True
    print(f"CUDA optimizations enabled for {torch.cuda.get_device_name(0)}")
from fastmcp.utilities.types import Audio
from pydantic import Field
from starlette.responses import FileResponse
from starlette.routing import Route

# Output directory for generated audio
OUTPUT_DIR = Path(r"C:\Users\tazzo\chatterbox-mcp\output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Voice reference directory for voice cloning
VOICES_DIR = Path(r"C:\Users\tazzo\chatterbox-mcp\voices")
VOICES_DIR.mkdir(exist_ok=True)

# Initialize FastMCP server
mcp = FastMCP(
    name="Chatterbox TTS",
    instructions="""
# Chatterbox TTS Server

You have access to a powerful text-to-speech system running on a remote GPU server. Use this to generate high-quality speech audio from text.

## Quick Start

For basic TTS, just call `text_to_speech` with your text:
```
text_to_speech(text="Hello, this is a test of the text to speech system.")
```

The tool returns a download URL. Use curl to save the file:
```bash
curl -s "<download_url>" -o output.wav
```

## Available Models

### 1. turbo (DEFAULT - recommended for most uses)
- Fastest generation, lowest VRAM usage
- English only
- Supports paralinguistic tags for expressive speech:
  - [laugh], [chuckle], [cough], [sigh], [gasp], [groan], [yawn], [clearing throat]
- Example: `text_to_speech(text="That's hilarious! [laugh] I can't believe it.")`

### 2. standard
- Original Chatterbox model, English only
- More control via `exaggeration` (0.0-1.0) and `cfg_weight` (0.0-1.0)
- Higher exaggeration = more expressive/dramatic
- Lower cfg_weight = slower, more deliberate pacing
- Example: `text_to_speech(text="...", model="standard", exaggeration=0.7, cfg_weight=0.3)`

### 3. multilingual
- Supports 23 languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
- Must specify `language` parameter
- Example: `text_to_speech(text="Bonjour, comment allez-vous?", model="multilingual", language="fr")`

## Voice Cloning

### Using Saved Voices (Recommended)
First, save a voice using `save_voice()`, then reference it by name:

```python
# Save a voice (one-time)
save_voice(name="david", audio_url="http://example.com/voice.wav")

# List available voices
list_voices()

# Use the saved voice
text_to_speech(text="Hello, this is my cloned voice", voice_name="david")
```

### Alternative: Base64 Audio
For one-off cloning without saving:

```python
text_to_speech(
    text="This will sound like the reference voice",
    voice_audio_base64="<base64 wav data>"
)
```

## Tips for Best Results

### General Use
- The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
- Punctuation matters: Use proper punctuation for natural pacing. Commas create pauses, periods create stops.
- If the reference speaker has a fast speaking style, lower `cfg_weight` to around 0.3 to improve pacing.

### Voice Cloning
- Best results with 10-15 seconds of clean speech, minimal background noise, consistent volume.
- Ensure the reference clip matches the target language. Otherwise, outputs may inherit the accent of the reference clip's language.
- To mitigate accent transfer in multilingual use, set `cfg_weight` to 0.

### Expressive or Dramatic Speech
- Use lower `cfg_weight` values (e.g., 0.3) and increase `exaggeration` to 0.7 or higher.
- Higher exaggeration speeds up speech; reducing `cfg_weight` compensates with slower, more deliberate pacing.
- Example: `text_to_speech(text="...", exaggeration=0.7, cfg_weight=0.3)`

### Long Text
- Long text is automatically split into chunks and concatenated seamlessly.
- The server handles texts of any length - no manual splitting needed.
- Each chunk generates up to ~30 seconds of audio, then they're joined with natural pauses.

## Workflow Example

When user asks for TTS:
1. Call text_to_speech with appropriate parameters
2. Use curl to download the file from the returned download_url
3. Tell the user where the file was saved

When user wants voice cloning:
1. Ask for or locate the reference audio file
2. Save it as a voice using save_voice() with audio_url or audio_base64
3. Use the voice_name parameter in text_to_speech
4. Download the output with curl
"""
)

# Global model cache (single instance for non-batch use)
_models = {}

# Model pool for concurrent batch processing (each thread gets its own model)
import threading
from queue import Queue
_model_pool = None
_pool_size = 4  # 4 instances - with caching + no watermark, PCIe should be fine

# Voice conditionals cache - avoids expensive CPU librosa preprocessing for each generation
# Key: (voice_path, mtime) -> Value: Conditionals object (GPU tensors)
_voice_conds_cache = {}
_voice_cache_lock = threading.Lock()


class NoOpWatermarker:
    """Dummy watermarker that skips CPU-intensive watermarking."""
    def apply_watermark(self, wav, sample_rate):
        return wav  # Pass through unchanged


def _disable_watermarker(model):
    """Replace watermarker with no-op to skip CPU processing."""
    model.watermarker = NoOpWatermarker()
    return model

# Local cache paths for models (avoids HuggingFace auth)
LOCAL_MODEL_PATHS = {
    "standard": r"C:\Users\tazzo\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\05e904af2b5c7f8e482687a9d7336c5c824467d9",
}

def get_model(model_type: str = "standard"):
    """Lazily load and cache models."""
    if model_type not in _models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_type} model on {device}...")

        if model_type == "standard":
            from chatterbox.tts import ChatterboxTTS
            local_path = LOCAL_MODEL_PATHS.get("standard")
            if local_path:
                model = ChatterboxTTS.from_local(local_path, device=device)
            else:
                model = ChatterboxTTS.from_pretrained(device=device)
            _models[model_type] = _disable_watermarker(model)
        elif model_type == "turbo":
            # Turbo model not cached locally - fall back to standard
            print("Turbo model not available locally, using standard model instead")
            return get_model("standard")
        elif model_type == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            _models[model_type] = _disable_watermarker(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"{model_type} model loaded successfully!")

    return _models[model_type]


def get_cached_conditionals(model, voice_path: str, exaggeration: float = 0.5):
    """
    Get cached voice conditionals, computing them only on first use.

    This eliminates the expensive CPU librosa preprocessing for each generation.
    Cache key includes file mtime so cache invalidates if voice file changes.
    """
    global _voice_conds_cache

    # Get file modification time for cache invalidation
    mtime = os.path.getmtime(voice_path)
    cache_key = (voice_path, mtime, exaggeration)

    with _voice_cache_lock:
        if cache_key in _voice_conds_cache:
            return _voice_conds_cache[cache_key]

    # Not in cache - compute conditionals (CPU-intensive librosa work)
    print(f"Computing voice conditionals for: {voice_path}")
    model.prepare_conditionals(voice_path, exaggeration=exaggeration)
    conds = model.conds

    with _voice_cache_lock:
        _voice_conds_cache[cache_key] = conds
        print(f"Cached conditionals for {voice_path} (cache size: {len(_voice_conds_cache)})")

    return conds


def get_model_pool():
    """Get or initialize the model pool for concurrent batch processing."""
    global _model_pool
    if _model_pool is None:
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_path = LOCAL_MODEL_PATHS.get("standard")

        print(f"Initializing model pool with {_pool_size} instances...")
        _model_pool = Queue()
        for i in range(_pool_size):
            print(f"  Loading model instance {i+1}/{_pool_size}...")
            if local_path:
                model = ChatterboxTTS.from_local(local_path, device=device)
            else:
                model = ChatterboxTTS.from_pretrained(device=device)
            _model_pool.put(_disable_watermarker(model))
        print(f"Model pool ready with {_pool_size} instances!")

    return _model_pool


def save_audio_to_bytes(wav_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to WAV bytes using scipy (avoids torchaudio FFmpeg issues)."""
    temp_path = tempfile.mktemp(suffix=".wav")
    try:
        # Convert to numpy and ensure correct shape for scipy
        audio_np = wav_tensor.squeeze().cpu().numpy()
        # Convert float32 [-1, 1] to int16 for WAV
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(temp_path, sample_rate, audio_int16)
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# Maximum characters per chunk (~250-300 chars gives ~30 sec audio)
MAX_CHUNK_CHARS = 280


def split_text_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split text into chunks based on sentence boundaries.
    Tries to keep chunks under max_chars while respecting sentence structure.
    """
    # Split on sentence boundaries (. ! ?)
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If single sentence is too long, split on secondary boundaries
        if len(sentence) > max_chars:
            # Split on semicolons, colons, or commas
            sub_parts = re.split(r'(?<=[;:,])\s+', sentence)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                if len(current_chunk) + len(part) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {part}".strip() if current_chunk else part
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # If part is still too long, force split by character count
                    if len(part) > max_chars:
                        words = part.split()
                        current_chunk = ""
                        for word in words:
                            if len(current_chunk) + len(word) + 1 <= max_chars:
                                current_chunk = f"{current_chunk} {word}".strip() if current_chunk else word
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = word
                    else:
                        current_chunk = part
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def concatenate_audio_tensors(tensors: List[torch.Tensor], sample_rate: int, silence_duration: float = 0.3) -> torch.Tensor:
    """
    Concatenate multiple audio tensors with short silence between them.
    """
    if not tensors:
        raise ValueError("No audio tensors to concatenate")

    if len(tensors) == 1:
        return tensors[0]

    # Create silence tensor (silence_duration seconds)
    silence_samples = int(sample_rate * silence_duration)
    silence = torch.zeros(1, silence_samples)

    # Concatenate all tensors with silence between
    result_parts = []
    for i, tensor in enumerate(tensors):
        # Ensure tensor is 2D (channels, samples)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        result_parts.append(tensor)
        if i < len(tensors) - 1:  # Don't add silence after last chunk
            result_parts.append(silence)

    return torch.cat(result_parts, dim=1)


@mcp.tool
def text_to_speech(
    text: str = Field(description="The text to convert to speech. Supports paralinguistic tags like [laugh], [cough], [chuckle] with turbo model."),
    model: Literal["standard", "multilingual"] = Field(
        default="standard",
        description="Model to use: 'standard' (English, CFG controls), 'multilingual' (23+ languages)"
    ),
    voice_name: Optional[str] = Field(
        default=None,
        description="Name of a saved voice to clone. Use list_voices() to see available voices. Takes priority over voice_audio_base64."
    ),
    voice_audio_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded WAV audio for voice cloning. Should be 5-15 seconds of clear speech. Prefer using voice_name instead."
    ),
    language: Optional[str] = Field(
        default=None,
        description="Language code for multilingual model (e.g., 'en', 'fr', 'es', 'de', 'zh', 'ja'). Only used with multilingual model."
    ),
    exaggeration: float = Field(
        default=0.5,
        description="Exaggeration level (0.0-1.0). Higher = more expressive. Only for standard/multilingual models."
    ),
    cfg_weight: float = Field(
        default=0.5,
        description="CFG weight (0.0-1.0). Lower = slower, more deliberate speech. Only for standard/multilingual models."
    )
) -> Audio:
    """
    Generate speech from text using Chatterbox TTS.

    Returns a WAV audio file that can be saved locally.

    For voice cloning, provide a voice_name (recommended) or base64-encoded WAV.

    Examples:
    - Basic TTS: text_to_speech(text="Hello, world!")
    - Voice cloning: text_to_speech(text="Hello", voice_name="david")
    - Multilingual: text_to_speech(text="Bonjour!", model="multilingual", language="fr")
    """
    tts_model = get_model(model)

    # Handle voice cloning reference audio
    audio_prompt_path = None
    temp_file = None

    # Priority: voice_name > voice_audio_base64
    if voice_name:
        # Look for voice file in voices directory
        voice_file = VOICES_DIR / f"{voice_name}.wav"
        if not voice_file.exists():
            # Try without extension in case they included it
            voice_file = VOICES_DIR / voice_name
            if not voice_file.exists():
                available = [f.stem for f in VOICES_DIR.glob("*.wav")]
                raise ValueError(f"Voice '{voice_name}' not found. Available voices: {available}")
        audio_prompt_path = str(voice_file)
        print(f"Using voice: {voice_file}")
    elif voice_audio_base64:
        # Decode base64 audio and save to temp file
        audio_bytes = base64.b64decode(voice_audio_base64)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.write(audio_bytes)
        temp_file.close()
        audio_prompt_path = temp_file.name

    try:
        # Build generation kwargs
        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
        if model == "multilingual" and language:
            kwargs["language_id"] = language

        # Use cached conditionals for voice cloning (avoids CPU librosa work)
        if audio_prompt_path:
            cached_conds = get_cached_conditionals(tts_model, audio_prompt_path, exaggeration)
            tts_model.conds = cached_conds
            # DON'T pass audio_prompt_path to generate() - we've already set conds

        # Split text into chunks to avoid 40-second generation limit
        chunks = split_text_into_chunks(text)
        num_chunks = len(chunks)

        if num_chunks > 1:
            print(f"Text split into {num_chunks} chunks for generation...")

        # Generate audio for each chunk
        audio_tensors = []
        for i, chunk in enumerate(chunks):
            if num_chunks > 1:
                print(f"Generating chunk {i + 1}/{num_chunks}: {chunk[:50]}...")
            wav = tts_model.generate(chunk, **kwargs)
            audio_tensors.append(wav)

        # Concatenate all audio chunks
        if num_chunks > 1:
            print("Concatenating audio chunks...")
            final_wav = concatenate_audio_tensors(audio_tensors, tts_model.sr)
        else:
            final_wav = audio_tensors[0]

        # Convert to bytes and save to file
        audio_bytes = save_audio_to_bytes(final_wav, tts_model.sr)

        # Generate unique filename and save
        timestamp = int(time.time())
        filename = f"tts_{timestamp}.wav"
        output_file = OUTPUT_DIR / filename
        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        print(f"Audio saved to: {output_file}")

        # Return download info only (no base64 to avoid token overflow)
        return {
            "status": "success",
            "filename": filename,
            "download_url": f"http://192.168.1.5:8765/download/{filename}",
            "size_bytes": len(audio_bytes),
            "message": "Use curl to download the file from download_url"
        }

    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def _generate_single_pooled(item: dict, model_pool: Queue, sample_rate: int) -> dict:
    """Internal function to generate a single TTS clip using a model from the pool."""
    # Acquire a model from the pool (blocks if none available)
    tts_model = model_pool.get()

    try:
        text = item.get("text", "")
        voice_name = item.get("voice_name")
        exaggeration = item.get("exaggeration", 0.5)
        cfg_weight = item.get("cfg_weight", 0.5)

        audio_prompt_path = None
        if voice_name:
            voice_file = VOICES_DIR / f"{voice_name}.wav"
            if voice_file.exists():
                audio_prompt_path = str(voice_file)

        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }

        # Use cached conditionals for voice cloning (avoids CPU librosa work)
        if audio_prompt_path:
            cached_conds = get_cached_conditionals(tts_model, audio_prompt_path, exaggeration)
            tts_model.conds = cached_conds
            # DON'T pass audio_prompt_path to generate() - we've already set conds

        # Generate audio
        chunks = split_text_into_chunks(text)
        audio_tensors = []
        for chunk in chunks:
            wav = tts_model.generate(chunk, **kwargs)
            audio_tensors.append(wav)

        if len(audio_tensors) > 1:
            final_wav = concatenate_audio_tensors(audio_tensors, sample_rate)
        else:
            final_wav = audio_tensors[0]

        audio_bytes = save_audio_to_bytes(final_wav, sample_rate)

        # Save to file
        timestamp = int(time.time() * 1000)  # Use milliseconds for uniqueness
        filename = f"tts_{timestamp}.wav"
        output_file = OUTPUT_DIR / filename
        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        return {
            "filename": filename,
            "download_url": f"http://192.168.1.5:8765/download/{filename}",
            "size_bytes": len(audio_bytes),
        }
    finally:
        # Always return the model to the pool
        model_pool.put(tts_model)


@mcp.tool
def batch_text_to_speech(
    items: List[dict] = Field(description="List of TTS items. Each item should have: text (required), voice_name (optional), exaggeration (optional, default 0.5), cfg_weight (optional, default 0.5)")
) -> dict:
    """
    Generate multiple TTS clips in parallel for faster batch processing.

    Perfect for conversations or multiple clips. Uses concurrent processing
    to maximize GPU utilization on high-VRAM cards like RTX 5090.

    Each item in the list should be a dict with:
    - text: The text to speak (required)
    - voice_name: Name of voice to use (optional)
    - exaggeration: 0.0-1.0 (optional, default 0.5)
    - cfg_weight: 0.0-1.0 (optional, default 0.5)

    Example:
    batch_text_to_speech(items=[
        {"text": "Hello from Trump", "voice_name": "trump", "exaggeration": 0.85},
        {"text": "Hello from Elon", "voice_name": "elon", "exaggeration": 0.75}
    ])
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not items:
        return {"status": "error", "message": "No items provided"}

    # Initialize model pool (loads multiple model instances on first use)
    model_pool = get_model_pool()

    # Get sample rate from one of the models
    temp_model = model_pool.get()
    sample_rate = temp_model.sr
    model_pool.put(temp_model)

    results = []
    errors = []

    # Use pool size as max workers (each worker gets its own model)
    max_workers = min(len(items), _pool_size)

    print(f"Batch processing {len(items)} items with {max_workers} workers (pool of {_pool_size} models)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_generate_single_pooled, item, model_pool, sample_rate): idx
            for idx, item in enumerate(items)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                result["index"] = idx
                results.append(result)
                print(f"  Completed item {idx + 1}/{len(items)}")
            except Exception as e:
                errors.append({"index": idx, "error": str(e)})
                print(f"  Error on item {idx + 1}: {e}")

    # Sort results by original index
    results.sort(key=lambda x: x["index"])

    return {
        "status": "success",
        "total": len(items),
        "completed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }


@mcp.tool
def list_voices() -> dict:
    """
    List all saved voices available for voice cloning.

    Returns a dictionary with voice names and their file info.
    Use these names with the voice_name parameter in text_to_speech.
    """
    voices = {}
    for voice_file in VOICES_DIR.glob("*.wav"):
        stat = voice_file.stat()
        voices[voice_file.stem] = {
            "filename": voice_file.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
        }
    return {
        "voices_directory": str(VOICES_DIR),
        "available_voices": voices,
        "count": len(voices),
        "usage": "Use voice_name parameter in text_to_speech, e.g., text_to_speech(text='Hello', voice_name='david')"
    }


@mcp.tool
def save_voice(
    name: str = Field(description="Name for the voice (will be saved as name.wav)"),
    audio_url: Optional[str] = Field(default=None, description="URL to download the voice audio from"),
) -> dict:
    """
    Save a voice reference audio for later use in voice cloning.

    For best performance, use the HTTP upload endpoint instead:
        curl -X POST "http://192.168.1.5:8765/upload_voice/name" -F "file=@audio.wav"

    Or provide audio_url to download from a URL.
    The audio should be 5-15 seconds of clear speech.

    Examples:
    - save_voice(name="david", audio_url="http://example.com/voice.wav")
    - Or use curl: curl -X POST "http://192.168.1.5:8765/upload_voice/david" -F "file=@voice.wav"
    """
    import urllib.request

    if not audio_url:
        raise ValueError("Must provide audio_url, or use the HTTP upload endpoint: curl -X POST 'http://192.168.1.5:8765/upload_voice/name' -F 'file=@audio.wav'")

    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()
    if not safe_name:
        raise ValueError("Name must contain at least one alphanumeric character")

    output_path = VOICES_DIR / f"{safe_name}.wav"

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


@mcp.tool
def delete_voice(
    name: str = Field(description="Name of the voice to delete")
) -> dict:
    """
    Delete a saved voice from the voices directory.
    """
    voice_file = VOICES_DIR / f"{name}.wav"
    if not voice_file.exists():
        available = [f.stem for f in VOICES_DIR.glob("*.wav")]
        raise ValueError(f"Voice '{name}' not found. Available voices: {available}")

    voice_file.unlink()
    return {
        "status": "success",
        "deleted": name,
        "message": f"Voice '{name}' has been deleted"
    }


@mcp.tool
def clone_voice_from_youtube(
    name: str = Field(description="Name for the voice (e.g., 'draymond', 'kobe')"),
    youtube_url: str = Field(description="YouTube video URL"),
    timestamp: str = Field(description="Start timestamp in MM:SS or HH:MM:SS format (e.g., '5:10' or '1:23:45')"),
    duration: int = Field(default=15, description="Duration in seconds to extract (default: 15, recommended: 10-15)")
) -> dict:
    """
    Clone a voice directly from a YouTube video.

    Downloads the audio, extracts a clip at the specified timestamp, and saves it as a voice.
    Much faster than manual download/upload workflow.

    Examples:
    - clone_voice_from_youtube(name="draymond", youtube_url="https://youtube.com/watch?v=xxx", timestamp="5:10")
    - clone_voice_from_youtube(name="kobe", youtube_url="https://youtu.be/xxx", timestamp="14:20", duration=12)
    """
    import subprocess
    import shutil

    # Check for required tools (with Windows-specific fallback paths)
    yt_dlp = shutil.which("yt-dlp")
    ffmpeg = shutil.which("ffmpeg")

    # Windows fallback paths
    if not yt_dlp:
        win_yt_dlp = Path(os.path.expanduser("~")) / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/Scripts/yt-dlp.exe"
        if win_yt_dlp.exists():
            yt_dlp = str(win_yt_dlp)
    if not ffmpeg:
        win_ffmpeg = Path(os.path.expanduser("~")) / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
        if win_ffmpeg.exists():
            ffmpeg = str(win_ffmpeg)

    if not yt_dlp:
        raise ValueError("yt-dlp not found. Install with: pip install yt-dlp")
    if not ffmpeg:
        raise ValueError("ffmpeg not found. Please install ffmpeg.")

    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_").lower()
    if not safe_name:
        raise ValueError("Name must contain at least one alphanumeric character")

    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp())
    temp_audio = temp_dir / "audio.wav"
    output_path = VOICES_DIR / f"{safe_name}.wav"

    try:
        # Download audio from YouTube
        print(f"Downloading audio from {youtube_url}...")
        result = subprocess.run([
            yt_dlp,
            "-x",  # Extract audio
            "--audio-format", "wav",
            "-o", str(temp_audio.with_suffix(".%(ext)s")),
            youtube_url
        ], capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise ValueError(f"yt-dlp failed: {result.stderr}")

        # Find the downloaded file (yt-dlp may add extension)
        downloaded_files = list(temp_dir.glob("audio.*"))
        if not downloaded_files:
            raise ValueError("No audio file downloaded")
        downloaded_audio = downloaded_files[0]

        # Parse timestamp to seconds for ffmpeg
        # Normalize timestamp format (add leading zeros if needed)
        parts = timestamp.split(":")
        if len(parts) == 2:
            timestamp_formatted = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
        elif len(parts) == 3:
            timestamp_formatted = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
        else:
            timestamp_formatted = timestamp

        # Extract clip with ffmpeg
        print(f"Extracting {duration}s clip at {timestamp}...")
        result = subprocess.run([
            ffmpeg,
            "-i", str(downloaded_audio),
            "-ss", timestamp_formatted,
            "-t", str(duration),
            "-ar", "24000",  # Sample rate for TTS
            "-ac", "1",      # Mono
            "-y",            # Overwrite
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
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@mcp.tool
def list_supported_languages() -> dict:
    """
    List all languages supported by the multilingual model.

    Returns a dictionary mapping language codes to language names.
    """
    return {
        "ar": "Arabic",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fi": "Finnish",
        "fr": "French",
        "he": "Hebrew",
        "hi": "Hindi",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "ms": "Malay",
        "nl": "Dutch",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ru": "Russian",
        "sv": "Swedish",
        "sw": "Swahili",
        "tr": "Turkish",
        "zh": "Chinese"
    }


@mcp.tool
def list_paralinguistic_tags() -> dict:
    """
    List paralinguistic tags supported by the turbo model.

    These tags can be embedded in text to add expressiveness.
    Example: "That's so funny! [laugh]"
    """
    return {
        "tags": [
            "[laugh]",
            "[chuckle]",
            "[cough]",
            "[sigh]",
            "[gasp]",
            "[groan]",
            "[yawn]",
            "[clearing throat]"
        ],
        "usage": "Embed tags in your text, e.g., 'Hello! [laugh] How are you?'",
        "note": "Only supported by the 'turbo' model"
    }


@mcp.tool
def get_model_info() -> dict:
    """
    Get information about available TTS models and their capabilities.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_available = torch.cuda.is_available()

    return {
        "device": device,
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "models": {
            "turbo": {
                "name": "Chatterbox-Turbo",
                "parameters": "350M",
                "languages": ["English"],
                "features": [
                    "Paralinguistic tags ([laugh], [cough], etc.)",
                    "Lower compute/VRAM requirements",
                    "Optimized for voice agents"
                ],
                "requires_reference_audio": True
            },
            "standard": {
                "name": "Chatterbox",
                "parameters": "500M",
                "languages": ["English"],
                "features": [
                    "CFG weight control",
                    "Exaggeration control",
                    "Zero-shot voice cloning"
                ],
                "requires_reference_audio": False
            },
            "multilingual": {
                "name": "Chatterbox-Multilingual",
                "parameters": "500M",
                "languages": "23+ languages",
                "features": [
                    "Multi-language support",
                    "Zero-shot voice cloning",
                    "CFG and exaggeration controls"
                ],
                "requires_reference_audio": False
            }
        },
        "loaded_models": list(_models.keys())
    }


if __name__ == "__main__":
    import socket

    # Get local IP for display
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    print("=" * 60)
    print("Chatterbox TTS MCP Server")
    print("=" * 60)
    print(f"Local URL:   http://127.0.0.1:8765/mcp")
    print(f"Network URL: http://{local_ip}:8765/mcp")
    print("-" * 60)
    print("To connect from Claude Code on another machine, add to your")
    print("MCP config (settings or .claude/settings.json):")
    print()
    print(f'  "chatterbox": {{')
    print(f'    "type": "url",')
    print(f'    "url": "http://{local_ip}:8765/mcp"')
    print(f'  }}')
    print("=" * 60)

    # Add download endpoint for audio files
    async def download_audio(request):
        filename = request.path_params["filename"]
        file_path = OUTPUT_DIR / filename
        if file_path.exists() and file_path.suffix == ".wav":
            return FileResponse(file_path, media_type="audio/wav", filename=filename)
        from starlette.responses import JSONResponse
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Add upload endpoint for voice cloning (much faster than base64 through MCP)
    async def upload_voice(request):
        from starlette.responses import JSONResponse
        voice_name = request.path_params["voice_name"]

        # Sanitize name
        safe_name = "".join(c for c in voice_name if c.isalnum() or c in "-_").lower()
        if not safe_name:
            return JSONResponse({"error": "Invalid voice name"}, status_code=400)

        # Parse multipart form data
        form = await request.form()
        uploaded_file = form.get("file")
        if not uploaded_file:
            return JSONResponse({"error": "No file uploaded. Use: curl -X POST 'url' -F 'file=@audio.wav'"}, status_code=400)

        # Save the file
        output_path = VOICES_DIR / f"{safe_name}.wav"
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

    # Get the underlying Starlette app and add our route
    from starlette.routing import Route
    download_route = Route("/download/{filename}", download_audio)
    upload_route = Route("/upload_voice/{voice_name}", upload_voice, methods=["POST"])

    print(f"Download URL: http://{local_ip}:8765/download/<filename>")
    print(f"Upload URL:   curl -X POST 'http://{local_ip}:8765/upload_voice/<name>' -F 'file=@audio.wav'")
    print("=" * 60)

    # Run server on all interfaces so it's accessible from other machines
    # stateless_http=True removes session management so connections don't expire
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount

    # Create a custom app that includes MCP, download, and upload routes
    # IMPORTANT: Must pass lifespan from MCP app for proper initialization
    mcp_http_app = mcp.http_app()
    app = Starlette(
        routes=[
            download_route,
            upload_route,
            Mount("/", app=mcp_http_app),
        ],
        lifespan=mcp_http_app.lifespan,
    )

    uvicorn.run(app, host="0.0.0.0", port=8765, ws="wsproto")
