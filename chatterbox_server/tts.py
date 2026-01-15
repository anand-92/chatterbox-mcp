"""Core TTS generation logic."""

import base64
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Callable, Generator, List, Optional, Tuple

import torch

from .audio import concatenate_audio_tensors, save_audio_to_bytes, split_text_into_chunks
from .config import config
from .models import get_cached_conditionals, get_model, get_model_pool
from .voices import get_voice_transcript


def _load_fish_voice_context(
    audio_prompt_path: Optional[str],
    voice_name: Optional[str],
    voice_text: Optional[str],
) -> Tuple[Optional[bytes], Optional[str]]:
    """Load reference audio and resolve transcript for fish model.

    Returns:
        Tuple of (reference_audio_bytes, voice_text)
    """
    reference_audio = None
    if audio_prompt_path:
        with open(audio_prompt_path, "rb") as f:
            reference_audio = f.read()

    # Auto-load transcript if not provided
    if audio_prompt_path and not voice_text and voice_name:
        saved_transcript = get_voice_transcript(voice_name)
        if saved_transcript:
            voice_text = saved_transcript
            print(f"Auto-loaded transcript for voice '{voice_name}'")
        else:
            raise ValueError(
                f"Fish model voice cloning requires voice_text (transcription of reference audio). "
                f"No saved transcript found for '{voice_name}'. "
                f"Use set_voice_transcript('{voice_name}', 'your transcript') to save one."
            )

    return reference_audio, voice_text


@contextmanager
def acquire_model(model_type: str):
    """Context manager to acquire and release a TTS model.

    Uses the model pool for standard models, direct loading for turbo/f5/fish.
    """
    use_pool = model_type == "standard"
    model_pool = get_model_pool() if use_pool else None
    tts_model = model_pool.get() if use_pool else get_model(model_type)

    try:
        yield tts_model
    finally:
        if use_pool and model_pool is not None:
            model_pool.put(tts_model)


def build_generation_kwargs(
    model_type: str,
    tts_model,
    audio_prompt_path: Optional[str],
    exaggeration: float,
    cfg_weight: float
) -> dict:
    """Build the kwargs dict for model.generate() based on model type."""
    if model_type == "turbo":
        return {
            "audio_prompt_path": audio_prompt_path,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }

    kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
    if audio_prompt_path:
        cached_conds = get_cached_conditionals(tts_model, audio_prompt_path, exaggeration)
        tts_model.conds = cached_conds
    return kwargs


def cleanup_old_outputs() -> dict:
    """Remove output files older than configured max age."""
    if config.OUTPUT_MAX_AGE_HOURS <= 0:
        return {"status": "disabled", "message": "Output cleanup disabled"}

    max_age_seconds = config.OUTPUT_MAX_AGE_HOURS * 3600
    now = time.time()
    removed = 0
    kept = 0
    errors = []

    for file in config.OUTPUT_DIR.glob("*.wav"):
        try:
            file_age = now - file.stat().st_mtime
            if file_age > max_age_seconds:
                file.unlink()
                removed += 1
            else:
                kept += 1
        except Exception as e:
            errors.append(f"{file.name}: {e}")

    if removed > 0:
        print(f"Cleanup: removed {removed} old files, kept {kept}")

    return {
        "status": "success",
        "removed": removed,
        "kept": kept,
        "errors": errors if errors else None
    }


def _resolve_voice(
    voice_name: Optional[str],
    voice_audio_base64: Optional[str]
) -> Tuple[Optional[str], Optional[tempfile.NamedTemporaryFile]]:
    """Resolve voice reference to a file path. Returns (path, temp_file)."""
    audio_prompt_path = None
    temp_file = None

    if voice_name:
        voice_file = config.VOICES_DIR / f"{voice_name}.wav"
        if not voice_file.exists():
            voice_file = config.VOICES_DIR / voice_name
            if not voice_file.exists():
                available = [f.stem for f in config.VOICES_DIR.glob("*.wav")]
                raise ValueError(f"Voice '{voice_name}' not found. Available: {available}")
        audio_prompt_path = str(voice_file)
        print(f"Using voice: {voice_file}")
    elif voice_audio_base64:
        audio_bytes = base64.b64decode(voice_audio_base64)
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.write(audio_bytes)
        temp_file.close()
        audio_prompt_path = temp_file.name

    return audio_prompt_path, temp_file


def generate_tts(
    text: str,
    model: str = "standard",
    voice_name: Optional[str] = None,
    voice_audio_base64: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    return_bytes: bool = False,
    # Fish Speech specific parameters
    voice_text: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
) -> dict | bytes:
    """Generate speech from text (thread-safe via model pool).

    Args:
        progress_callback: Optional callback(current_chunk, total_chunks) for progress updates
        return_bytes: If True, return raw audio bytes instead of dict with download URL
        voice_text: Transcription of reference audio (required for fish model voice cloning)
        temperature: Sampling temperature for fish model (0.1-1.0)
        top_p: Top-p sampling for fish model (0.1-1.0)
        repetition_penalty: Repetition penalty for fish model (0.9-2.0)
    """
    audio_prompt_path, temp_file = _resolve_voice(voice_name, voice_audio_base64)

    if model == "turbo" and not audio_prompt_path:
        raise ValueError("Turbo model requires a voice reference (voice_name or voice_audio_base64)")

    # Load fish model voice context (reference audio + transcript)
    if model == "fish":
        reference_audio, voice_text = _load_fish_voice_context(audio_prompt_path, voice_name, voice_text)

    try:
        with acquire_model(model) as tts_model:
            if model == "fish":
                print(f"Generating with Fish Speech: {text[:50]}...")
                wav = tts_model.generate(
                    text=text,
                    reference_audio=reference_audio,
                    reference_text=voice_text,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=config.FISH_MAX_NEW_TOKENS,
                )
                final_wav = wav
                if progress_callback:
                    progress_callback(1, 1)
            else:
                # Standard/Turbo model generation
                kwargs = build_generation_kwargs(model, tts_model, audio_prompt_path, exaggeration, cfg_weight)
                chunks = split_text_into_chunks(text)
                num_chunks = len(chunks)

                if num_chunks > 1:
                    print(f"Text split into {num_chunks} chunks...")

                audio_tensors = []
                for i, chunk in enumerate(chunks):
                    if num_chunks > 1:
                        print(f"Generating chunk {i + 1}/{num_chunks}: {chunk[:50]}...")
                    wav = tts_model.generate(chunk, **kwargs)
                    audio_tensors.append(wav)
                    if progress_callback:
                        progress_callback(i + 1, num_chunks)

                if num_chunks > 1:
                    print("Concatenating audio chunks...")
                    final_wav = concatenate_audio_tensors(audio_tensors, tts_model.sr)
                else:
                    final_wav = audio_tensors[0]

            audio_bytes = save_audio_to_bytes(final_wav, tts_model.sr)

        if return_bytes:
            return audio_bytes

        timestamp = int(time.time())
        filename = f"tts_{timestamp}.wav"
        output_file = config.OUTPUT_DIR / filename
        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        print(f"Audio saved to: {output_file}")

        return {
            "status": "success",
            "filename": filename,
            "download_url": f"{config.get_base_url()}/download/{filename}",
            "size_bytes": len(audio_bytes),
            "message": "Use curl to download the file from download_url"
        }
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def generate_tts_streaming(
    text: str,
    model: str = "standard",
    voice_name: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    # Fish Speech parameters
    voice_text: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
) -> Generator[Tuple[int, int, bytes], None, None]:
    """Generate speech and yield each chunk as it completes.

    Yields: (chunk_index, total_chunks, audio_bytes) for each chunk
    """
    audio_prompt_path, temp_file = _resolve_voice(voice_name, None)

    if model == "turbo" and not audio_prompt_path:
        raise ValueError("Turbo model requires a voice reference (voice_name)")

    # Load fish model voice context (reference audio + transcript)
    if model == "fish":
        reference_audio, voice_text = _load_fish_voice_context(audio_prompt_path, voice_name, voice_text)

    try:
        with acquire_model(model) as tts_model:
            if model == "fish":
                print(f"Streaming Fish Speech: {text[:50]}...")

                for seg_idx, audio_tensor in tts_model.generate_streaming(
                    text=text,
                    reference_audio=reference_audio,
                    reference_text=voice_text,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=config.FISH_MAX_NEW_TOKENS,
                ):
                    print(f"Streaming fish segment {seg_idx + 1}...")
                    chunk_bytes = save_audio_to_bytes(audio_tensor, tts_model.sr)
                    yield (seg_idx, -1, chunk_bytes)  # -1 for total since fish doesn't know upfront

            else:
                # Standard/Turbo model streaming
                kwargs = build_generation_kwargs(model, tts_model, audio_prompt_path, exaggeration, cfg_weight)
                chunks = split_text_into_chunks(text)
                num_chunks = len(chunks)

                print(f"Streaming TTS: {num_chunks} chunks...")

                for i, chunk in enumerate(chunks):
                    print(f"Streaming chunk {i + 1}/{num_chunks}: {chunk[:50]}...")
                    wav = tts_model.generate(chunk, **kwargs)
                    chunk_bytes = save_audio_to_bytes(wav, tts_model.sr)
                    yield (i, num_chunks, chunk_bytes)
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def _resolve_voice_path(voice_name: Optional[str]) -> Optional[str]:
    """Resolve a voice name to its file path, or None if not found."""
    if not voice_name:
        return None
    voice_file = config.VOICES_DIR / f"{voice_name}.wav"
    return str(voice_file) if voice_file.exists() else None


def _generate_single_clip(
    item: dict,
    sample_rate: int,
    silence_duration: float = 0.0
) -> torch.Tensor:
    """Generate a single TTS clip and return the audio tensor."""
    text = item.get("text", "")
    voice_name = item.get("voice_name")
    model_type = item.get("model", "standard")
    exaggeration = item.get("exaggeration", 0.5)
    cfg_weight = item.get("cfg_weight", 0.5)

    audio_prompt_path = _resolve_voice_path(voice_name)

    if model_type == "turbo" and not audio_prompt_path:
        raise ValueError("Turbo requires voice_name")

    with acquire_model(model_type) as tts_model:
        kwargs = build_generation_kwargs(model_type, tts_model, audio_prompt_path, exaggeration, cfg_weight)
        chunks = split_text_into_chunks(text)

        audio_tensors = [tts_model.generate(chunk, **kwargs) for chunk in chunks]

        if len(audio_tensors) > 1:
            return concatenate_audio_tensors(audio_tensors, sample_rate, silence_duration=silence_duration)
        return audio_tensors[0]


def _get_sample_rate() -> int:
    """Get the sample rate from a pooled model."""
    with acquire_model("standard") as model:
        return model.sr


def generate_batch(items: List[dict]) -> dict:
    """Generate multiple TTS clips sequentially."""
    if not items:
        return {"status": "error", "message": "No items provided"}

    sample_rate = _get_sample_rate()
    results = []
    errors = []

    print(f"Batch processing {len(items)} items...")

    for idx, item in enumerate(items):
        try:
            tensor = _generate_single_clip(item, sample_rate)
            audio_bytes = save_audio_to_bytes(tensor, sample_rate)

            timestamp = int(time.time() * 1000)
            filename = f"tts_{timestamp}.wav"
            output_file = config.OUTPUT_DIR / filename
            with open(output_file, "wb") as f:
                f.write(audio_bytes)

            results.append({
                "index": idx,
                "filename": filename,
                "download_url": f"{config.get_base_url()}/download/{filename}",
                "size_bytes": len(audio_bytes),
            })
            print(f"  Completed item {idx + 1}/{len(items)}")
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
            print(f"  Error on item {idx + 1}: {e}")

    return {
        "status": "success",
        "total": len(items),
        "completed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }


def generate_conversation(
    items: List[dict],
    output_name: Optional[str] = None,
    silence_between: float = 0.4
) -> dict:
    """Generate a multi-voice conversation."""
    if not items:
        return {"status": "error", "message": "No items provided"}

    sample_rate = _get_sample_rate()
    print(f"Generating conversation: {len(items)} clips...")

    audio_tensors = []
    errors = []

    for idx, item in enumerate(items):
        try:
            tensor = _generate_single_clip(item, sample_rate, silence_duration=0.15)
            audio_tensors.append(tensor)
            print(f"  Generated clip {idx + 1}/{len(items)}")
        except Exception as e:
            errors.append({"index": idx, "error": str(e)})
            print(f"  Error on clip {idx + 1}: {e}")

    if errors:
        return {"status": "error", "message": f"Failed to generate {len(errors)} clips", "errors": errors}

    print("Stitching clips together...")
    final_audio = concatenate_audio_tensors(audio_tensors, sample_rate, silence_duration=silence_between)
    audio_bytes = save_audio_to_bytes(final_audio, sample_rate)

    filename = f"{output_name}.wav" if output_name else f"conversation_{int(time.time())}.wav"
    output_file = config.OUTPUT_DIR / filename
    with open(output_file, "wb") as f:
        f.write(audio_bytes)

    duration_seconds = final_audio.shape[-1] / sample_rate
    print(f"Conversation saved: {filename} ({duration_seconds:.1f}s)")

    return {
        "status": "success",
        "filename": filename,
        "download_url": f"{config.get_base_url()}/download/{filename}",
        "size_bytes": len(audio_bytes),
        "duration_seconds": round(duration_seconds, 2),
        "clips_generated": len(items)
    }
