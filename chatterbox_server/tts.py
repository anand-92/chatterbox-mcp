"""Core TTS generation logic."""

import base64
import os
import tempfile
import time
from queue import Queue
from typing import List, Optional

import torch

from .audio import concatenate_audio_tensors, save_audio_to_bytes, split_text_into_chunks
from .config import config
from .models import get_cached_conditionals, get_model, get_model_pool


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


def generate_tts(
    text: str,
    model: str = "standard",
    voice_name: Optional[str] = None,
    voice_audio_base64: Optional[str] = None,
    language: Optional[str] = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5
) -> dict:
    """Generate speech from text (thread-safe via model pool)."""
    audio_prompt_path = None
    temp_file = None

    # Handle voice reference
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

    # Turbo requires voice reference
    if model == "turbo" and not audio_prompt_path:
        raise ValueError("Turbo model requires a voice reference (voice_name or voice_audio_base64)")

    # Use model pool for standard, direct load for turbo/multilingual
    use_pool = model == "standard"
    model_pool = None
    tts_model = None

    if use_pool:
        model_pool = get_model_pool()
        tts_model = model_pool.get()
    else:
        tts_model = get_model(model)

    try:
        # Build generation kwargs based on model type
        if model == "turbo":
            # Turbo uses audio_prompt_path directly in generate()
            kwargs = {
                "audio_prompt_path": audio_prompt_path,
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
            }
        else:
            kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
            if model == "multilingual" and language:
                kwargs["language_id"] = language
            # Standard/multilingual use cached conditionals
            if audio_prompt_path:
                cached_conds = get_cached_conditionals(tts_model, audio_prompt_path, exaggeration)
                tts_model.conds = cached_conds

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

        if num_chunks > 1:
            print("Concatenating audio chunks...")
            final_wav = concatenate_audio_tensors(audio_tensors, tts_model.sr)
        else:
            final_wav = audio_tensors[0]

        audio_bytes = save_audio_to_bytes(final_wav, tts_model.sr)

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
        if use_pool and model_pool is not None and tts_model is not None:
            model_pool.put(tts_model)
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def _generate_single_pooled(item: dict, model_pool: Queue, sample_rate: int) -> dict:
    """Generate a single TTS clip using a model from the pool."""
    tts_model = model_pool.get()

    try:
        text = item.get("text", "")
        voice_name = item.get("voice_name")
        exaggeration = item.get("exaggeration", 0.5)
        cfg_weight = item.get("cfg_weight", 0.5)

        audio_prompt_path = None
        if voice_name:
            voice_file = config.VOICES_DIR / f"{voice_name}.wav"
            if voice_file.exists():
                audio_prompt_path = str(voice_file)

        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}

        if audio_prompt_path:
            cached_conds = get_cached_conditionals(tts_model, audio_prompt_path, exaggeration)
            tts_model.conds = cached_conds

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

        timestamp = int(time.time() * 1000)
        filename = f"tts_{timestamp}.wav"
        output_file = config.OUTPUT_DIR / filename
        with open(output_file, "wb") as f:
            f.write(audio_bytes)

        return {
            "filename": filename,
            "download_url": f"{config.get_base_url()}/download/{filename}",
            "size_bytes": len(audio_bytes),
        }
    finally:
        model_pool.put(tts_model)


def _generate_single_tensor(item: dict, model_pool: Queue, sample_rate: int) -> torch.Tensor:
    """Generate a single TTS clip and return the audio tensor."""
    tts_model = model_pool.get()

    try:
        text = item.get("text", "")
        voice_name = item.get("voice_name")
        exaggeration = item.get("exaggeration", 0.5)
        cfg_weight = item.get("cfg_weight", 0.5)

        audio_prompt_path = None
        if voice_name:
            voice_file = config.VOICES_DIR / f"{voice_name}.wav"
            if voice_file.exists():
                audio_prompt_path = str(voice_file)

        kwargs = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}

        if audio_prompt_path:
            cached_conds = get_cached_conditionals(tts_model, audio_prompt_path, exaggeration)
            tts_model.conds = cached_conds

        chunks = split_text_into_chunks(text)
        audio_tensors = []
        for chunk in chunks:
            wav = tts_model.generate(chunk, **kwargs)
            audio_tensors.append(wav)

        if len(audio_tensors) > 1:
            return concatenate_audio_tensors(audio_tensors, sample_rate, silence_duration=0.15)
        return audio_tensors[0]

    finally:
        model_pool.put(tts_model)


def generate_batch(items: List[dict]) -> dict:
    """Generate multiple TTS clips sequentially."""
    if not items:
        return {"status": "error", "message": "No items provided"}

    model_pool = get_model_pool()
    temp_model = model_pool.get()
    sample_rate = temp_model.sr
    model_pool.put(temp_model)

    results = []
    errors = []

    print(f"Batch processing {len(items)} items...")

    for idx, item in enumerate(items):
        try:
            result = _generate_single_pooled(item, model_pool, sample_rate)
            result["index"] = idx
            results.append(result)
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

    model_pool = get_model_pool()
    temp_model = model_pool.get()
    sample_rate = temp_model.sr
    model_pool.put(temp_model)

    print(f"Generating conversation: {len(items)} clips...")

    audio_tensors = []
    errors = []

    for idx, item in enumerate(items):
        try:
            tensor = _generate_single_tensor(item, model_pool, sample_rate)
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

    if output_name:
        filename = f"{output_name}.wav"
    else:
        filename = f"conversation_{int(time.time())}.wav"

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
