"""Model loading, caching, and pool management."""

import os
import threading
from collections import OrderedDict
from queue import Queue
from typing import Optional

import torch

from .config import config


# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"CUDA optimizations enabled for {torch.cuda.get_device_name(0)}")


class NoOpWatermarker:
    """Dummy watermarker that skips CPU-intensive watermarking."""
    def apply_watermark(self, wav, sample_rate):
        return wav


def _disable_watermarker(model):
    """Replace watermarker with no-op to skip CPU processing."""
    model.watermarker = NoOpWatermarker()
    return model


# Global state
_models = {}
_models_lock = threading.Lock()
_model_pool: Optional[Queue] = None
_pool_lock = threading.Lock()

# Voice conditionals cache with LRU eviction
_voice_conds_cache: OrderedDict = OrderedDict()
_voice_cache_lock = threading.Lock()


def get_model(model_type: str = "standard"):
    """Lazily load and cache models (thread-safe)."""
    global _models

    if model_type in _models:
        return _models[model_type]

    with _models_lock:
        if model_type in _models:
            return _models[model_type]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_type} model on {device}...")

        if model_type == "standard":
            from chatterbox.tts import ChatterboxTTS
            if config.LOCAL_MODEL_PATH.exists():
                model = ChatterboxTTS.from_local(str(config.LOCAL_MODEL_PATH), device=device)
            else:
                model = ChatterboxTTS.from_pretrained(device=device)
            _models[model_type] = _disable_watermarker(model)
        elif model_type == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(device=device)
            _models[model_type] = _disable_watermarker(model)
        elif model_type == "multilingual":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            _models[model_type] = _disable_watermarker(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"{model_type} model loaded!")

    return _models[model_type]


def get_model_pool() -> Queue:
    """Get or initialize the model pool (thread-safe)."""
    global _model_pool

    if _model_pool is not None:
        return _model_pool

    with _pool_lock:
        if _model_pool is not None:
            return _model_pool

        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Initializing model pool ({config.POOL_SIZE} instances)...")
        _model_pool = Queue()

        for i in range(config.POOL_SIZE):
            print(f"  Loading instance {i+1}/{config.POOL_SIZE}...")
            if config.LOCAL_MODEL_PATH.exists():
                model = ChatterboxTTS.from_local(str(config.LOCAL_MODEL_PATH), device=device)
            else:
                model = ChatterboxTTS.from_pretrained(device=device)
            _model_pool.put(_disable_watermarker(model))

        print(f"Model pool ready!")

    return _model_pool


def get_cached_conditionals(model, voice_path: str, exaggeration: float = 0.5):
    """Get cached voice conditionals with LRU eviction."""
    global _voice_conds_cache

    mtime = os.path.getmtime(voice_path)
    cache_key = (voice_path, mtime, exaggeration)

    with _voice_cache_lock:
        if cache_key in _voice_conds_cache:
            # Move to end (most recently used)
            _voice_conds_cache.move_to_end(cache_key)
            return _voice_conds_cache[cache_key]

    # Compute conditionals outside lock (CPU-intensive)
    print(f"Computing voice conditionals: {voice_path}")
    model.prepare_conditionals(voice_path, exaggeration=exaggeration)
    conds = model.conds

    with _voice_cache_lock:
        _voice_conds_cache[cache_key] = conds
        # Evict oldest if over limit
        while len(_voice_conds_cache) > config.VOICE_CACHE_MAX_SIZE:
            oldest = next(iter(_voice_conds_cache))
            del _voice_conds_cache[oldest]
            print(f"Evicted from cache: {oldest[0]}")
        print(f"Cached conditionals (size: {len(_voice_conds_cache)})")

    return conds


def get_status() -> dict:
    """Get server status for health check."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": list(_models.keys()),
        "pool_initialized": _model_pool is not None,
        "voice_cache_size": len(_voice_conds_cache),
    }
