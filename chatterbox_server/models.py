"""Model loading, caching, and pool management."""

import os
import threading
from collections import OrderedDict
from queue import Queue
from typing import Optional, Tuple

import torch
import numpy as np

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


class FishSpeechWrapper:
    """Wrapper for Fish Speech (OpenAudio S1) model to provide a consistent interface."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda", compile: bool = False):
        """Initialize Fish Speech model.

        Args:
            checkpoint_dir: Path to the openaudio-s1-mini checkpoint directory
            device: Device to run on (cuda or cpu)
            compile: Whether to use torch.compile for faster inference
        """
        # Set up pyrootutils before importing fish_speech modules
        import pyrootutils
        pyrootutils.setup_root(
            search_from=checkpoint_dir,
            indicator=[".project-root", "pyproject.toml", ".git"],
            pythonpath=True,
            dotenv=False,
            cwd=False,
        )

        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        self.device = device
        self.compile = compile
        self.checkpoint_dir = checkpoint_dir

        # Determine precision based on device
        if device == "cuda" and torch.cuda.is_available():
            self.precision = torch.bfloat16
        else:
            self.precision = torch.float32

        # Load LLAMA model (text-to-semantic)
        print(f"Loading Fish Speech LLAMA model from {checkpoint_dir}...")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=checkpoint_dir,
            device=device,
            precision=self.precision,
            compile=compile,
        )

        # Load DAC decoder model
        decoder_checkpoint = os.path.join(checkpoint_dir, "codec.pth")
        print(f"Loading Fish Speech decoder from {decoder_checkpoint}...")
        self.decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=decoder_checkpoint,
            device=device,
        )

        # Create the inference engine
        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=self.precision,
            compile=compile,
        )

        # Get sample rate from decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            self.sr = self.decoder_model.spec_transform.sample_rate
        else:
            self.sr = self.decoder_model.sample_rate

        print(f"Fish Speech model loaded! Sample rate: {self.sr}")

    def _build_request(
        self,
        text: str,
        reference_audio: Optional[bytes],
        reference_text: Optional[str],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_new_tokens: int,
        streaming: bool,
    ):
        """Build a ServeTTSRequest with the given parameters."""
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

        references = []
        if reference_audio and reference_text:
            references.append(ServeReferenceAudio(audio=reference_audio, text=reference_text))

        return ServeTTSRequest(
            text=text,
            references=references,
            reference_id=None,
            max_new_tokens=max_new_tokens,
            chunk_length=200,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            format="wav",
            streaming=streaming,
        )

    def generate(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 2048,
    ) -> torch.Tensor:
        """Generate speech from text.

        Args:
            text: The text to synthesize
            reference_audio: Optional reference audio bytes for voice cloning
            reference_text: Optional transcription of reference audio
            temperature: Sampling temperature (0.1-1.0)
            top_p: Top-p sampling parameter (0.1-1.0)
            repetition_penalty: Repetition penalty (0.9-2.0)
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Audio tensor of shape (samples,)
        """
        request = self._build_request(
            text, reference_audio, reference_text,
            temperature, top_p, repetition_penalty, max_new_tokens,
            streaming=False,
        )

        final_audio = None
        for result in self.engine.inference(request):
            if result.code == "error":
                raise RuntimeError(f"Fish Speech error: {result.error}")
            if result.code == "final" and isinstance(result.audio, tuple):
                _, final_audio = result.audio

        if final_audio is None:
            raise RuntimeError("Fish Speech generated no audio")

        return torch.from_numpy(final_audio).float()

    def generate_streaming(
        self,
        text: str,
        reference_audio: Optional[bytes] = None,
        reference_text: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 2048,
    ):
        """Generate speech from text with streaming.

        Yields:
            Tuple of (segment_index, audio_tensor) for each segment
        """
        request = self._build_request(
            text, reference_audio, reference_text,
            temperature, top_p, repetition_penalty, max_new_tokens,
            streaming=True,
        )

        segment_index = 0
        for result in self.engine.inference(request):
            if result.code == "error":
                raise RuntimeError(f"Fish Speech error: {result.error}")
            if result.code == "segment" and isinstance(result.audio, tuple):
                _, audio_np = result.audio
                yield (segment_index, torch.from_numpy(audio_np).float())
                segment_index += 1


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
        elif model_type == "f5":
            from f5_tts.api import F5TTS
            model = F5TTS(
                model=config.F5_TTS_MODEL,
                device=device,
            )
            _models[model_type] = model  # F5-TTS doesn't have watermarker
        elif model_type == "fish":
            checkpoint_dir = str(config.FISH_CHECKPOINT_DIR)
            if not os.path.exists(checkpoint_dir):
                raise ValueError(
                    f"Fish Speech checkpoint not found at {checkpoint_dir}. "
                    "Please download from HuggingFace: fishaudio/openaudio-s1-mini"
                )
            model = FishSpeechWrapper(
                checkpoint_dir=checkpoint_dir,
                device=device,
                compile=config.FISH_COMPILE,
            )
            _models[model_type] = model
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
