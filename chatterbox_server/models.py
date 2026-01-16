"""Model loading, caching, and pool management."""

import os
import glob
import threading
from collections import OrderedDict
from queue import Queue
from typing import Optional, Tuple

# Must import config and set env var BEFORE importing torch
from .config import config


def _find_msvc_compiler():
    """Find MSVC cl.exe on Windows and set CC environment variable."""
    if os.name != 'nt' or os.environ.get('CC'):
        return  # Not Windows or CC already set

    # Common Visual Studio installation paths
    vs_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
    ]

    for pattern in vs_paths:
        matches = glob.glob(pattern)
        if matches:
            # Use the most recent version (last in sorted order)
            cl_path = sorted(matches)[-1]
            os.environ['CC'] = cl_path
            print(f"Found MSVC compiler: {cl_path}")
            return

    print("No MSVC compiler found - torch.compile may not work")


# Fix for Windows PyTorch 2.9 OverflowError with torch.compile
# See: https://github.com/pytorch/pytorch/issues/166886
# Must be set BEFORE importing torch
if os.name == 'nt':
    os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"

# Try to find and configure MSVC compiler for torch.compile
if config.FISH_COMPILE:
    _find_msvc_compiler()
else:
    # Disable torch.compile if no C compiler available
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import numpy as np

# Also disable via config after import (belt and suspenders)
if os.name == 'nt':
    import torch._inductor.config
    torch._inductor.config.triton.cudagraphs = False

# CUDA optimizations
if torch.cuda.is_available():
    # Use new TF32 API (PyTorch 2.9+)
    if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    else:
        # Fallback for older PyTorch
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
        reference_id: Optional[str] = None,
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
            chunk_length=300,  # Max recommended for long texts
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            format="wav",
            streaming=streaming,
            use_memory_cache="off",
        )

    def _get_reference_id(self, reference_audio: Optional[bytes]) -> Optional[str]:
        """Generate a stable reference ID for caching based on audio content."""
        if not reference_audio:
            return None
        import hashlib
        return hashlib.md5(reference_audio).hexdigest()[:16]

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
        ref_id = self._get_reference_id(reference_audio)
        request = self._build_request(
            text, reference_audio, reference_text,
            temperature, top_p, repetition_penalty, max_new_tokens,
            streaming=False,
            reference_id=ref_id,
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
        ref_id = self._get_reference_id(reference_audio)
        request = self._build_request(
            text, reference_audio, reference_text,
            temperature, top_p, repetition_penalty, max_new_tokens,
            streaming=True,
            reference_id=ref_id,
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

# Fish Speech reference audio cache (voice_path -> bytes)
_fish_reference_cache: OrderedDict = OrderedDict()
_fish_cache_lock = threading.Lock()


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


def get_cached_fish_reference(voice_path: str) -> bytes:
    """Get cached reference audio bytes for Fish Speech with LRU eviction."""
    global _fish_reference_cache

    mtime = os.path.getmtime(voice_path)
    cache_key = (voice_path, mtime)

    with _fish_cache_lock:
        if cache_key in _fish_reference_cache:
            _fish_reference_cache.move_to_end(cache_key)
            return _fish_reference_cache[cache_key]

    # Read file outside lock
    with open(voice_path, "rb") as f:
        audio_bytes = f.read()

    with _fish_cache_lock:
        _fish_reference_cache[cache_key] = audio_bytes
        # Evict oldest if over limit
        while len(_fish_reference_cache) > config.VOICE_CACHE_MAX_SIZE:
            oldest = next(iter(_fish_reference_cache))
            del _fish_reference_cache[oldest]
        print(f"Cached fish reference: {voice_path} (cache size: {len(_fish_reference_cache)})")

    return audio_bytes


def get_status() -> dict:
    """Get server status for health check."""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models_loaded": list(_models.keys()),
        "pool_initialized": _model_pool is not None,
        "voice_cache_size": len(_voice_conds_cache),
        "fish_reference_cache_size": len(_fish_reference_cache),
    }
