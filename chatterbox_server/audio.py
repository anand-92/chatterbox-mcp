"""Audio processing utilities."""

import os
import re
import tempfile
from typing import List

import numpy as np
import torch
from scipy.io import wavfile

from .config import config


def save_audio_to_bytes(wav_tensor: torch.Tensor, sample_rate: int) -> bytes:
    """Convert audio tensor to WAV bytes using scipy."""
    temp_path = tempfile.mktemp(suffix=".wav")
    try:
        audio_np = wav_tensor.squeeze().cpu().numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(temp_path, sample_rate, audio_int16)
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def split_text_into_chunks(text: str, max_chars: int = None) -> List[str]:
    """
    Split text into chunks based on sentence boundaries.
    Keeps chunks under max_chars while respecting sentence structure.
    """
    if max_chars is None:
        max_chars = config.MAX_CHUNK_CHARS

    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

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
                    if len(part) > max_chars:
                        # Force split by words
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

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def concatenate_audio_tensors(
    tensors: List[torch.Tensor],
    sample_rate: int,
    silence_duration: float = 0.3
) -> torch.Tensor:
    """Concatenate multiple audio tensors with silence between them."""
    if not tensors:
        raise ValueError("No audio tensors to concatenate")

    if len(tensors) == 1:
        return tensors[0]

    silence_samples = int(sample_rate * silence_duration)
    silence = torch.zeros(1, silence_samples)

    result_parts = []
    for i, tensor in enumerate(tensors):
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        result_parts.append(tensor)
        if i < len(tensors) - 1:
            result_parts.append(silence)

    return torch.cat(result_parts, dim=1)
