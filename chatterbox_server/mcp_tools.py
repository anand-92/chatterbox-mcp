"""MCP tool definitions for Chatterbox TTS."""

import asyncio
from typing import List, Literal, Optional

from fastmcp import FastMCP
from fastmcp.utilities.types import Audio
from pydantic import Field

from . import tts, voices

# MCP server instructions
MCP_INSTRUCTIONS = """
# Chatterbox TTS Server

Text-to-speech with voice cloning. Returns download URLs.

## Quick Start

```python
text_to_speech(text="Hello!", voice_name="david")
```

Download: `curl -s "<download_url>" -o output.wav`

## Models

### standard (default)
- English only, best quality
- Use `exaggeration` (0-1) and `cfg_weight` (0-1) to control style
- Higher exaggeration = more expressive
- Lower cfg_weight = slower, more deliberate

### turbo
- Faster generation, supports paralinguistic tags
- **REQUIRES voice_name** (voice cloning mandatory)
- Embed tags in text for expressiveness:
  - `[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, `[gasp]`, `[groan]`, `[yawn]`, `[clearing throat]`
- Example: `text_to_speech(text="That's hilarious! [laugh]", model="turbo", voice_name="david")`

### multilingual
- 23 languages: ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
- Set `language` parameter (e.g., "fr", "es", "ja")

## Voice Cloning

```python
save_voice(name="david", audio_url="http://example.com/voice.wav")
text_to_speech(text="Hello", voice_name="david")
```

## Tips

- 10-15 seconds of clean speech works best for cloning
- For turbo with paralinguistic tags, place tags after relevant text
- Lower cfg_weight (0.3) + higher exaggeration (0.7) for dramatic speech
"""


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP(name="Chatterbox TTS", instructions=MCP_INSTRUCTIONS)

    @mcp.tool
    async def text_to_speech(
        text: str = Field(description="The text to convert to speech."),
        model: Literal["standard", "turbo", "multilingual"] = Field(
            default="standard",
            description="Model: 'standard' (default), 'turbo' (fast, requires voice, supports [laugh] tags), or 'multilingual'"
        ),
        voice_name: Optional[str] = Field(
            default=None,
            description="Name of a saved voice to clone."
        ),
        voice_audio_base64: Optional[str] = Field(
            default=None,
            description="Base64-encoded WAV audio for voice cloning (5-15 seconds)."
        ),
        language: Optional[str] = Field(
            default=None,
            description="Language code for multilingual model (e.g., 'en', 'fr', 'es')."
        ),
        exaggeration: float = Field(
            default=0.5,
            description="Exaggeration level (0.0-1.0). Higher = more expressive."
        ),
        cfg_weight: float = Field(
            default=0.5,
            description="CFG weight (0.0-1.0). Lower = slower, more deliberate speech."
        )
    ) -> Audio:
        """Generate speech from text using Chatterbox TTS."""
        return await asyncio.to_thread(
            tts.generate_tts, text, model, voice_name, voice_audio_base64,
            language, exaggeration, cfg_weight
        )

    @mcp.tool
    async def batch_text_to_speech(
        items: List[dict] = Field(
            description="List of TTS items. Each: {text, voice_name, exaggeration, cfg_weight}"
        )
    ) -> dict:
        """Generate multiple TTS clips in batch."""
        return await asyncio.to_thread(tts.generate_batch, items)

    @mcp.tool
    async def generate_conversation(
        items: List[dict] = Field(
            description="List of dialogue items. Each: {text, voice_name, exaggeration, cfg_weight}"
        ),
        output_name: Optional[str] = Field(
            default=None,
            description="Output filename (without .wav)."
        ),
        silence_between: float = Field(
            default=0.4,
            description="Seconds of silence between clips."
        )
    ) -> dict:
        """Generate a multi-voice conversation as a single audio file."""
        return await asyncio.to_thread(
            tts.generate_conversation, items, output_name, silence_between
        )

    @mcp.tool
    async def list_voices() -> dict:
        """List all saved voices available for voice cloning."""
        return await asyncio.to_thread(voices.list_voices)

    @mcp.tool
    async def save_voice(
        name: str = Field(description="Name for the voice (saved as name.wav)"),
        audio_url: Optional[str] = Field(
            default=None,
            description="URL to download the voice audio from"
        ),
    ) -> dict:
        """Save a voice reference audio for later use in voice cloning."""
        return await asyncio.to_thread(voices.save_voice, name, audio_url)

    @mcp.tool
    async def delete_voice(
        name: str = Field(description="Name of the voice to delete")
    ) -> dict:
        """Delete a saved voice from the voices directory."""
        return await asyncio.to_thread(voices.delete_voice, name)

    @mcp.tool
    async def clone_voice_from_youtube(
        name: str = Field(description="Name for the voice (e.g., 'kobe')"),
        youtube_url: str = Field(description="YouTube video URL"),
        timestamp: str = Field(description="Start timestamp (MM:SS or HH:MM:SS)"),
        duration: int = Field(default=15, description="Duration in seconds (10-15 recommended)")
    ) -> dict:
        """Clone a voice directly from a YouTube video."""
        return await asyncio.to_thread(
            voices.clone_voice_from_youtube, name, youtube_url, timestamp, duration
        )

    return mcp
