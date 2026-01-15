"""MCP tool definitions for SolSpeak TTS."""

import asyncio
from typing import List, Literal, Optional

from fastmcp import Context, FastMCP
from fastmcp.utilities.types import Audio
from pydantic import Field

from . import tts, voices

# MCP server instructions
MCP_INSTRUCTIONS = """
# SolSpeak TTS Server

Text-to-speech with voice cloning. Returns audio directly.

## Quick Start

```python
text_to_speech(text="Hello!", voice_name="david")
```

## Models

### standard (default) - Chatterbox
- English only, best quality
- Use `exaggeration` (0-1) and `cfg_weight` (0-1) to control style
- Higher exaggeration = more expressive
- Lower cfg_weight = slower, more deliberate

### turbo - Chatterbox Turbo
- Faster generation, supports paralinguistic tags
- **REQUIRES voice_name** (voice cloning mandatory)
- Embed tags in text for expressiveness:
  - `[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, `[gasp]`, `[groan]`, `[yawn]`, `[clearing throat]`
- Example: `text_to_speech(text="That's hilarious! [laugh]", model="turbo", voice_name="david")`

### fish - Fish Speech (OpenAudio S1)
- Multilingual (13 languages), expressive speech synthesis
- Supports 45+ emotional markers: (angry), (sad), (excited), (laughing), etc.
- **For voice cloning: REQUIRES voice_name AND voice_text** (transcription of reference)
- **Transcripts are auto-loaded** if saved with `set_voice_transcript` or during cloning
- Without voice reference: generates with random voice
- Use `temperature` (0.1-1.0), `top_p` (0.1-1.0), `repetition_penalty` (0.9-2.0)
- Example: `text_to_speech(text="(excited) This is amazing!", model="fish")`
- Example with cloning: `text_to_speech(text="Hello", model="fish", voice_name="david")`  # transcript auto-loaded

## Voice Cloning

```python
# Clone with transcript for fish model (transcript persists)
clone_voice_from_youtube(name="david", youtube_url="https://youtube.com/...", timestamp="1:30", transcript="What they said in the clip")
text_to_speech(text="Hello", model="fish", voice_name="david")  # transcript auto-loaded!

# Set transcript for existing voice
set_voice_transcript(voice_name="david", transcript="What they said in the clip")
```

## Tips

- 10-15 seconds of clean speech works best for cloning
- For turbo with paralinguistic tags, place tags after relevant text
- Lower cfg_weight (0.3) + higher exaggeration (0.7) for dramatic speech
- For fish: use emotional markers in parentheses for expressive speech
"""


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server with all tools."""
    mcp = FastMCP(name="SolSpeak TTS", instructions=MCP_INSTRUCTIONS)

    @mcp.tool
    async def text_to_speech(
        text: str = Field(description="The text to convert to speech."),
        model: Literal["standard", "turbo", "fish"] = Field(
            default="standard",
            description="Model: 'standard' (default), 'turbo' (fast, requires voice, paralinguistic tags), or 'fish' (multilingual, emotional markers)"
        ),
        voice_name: Optional[str] = Field(
            default=None,
            description="Name of a saved voice to clone."
        ),
        voice_audio_base64: Optional[str] = Field(
            default=None,
            description="Base64-encoded WAV audio for voice cloning (5-15 seconds)."
        ),
        exaggeration: float = Field(
            default=0.5,
            description="Exaggeration level (0.0-1.0). Higher = more expressive. (standard/turbo only)"
        ),
        cfg_weight: float = Field(
            default=0.5,
            description="CFG weight (0.0-1.0). Lower = slower, more deliberate speech. (standard/turbo only)"
        ),
        voice_text: Optional[str] = Field(
            default=None,
            description="Transcription of reference audio. REQUIRED for fish model voice cloning."
        ),
        temperature: float = Field(
            default=0.7,
            description="Sampling temperature (0.1-1.0). Higher = more varied. (fish only)"
        ),
        top_p: float = Field(
            default=0.8,
            description="Top-p sampling (0.1-1.0). (fish only)"
        ),
        repetition_penalty: float = Field(
            default=1.1,
            description="Repetition penalty (0.9-2.0). (fish only)"
        ),
        ctx: Context = None
    ) -> Audio:
        """Generate speech from text using SolSpeak TTS. Returns audio directly."""
        loop = asyncio.get_running_loop()

        def progress_callback(current: int, total: int) -> None:
            if ctx:
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=current, total=total), loop
                )

        audio_bytes = await asyncio.to_thread(
            tts.generate_tts,
            text=text,
            model=model,
            voice_name=voice_name,
            voice_audio_base64=voice_audio_base64,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            progress_callback=progress_callback,
            return_bytes=True,
            voice_text=voice_text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return Audio(data=audio_bytes, format="wav")

    @mcp.tool
    async def generate_conversation(
        items: List[dict] = Field(
            description="List of dialogue items. Each: {text, voice_name, model, exaggeration, cfg_weight, voice_text, temperature}. Model can be 'standard', 'turbo', or 'fish'."
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
    async def delete_voice(
        name: str = Field(description="Name of the voice to delete"),
        password: str = Field(description="Password required for deletion")
    ) -> dict:
        """Delete a saved voice from the voices directory. Requires password."""
        return await asyncio.to_thread(voices.delete_voice, name, password)

    @mcp.tool
    async def clone_voice_from_youtube(
        name: str = Field(description="Name for the voice (e.g., 'kobe')"),
        youtube_url: str = Field(description="YouTube video URL"),
        timestamp: str = Field(default="0:00", description="Start timestamp (MM:SS or HH:MM:SS)"),
        duration: int = Field(default=15, description="Duration in seconds (10-15 recommended)"),
        transcript: Optional[str] = Field(
            default=None,
            description="Transcription of what's said in the clip. REQUIRED for fish model. Saves persistently."
        )
    ) -> dict:
        """Clone a voice directly from a YouTube video. Include transcript for fish model support."""
        return await asyncio.to_thread(
            voices.clone_voice_from_youtube, name, youtube_url, timestamp, duration, transcript
        )

    @mcp.tool
    async def set_voice_transcript(
        voice_name: str = Field(description="Name of an existing saved voice"),
        transcript: str = Field(description="Transcription of what's said in the voice's reference audio")
    ) -> dict:
        """Set or update the transcript for a saved voice. Required for fish model voice cloning."""
        return await asyncio.to_thread(voices.set_voice_transcript, voice_name, transcript)

    return mcp
