"""MCP tool definitions for SolSpeak TTS."""

import asyncio
from typing import List, Literal, Optional

from fastmcp import Context, FastMCP
from pydantic import Field

from . import tts, voices

# MCP server instructions
MCP_INSTRUCTIONS = """
# SolSpeak TTS Server

Text-to-speech with voice cloning. Returns download URL.

## Quick Start

```python
text_to_speech(text="Hello!", voice_name="david", model="fish")
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
- Embed tags in text: `[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, `[gasp]`, `[groan]`, `[yawn]`
- Example: `text_to_speech(text="That's hilarious! [laugh]", model="turbo", voice_name="david")`

### fish - Fish Speech (OpenAudio S1) - RECOMMENDED
- Multilingual: English, Chinese, Japanese, Korean, French, German, Spanish, Arabic, Russian, Dutch, Italian, Polish, Portuguese
- 64+ emotional markers with fine-grained control
- **For voice cloning: provide voice_name** (transcript auto-loads if saved)
- Without voice reference: generates with random voice

## Fish Speech Emotion Guide

**Place markers at the START of sentences in parentheses.**

### Basic Emotions
`(happy)`, `(sad)`, `(angry)`, `(excited)`, `(calm)`, `(nervous)`, `(confident)`, `(surprised)`, `(satisfied)`, `(delighted)`, `(scared)`, `(worried)`, `(upset)`, `(frustrated)`, `(depressed)`, `(proud)`, `(relaxed)`, `(grateful)`, `(curious)`, `(confused)`, `(joyful)`

### Intense Emotions
`(hysterical)`, `(furious)`, `(panicked)`, `(astonished)`, `(disdainful)`, `(scornful)`, `(sarcastic)`, `(sneering)`

### Tone Modifiers
`(whispering)` - soft, secretive
`(soft tone)` - gentle, calm
`(shouting)` - loud, energetic
`(screaming)` - high intensity
`(in a hurry tone)` - rushed speech

### Paralinguistic Effects
`(laughing)`, `(chuckling)`, `(sobbing)`, `(crying loudly)`, `(sighing)`, `(groaning)`, `(panting)`, `(gasping)`, `(yawning)`

### Audience Effects
`(crowd laughing)`, `(background laughter)`, `(audience laughing)`

### Manual Laughter
Use "Ha, ha, ha" or "Hahaha" directly in text for natural laughter.

## Fish Examples

```python
# Single emotion
text_to_speech(text="(excited) We won the championship!", model="fish", voice_name="david")

# Emotion + effect
text_to_speech(text="(happy)(laughing) That joke was hilarious! Ha ha ha!", model="fish", voice_name="david")

# Changing emotions mid-speech
text_to_speech(text="(sad) I can't believe it's over. (angry) But I won't give up! (confident) I'll come back stronger.", model="fish", voice_name="david")

# Whispering
text_to_speech(text="(whispering) Don't tell anyone, but I know the secret.", model="fish", voice_name="david")

# Intense delivery
text_to_speech(text="(furious)(shouting) How could you do this to me?!", model="fish", voice_name="david")
```

## Voice Cloning

```python
# Clone with transcript (required for fish voice cloning quality)
clone_voice_from_youtube(name="david", youtube_url="https://youtube.com/...", timestamp="1:30", transcript="Exact words spoken in clip")

# Set transcript for existing voice
set_voice_transcript(voice_name="david", transcript="Exact words spoken in clip")

# Use cloned voice
text_to_speech(text="(excited) Hello world!", model="fish", voice_name="david")
```

## Tips

- 10-15 seconds of clean speech works best for cloning
- Place emotion markers at sentence START, not middle
- Stack markers: `(sad)(whispering)` for combined effects
- Use one primary emotion per sentence for best results
- Transcript must match reference audio exactly for best voice cloning
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
    ) -> dict:
        """Generate speech from text using SolSpeak TTS. Returns download URL."""
        loop = asyncio.get_running_loop()

        def progress_callback(current: int, total: int) -> None:
            if ctx:
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(progress=current, total=total), loop
                )

        result = await asyncio.to_thread(
            tts.generate_tts,
            text=text,
            model=model,
            voice_name=voice_name,
            voice_audio_base64=voice_audio_base64,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            progress_callback=progress_callback,
            return_bytes=False,  # Save file and return URL
            voice_text=voice_text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return result

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
