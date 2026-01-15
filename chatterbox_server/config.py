"""Configuration settings for SolSpeak TTS server."""

import os
import re
import socket
import subprocess
from pathlib import Path


def _get_lan_ip() -> str:
    """Get the LAN IP address (prefer 192.168.x.x or 10.x.x.x ranges)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip.startswith(("192.168.", "10.")):
            return ip
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["ipconfig"], capture_output=True, text=True, timeout=5, shell=True
        )
        if result.returncode == 0:
            matches = re.findall(r"IPv4[^:]*:\s*(\d+\.\d+\.\d+\.\d+)", result.stdout)
            for ip in matches:
                if ip.startswith(("192.168.", "10.")):
                    return ip
            for ip in matches:
                if not ip.startswith("127."):
                    return ip
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                if ip.startswith(("192.168.", "10.")):
                    return ip
            if ips:
                return ips[0]
    except Exception:
        pass

    return "127.0.0.1"


class Config:
    """Server configuration. Override with SOLSPEAK_* environment variables."""

    def __init__(self):
        # Paths
        self.BASE_DIR = Path(os.getenv("SOLSPEAK_BASE_DIR", Path(__file__).parent.parent))
        self.OUTPUT_DIR = Path(os.getenv("SOLSPEAK_OUTPUT_DIR", self.BASE_DIR / "output"))
        self.VOICES_DIR = Path(os.getenv("SOLSPEAK_VOICES_DIR", self.BASE_DIR / "voices"))
        self.UI_DIR = Path(os.getenv("SOLSPEAK_UI_DIR", self.BASE_DIR / "ui"))

        # Server
        self.PORT = int(os.getenv("SOLSPEAK_PORT", "8765"))
        self.HOST = os.getenv("SOLSPEAK_HOST", "0.0.0.0")
        self.SERVER_IP = _get_lan_ip()

        # Public URL (set to "" or "none" to disable)
        public_url = os.getenv("SOLSPEAK_PUBLIC_URL", "https://mcp.thethirdroom.xyz")
        self.PUBLIC_URL = None if public_url.lower() in ("", "none", "null") else public_url

        # Model
        self.POOL_SIZE = int(os.getenv("SOLSPEAK_POOL_SIZE", "3"))
        self.LOCAL_MODEL_PATH = Path(os.getenv(
            "SOLSPEAK_MODEL_PATH",
            os.path.expanduser("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/"
                              "snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9")
        ))

        # Audio
        self.MAX_CHUNK_CHARS = int(os.getenv("SOLSPEAK_MAX_CHUNK_CHARS", "280"))

        # Voice cache (LRU eviction)
        self.VOICE_CACHE_MAX_SIZE = int(os.getenv("SOLSPEAK_VOICE_CACHE_SIZE", "50"))

        # Output cleanup (hours, 0 = disabled)
        self.OUTPUT_MAX_AGE_HOURS = int(os.getenv("SOLSPEAK_OUTPUT_MAX_AGE_HOURS", "24"))

        # F5-TTS settings
        self.F5_TTS_MODEL = os.getenv("SOLSPEAK_F5_MODEL", "F5TTS_v1_Base")
        self.F5_TTS_NFE_STEPS = int(os.getenv("SOLSPEAK_F5_NFE_STEPS", "32"))  # Quality vs speed
        self.F5_TTS_CFG_STRENGTH = float(os.getenv("SOLSPEAK_F5_CFG_STRENGTH", "2.0"))

        # Fish Speech (OpenAudio S1) settings
        self.FISH_CHECKPOINT_DIR = Path(os.getenv(
            "SOLSPEAK_FISH_CHECKPOINT_DIR",
            self.BASE_DIR / "checkpoints" / "openaudio-s1-mini"
        ))
        self.FISH_TEMPERATURE = float(os.getenv("SOLSPEAK_FISH_TEMPERATURE", "0.7"))
        self.FISH_TOP_P = float(os.getenv("SOLSPEAK_FISH_TOP_P", "0.8"))
        self.FISH_REPETITION_PENALTY = float(os.getenv("SOLSPEAK_FISH_REPETITION_PENALTY", "1.1"))
        self.FISH_MAX_NEW_TOKENS = int(os.getenv("SOLSPEAK_FISH_MAX_NEW_TOKENS", "2048"))
        self.FISH_COMPILE = os.getenv("SOLSPEAK_FISH_COMPILE", "false").lower() == "true"

        # Create directories
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.VOICES_DIR.mkdir(parents=True, exist_ok=True)
        self.UI_DIR.mkdir(parents=True, exist_ok=True)
        self.FISH_CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)

    def get_base_url(self) -> str:
        """Get the base URL for download links."""
        if self.PUBLIC_URL:
            return self.PUBLIC_URL
        return f"http://{self.SERVER_IP}:{self.PORT}"


config = Config()
