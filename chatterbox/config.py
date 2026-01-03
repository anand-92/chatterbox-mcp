"""Configuration settings for Chatterbox TTS server."""

import os
import re
import socket
import subprocess
from pathlib import Path


def _get_lan_ip() -> str:
    """Get the LAN IP address (prefer 192.168.x.x or 10.x.x.x ranges)."""
    # Method 1: Connect to external address to find default route interface
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip.startswith("192.168."):
            return ip
    except Exception:
        pass

    # Method 2: Windows ipconfig
    try:
        result = subprocess.run(
            ["ipconfig"], capture_output=True, text=True, timeout=5, shell=True
        )
        if result.returncode == 0:
            matches = re.findall(r"IPv4[^:]*:\s*(\d+\.\d+\.\d+\.\d+)", result.stdout)
            for ip in matches:
                if ip.startswith("192.168."):
                    return ip
            for ip in matches:
                if not ip.startswith("127."):
                    return ip
    except Exception:
        pass

    # Method 3: Linux hostname -I
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            for ip in ips:
                if ip.startswith("192.168."):
                    return ip
            if ips:
                return ips[0]
    except Exception:
        pass

    return "127.0.0.1"


class Config:
    """Server configuration."""

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "output"
    VOICES_DIR = BASE_DIR / "voices"
    UI_DIR = BASE_DIR / "ui"

    # Server settings
    PORT = 8765
    HOST = "0.0.0.0"
    SERVER_IP = _get_lan_ip()

    # Public URL for when running behind a tunnel (e.g., Cloudflare Tunnel)
    # Set to None to use local IP
    PUBLIC_URL = "https://mcp.thethirdroom.xyz"

    # Model settings
    POOL_SIZE = 3  # RTX 5090 (32GB VRAM) can handle 3+ instances
    LOCAL_MODEL_PATH = Path(os.path.expanduser(
        "~/.cache/huggingface/hub/models--ResembleAI--chatterbox/"
        "snapshots/05e904af2b5c7f8e482687a9d7336c5c824467d9"
    ))

    # Audio settings
    MAX_CHUNK_CHARS = 280  # ~30 sec audio per chunk

    def __init__(self):
        # Ensure directories exist
        self.OUTPUT_DIR.mkdir(exist_ok=True)
        self.VOICES_DIR.mkdir(exist_ok=True)
        self.UI_DIR.mkdir(exist_ok=True)

    def get_base_url(self) -> str:
        """Get the base URL for download links."""
        if self.PUBLIC_URL:
            return self.PUBLIC_URL
        return f"http://{self.SERVER_IP}:{self.PORT}"


# Global config instance
config = Config()
