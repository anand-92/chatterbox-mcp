"""
Chatterbox TTS Proxy for Claude Desktop

This proxy connects Claude Desktop (via STDIO) to the remote Chatterbox HTTP server.
"""

from fastmcp import FastMCP

# Create a proxy to the remote Chatterbox server
proxy = FastMCP.as_proxy(
    "http://192.168.1.5:8765/mcp",
    name="Chatterbox TTS"
)

if __name__ == "__main__":
    proxy.run()  # Runs via STDIO for Claude Desktop
