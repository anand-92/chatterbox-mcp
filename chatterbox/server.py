"""Main server entry point."""

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from .api import get_api_routes
from .config import config
from .mcp_tools import create_mcp_server


def create_app():
    """Create the combined Starlette application."""
    mcp = create_mcp_server()
    mcp_http_app = mcp.http_app()

    app = Starlette(
        routes=[
            *get_api_routes(),
            Mount("/ui", app=StaticFiles(directory=str(config.UI_DIR), html=True), name="ui"),
            Mount("/", app=mcp_http_app),
        ],
        lifespan=mcp_http_app.lifespan,
    )

    # Add CORS middleware
    app = CORSMiddleware(
        app,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def print_startup_info():
    """Print server startup information."""
    print("=" * 60)
    print("Chatterbox TTS MCP Server")
    print("=" * 60)
    print(f"Local URL:   http://127.0.0.1:{config.PORT}/mcp")
    print(f"Network URL: http://{config.SERVER_IP}:{config.PORT}/mcp")
    print("-" * 60)
    print("MCP Config:")
    print(f'  "chatterbox": {{')
    print(f'    "type": "http",')
    print(f'    "url": "http://{config.SERVER_IP}:{config.PORT}/mcp"')
    print(f'  }}')
    print("-" * 60)
    print(f"Download:    {config.get_base_url()}/download/<filename>")
    print(f"Upload:      curl -X POST '{config.get_base_url()}/upload_voice/<name>' -F 'file=@audio.wav'")
    print("-" * 60)
    print(f"Web UI:      http://{config.SERVER_IP}:{config.PORT}/ui/")
    print(f"             http://127.0.0.1:{config.PORT}/ui/")
    print("=" * 60)


def main():
    """Run the server."""
    print_startup_info()
    app = create_app()
    uvicorn.run(app, host=config.HOST, port=config.PORT, ws="wsproto")


if __name__ == "__main__":
    main()
