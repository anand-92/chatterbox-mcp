"""Main server entry point."""

import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from .api import create_api_app
from .config import config
from .mcp_tools import create_mcp_server
from .tts import cleanup_old_outputs


class PathDispatcher:
    """ASGI dispatcher that routes requests based on path prefix."""

    def __init__(self, api_app, mcp_app):
        self.api_app = api_app
        self.mcp_app = mcp_app
        # Paths that go to FastAPI (swagger, api, download, upload)
        self.api_paths = ("/api", "/docs", "/redoc", "/openapi.json", "/download")

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "/")
            # Route to API app if path matches any API prefix
            for prefix in self.api_paths:
                if path.startswith(prefix):
                    return await self.api_app(scope, receive, send)
        # Everything else goes to MCP
        return await self.mcp_app(scope, receive, send)


def create_app():
    """Create the combined application with FastAPI (Swagger) + MCP at root."""
    from starlette.applications import Starlette

    mcp = create_mcp_server()
    mcp_http_app = mcp.http_app()

    # Create FastAPI app with Swagger docs
    api_app = create_api_app()

    # Create Starlette app for static files
    static_app = Starlette(
        routes=[
            Mount("/ui", app=StaticFiles(directory=str(config.UI_DIR), html=True), name="ui"),
        ]
    )

    # Combine: static files checked first, then dispatch between API and MCP
    class CombinedApp:
        def __init__(self):
            self.static = static_app
            self.dispatcher = PathDispatcher(api_app, mcp_http_app)

        async def __call__(self, scope, receive, send):
            path = scope.get("path", "/")
            # UI static files
            if path.startswith("/ui"):
                return await self.static(scope, receive, send)
            # Everything else goes through dispatcher
            return await self.dispatcher(scope, receive, send)

    app = CombinedApp()

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
    base_url = config.get_base_url()
    print("=" * 60)
    print("SolSpeak TTS Server")
    print("=" * 60)
    print(f"Local URL:   http://127.0.0.1:{config.PORT}")
    print(f"Network URL: http://{config.SERVER_IP}:{config.PORT}")
    print("-" * 60)
    print("API Docs:")
    print(f"  Swagger UI: {base_url}/docs")
    print(f"  ReDoc:      {base_url}/redoc")
    print(f"  OpenAPI:    {base_url}/openapi.json")
    print("-" * 60)
    print("MCP Config:")
    print(f'  "solspeak": {{')
    print(f'    "type": "http",')
    print(f'    "url": "{base_url}/mcp"')
    print(f'  }}')
    print("-" * 60)
    print(f"Web UI:      {base_url}/ui/")
    print("=" * 60)


def main():
    """Run the server."""
    print_startup_info()
    cleanup_old_outputs()  # Clean up old files on startup
    app = create_app()
    uvicorn.run(app, host=config.HOST, port=config.PORT, ws="wsproto")


if __name__ == "__main__":
    main()
