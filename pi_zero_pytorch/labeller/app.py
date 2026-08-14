from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .routes import router
from .errors import ApiError
from .state import AppState, app_state


def create_app(state: AppState = None) -> FastAPI:
    """Application factory: builds the FastAPI app and mounts static assets."""
    if state is None:
        state = app_state

    app = FastAPI(title="Pi0 Trajectory Labeller")
    app.state.labeller = state
    app.include_router(router)

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"error": exc.message}, status_code=exc.status_code)

    # Mount the frame cache directory
    state.cache_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/cache", StaticFiles(directory=str(state.cache_dir)), name="cache")

    # Mount the UI assets
    ui_dir = Path(__file__).parent / "web_ui"
    app.mount("/ui", StaticFiles(directory=str(ui_dir)), name="ui")

    return app
