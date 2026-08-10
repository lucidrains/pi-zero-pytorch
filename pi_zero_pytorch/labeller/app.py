from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .state import AppState, app_state


def create_app(state: AppState = None) -> FastAPI:
    """Application factory: builds the FastAPI app and mounts static assets."""
    if state is None:
        state = app_state

    app = FastAPI(title="Pi0 Trajectory Labeller")
    app.state.labeller = state
    app.include_router(router)

    # Mount the frame cache directory
    state.cache_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/cache", StaticFiles(directory=str(state.cache_dir)), name="cache")

    # Mount the UI assets
    ui_dir = Path(__file__).parent / "web_ui"
    app.mount("/ui", StaticFiles(directory=str(ui_dir)), name="ui")

    return app
