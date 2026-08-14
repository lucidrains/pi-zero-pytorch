import logging
import threading
from pathlib import Path

import click
import uvicorn

from .app import create_app
from .config import load_recap_config
from .networks import SmallValueNetwork
from .state import app_state
from .storage import init_replay_buffer

logger = logging.getLogger(__name__)


@click.command()
@click.option('--port', default=8000, help='Port to run the server on.')
@click.option('--folder', 'folders', multiple=True, help='Path to video directory for standalone mode.')
@click.option('--recap-workspace', default=None, help='Path to RECAP algorithm workspace folder.')
def main(port, folders, recap_workspace):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    state = app_state
    state.recap_config = load_recap_config()

    # Initialize RECAP workspace
    if recap_workspace:
        state.recap_workspace = Path(recap_workspace)
        state.recap_workspace.mkdir(parents=True, exist_ok=True)
        logger.info("RECAP workspace enabled: %s", state.recap_workspace)
    else:
        state.recap_workspace = None

    valid_dirs = []
    if folders:
        for vdir in [Path(f) for f in folders]:
            if not vdir.exists():
                logger.warning("folder %s does not exist", vdir)
            else:
                valid_dirs.append(vdir)

        if not valid_dirs:
            logger.error("no valid folders provided")
            return

    # Initialize conversion status
    state.conversion_status = {
        "is_converting": bool(folders),  # Set to True immediately if in standalone mode
        "progress": 0,
        "total": 0,
        "current_video": ""
    }

    if valid_dirs:
        state.video_dirs = valid_dirs
        logger.info("standalone mode: loading videos from %d folders", len(state.video_dirs))
        # Initialize buffer immediately for standalone mode
        threading.Thread(target=init_replay_buffer, args=(state, state.video_dirs), daemon=True).start()

    logger.info("initializing SmallValueNetwork on %s", state.device)
    state.value_network = SmallValueNetwork().to(state.device)

    app = create_app(state)
    logger.info("starting trajectory labeller at http://localhost:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
