import json
from pathlib import Path

RECAP_CONFIG_PATH = Path(__file__).resolve().parent.parent / "recap_config.json"


def load_recap_config() -> dict:
    if not RECAP_CONFIG_PATH.exists():
        return {}
    with open(RECAP_CONFIG_PATH) as f:
        return json.load(f)
