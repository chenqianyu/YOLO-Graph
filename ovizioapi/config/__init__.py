import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]  # Root folder of the project
path_to_config = (
    PROJECT_ROOT / "config" / "config.json"
)  # Absolute path to the config file

# Load the main config variable
with path_to_config.open() as f:
    cfg = json.load(f)
