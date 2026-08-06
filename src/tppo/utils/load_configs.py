"""
Loads agent configs for currently .json and .toml files.
"""

import json 
import tomllib 
from pathlib import Path
from typing import Any 

CONFIG_DIR = Path(__file__).parent.parent / "agent_configs"

def load_config(path: str | Path) -> dict[str, Any]: 
    """
    Loads .toml or .json configs. 

    Path should be the entire file (ie. "my_model.json"). 
    Returns a dictionary of your config. 
    """

    path = Path(CONFIG_DIR / path)
    suffix = path.suffix.lower() 

    if suffix == ".json": 
        with open(path) as f: 
            return json.load(f)
    elif suffix == ".toml": 
        with open(path, "rb") as f: 
            return tomllib.load(f)
    else: 
        raise ValueError(f"Unsupported config format: {suffix}. Only .toml or .json configs can be loaded.")