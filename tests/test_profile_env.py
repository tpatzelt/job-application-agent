import os
from pathlib import Path

from src.config_manager import load_config


def test_load_profile_env(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    # ensure loading base config works
    cfg = load_config(root)
    assert cfg.max_results >= 1
