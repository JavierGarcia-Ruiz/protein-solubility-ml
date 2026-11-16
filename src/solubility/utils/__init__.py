"""Configuration loading, path resolution, seeding and logging helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load a YAML config and resolve every entry of ``paths`` to an absolute directory.

    Relative paths are interpreted with respect to the project root, and the directories are
    created if they do not exist.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    resolved = {}
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        p = p if p.is_absolute() else PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        resolved[key] = p
    cfg["paths"] = resolved
    return cfg


def seed_everything(seed: int) -> None:
    """Seed python, numpy and hash-based randomness for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = "solubility", log_file: str | Path | None = None) -> logging.Logger:
    """Return a configured logger that writes to stderr (and optionally to ``log_file``)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
