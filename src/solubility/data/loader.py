"""Dataset loaders for DeepSol (PSI-Biology) and eSOL (E. coli cell-free)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import pandas as pd


def load_deepsol_split(
    raw_dir: str | Path,
    split: Literal["train", "val", "test"] = "train",
) -> pd.DataFrame:
    """Load a DeepSol split (Khurana et al. 2018) into a DataFrame.

    Returns:
        DataFrame with columns: ``sequence`` (str), ``label`` (int, 0=insoluble, 1=soluble).
    """
    raw_dir = Path(raw_dir)
    deepsol_dir = raw_dir / "deepsol" if (raw_dir / "deepsol").exists() else raw_dir

    src_file = deepsol_dir / f"{split}_src"
    tgt_file = deepsol_dir / f"{split}_tgt"

    if not src_file.exists() or not tgt_file.exists():
        raise FileNotFoundError(f"DeepSol split files not found at {src_file} or {tgt_file}")

    with src_file.open("r", encoding="utf-8") as f:
        seqs = [line.strip().upper() for line in f if line.strip()]

    with tgt_file.open("r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    if len(seqs) != len(labels):
        raise ValueError(f"Mismatch between sequences ({len(seqs)}) and labels ({len(labels)})")

    df = pd.DataFrame({"sequence": seqs, "label": labels})
    df["split"] = split
    return df


def load_esol_split(
    raw_dir: str | Path,
    split: Literal["train", "test"] = "train",
) -> pd.DataFrame:
    """Load an eSOL split (Niwa et al. 2009; GraphSol split) into a DataFrame.

    Returns:
        DataFrame with columns: ``gene``, ``solubility`` (float in [0, 1]), ``sequence``, ``label`` (binary, >=0.5).
    """
    raw_dir = Path(raw_dir)
    esol_dir = raw_dir / "esol" if (raw_dir / "esol").exists() else raw_dir

    filename = f"eSol_{split}.csv"
    filepath = esol_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"eSOL file not found at {filepath}")

    df = pd.read_csv(filepath)
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["solubility"] = df["solubility"].astype(float)
    df["label"] = (df["solubility"] >= 0.5).astype(int)
    df["split"] = split
    return df

