"""Preprocess raw DeepSol and eSOL datasets and cache featurized parquets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import pandas as pd

from solubility.data.loader import load_deepsol_split, load_esol_split
from solubility.data.features import featurize_sequences
from solubility.utils import get_logger, load_config, seed_everything


def preprocess_all(config_path: str = "configs/default.yaml", n_jobs: int = 12) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    logger = get_logger("solubility.preprocess", log_file=cfg["paths"]["logs"] / "preprocess.log")
    
    raw_dir = Path(cfg["paths"]["raw"])
    proc_dir = Path(cfg["paths"]["processed"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. DeepSol splits
    for split in ["test", "val", "train"]:
        out_file = proc_dir / f"deepsol_{split}.parquet"
        if out_file.exists():
            logger.info(f"Skipping existing {out_file}")
            continue
            
        logger.info(f"Loading DeepSol {split} split...")
        df_raw = load_deepsol_split(raw_dir, split=split)
        logger.info(f"Extracting features for {len(df_raw)} sequences (n_jobs={n_jobs})...")
        t0 = time.time()
        X_df = featurize_sequences(df_raw["sequence"].tolist(), n_jobs=n_jobs)
        elapsed = time.time() - t0
        logger.info(f"Featurized {len(df_raw)} sequences in {elapsed:.1f}s ({len(df_raw)/elapsed:.1f} seq/s)")
        
        # Merge target and metadata
        X_df.insert(0, "sequence", df_raw["sequence"].values)
        X_df.insert(1, "label", df_raw["label"].values)
        X_df.insert(2, "split", df_raw["split"].values)
        
        X_df.to_parquet(out_file, index=False)
        logger.info(f"Saved {out_file} (shape: {X_df.shape})")

    # 2. eSOL splits
    for split in ["test", "train"]:
        out_file = proc_dir / f"esol_{split}.parquet"
        if out_file.exists():
            logger.info(f"Skipping existing {out_file}")
            continue
            
        logger.info(f"Loading eSOL {split} split...")
        df_raw = load_esol_split(raw_dir, split=split)
        logger.info(f"Extracting features for {len(df_raw)} sequences...")
        t0 = time.time()
        X_df = featurize_sequences(df_raw["sequence"].tolist(), n_jobs=n_jobs)
        elapsed = time.time() - t0
        logger.info(f"Featurized eSOL {split} in {elapsed:.1f}s")
        
        X_df.insert(0, "gene", df_raw["gene"].values)
        X_df.insert(1, "sequence", df_raw["sequence"].values)
        X_df.insert(2, "solubility", df_raw["solubility"].values)
        X_df.insert(3, "label", df_raw["label"].values)
        X_df.insert(4, "split", df_raw["split"].values)
        
        X_df.to_parquet(out_file, index=False)
        logger.info(f"Saved {out_file} (shape: {X_df.shape})")

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DeepSol and eSOL")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()
    preprocess_all(config_path=args.config, n_jobs=args.n_jobs)

