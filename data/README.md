# Dataset Provenance & Split Strategy

This project benchmarks protein solubility across two complementary experimental paradigms:

## 1. DeepSol / PSI-Biology Dataset (Binary Classification)
- **Source**: Khurana et al. (2018), *Bioinformatics*, 34(15):2605–2613. [DOI: 10.1093/bioinformatics/bty166](https://doi.org/10.1093/bioinformatics/bty166).
- **GitHub Repository**: [sameerkhurana10/DSOL_rv0.2](https://github.com/sameerkhurana10/DSOL_rv0.2)
- **Experimental System**: Cell-free and in vivo expression from the Protein Structure Initiative (PSI-Biology / TargetDB / PepcDB).
- **Target**: Binary solubility (1 = soluble upon expression/cell lysis, 0 = insoluble/inclusion bodies).
- **Splits & Homology Separation**:
  - `train`: 62,478 sequences (development split)
  - `val`: 6,942 sequences (development split)
  - `test`: 2,001 sequences (held-out evaluation set)
  - All splits were strictly homology-reduced at 25% sequence identity using CD-HIT to prevent data leakage between development and test sets.

## 2. eSOL Dataset (Continuous Solubility Ratio Regression)
- **Source**: Niwa et al. (2009), *PNAS*, 106(11):4201–4206. [DOI: 10.1073/pnas.0811922106](https://doi.org/10.1073/pnas.0811922106).
- **Split Version**: As curated and split by GraphSol (Chen et al. 2021, *J. Cheminform.*, 13:7, [DOI: 10.1186/s13321-021-00488-1](https://doi.org/10.1186/s13321-021-00488-1)). [GitHub jcchan23/GraphSol](https://github.com/jcchan23/GraphSol).
- **Experimental System**: Reconstituted cell-free transcription-translation system (PURE system) spanning 3,147 cytosolic *E. coli* proteins.
- **Target**: Continuous solubility ratio $S \in [0.0, 1.0]$, defined as the ratio of supernatant protein concentration to total synthesized protein.
- **Splits**:
  - `eSol_train.csv`: 2,364 proteins
  - `eSol_test.csv`: 783 proteins

## Featurization
All raw sequences are converted into 935-dimensional vectorized physicochemical representations using `scripts/preprocess.py` and cached locally as `.parquet` files under `data/processed/` (which are excluded from version control via `.gitignore`).

