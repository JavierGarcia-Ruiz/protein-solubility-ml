"""Vectorized and parallel feature extraction for protein sequences."""

from __future__ import annotations

import math
from itertools import product
from typing import Sequence
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from joblib import Parallel, delayed

STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(STANDARD_AAS)

DIPEPTIDES = ["".join(p) for p in product(STANDARD_AAS, repeat=2)]

# Conjoint Triad classification (Shen et al., 2007)
CONJOINT_MAP = {
    "A": 0, "G": 0, "V": 0,
    "I": 1, "L": 1, "F": 1, "P": 1,
    "Y": 2, "M": 2, "T": 2, "S": 2,
    "H": 3, "N": 3, "Q": 3, "W": 3,
    "R": 4, "K": 4,
    "D": 5, "E": 5,
    "C": 6,
}
TRIAD_KEYS = ["".join(map(str, p)) for p in product(range(7), repeat=3)]

# CTD 7 Property definitions (Tomii & Kanehisa 1996; Dubchak et al. 1995)
CTD_PROPERTIES = {
    "hydrophobicity": {
        1: set("RKEDQN"),
        2: set("GASTPHY"),
        3: set("CLVIMFW"),
    },
    "vdw_volume": {
        1: set("GASTPDC"),
        2: set("NVEQIL"),
        3: set("MHKFRYW"),
    },
    "polarity": {
        1: set("LIFWCMVY"),
        2: set("PATGS"),
        3: set("HQRK NED".replace(" ", "")),
    },
    "polarizability": {
        1: set("GASDT"),
        2: set("CPNVEQIL"),
        3: set("KMHFRYW"),
    },
    "charge": {
        1: set("KR"),
        2: set("ANCQGHILMFPSTWYV"),
        3: set("DE"),
    },
    "secondary_struct": {
        1: set("EALMQKRH"),
        2: set("VIYCWFT"),
        3: set("GNPSD"),
    },
    "solvent_access": {
        1: set("ALFCGIVW"),
        2: set("RKQEND"),
        3: set("MSPTHY"),
    },
}

# Kyte-Doolittle hydropathy scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# pKa values for net charge calculation
PKA = {
    "N_term": 9.69, "C_term": 2.34,
    "C": 8.33, "D": 3.86, "E": 4.25,
    "H": 6.00, "K": 10.50, "R": 12.48, "Y": 10.00,
}


def clean_sequence(seq: str) -> str:
    """Filter out non-standard amino acids."""
    return "".join(c for c in seq.upper() if c in AA_SET)


def compute_net_charge(seq: str, ph: float = 7.0) -> float:
    """Compute theoretical net charge of a sequence at a given pH."""
    # N-terminus
    charge = 1.0 / (1.0 + 10 ** (ph - PKA["N_term"]))
    # C-terminus
    charge -= 1.0 / (1.0 + 10 ** (PKA["C_term"] - ph))
    
    counts = {aa: seq.count(aa) for aa in "CDEHKRY"}
    
    # Positive side chains (Arg, Lys, His)
    charge += counts["R"] / (1.0 + 10 ** (ph - PKA["R"]))
    charge += counts["K"] / (1.0 + 10 ** (ph - PKA["K"]))
    charge += counts["H"] / (1.0 + 10 ** (ph - PKA["H"]))
    
    # Negative side chains (Asp, Glu, Cys, Tyr)
    charge -= counts["D"] / (1.0 + 10 ** (PKA["D"] - ph))
    charge -= counts["E"] / (1.0 + 10 ** (PKA["E"] - ph))
    charge -= counts["C"] / (1.0 + 10 ** (PKA["C"] - ph))
    charge -= counts["Y"] / (1.0 + 10 ** (PKA["Y"] - ph))
    
    return float(charge)


def extract_single_sequence_features(seq: str) -> dict[str, float]:
    """Extract all features for a single protein sequence."""
    clean_seq = clean_sequence(seq)
    L = len(clean_seq)
    if L < 5:
        # Fallback for empty or ultra-short sequences
        clean_seq = "ACDEF"
        L = len(clean_seq)

    feats: dict[str, float] = {}
    
    # 1. Amino acid composition (20)
    for aa in STANDARD_AAS:
        feats[f"aac_{aa}"] = clean_seq.count(aa) / L
        
    # 2. Physicochemical & global descriptors (~20)
    feats["length"] = float(L)
    feats["log_length"] = math.log(L)
    
    # Hydropathy (GRAVY)
    feats["gravy"] = sum(KD_SCALE[aa] for aa in clean_seq) / L
    
    # Net charge at pH 7
    net_charge = compute_net_charge(clean_seq, ph=7.0)
    feats["net_charge_ph7"] = net_charge
    feats["net_charge_per_res"] = net_charge / L
    
    # Charged fractions
    pos_res = clean_seq.count("K") + clean_seq.count("R")
    neg_res = clean_seq.count("D") + clean_seq.count("E")
    feats["frac_pos_charge"] = pos_res / L
    feats["frac_neg_charge"] = neg_res / L
    feats["frac_charged"] = (pos_res + neg_res) / L
    feats["charge_balance"] = (pos_res - neg_res) / (pos_res + neg_res + 1e-5)
    feats["frac_K"] = clean_seq.count("K") / L
    feats["frac_R"] = clean_seq.count("R") / L
    feats["lys_arg_ratio"] = clean_seq.count("K") / (pos_res + 1e-5)
    
    # Hydrophobic / Aromatic / Aliphatic / Turn-forming fractions
    hydro_res = sum(clean_seq.count(aa) for aa in "AILVMFW")
    feats["frac_hydrophobic"] = hydro_res / L
    feats["frac_aromatic"] = sum(clean_seq.count(aa) for aa in "FWY") / L
    feats["aliphatic_index"] = (
        clean_seq.count("A") + 2.9 * clean_seq.count("V") + 3.9 * (clean_seq.count("I") + clean_seq.count("L"))
    ) / L
    feats["frac_turn_forming"] = sum(clean_seq.count(aa) for aa in "NGPS") / L
    feats["frac_tiny"] = sum(clean_seq.count(aa) for aa in "ACGST") / L
    feats["frac_polar_uncharged"] = sum(clean_seq.count(aa) for aa in "STNQYC") / L
    
    # ProtParam descriptors
    try:
        analysed = ProteinAnalysis(clean_seq)
        feats["mw"] = analysed.molecular_weight()
        feats["mw_per_res"] = feats["mw"] / L
        feats["pi"] = analysed.isoelectric_point()
        feats["instability_index"] = analysed.instability_index()
        ss_frac = analysed.secondary_structure_fraction()
        feats["ss_helix"] = ss_frac[0]
        feats["ss_turn"] = ss_frac[1]
        feats["ss_sheet"] = ss_frac[2]
    except Exception:
        feats["mw"] = L * 110.0
        feats["mw_per_res"] = 110.0
        feats["pi"] = 6.5
        feats["instability_index"] = 40.0
        feats["ss_helix"] = 0.3
        feats["ss_turn"] = 0.2
        feats["ss_sheet"] = 0.2

    # 3. Dipeptide composition (400)
    if L > 1:
        dipep_denom = L - 1
        d_counts = {dp: 0 for dp in DIPEPTIDES}
        for i in range(L - 1):
            pair = clean_seq[i : i + 2]
            if pair in d_counts:
                d_counts[pair] += 1
        for dp, cnt in d_counts.items():
            feats[f"dpc_{dp}"] = cnt / dipep_denom
    else:
        for dp in DIPEPTIDES:
            feats[f"dpc_{dp}"] = 0.0

    # 4. CTD (Composition, Transition, Distribution) (147)
    for prop_name, groups in CTD_PROPERTIES.items():
        # Map sequence to group 1, 2, 3
        mapped = []
        for aa in clean_seq:
            if aa in groups[1]:
                mapped.append(1)
            elif aa in groups[2]:
                mapped.append(2)
            elif aa in groups[3]:
                mapped.append(3)
            else:
                mapped.append(2)  # default group
        
        # Composition (3)
        n1 = mapped.count(1)
        n2 = mapped.count(2)
        n3 = mapped.count(3)
        feats[f"ctd_C_{prop_name}_g1"] = n1 / L
        feats[f"ctd_C_{prop_name}_g2"] = n2 / L
        feats[f"ctd_C_{prop_name}_g3"] = n3 / L

        # Transition (3)
        if L > 1:
            t12 = t13 = t23 = 0
            for i in range(L - 1):
                p = (mapped[i], mapped[i + 1])
                if p in ((1, 2), (2, 1)):
                    t12 += 1
                elif p in ((1, 3), (3, 1)):
                    t13 += 1
                elif p in ((2, 3), (3, 2)):
                    t23 += 1
            denom = L - 1
            feats[f"ctd_T_{prop_name}_12"] = t12 / denom
            feats[f"ctd_T_{prop_name}_13"] = t13 / denom
            feats[f"ctd_T_{prop_name}_23"] = t23 / denom
        else:
            feats[f"ctd_T_{prop_name}_12"] = 0.0
            feats[f"ctd_T_{prop_name}_13"] = 0.0
            feats[f"ctd_T_{prop_name}_23"] = 0.0

        # Distribution (15: 1st, 25%, 50%, 75%, 100% position for each group)
        for g_idx, g_count in [(1, n1), (2, n2), (3, n3)]:
            prefix = f"ctd_D_{prop_name}_g{g_idx}"
            if g_count == 0:
                for frac in [0, 25, 50, 75, 100]:
                    feats[f"{prefix}_{frac}"] = 0.0
            else:
                # find indices of occurrences
                indices = [idx + 1 for idx, g in enumerate(mapped) if g == g_idx]
                feats[f"{prefix}_0"] = indices[0] / L
                feats[f"{prefix}_25"] = indices[max(0, int(math.ceil(0.25 * g_count)) - 1)] / L
                feats[f"{prefix}_50"] = indices[max(0, int(math.ceil(0.50 * g_count)) - 1)] / L
                feats[f"{prefix}_75"] = indices[max(0, int(math.ceil(0.75 * g_count)) - 1)] / L
                feats[f"{prefix}_100"] = indices[-1] / L

    # 5. Conjoint Triad (343)
    if L >= 3:
        triad_mapped = [CONJOINT_MAP.get(aa, 0) for aa in clean_seq]
        triad_counts = {k: 0 for k in TRIAD_KEYS}
        for i in range(L - 2):
            key = f"{triad_mapped[i]}{triad_mapped[i+1]}{triad_mapped[i+2]}"
            if key in triad_counts:
                triad_counts[key] += 1
        denom = L - 2
        for k, cnt in triad_counts.items():
            feats[f"triad_{k}"] = cnt / denom
    else:
        for k in TRIAD_KEYS:
            feats[f"triad_{k}"] = 0.0

    return feats


def featurize_sequences(
    sequences: Sequence[str],
    n_jobs: int = 6,
    batch_size: int = 250,
) -> pd.DataFrame:
    """Extract features for a list of sequences in parallel.

    Returns:
        DataFrame of shape (N, n_features) with float32 values.
    """
    records = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
        delayed(extract_single_sequence_features)(seq) for seq in sequences
    )
    df = pd.DataFrame(records)
    return df.astype(np.float32)

