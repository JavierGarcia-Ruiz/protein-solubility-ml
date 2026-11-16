"""Unit tests for feature engineering."""

import numpy as np
import pytest
from solubility.data.features import (
    clean_sequence,
    compute_net_charge,
    extract_single_sequence_features,
    featurize_sequences,
)


def test_clean_sequence():
    assert clean_sequence("MKWVXUBZ") == "MKWV"
    assert clean_sequence("acdef") == "ACDEF"


def test_compute_net_charge():
    # Poly-lysine should be strongly positive
    poly_k = "KKKKKKKKKK"
    charge_k = compute_net_charge(poly_k, ph=7.0)
    assert charge_k > 9.0

    # Poly-glutamate should be strongly negative
    poly_e = "EEEEEEEEEE"
    charge_e = compute_net_charge(poly_e, ph=7.0)
    assert charge_e < -9.0


def test_extract_single_sequence_features():
    seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPF"
    feats = extract_single_sequence_features(seq)

    assert len(feats) == 935
    assert feats["length"] == len(seq)
    assert 0.0 <= feats["aac_A"] <= 1.0
    assert 0.0 <= feats["frac_charged"] <= 1.0
    assert not np.isnan(feats["gravy"])
    assert not np.isnan(feats["net_charge_ph7"])


def test_featurize_sequences_batch():
    seqs = ["MKWVTFIS", "ACDEFGHIKLMNPQRSTVWY"]
    df = featurize_sequences(seqs, n_jobs=2)
    assert df.shape == (2, 935)
    assert not df.isna().any().any()

