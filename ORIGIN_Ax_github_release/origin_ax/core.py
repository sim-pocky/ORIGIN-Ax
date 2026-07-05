"""Shared utilities for ORIGIN-Ax v1.

ORIGIN-Ax is a consensus axis inference layer. It keeps v0 MalAxis-OA intact and adds
multi-candidate axis families with origin-aware additive scoring.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V0_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../malaxis_oa"))
_spec = importlib.util.spec_from_file_location("malaxis_oa_v0_core", os.path.join(V0_DIR, "core.py"))
if _spec is None or _spec.loader is None:
    raise ImportError("Cannot load MalAxis-OA v0 core.py")
_v0 = importlib.util.module_from_spec(_spec)
sys.modules["malaxis_oa_v0_core"] = _v0
_spec.loader.exec_module(_v0)

EPI = _v0.EPI
NORMAL_EPI = _v0.NORMAL_EPI
TME = _v0.TME
bootstrap_spatial_stability = _v0.bootstrap_spatial_stability
gate_clone_niche_overlap = _v0.gate_clone_niche_overlap
gate_clone_reliability = _v0.gate_clone_reliability
marker_score = _v0.marker_score
marker_clone_proxy = _v0.marker_clone_proxy
local_niche_score = _v0.local_niche_score
moran_i = _v0.moran_i
rank01 = _v0.rank01
zscore = _v0.zscore
GateThresholds = _v0.GateThresholds


RESULT_COLUMNS = [
    "family_id",
    "member_axes",
    "source_methods",
    "n_members",
    "top_genes",
    "consensus_top_genes",
    "mean_loading_similarity",
    "family_stability",
    "orientation_flipped",
    "origin_margin",
    "identifiability_margin",
    "dominant_offtarget_source",
    "origin_margin_norm",
    "stability",
    "stability_norm",
    "normal_epithelial_affinity",
    "tme_affinity",
    "technical_affinity",
    "off_target_affinity",
    "off_target_affinity_norm",
    "origin_core_score",
    "mean_origin_compatibility",
    "mean_origin_conflict",
    "anchor_alignment",
    "identifiability_score",
    "family_label",
    "attribution_label",
    "final_call",
    "rejection_or_downgrade_reason",
]


@dataclass
class OriginAxConfig:
    k: int = 6
    seeds: tuple[int, ...] = (0, 1)
    similarity_threshold: float = 0.35
    origin_graph_enabled: bool = True
    origin_compatibility_weight: float = 0.15
    origin_conflict_penalty: float = 0.60
    target_threshold: float = 0.20
    off_target_threshold: float = -0.20
    attribution_r: float = 0.15
    identifiability_min: float = 0.50


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else 0.0


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else 0.0


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.nan_to_num(values), weights=np.abs(weights) + 1e-9))


def gene_weight_arrays_from_table(gene_weight: pd.DataFrame, genes: np.ndarray) -> dict[str, np.ndarray]:
    by_gene = gene_weight.drop_duplicates("gene").set_index("gene")
    aliases = {
        "target_support": ["target_support", "malignant_support"],
        "normal_epithelial_penalty": ["normal_epithelial_penalty"],
        "tme_penalty": ["tme_penalty"],
        "final_weight": ["final_weight"],
    }
    out: dict[str, np.ndarray] = {}
    for out_col, candidates in aliases.items():
        vals = []
        for g in genes.astype(str):
            v = 0.0
            if g in by_gene.index:
                for c in candidates:
                    if c in by_gene.columns:
                        v = float(by_gene.loc[g, c])
                        break
            vals.append(v)
        out[out_col] = np.nan_to_num(np.asarray(vals, dtype=float))
    if "final_weight" not in out or not np.any(out["final_weight"]):
        out["final_weight"] = out["target_support"] - out["normal_epithelial_penalty"] - out["tme_penalty"]
    return out


def make_gene_weight_table(
    genes: np.ndarray,
    target_support: np.ndarray,
    normal_penalty: np.ndarray | None = None,
    tme_penalty: np.ndarray | None = None,
    source: str = "origin_ax",
) -> pd.DataFrame:
    normal_penalty = np.zeros(len(genes)) if normal_penalty is None else normal_penalty
    tme_penalty = np.zeros(len(genes)) if tme_penalty is None else tme_penalty
    final = target_support - normal_penalty - tme_penalty
    return pd.DataFrame({
        "gene": genes.astype(str),
        "target_support": target_support,
        "malignant_support": target_support,
        "normal_epithelial_penalty": normal_penalty,
        "tme_penalty": tme_penalty,
        "final_weight": final,
        "source": source,
    })


def top_genes_from_loading(genes: np.ndarray, loading: np.ndarray, n: int = 30) -> list[str]:
    idx = np.argsort(-loading)[:n]
    # If the positive side is weak, fall back to absolute ranking for reporting.
    if len(idx) == 0 or loading[idx[0]] <= 0:
        idx = np.argsort(-np.abs(loading))[:n]
    return [str(genes[i]) for i in idx]


def target_support_score(x: np.ndarray, gene_weights: dict[str, np.ndarray]) -> tuple[np.ndarray | None, dict]:
    idx = np.where(gene_weights["target_support"] > 0)[0]
    if len(idx) < 10:
        return None, {
            "support_gene_count": int(len(idx)),
            "subset_size": 0,
            "score_quantile_threshold": np.nan,
            "skipped_reason": "support_genes_lt_10",
        }
    score = np.log1p(np.clip(x[:, idx], 0, None)).mean(axis=1)
    return rank01(score), {
        "support_gene_count": int(len(idx)),
        "subset_size": 0,
        "score_quantile_threshold": float(np.quantile(score, 0.70)),
        "skipped_reason": "",
    }


def global_gate_summary(clone: np.ndarray, niche: np.ndarray, coords: np.ndarray, attribution_valid: bool) -> dict:
    th = GateThresholds()
    g1 = gate_clone_reliability(clone, coords, th)
    g2 = gate_clone_niche_overlap(clone, niche, th)
    return {
        "clone_gate_pass": bool(g1["passed"]),
        "overlap_gate_pass": bool(g2["passed"]),
        "global_gates_pass": bool(g1["passed"] and g2["passed"] and attribution_valid),
        "clone_moran": float(g1["clone_moran"]),
        "clone_bootstrap": float(g1["clone_bootstrap"]),
        "overlap_coverage": float(g2["overlap_coverage"]),
        "overlap_entropy": float(g2["overlap_entropy"]),
        "attribution_valid": bool(attribution_valid),
    }
