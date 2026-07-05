"""Axis family clustering and consensus for ORIGIN-Ax v1."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from candidate_generation import CandidateAxis
from core import safe_corr, top_genes_from_loading, zscore


@dataclass
class AxisFamily:
    family_id: str
    members: list[CandidateAxis]
    consensus_loading: np.ndarray
    consensus_score: np.ndarray
    top_genes: list[str]
    mean_loading_similarity: float
    mean_origin_compatibility: float
    mean_origin_conflict: float
    orientation_flipped: list[bool]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a[:30]), set(b[:30])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def axis_similarity(a: CandidateAxis, b: CandidateAxis) -> float:
    load = abs(safe_corr(a.loading, b.loading))
    jac = _jaccard(a.top_genes, b.top_genes)
    return float(0.6 * load + 0.4 * jac)


def _weighted_axis_evidence(values: np.ndarray, loading: np.ndarray) -> float:
    weights = np.clip(np.nan_to_num(loading, nan=0.0), 0.0, None)
    if np.sum(weights) < 1e-9:
        weights = np.abs(np.nan_to_num(loading, nan=0.0))
    if np.sum(weights) < 1e-9:
        return 0.0
    keep = np.argsort(-weights)[: min(50, len(weights))]
    mask = np.zeros_like(weights, dtype=bool)
    mask[keep] = True
    weights = np.where(mask, weights, 0.0)
    return float(np.average(np.nan_to_num(values), weights=weights + 1e-9))


def axis_origin_profile(axis: CandidateAxis, gene_weights: dict[str, np.ndarray]) -> dict[str, float]:
    """Return a compact competing-origin profile for graph refinement."""
    target = _weighted_axis_evidence(gene_weights["target_support"], axis.loading)
    normal = _weighted_axis_evidence(gene_weights["normal_epithelial_penalty"], axis.loading)
    tme = _weighted_axis_evidence(gene_weights["tme_penalty"], axis.loading)
    technical = 0.0
    off = max(normal, tme, technical)
    dominant = "target"
    if normal >= max(target, tme, technical):
        dominant = "normal_epithelial"
    elif tme >= max(target, normal, technical):
        dominant = "tme"
    elif technical >= max(target, normal, tme):
        dominant = "technical"
    return {
        "target": float(target),
        "normal_epithelial": float(normal),
        "tme": float(tme),
        "technical": float(technical),
        "offtarget": float(off),
        "margin": float(target - off),
        "dominant_origin": dominant,
    }


def _origin_compatibility(pi: dict[str, float], pj: dict[str, float]) -> float:
    if pi["dominant_origin"] != pj["dominant_origin"]:
        return 0.0
    vi = np.asarray([pi["target"], pi["normal_epithelial"], pi["tme"], pi["technical"]], dtype=float)
    vj = np.asarray([pj["target"], pj["normal_epithelial"], pj["tme"], pj["technical"]], dtype=float)
    if np.linalg.norm(vi) < 1e-9 or np.linalg.norm(vj) < 1e-9:
        return 0.0
    cos = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9))
    return float(np.clip(cos, 0.0, 1.0))


def _origin_conflict(pi: dict[str, float], pj: dict[str, float]) -> float:
    max_i = max(pi["target"], pi["normal_epithelial"], pi["tme"], pi["technical"])
    max_j = max(pj["target"], pj["normal_epithelial"], pj["tme"], pj["technical"])
    if pi["dominant_origin"] != pj["dominant_origin"] and min(max_i, max_j) >= 0.05:
        return 1.0
    target_vs_off = max(
        min(pi["target"], pj["offtarget"]),
        min(pj["target"], pi["offtarget"]),
    )
    margin_flip = 1.0 if pi["margin"] * pj["margin"] < 0 else 0.0
    different_dominant = 1.0 if pi["dominant_origin"] != pj["dominant_origin"] else 0.0
    return float(np.clip(0.5 * target_vs_off + 0.25 * margin_flip + 0.25 * different_dominant, 0.0, 1.0))


def _components(sim: np.ndarray, threshold: float) -> list[list[int]]:
    n = sim.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.where(sim[u] >= threshold)[0]:
                if v != u and not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        comps.append(comp)
    return comps


def build_axis_families(
    candidates: list[CandidateAxis],
    genes: np.ndarray,
    similarity_threshold: float = 0.35,
    gene_weights: dict[str, np.ndarray] | None = None,
    origin_graph_enabled: bool = False,
    origin_compatibility_weight: float = 0.15,
    origin_conflict_penalty: float = 0.25,
) -> tuple[list[AxisFamily], pd.DataFrame]:
    n = len(candidates)
    sim = np.eye(n)
    expr_sim = np.eye(n)
    origin_compat = np.zeros((n, n), dtype=float)
    origin_conflict = np.zeros((n, n), dtype=float)
    profiles = [axis_origin_profile(c, gene_weights) for c in candidates] if gene_weights is not None else []
    for i in range(n):
        for j in range(i + 1, n):
            s = axis_similarity(candidates[i], candidates[j])
            expr_sim[i, j] = expr_sim[j, i] = s
            w = s
            if origin_graph_enabled and gene_weights is not None:
                c_origin = _origin_compatibility(profiles[i], profiles[j])
                h_conflict = _origin_conflict(profiles[i], profiles[j])
                origin_compat[i, j] = origin_compat[j, i] = c_origin
                origin_conflict[i, j] = origin_conflict[j, i] = h_conflict
                w = s + origin_compatibility_weight * c_origin - origin_conflict_penalty * h_conflict
            sim[i, j] = sim[j, i] = float(np.clip(w, 0.0, 1.0))
    comps = _components(sim, similarity_threshold)
    families: list[AxisFamily] = []
    for fid, comp in enumerate(comps):
        members = [candidates[i] for i in comp]
        # Reference: highest positive target-oriented loading magnitude proxy, then first.
        ref = members[0].loading.copy()
        aligned_loadings, aligned_scores, flips = [], [], []
        for m in members:
            loading = m.loading.copy()
            score = m.score.copy()
            flipped = False
            if safe_corr(loading, ref) < 0:
                loading = -loading
                score = -score
                flipped = True
            aligned_loadings.append(loading)
            aligned_scores.append(score)
            flips.append(bool(m.orientation_flipped or flipped))
        cons_loading = np.mean(np.vstack(aligned_loadings), axis=0)
        cons_score = np.mean(np.vstack(aligned_scores), axis=0)
        pair_vals = []
        for i in range(len(aligned_loadings)):
            for j in range(i + 1, len(aligned_loadings)):
                pair_vals.append(abs(safe_corr(aligned_loadings[i], aligned_loadings[j])))
        mean_sim = float(np.mean(pair_vals)) if pair_vals else 1.0
        comp_pairs = [(a, b) for x, a in enumerate(comp) for b in comp[x + 1:]]
        mean_origin_compat = float(np.mean([origin_compat[a, b] for a, b in comp_pairs])) if comp_pairs and gene_weights is not None else 1.0
        mean_origin_conflict = float(np.mean([origin_conflict[a, b] for a, b in comp_pairs])) if comp_pairs and gene_weights is not None else 0.0
        families.append(AxisFamily(
            family_id=f"F{fid}",
            members=members,
            consensus_loading=cons_loading,
            consensus_score=cons_score,
            top_genes=top_genes_from_loading(genes, cons_loading, n=30),
            mean_loading_similarity=mean_sim,
            mean_origin_compatibility=mean_origin_compat,
            mean_origin_conflict=mean_origin_conflict,
            orientation_flipped=flips,
        ))
    sim_df = pd.DataFrame(sim, index=[c.axis_id for c in candidates], columns=[c.axis_id for c in candidates])
    return families, sim_df


def origin_graph_edges(
    candidates: list[CandidateAxis],
    gene_weights: dict[str, np.ndarray],
    similarity_threshold: float = 0.35,
    origin_compatibility_weight: float = 0.15,
    origin_conflict_penalty: float = 0.25,
) -> pd.DataFrame:
    """Audit table for expression-only vs origin-constrained graph edges."""
    profiles = [axis_origin_profile(c, gene_weights) for c in candidates]
    rows = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            s = axis_similarity(candidates[i], candidates[j])
            compat = _origin_compatibility(profiles[i], profiles[j])
            conflict = _origin_conflict(profiles[i], profiles[j])
            w = float(np.clip(s + origin_compatibility_weight * compat - origin_conflict_penalty * conflict, 0.0, 1.0))
            rows.append({
                "axis_i": candidates[i].axis_id,
                "axis_j": candidates[j].axis_id,
                "expression_similarity": s,
                "origin_compatibility": compat,
                "origin_conflict": conflict,
                "origin_constrained_weight": w,
                "expression_edge": bool(s >= similarity_threshold),
                "origin_constrained_edge": bool(w >= similarity_threshold),
                "origin_i": profiles[i]["dominant_origin"],
                "origin_j": profiles[j]["dominant_origin"],
                "margin_i": profiles[i]["margin"],
                "margin_j": profiles[j]["margin"],
            })
    return pd.DataFrame(rows)
