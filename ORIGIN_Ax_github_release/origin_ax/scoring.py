"""Additive ORIGIN-Ax v1 family scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from axis_family import AxisFamily
from core import RESULT_COLUMNS, safe_spearman


def percentile_norm(values: np.ndarray, fallback: tuple[float, float]) -> tuple[np.ndarray, dict]:
    values = np.asarray(values, dtype=float)
    if len(values) < 4 or np.nanstd(values) < 1e-9:
        lo, hi = fallback
    else:
        lo, hi = np.nanpercentile(values, [5, 95])
        if hi - lo < 1e-9:
            lo, hi = fallback
    norm = np.clip((values - lo) / (hi - lo + 1e-9), 0, 1)
    return norm, {"p5": float(lo), "p95": float(hi)}


def _family_stability(f: AxisFamily) -> float:
    methods = {m.source_method for m in f.members}
    n_score = min(1.0, len(f.members) / 4.0)
    method_score = min(1.0, len(methods) / 3.0)
    return float(0.4 * n_score + 0.3 * method_score + 0.3 * f.mean_loading_similarity)


def _axis_tail_weighted_mean(values: np.ndarray, loading: np.ndarray) -> float:
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


def score_families(
    families: list[AxisFamily],
    genes: np.ndarray,
    gene_weights: dict[str, np.ndarray],
    clone: np.ndarray,
    niche: np.ndarray,
    global_gates: dict,
    truth_eval_genes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    for f in families:
        load = f.consensus_loading
        target = _axis_tail_weighted_mean(gene_weights["target_support"], load)
        normal = _axis_tail_weighted_mean(gene_weights["normal_epithelial_penalty"], load)
        tme = _axis_tail_weighted_mean(gene_weights["tme_penalty"], load)
        technical = 0.0
        origin_margin = target - normal - tme
        off_target = max(normal, tme, technical)
        identifiability_margin = target - off_target
        dominant_offtarget_source = "normal_epithelial"
        if tme >= max(normal, technical):
            dominant_offtarget_source = "tme"
        elif technical >= max(normal, tme):
            dominant_offtarget_source = "technical"
        stability = _family_stability(f)
        anchor_alignment = abs(safe_spearman(f.consensus_score, clone))
        niche_alignment = abs(safe_spearman(f.consensus_score, niche))
        identifiability = float(np.mean([
            global_gates.get("clone_gate_pass", False),
            global_gates.get("overlap_gate_pass", False),
            global_gates.get("attribution_valid", False),
        ]))
        truth_overlap = 0
        if truth_eval_genes:
            truth_overlap = len(set(f.top_genes[:30]) & truth_eval_genes)
        raw_rows.append({
            "family": f,
            "origin_margin": origin_margin,
            "identifiability_margin": identifiability_margin,
            "dominant_offtarget_source": dominant_offtarget_source,
            "stability": stability,
            "normal_epithelial_affinity": normal,
            "tme_affinity": tme,
            "technical_affinity": technical,
            "off_target_affinity": off_target,
            "anchor_alignment": anchor_alignment,
            "niche_alignment": niche_alignment,
            "identifiability_score": identifiability,
            "heldout_truth_overlap": truth_overlap,
        })
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS), pd.DataFrame()

    origin_norm, odiag = percentile_norm(raw["origin_margin"].to_numpy(), fallback=(-1, 1))
    stability_norm, sdiag = percentile_norm(raw["stability"].to_numpy(), fallback=(0, 1))
    off_norm, fdiag = percentile_norm(raw["off_target_affinity"].to_numpy(), fallback=(0, 1))
    rows = []
    for i, r in raw.iterrows():
        fam: AxisFamily = r["family"]
        score = float(0.6 * origin_norm[i] + 0.4 * stability_norm[i] - 1.0 * off_norm[i])
        score = float(np.clip(score, -1, 1))
        max_off_name = "normal_epithelial_like"
        if r["tme_affinity"] >= max(r["normal_epithelial_affinity"], r["technical_affinity"]):
            max_off_name = "tme_like"
        elif r["technical_affinity"] >= max(r["normal_epithelial_affinity"], r["tme_affinity"]):
            max_off_name = "technical_like"

        reason = ""
        conflict_high = fam.mean_origin_conflict > 0.35
        margin_close = abs(r["identifiability_margin"]) < 0.05
        if score > 0.20 and r["identifiability_margin"] > 0 and not conflict_high:
            family_label = "target_origin_supported"
        elif score < -0.20 or r["identifiability_margin"] < -0.05:
            family_label = max_off_name
        else:
            family_label = "mixed_origin" if (r["off_target_affinity"] > 0.15 and r["origin_margin"] > -0.1 and not margin_close) else "non_identifiable"
        if conflict_high and family_label == "target_origin_supported":
            family_label = "non_identifiable"
            reason = "family_origin_conflict_high"

        if family_label == "target_origin_supported" and r["identifiability_score"] < 0.50:
            family_label = "target_origin_supported_but_non_identifiable"
            reason = "identifiability_low"
        elif family_label == "target_origin_supported" and r["anchor_alignment"] < 0.15 and r["niche_alignment"] < 0.15:
            family_label = "target_origin_supported_but_attribution_unresolved"
            reason = "anchor_and_niche_alignment_low"

        attribution_label = "attribution_not_allowed"
        if family_label.startswith("target_origin_supported"):
            if not global_gates.get("attribution_valid", False):
                attribution_label = "attribution_not_allowed"
            elif r["identifiability_score"] < 0.50:
                attribution_label = "attribution_unresolved"
            else:
                clone_hit = r["anchor_alignment"] >= 0.15
                niche_hit = r["niche_alignment"] >= 0.15
                if clone_hit and niche_hit:
                    attribution_label = "mixed_clone_niche"
                elif clone_hit:
                    attribution_label = "clone_associated"
                elif niche_hit:
                    attribution_label = "niche_associated"
                else:
                    attribution_label = "attribution_unresolved"
        final_call = "ambiguous_axis"
        if family_label == "target_origin_supported":
            final_call = "target_axis"
        elif family_label in {"target_origin_supported_but_attribution_unresolved", "target_origin_supported_but_non_identifiable"}:
            final_call = "ambiguous_axis"
        elif family_label in {"normal_epithelial_like", "tme_like", "technical_like", "off_target_like"}:
            final_call = "off_target_axis"
        elif family_label == "non_identifiable":
            final_call = "not_target_axis"

        rows.append({
            "family_id": fam.family_id,
            "member_axes": ";".join(m.axis_id for m in fam.members),
            "source_methods": ";".join(sorted({m.source_method for m in fam.members})),
            "n_members": len(fam.members),
            "top_genes": ",".join(fam.top_genes[:12]),
            "consensus_top_genes": ",".join(fam.top_genes[:12]),
            "mean_loading_similarity": fam.mean_loading_similarity,
            "family_stability": r["stability"],
            "orientation_flipped": ";".join(map(str, fam.orientation_flipped)),
            "origin_margin": r["origin_margin"],
            "identifiability_margin": r["identifiability_margin"],
            "dominant_offtarget_source": r["dominant_offtarget_source"],
            "origin_margin_norm": origin_norm[i],
            "stability": r["stability"],
            "stability_norm": stability_norm[i],
            "normal_epithelial_affinity": r["normal_epithelial_affinity"],
            "tme_affinity": r["tme_affinity"],
            "technical_affinity": r["technical_affinity"],
            "off_target_affinity": r["off_target_affinity"],
            "off_target_affinity_norm": off_norm[i],
            "origin_core_score": score,
            "mean_origin_compatibility": fam.mean_origin_compatibility,
            "mean_origin_conflict": fam.mean_origin_conflict,
            "anchor_alignment": r["anchor_alignment"],
            "identifiability_score": r["identifiability_score"],
            "family_label": family_label,
            "attribution_label": attribution_label,
            "final_call": final_call,
            "rejection_or_downgrade_reason": reason,
            "heldout_truth_overlap": int(r["heldout_truth_overlap"]),
        })
    diag = pd.DataFrame([
        {"score": "origin_margin", **odiag},
        {"score": "stability", **sdiag},
        {"score": "off_target_affinity", **fdiag},
    ])
    return pd.DataFrame(rows), diag
