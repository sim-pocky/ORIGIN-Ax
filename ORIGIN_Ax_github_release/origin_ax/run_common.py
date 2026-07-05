"""Common runner helpers for ORIGIN-Ax v1 experiments."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from axis_family import build_axis_families, origin_graph_edges
from candidate_generation import candidate_table, generate_candidates
from core import OriginAxConfig, global_gate_summary
from scoring import score_families


def run_origin_ax(
    *,
    dataset: str,
    x: np.ndarray,
    genes: np.ndarray,
    coords: np.ndarray,
    clone: np.ndarray,
    niche: np.ndarray,
    gene_weights: dict[str, np.ndarray],
    out_dir: str | Path,
    attribution_valid: bool = True,
    truth_eval_genes: set[str] | None = None,
    config: OriginAxConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    config = config or OriginAxConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, subset_diag = generate_candidates(x, genes, gene_weights, k=config.k, seeds=config.seeds)
    if candidates:
        np.savez_compressed(
            out_dir / "candidate_axis_arrays.npz",
            axis_id=np.array([c.axis_id for c in candidates], dtype=object),
            source_method=np.array([c.source_method for c in candidates], dtype=object),
            gene_loadings=np.vstack([c.loading for c in candidates]),
            bin_scores=np.vstack([c.score for c in candidates]),
        )
    cand_df = candidate_table(candidates)
    cand_df.insert(0, "dataset", dataset)
    cand_df.to_csv(out_dir / "candidate_axis_table.csv", index=False)
    subset_diag.insert(0, "dataset", dataset)
    subset_diag.to_csv(out_dir / "candidate_generation_summary.csv", index=False)

    if candidates:
        edges = origin_graph_edges(
            candidates,
            gene_weights,
            similarity_threshold=config.similarity_threshold,
            origin_compatibility_weight=config.origin_compatibility_weight,
            origin_conflict_penalty=config.origin_conflict_penalty,
        )
        edges.to_csv(out_dir / "candidate_origin_graph_edges.csv", index=False)
        expr_families, _ = build_axis_families(
            candidates,
            genes,
            similarity_threshold=config.similarity_threshold,
            gene_weights=gene_weights,
            origin_graph_enabled=False,
        )
        pd.DataFrame([{
            "dataset": dataset,
            "graph_variant": "expression_only",
            "n_axis_families": len(expr_families),
            "n_candidate_edges": int(edges["expression_edge"].sum()),
        }, {
            "dataset": dataset,
            "graph_variant": "origin_constrained",
            "n_axis_families": np.nan,
            "n_candidate_edges": int(edges["origin_constrained_edge"].sum()),
        }]).to_csv(out_dir / "origin_graph_ablation.csv", index=False)
    families, sim_df = build_axis_families(
        candidates,
        genes,
        similarity_threshold=config.similarity_threshold,
        gene_weights=gene_weights,
        origin_graph_enabled=config.origin_graph_enabled,
        origin_compatibility_weight=config.origin_compatibility_weight,
        origin_conflict_penalty=config.origin_conflict_penalty,
    )
    if candidates:
        ab = pd.read_csv(out_dir / "origin_graph_ablation.csv")
        ab.loc[ab["graph_variant"] == "origin_constrained", "n_axis_families"] = len(families)
        ab.to_csv(out_dir / "origin_graph_ablation.csv", index=False)
    sim_df.to_csv(out_dir / "candidate_similarity.csv")
    gates = global_gate_summary(clone, niche, coords, attribution_valid=attribution_valid)
    fam_df, norm_diag = score_families(
        families,
        genes,
        gene_weights,
        clone,
        niche,
        gates,
        truth_eval_genes=truth_eval_genes,
    )
    fam_df.insert(0, "dataset", dataset)
    fam_df.to_csv(out_dir / "axis_family_table.csv", index=False)
    norm_diag.insert(0, "dataset", dataset)
    norm_diag.to_csv(out_dir / "score_normalization.csv", index=False)
    pd.DataFrame([{**{"dataset": dataset}, **gates}]).to_csv(out_dir / "global_gate_report.csv", index=False)

    summary = {
        "dataset": dataset,
        "gates_passed": bool(gates["global_gates_pass"]),
        "n_candidates": int(len(candidates)),
        "n_axis_families": int(len(fam_df)),
        "n_target_families": int((fam_df["final_call"] == "target_axis").sum()) if not fam_df.empty else 0,
        "n_truth_target_families": int((fam_df.get("heldout_truth_overlap", pd.Series(dtype=int)) >= 4).sum()) if not fam_df.empty else 0,
        "n_passed_truth_mal_axis": int(((fam_df["final_call"] == "target_axis") & (fam_df.get("heldout_truth_overlap", 0) >= 4)).sum()) if not fam_df.empty else 0,
        "n_false_target_calls": int(((fam_df["final_call"] == "target_axis") & (fam_df.get("heldout_truth_overlap", 0) < 4)).sum()) if truth_eval_genes and not fam_df.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    return fam_df, summary
