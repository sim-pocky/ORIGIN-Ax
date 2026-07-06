"""Calibration, baseline, and reference-coverage audits for ORIGIN-Ax v1.

This script does not change ORIGIN-Ax defaults. It audits whether the current
defaults sit inside stable regions of already-generated development outputs and
creates lightweight tables for the paper.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_v1"
CALIBRATION_DATASETS = {
    "controlled_positive": RESULTS / "controlled_positive",
    "pseudo_st_balanced": RESULTS / "pseudo_st_positive" / "balanced",
    "pseudo_st_low_purity": RESULTS / "pseudo_st_positive" / "low_purity",
}


def _components_from_edges(nodes: list[str], edges: list[tuple[str, str]]) -> int:
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen: set[str] = set()
    n_comp = 0
    for n in nodes:
        if n in seen:
            continue
        n_comp += 1
        seen.add(n)
        q: deque[str] = deque([n])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
    return n_comp


def graph_parameter_calibration() -> pd.DataFrame:
    rows = []
    rhos = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    kappas = [0.20, 0.40, 0.60, 0.80, 1.00]
    edge_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45]
    for dataset, base in CALIBRATION_DATASETS.items():
        edge_path = base / "candidate_origin_graph_edges.csv"
        if not edge_path.exists():
            continue
        edges = pd.read_csv(edge_path)
        nodes = sorted(set(edges["axis_i"].astype(str)) | set(edges["axis_j"].astype(str)))
        default_w = np.clip(
            edges["expression_similarity"].to_numpy(float)
            + 0.15 * edges["origin_compatibility"].to_numpy(float)
            - 0.60 * edges["origin_conflict"].to_numpy(float),
            0.0,
            1.0,
        )
        default_edges = list(zip(edges.loc[default_w >= 0.35, "axis_i"].astype(str), edges.loc[default_w >= 0.35, "axis_j"].astype(str)))
        default_n_families = _components_from_edges(nodes, default_edges)
        for rho in rhos:
            for kappa in kappas:
                w_raw = (
                    edges["expression_similarity"].to_numpy(float)
                    + rho * edges["origin_compatibility"].to_numpy(float)
                    - kappa * edges["origin_conflict"].to_numpy(float)
                )
                w = np.clip(w_raw, 0.0, 1.0)
                for th in edge_thresholds:
                    keep = w >= th
                    expr_keep = edges["expression_similarity"].to_numpy(float) >= th
                    kept_edges = list(zip(edges.loc[keep, "axis_i"].astype(str), edges.loc[keep, "axis_j"].astype(str)))
                    n_families = _components_from_edges(nodes, kept_edges)
                    conflict = edges["origin_conflict"].to_numpy(float)
                    compat = edges["origin_compatibility"].to_numpy(float)
                    rows.append({
                        "dataset": dataset,
                        "rho": rho,
                        "kappa": kappa,
                        "edge_threshold": th,
                        "n_candidate_axes": len(nodes),
                        "n_axis_families": n_families,
                        "n_edges_retained": int(keep.sum()),
                        "n_expression_edges": int(expr_keep.sum()),
                        "n_conflict_edges_pruned": int((expr_keep & ~keep & (conflict >= 0.35)).sum()),
                        "n_compatible_edges_added": int((~expr_keep & keep & (compat >= 0.50)).sum()),
                        "default_n_axis_families": default_n_families,
                        "family_count_delta_from_default": int(n_families - default_n_families),
                        "in_default_stability_band": bool(abs(n_families - default_n_families) <= 1),
                        "is_default": bool(rho == 0.15 and kappa == 0.60 and th == 0.35),
                    })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "threshold_graph_calibration_audit.csv", index=False)
    return out


def score_weight_calibration() -> pd.DataFrame:
    rows = []
    weights = [0.50, 0.60, 0.70]
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    for dataset, base in CALIBRATION_DATASETS.items():
        fam_path = base / "axis_family_table.csv"
        if not fam_path.exists():
            continue
        fam = pd.read_csv(fam_path)
        for wt in weights:
            ws = 1.0 - wt
            score = wt * fam["origin_margin_norm"] + ws * fam["stability_norm"] - fam["off_target_affinity_norm"]
            for th in thresholds:
                target = (score > th) & (fam["identifiability_margin"] > 0) & (fam["mean_origin_conflict"] <= 0.35)
                truth = fam["heldout_truth_overlap"].fillna(0) >= 4
                rows.append({
                    "dataset": dataset,
                    "target_margin_weight": wt,
                    "stability_weight": ws,
                    "score_threshold": th,
                    "n_axis_families": len(fam),
                    "target_calls": int(target.sum()),
                    "truth_supported_calls": int((target & truth).sum()),
                    "unsupported_target_calls": int((target & ~truth).sum()),
                    "truth_recovery_retained": bool((target & truth).sum() >= 1),
                    "no_unsupported_calls": bool((target & ~truth).sum() == 0),
                    "is_default_score_rule": bool(wt == 0.60 and th == 0.20),
                })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "threshold_score_calibration_audit.csv", index=False)
    return out


def summarize_threshold_plateau(graph: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    graph_plateau = graph.groupby(["rho", "kappa", "edge_threshold"], as_index=False).agg(
        n_datasets=("dataset", "nunique"),
        stable_datasets=("in_default_stability_band", "sum"),
        mean_family_delta=("family_count_delta_from_default", "mean"),
        total_conflict_edges_pruned=("n_conflict_edges_pruned", "sum"),
        total_compatible_edges_added=("n_compatible_edges_added", "sum"),
    )
    graph_plateau["stable_all_calibration_datasets"] = graph_plateau["stable_datasets"].eq(graph_plateau["n_datasets"])
    graph_plateau["default_tuple"] = (
        graph_plateau["rho"].eq(0.15)
        & graph_plateau["kappa"].eq(0.60)
        & graph_plateau["edge_threshold"].eq(0.35)
    )

    score_plateau = score.groupby(["target_margin_weight", "stability_weight", "score_threshold"], as_index=False).agg(
        n_datasets=("dataset", "nunique"),
        recovery_datasets=("truth_recovery_retained", "sum"),
        specificity_datasets=("no_unsupported_calls", "sum"),
        total_truth_supported=("truth_supported_calls", "sum"),
        total_unsupported=("unsupported_target_calls", "sum"),
    )
    score_plateau["recovery_and_specificity_all"] = (
        score_plateau["recovery_datasets"].eq(score_plateau["n_datasets"])
        & score_plateau["specificity_datasets"].eq(score_plateau["n_datasets"])
    )
    score_plateau["default_tuple"] = (
        score_plateau["target_margin_weight"].eq(0.60)
        & score_plateau["score_threshold"].eq(0.20)
    )

    graph_plateau.to_csv(RESULTS / "threshold_graph_stability_plateau.csv", index=False)
    score_plateau.to_csv(RESULTS / "threshold_score_stability_plateau.csv", index=False)
    return pd.DataFrame([
        {
            "audit": "graph_parameter_plateau",
            "n_parameter_tuples": len(graph_plateau),
            "n_stable_tuples": int(graph_plateau["stable_all_calibration_datasets"].sum()),
            "default_in_stable_plateau": bool(graph_plateau.loc[graph_plateau["default_tuple"], "stable_all_calibration_datasets"].any()),
            "selected_default": "rho=0.15,kappa=0.60,edge_threshold=0.35",
        },
        {
            "audit": "score_weight_plateau",
            "n_parameter_tuples": len(score_plateau),
            "n_stable_tuples": int(score_plateau["recovery_and_specificity_all"].sum()),
            "default_in_stable_plateau": bool(score_plateau.loc[score_plateau["default_tuple"], "recovery_and_specificity_all"].any()),
            "selected_default": "target_margin_weight=0.60,stability_weight=0.40,score_threshold=0.20",
        },
    ])


def external_baseline_comparison() -> pd.DataFrame:
    bench = pd.read_csv(RESULTS / "benchmark_refinement_summary.csv")
    abl = pd.read_csv(RESULTS / "ablation_summary.csv")
    dataset_alias = {
        "pseudo_st_positive": "pseudo_ST_positive",
        "pseudo_ST_positive": "pseudo_ST_positive",
        "real_xenium_myeloid": "real_positive_xenium_myeloid",
        "real_positive_xenium_myeloid": "real_positive_xenium_myeloid",
        "visium_hd_crc_stress": "visium_hd_crc_stress",
    }
    rows = []
    for dataset, sub in bench.groupby("dataset"):
        dataset = dataset_alias.get(dataset, dataset)
        rows.append({
            "dataset": dataset,
            "baseline": "top_marker_candidate_level",
            "target_calls": int(sub["benchmark_target_like_calls"].sum()),
            "truth_supported_calls": int(sub["truth_axis_hits_before"].fillna(0).sum()),
            "unsupported_promotions": int(sub["benchmark_false_target_calls"].sum()),
            "offtarget_rejected": "",
            "non_identifiable_calls": "",
            "comment": "candidate axes promoted by top-marker/naive target-like evidence before refinement",
        })
        full = sub
        rows.append({
            "dataset": dataset,
            "baseline": "origin_ax_full",
            "target_calls": int(full["origin_ax_target_calls"].sum()),
            "truth_supported_calls": int(full["truth_axis_hits_after"].fillna(0).sum()),
            "unsupported_promotions": int(full["origin_ax_false_target_calls"].sum()),
            "offtarget_rejected": int(full["off_target_rejected"].map(lambda x: str(x).lower() == "true").sum()) if "off_target_rejected" in full else "",
            "non_identifiable_calls": "",
            "comment": "origin-constrained graph refinement with off-target rejection and family consensus",
        })

    name_map = {
        "no_offtarget_penalty": "target_only_family_evidence",
        "no_stability_term": "no_family_stability_score",
    }
    for (dataset, ablation), sub in abl[abl["ablation"].isin(name_map)].groupby(["dataset", "ablation"]):
        dataset = dataset_alias.get(dataset, dataset)
        rows.append({
            "dataset": dataset,
            "baseline": name_map[ablation],
            "target_calls": int(sub["target_calls"].sum()),
            "truth_supported_calls": int(sub["truth_hits"].sum()),
            "unsupported_promotions": int(sub["false_target_calls"].sum()),
            "offtarget_rejected": "",
            "non_identifiable_calls": "",
            "comment": "family-level baseline with one ORIGIN-Ax evidence component removed",
        })
    out = pd.DataFrame(rows).sort_values(["dataset", "baseline"])
    out.to_csv(RESULTS / "external_baseline_comparison.csv", index=False)
    return out


def _split_genes(text: str) -> list[str]:
    return [g.strip() for g in str(text).split(",") if g.strip()]


def _reference_diagnostics_from_weight_table(
    *,
    dataset: str,
    target_setting: str,
    gene_weight_path: Path,
    family_table_path: Path | None = None,
    reference_gene_set_path: Path | None = None,
) -> list[dict]:
    gw = pd.read_csv(gene_weight_path)
    universe = set(gw["gene"].astype(str))
    target = set(gw.loc[gw.get("target_support", gw.get("malignant_support", 0)).astype(float) > 0, "gene"].astype(str))
    normal = set(gw.loc[gw.get("normal_epithelial_penalty", 0).astype(float) > 0, "gene"].astype(str))
    tme = set(gw.loc[gw.get("tme_penalty", 0).astype(float) > 0, "gene"].astype(str))
    ref_categories: dict[str, set[str]] = {}
    if target:
        ref_categories["target"] = target
    if normal:
        ref_categories["normal_epithelial"] = normal
    if tme:
        ref_categories["tme"] = tme

    if reference_gene_set_path is not None and reference_gene_set_path.exists():
        ref = pd.read_csv(reference_gene_set_path)
        for cat, sub in ref.groupby("reference_category"):
            ref_categories[str(cat)] = set(sub["gene"].astype(str)) | ref_categories.get(str(cat), set())

    all_ref = set().union(*ref_categories.values()) if ref_categories else set()
    off_ref = set().union(*(v for k, v in ref_categories.items() if k != "target")) if ref_categories else set()
    overlap = target & off_ref
    category_hits = {k: len(v & universe) for k, v in ref_categories.items()}
    family_hit_rates = []
    if family_table_path is not None and family_table_path.exists():
        fam = pd.read_csv(family_table_path)
        for _, row in fam.iterrows():
            top = set(_split_genes(row.get("top_genes", ""))[:12])
            if top:
                family_hit_rates.append(len(top & all_ref) / len(top))
    mean_top_hit = float(np.mean(family_hit_rates)) if family_hit_rates else np.nan
    warnings = []
    off_total = max(1, len(off_ref))
    off_covered = len(off_ref & universe)
    target_total = max(1, len(target))
    target_covered = len(target & universe)
    decoy_diversity = sum(1 for k, v in ref_categories.items() if k != "target" and len(v & universe) > 0)
    ambiguous_fraction = len(overlap) / max(1, len(target | off_ref))
    if off_covered / off_total < 0.30:
        warnings.append("low_offtarget_reference_coverage")
    if decoy_diversity < 2:
        warnings.append("insufficient_decoy_diversity")
    if ambiguous_fraction > 0.25:
        warnings.append("high_marker_ambiguity")
    if not np.isnan(mean_top_hit) and mean_top_hit < 0.15:
        warnings.append("low_top_gene_reference_hit_rate")

    return [{
        "dataset": dataset,
        "target_setting": target_setting,
        "n_gene_universe": len(universe),
        "n_target_reference_genes": len(target),
        "n_target_reference_genes_in_universe": target_covered,
        "target_reference_coverage": target_covered / target_total,
        "n_offtarget_reference_genes": len(off_ref),
        "n_offtarget_reference_genes_in_universe": off_covered,
        "offtarget_reference_coverage": off_covered / off_total,
        "decoy_diversity": decoy_diversity,
        "ambiguous_marker_fraction": ambiguous_fraction,
        "mean_top_gene_reference_hit_rate": mean_top_hit,
        "category_coverage": ";".join(f"{k}:{v}" for k, v in sorted(category_hits.items())),
        "warnings": ";".join(warnings) if warnings else "none",
    }]


def reference_coverage_diagnostics() -> pd.DataFrame:
    rows: list[dict] = []
    rows += _reference_diagnostics_from_weight_table(
        dataset="real_positive_xenium_myeloid",
        target_setting="myeloid_inflammatory",
        gene_weight_path=RESULTS / "real_positive_xenium_melanoma" / "gene_weight_anchor.csv",
        family_table_path=RESULTS / "real_positive_xenium_melanoma" / "axis_family_table.csv",
        reference_gene_set_path=RESULTS / "real_positive_xenium_melanoma" / "real_positive_reference_gene_sets.csv",
    )
    rows += _reference_diagnostics_from_weight_table(
        dataset="target_swap_xenium",
        target_setting="myeloid_inflammatory",
        gene_weight_path=RESULTS / "target_swap_generalization" / "gene_weight_myeloid.csv",
        family_table_path=RESULTS / "target_swap_generalization" / "myeloid_inflammatory" / "axis_family_table.csv",
    )
    rows += _reference_diagnostics_from_weight_table(
        dataset="target_swap_xenium",
        target_setting="stromal_fibroblast_ecm",
        gene_weight_path=RESULTS / "target_swap_generalization" / "gene_weight_stromal.csv",
        family_table_path=RESULTS / "target_swap_generalization" / "stromal_fibroblast_ecm" / "axis_family_table.csv",
    )
    crc_gw = HERE.parent / "malaxis_oa" / "results" / "gene_weight_anchor.csv"
    if crc_gw.exists():
        rows += _reference_diagnostics_from_weight_table(
            dataset="visium_hd_crc_stress",
            target_setting="malignant_crc",
            gene_weight_path=crc_gw,
            family_table_path=RESULTS / "visium_hd_crc" / "axis_family_table.csv",
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "reference_coverage_diagnostics.csv", index=False)
    return out


def write_summary(
    graph: pd.DataFrame,
    score: pd.DataFrame,
    plateau: pd.DataFrame,
    baselines: pd.DataFrame,
    refdiag: pd.DataFrame,
) -> None:
    default_graph = graph[graph["is_default"]]
    default_score = score[score["is_default_score_rule"]]
    full = baselines[baselines["baseline"].eq("origin_ax_full")]
    top = baselines[baselines["baseline"].eq("top_marker_candidate_level")]
    target_only = baselines[baselines["baseline"].eq("target_only_family_evidence")]
    lines = [
        "# ORIGIN-Ax Calibration, Baseline, and Reference Diagnostics",
        "",
        "## Threshold calibration",
        "",
        f"- Graph calibration tuples scanned: {graph[['rho','kappa','edge_threshold']].drop_duplicates().shape[0]}.",
        f"- Default graph tuple family-count delta range across calibration datasets: "
        f"{int(default_graph['family_count_delta_from_default'].min())} to {int(default_graph['family_count_delta_from_default'].max())}.",
        f"- Score calibration tuples scanned: {score[['target_margin_weight','stability_weight','score_threshold']].drop_duplicates().shape[0]}.",
        f"- Default score rule total truth-supported calls: {int(default_score['truth_supported_calls'].sum())}; "
        f"unsupported target calls: {int(default_score['unsupported_target_calls'].sum())}.",
        f"- Default graph in stability plateau: {bool(plateau.loc[plateau['audit'].eq('graph_parameter_plateau'), 'default_in_stable_plateau'].iloc[0])}.",
        f"- Default score in recovery/specificity plateau: {bool(plateau.loc[plateau['audit'].eq('score_weight_plateau'), 'default_in_stable_plateau'].iloc[0])}.",
        "",
        "## Baseline comparison",
        "",
        f"- Top-marker candidate-level unsupported promotions: {int(top['unsupported_promotions'].sum())}.",
        f"- Target-only family-evidence unsupported promotions: {int(target_only['unsupported_promotions'].sum())}.",
        f"- ORIGIN-Ax full unsupported promotions: {int(full['unsupported_promotions'].sum())}.",
        "",
        "## Reference coverage",
        "",
    ]
    for row in refdiag.itertuples(index=False):
        lines.append(
            f"- {row.dataset} / {row.target_setting}: target coverage={row.target_reference_coverage:.2f}, "
            f"off-target coverage={row.offtarget_reference_coverage:.2f}, decoy diversity={row.decoy_diversity}, "
            f"warnings={row.warnings}."
        )
    lines += [
        "",
        "These audits are supplemental diagnostics. They do not change ORIGIN-Ax v1 defaults or relabel existing results.",
        "",
    ]
    (RESULTS / "CALIBRATION_BASELINE_REFERENCE_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    graph = graph_parameter_calibration()
    score = score_weight_calibration()
    plateau = summarize_threshold_plateau(graph, score)
    plateau.to_csv(RESULTS / "threshold_calibration_summary.csv", index=False)
    baselines = external_baseline_comparison()
    refdiag = reference_coverage_diagnostics()
    write_summary(graph, score, plateau, baselines, refdiag)
    print(plateau.to_string(index=False))
    print(baselines.groupby("baseline")["unsupported_promotions"].sum().to_string())
    print(refdiag[["dataset", "target_setting", "warnings"]].to_string(index=False))


if __name__ == "__main__":
    main()
