"""Paper-level supplements for ORIGIN-Ax v1.

This script does not change ORIGIN-Ax scoring, thresholds, clustering, or
candidate generation. It reads existing benchmark-refinement outputs and
creates lightweight ablation, sensitivity, scalability, and user-guidance
artifacts for the paper.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_v1"
DETAIL = RESULTS / "benchmark_refinement"
METHODS = ["raw_ICA_only", "NMF_only", "cNMF_like_repeated_NMF"]
DATASETS = [
    "benchmark_pseudo_st_positive",
    "benchmark_real_xenium_myeloid",
    "benchmark_visium_hd_crc_stress",
]

REAL_FORBIDDEN = {"stromal", "fibroblast", "ECM", "endothelial", "malignant_melanoma"}


def _read_family_table(dataset: str, method: str) -> pd.DataFrame:
    base = DETAIL / dataset / method
    for name in [
        "axis_family_table_after_target_specific_report.csv",
        "axis_family_table_after_crc_axis5_report.csv",
        "axis_family_table_after_refinement.csv",
    ]:
        path = base / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(base)


def _read_summary_row(dataset: str, method: str) -> pd.Series:
    summary = pd.read_csv(RESULTS / "benchmark_refinement_summary.csv")
    dataset_label = {
        "benchmark_pseudo_st_positive": "pseudo_ST_positive",
        "benchmark_real_xenium_myeloid": "real_positive_xenium_myeloid",
        "benchmark_visium_hd_crc_stress": "visium_hd_crc_stress",
    }[dataset]
    row = summary.loc[(summary["dataset"] == dataset_label) & (summary["benchmark_method"] == method)]
    if row.empty:
        raise ValueError(f"missing summary row: {dataset} {method}")
    return row.iloc[0]


def _metrics(dataset: str, fam: pd.DataFrame, call_col: str = "final_call") -> dict[str, int]:
    is_target = fam[call_col].eq("target_axis") if call_col in fam else pd.Series(False, index=fam.index)
    if dataset == "benchmark_pseudo_st_positive":
        truth = fam.get("heldout_truth_overlap", 0) >= 4
        false = is_target & ~truth
        return {
            "target_calls": int(is_target.sum()),
            "false_target_calls": int(false.sum()),
            "truth_hits": int((is_target & truth).sum()),
        }
    if dataset == "benchmark_real_xenium_myeloid":
        heldout = fam.get("heldout_truth_overlap", 0)
        max_off = fam.get("max_offtarget_source", pd.Series("none", index=fam.index)).fillna("none")
        strict = is_target & (heldout >= 2) & ~max_off.isin(REAL_FORBIDDEN)
        false = is_target & ((heldout == 0) | max_off.isin(REAL_FORBIDDEN))
        return {
            "target_calls": int(is_target.sum()),
            "false_target_calls": int(false.sum()),
            "truth_hits": int(strict.sum()),
        }
    if dataset == "benchmark_visium_hd_crc_stress":
        return {
            "target_calls": int(is_target.sum()),
            "false_target_calls": int(is_target.sum()),
            "truth_hits": 0,
        }
    raise ValueError(dataset)


def _call_from_score(fam: pd.DataFrame, score: pd.Series, threshold: float) -> pd.Series:
    calls = pd.Series("not_target_axis", index=fam.index)
    calls.loc[score > threshold] = "target_axis"
    return calls


def make_ablation_summary() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            fam = _read_family_table(dataset, method)
            base = _read_summary_row(dataset, method)

            variants = {
                "full_origin_ax": fam["final_call"],
                "no_offtarget_penalty": _call_from_score(
                    fam,
                    0.6 * fam["origin_margin_norm"] + 0.4 * fam["stability_norm"],
                    0.20,
                ),
                "no_stability_term": _call_from_score(
                    fam,
                    fam["origin_margin_norm"] - fam["off_target_affinity_norm"],
                    0.20,
                ),
            }
            for ablation, calls in variants.items():
                tmp = fam.copy()
                tmp["ablation_call"] = calls
                m = _metrics(dataset, tmp, "ablation_call")
                rows.append({
                    "dataset": dataset.replace("benchmark_", ""),
                    "benchmark_method": method,
                    "ablation": ablation,
                    "n_axis_families": len(fam),
                    **m,
                    "interpretation": _ablation_interpretation(ablation, m),
                })

            rows.append({
                "dataset": dataset.replace("benchmark_", ""),
                "benchmark_method": method,
                "ablation": "no_family_refinement_candidate_level",
                "n_axis_families": "",
                "target_calls": int(base["benchmark_target_like_calls"]),
                "false_target_calls": int(base["benchmark_false_target_calls"]),
                "truth_hits": int(base["truth_axis_hits_before"]) if not pd.isna(base["truth_axis_hits_before"]) else 0,
                "interpretation": "candidate-level interpretation before family refinement",
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "ablation_summary.csv", index=False)
    return out


def _ablation_interpretation(ablation: str, m: dict[str, int]) -> str:
    if ablation == "full_origin_ax":
        return "reported ORIGIN-Ax v1 output"
    if ablation == "no_offtarget_penalty":
        return "tests whether target support alone overcalls off-target families"
    if ablation == "no_stability_term":
        return "tests contribution of family consensus stability"
    return ""


def make_score_threshold_sensitivity() -> pd.DataFrame:
    rows = []
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]
    for dataset in DATASETS:
        for method in METHODS:
            fam = _read_family_table(dataset, method)
            for th in thresholds:
                tmp = fam.copy()
                tmp["threshold_call"] = _call_from_score(fam, fam["origin_core_score"], th)
                m = _metrics(dataset, tmp, "threshold_call")
                rows.append({
                    "dataset": dataset.replace("benchmark_", ""),
                    "benchmark_method": method,
                    "score_threshold": th,
                    **m,
                })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "threshold_sensitivity_summary.csv", index=False)
    return out


def _n_components(sim: np.ndarray, threshold: float) -> int:
    n = sim.shape[0]
    seen = np.zeros(n, dtype=bool)
    n_comp = 0
    for i in range(n):
        if seen[i]:
            continue
        n_comp += 1
        seen[i] = True
        queue: deque[int] = deque([i])
        while queue:
            u = queue.popleft()
            for v in np.where(sim[u] >= threshold)[0]:
                if v != u and not seen[v]:
                    seen[v] = True
                    queue.append(int(v))
    return n_comp


def make_similarity_sensitivity() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            sim_path = DETAIL / dataset / method / "benchmark_candidate_similarity.csv"
            sim = pd.read_csv(sim_path, index_col=0).to_numpy(dtype=float)
            for th in [0.25, 0.30, 0.35, 0.40, 0.45]:
                rows.append({
                    "dataset": dataset.replace("benchmark_", ""),
                    "benchmark_method": method,
                    "similarity_threshold": th,
                    "n_input_axes": sim.shape[0],
                    "n_axis_families": _n_components(sim, th),
                })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "family_similarity_sensitivity_summary.csv", index=False)
    return out


def make_scalability_summary() -> pd.DataFrame:
    rows = []
    for n_axes in [24, 50, 100, 250, 500, 1000]:
        pairs = n_axes * (n_axes - 1) // 2
        rows.append({
            "n_candidate_axes": n_axes,
            "pairwise_axis_comparisons": pairs,
            "similarity_matrix_mb_float64": round(n_axes * n_axes * 8 / 1e6, 3),
            "dominant_step": "pairwise loading/top-gene similarity",
            "asymptotic_cost": "O(A^2 * G) for A axes and G genes; scoring is O(F * G)",
            "practical_note": "axis count, not bin count, dominates refinement after decomposition",
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "scalability_summary.csv", index=False)
    return out


def write_guidance(ablation: pd.DataFrame, score_sens: pd.DataFrame, sim_sens: pd.DataFrame) -> None:
    path = RESULTS / "PAPER_SUPPLEMENT_SUMMARY.md"
    full = ablation.loc[ablation["ablation"] == "full_origin_ax"]
    no_off = ablation.loc[ablation["ablation"] == "no_offtarget_penalty"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Paper Supplement Summary\n\n")
        fh.write("These analyses are paper-level supplements. They do not modify ORIGIN-Ax v1 core scoring, thresholds, clustering, or candidate generation.\n\n")
        fh.write("## Ablation Findings\n\n")
        fh.write(f"- Full ORIGIN-Ax target calls across benchmark rows: {int(full['target_calls'].sum())}.\n")
        fh.write(f"- Full ORIGIN-Ax false target calls across benchmark rows: {int(full['false_target_calls'].sum())}.\n")
        fh.write(f"- Removing the off-target penalty yields {int(no_off['false_target_calls'].sum())} false target calls across benchmark rows.\n")
        fh.write("- Candidate-level interpretation is retained as the no-family-refinement baseline; this quantifies the false-call reduction provided by the refinement layer.\n\n")
        fh.write("## Threshold Sensitivity\n\n")
        fh.write("The threshold sensitivity table reports target calls, false target calls, and truth hits for fixed score thresholds from 0.10 to 0.30. It is intended as a robustness audit, not as post-hoc threshold tuning.\n\n")
        fh.write("## Similarity Threshold Sensitivity\n\n")
        fh.write("The family-similarity table reports how the number of connected-component families changes when the axis graph threshold varies from 0.25 to 0.45.\n\n")
        fh.write("## Reference-Gene-Set Guidance\n\n")
        fh.write("- Define target support and off-target references before inspecting final calls.\n")
        fh.write("- Include plausible false-boundary programs, not only canonical cell-type markers. For tumors, this includes normal epithelial, stromal/ECM, endothelial, immune, and platform-specific technical programs.\n")
        fh.write("- Split target genes into anchor and held-out subsets when possible. Anchor genes may guide scoring; held-out genes should only be used for evaluation.\n")
        fh.write("- Treat incomplete off-target references as a failure mode. A target call should be downgraded if the off-target reference lacks known confounders for the tissue.\n")
        fh.write("- Use target-swap controls when a rejected family is biologically coherent. A family rejected under one target definition can be a valid target under another.\n\n")
        fh.write("## Non-Identifiable Follow-up\n\n")
        fh.write("A non-identifiable family should trigger an evidence-acquisition decision rather than a biological conclusion. Recommended next steps are: add orthogonal anchors such as CNV or cell labels, improve off-target references, split target genes into held-out validation sets, or report the family as unresolved instead of forcing a target label.\n")
    print(path)


def main() -> None:
    ablation = make_ablation_summary()
    score_sens = make_score_threshold_sensitivity()
    sim_sens = make_similarity_sensitivity()
    make_scalability_summary()
    write_guidance(ablation, score_sens, sim_sens)
    print(RESULTS / "ablation_summary.csv")
    print(RESULTS / "threshold_sensitivity_summary.csv")
    print(RESULTS / "family_similarity_sensitivity_summary.csv")
    print(RESULTS / "scalability_summary.csv")


if __name__ == "__main__":
    main()
