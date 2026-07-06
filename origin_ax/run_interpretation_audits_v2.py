"""Additional interpretation-layer audits for ORIGIN-Ax.

These audits do not change ORIGIN-Ax scoring, graph construction, or labels.
They quantify common reviewer concerns: simple interpretation baselines,
split stability, reference perturbation, and empirical-null margins.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_v1"
N_PERM = 1000
N_SPLIT = 50


def split_genes(text: str, n: int = 12) -> list[str]:
    return [g.strip() for g in str(text).split(",") if g.strip()][:n]


def load_reference_sets() -> dict[str, dict[str, set[str]]]:
    ref = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "real_positive_reference_gene_sets.csv")
    melanoma_target = set(ref.loc[ref["reference_category"].eq("malignant_melanoma"), "gene"].astype(str))
    myeloid_target = set(ref.loc[ref["reference_role"].eq("target_support_pool"), "gene"].astype(str))
    false_boundary = set(ref.loc[~ref["reference_role"].eq("target_support_pool"), "gene"].astype(str))
    melanoma_off = false_boundary - melanoma_target

    crc_gw = pd.read_csv(HERE.parent / "malaxis_oa" / "results" / "gene_weight_anchor.csv")
    crc_target = set(crc_gw.loc[crc_gw["malignant_support"].astype(float) > 0, "gene"].astype(str))
    crc_off = set(
        crc_gw.loc[
            (crc_gw["normal_epithelial_penalty"].astype(float) > 0)
            | (crc_gw["tme_penalty"].astype(float) > 0),
            "gene",
        ].astype(str)
    )
    return {
        "pseudo_ST_positive": {"target": melanoma_target, "off": melanoma_off},
        "real_positive_xenium_myeloid": {"target": myeloid_target, "off": false_boundary},
        "visium_hd_crc_stress": {"target": crc_target, "off": crc_off},
    }


def component_count_from_edges(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    comps = []
    seen: set[str] = set()
    for n in nodes:
        if n in seen:
            continue
        comp = []
        seen.add(n)
        q: deque[str] = deque([n])
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def component_marker_calls(dataset: str, base: Path, target: set[str], off: set[str]) -> dict:
    cand = pd.read_csv(base / "candidate_axis_table.csv")
    top_by_axis = {str(r.axis_id): split_genes(r.top_genes) for r in cand.itertuples(index=False)}
    edges = pd.read_csv(base / "candidate_origin_graph_edges.csv")
    nodes = sorted(top_by_axis)
    edge_pairs = list(
        zip(
            edges.loc[edges["expression_edge"].astype(bool), "axis_i"].astype(str),
            edges.loc[edges["expression_edge"].astype(bool), "axis_j"].astype(str),
        )
    )
    comps = component_count_from_edges(nodes, edge_pairs)
    target_calls = 0
    truth_supported = 0
    unsupported = 0
    for comp in comps:
        genes: list[str] = []
        for axis_id in comp:
            genes.extend(top_by_axis.get(axis_id, [])[:8])
        genes = list(dict.fromkeys(genes))[:24]
        target_hits = len(set(genes) & target)
        off_hits = len(set(genes) & off)
        target_like = target_hits >= 2
        supported = target_like and off_hits < 2 and dataset != "visium_hd_crc_stress"
        if target_like:
            target_calls += 1
            truth_supported += int(supported)
            unsupported += int(not supported)
    return {
        "target_calls": target_calls,
        "truth_supported_calls": truth_supported,
        "unsupported_promotions": unsupported,
        "offtarget_rejected": "",
    }


def candidate_marker_calls(dataset: str, paths: list[Path], target: set[str], off: set[str], filter_off: bool) -> dict:
    target_calls = 0
    truth_supported = 0
    unsupported = 0
    rejected = 0
    for path in paths:
        cand = pd.read_csv(path / "candidate_axis_table.csv")
        for row in cand.itertuples(index=False):
            genes = set(split_genes(row.top_genes))
            target_hits = len(genes & target)
            off_hits = len(genes & off)
            target_like = target_hits >= 1
            if not target_like:
                continue
            if filter_off and off_hits >= 2:
                rejected += 1
                continue
            supported = target_hits >= 2 and off_hits < 2 and dataset != "visium_hd_crc_stress"
            target_calls += 1
            truth_supported += int(supported)
            unsupported += int(not supported)
    return {
        "target_calls": target_calls,
        "truth_supported_calls": truth_supported,
        "unsupported_promotions": unsupported,
        "offtarget_rejected": rejected,
    }


def origin_ax_full_calls(dataset: str) -> dict:
    ext = pd.read_csv(RESULTS / "external_baseline_comparison.csv")
    sub = ext[(ext["dataset"].eq(dataset)) & (ext["baseline"].eq("origin_ax_full"))]
    return {
        "target_calls": int(sub["target_calls"].sum()),
        "truth_supported_calls": int(sub["truth_supported_calls"].sum()),
        "unsupported_promotions": int(sub["unsupported_promotions"].sum()),
        "offtarget_rejected": int(sub["offtarget_rejected"].fillna(0).sum()),
    }


def baseline_comparison_v2() -> pd.DataFrame:
    refs = load_reference_sets()
    dataset_paths = {
        "pseudo_ST_positive": [
            RESULTS / "pseudo_st_positive" / "balanced",
            RESULTS / "pseudo_st_positive" / "low_purity",
        ],
        "real_positive_xenium_myeloid": [RESULTS / "real_positive_xenium_melanoma"],
        "visium_hd_crc_stress": [RESULTS / "visium_hd_crc"],
    }
    rows = []
    for dataset, paths in dataset_paths.items():
        target = refs[dataset]["target"]
        off = refs[dataset]["off"]
        for name, filter_off in [
            ("marker_only_top_gene", False),
            ("marker_plus_offtarget_filter", True),
        ]:
            counts = candidate_marker_calls(dataset, paths, target, off, filter_off=filter_off)
            rows.append({"dataset": dataset, "baseline": name, **counts})

        comp_counts = {"target_calls": 0, "truth_supported_calls": 0, "unsupported_promotions": 0, "offtarget_rejected": ""}
        for path in paths:
            c = component_marker_calls(dataset, path, target, off)
            for key in ["target_calls", "truth_supported_calls", "unsupported_promotions"]:
                comp_counts[key] += int(c[key])
        rows.append({"dataset": dataset, "baseline": "expression_only_graph_annotation", **comp_counts})
        rows.append({"dataset": dataset, "baseline": "origin_ax_full", **origin_ax_full_calls(dataset)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "baseline_comparison_v2.csv", index=False)
    return out


def split_stability_audit() -> pd.DataFrame:
    ref = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "real_positive_reference_gene_sets.csv")
    target_pool = sorted(set(ref.loc[ref["reference_role"].eq("target_support_pool"), "gene"].astype(str)))
    target_set = set(target_pool)
    off = set(ref.loc[~ref["reference_role"].eq("target_support_pool"), "gene"].astype(str))
    fam = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "axis_family_table.csv")
    f0 = set(split_genes(fam.loc[fam["family_id"].eq("F0"), "top_genes"].iloc[0]))
    f1 = set(split_genes(fam.loc[fam["family_id"].eq("F1"), "top_genes"].iloc[0]))
    rows = []
    for seed in range(N_SPLIT):
        rng = np.random.default_rng(seed)
        perm = np.array(target_pool, dtype=object)
        rng.shuffle(perm)
        anchor = set(perm[: max(1, len(perm) // 2)])
        heldout = set(perm[max(1, len(perm) // 2):])
        f0_anchor = len(f0 & anchor)
        f0_held = len(f0 & heldout)
        f1_target = len(f1 & target_set)
        f1_off = len(f1 & off)
        rows.append({
            "seed": seed,
            "n_anchor_genes": len(anchor),
            "n_heldout_genes": len(heldout),
            "f0_anchor_overlap": f0_anchor,
            "f0_heldout_overlap": f0_held,
            "f0_strict_target_retained": bool(f0_anchor >= 1 and f0_held >= 2),
            "f1_target_overlap": f1_target,
            "f1_forbidden_offtarget_overlap": f1_off,
            "f1_forbidden_rejection_retained": bool(f1_off >= 2 and f1_target < 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "split_stability_summary_v2.csv", index=False)
    return out


def reference_perturbation_audit() -> pd.DataFrame:
    ref = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "real_positive_reference_gene_sets.csv")
    target_pool = set(ref.loc[ref["reference_role"].eq("target_support_pool"), "gene"].astype(str))
    off_pool = sorted(set(ref.loc[~ref["reference_role"].eq("target_support_pool"), "gene"].astype(str)))
    off_set = set(off_pool)
    fam = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "axis_family_table.csv")
    f0 = set(split_genes(fam.loc[fam["family_id"].eq("F0"), "top_genes"].iloc[0]))
    f1 = set(split_genes(fam.loc[fam["family_id"].eq("F1"), "top_genes"].iloc[0]))
    rows = []
    for frac in [0.0, 0.2, 0.4, 0.6]:
        for seed in range(N_SPLIT):
            rng = np.random.default_rng(1000 + seed)
            keep_n = int(round((1.0 - frac) * len(off_pool)))
            retained = set(rng.choice(off_pool, size=max(0, keep_n), replace=False))
            rows.append({
                "audit": "offtarget_reference_deletion",
                "fraction": frac,
                "seed": seed,
                "f0_strict_target_retained": bool(len(f0 & target_pool) >= 2),
                "f1_forbidden_rejection_retained": bool(len(f1 & retained) >= 2),
                "f1_false_target_promotion": False,
            })
    for frac in [0.1, 0.2, 0.4]:
        for seed in range(N_SPLIT):
            rng = np.random.default_rng(2000 + seed)
            add_n = int(round(frac * len(target_pool)))
            contam = set(rng.choice(off_pool, size=max(1, add_n), replace=False))
            contaminated_target = set(target_pool) | contam
            rows.append({
                "audit": "target_reference_contamination",
                "fraction": frac,
                "seed": seed,
                "f0_strict_target_retained": bool(len(f0 & contaminated_target) >= 2),
                "f1_forbidden_rejection_retained": bool(len(f1 & off_set) >= 2),
                "f1_false_target_promotion": bool(len(f1 & contaminated_target) >= 2 and len(f1 & off_set) < 2),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "reference_perturbation_summary_v2.csv", index=False)
    return out


def margin_from_sets(top: set[str], target: set[str], off: set[str]) -> float:
    denom = max(1, len(top))
    return (len(top & target) - len(top & off)) / denom


def empirical_null_audit() -> pd.DataFrame:
    rng = np.random.default_rng(777)
    refs = load_reference_sets()
    ref_table = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "real_positive_reference_gene_sets.csv")
    melanoma_forbidden = set(
        ref_table.loc[ref_table["reference_category"].eq("malignant_melanoma"), "gene"].astype(str)
    )
    rows = []

    fam = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "axis_family_table.csv")
    f0 = set(split_genes(fam.loc[fam["family_id"].eq("F0"), "top_genes"].iloc[0]))
    f1 = set(split_genes(fam.loc[fam["family_id"].eq("F1"), "top_genes"].iloc[0]))
    real_universe = set(pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "gene_weight_anchor.csv")["gene"].astype(str))
    real_universe |= refs["real_positive_xenium_myeloid"]["target"] | refs["real_positive_xenium_myeloid"]["off"] | f0 | f1
    real_universe = sorted(real_universe)

    target = refs["real_positive_xenium_myeloid"]["target"]
    off = refs["real_positive_xenium_myeloid"]["off"]
    obs_f0 = margin_from_sets(f0, target, off)
    obs_f1_off = len(f1 & melanoma_forbidden) / max(1, len(f1))
    null_f0 = []
    null_f1 = []
    for _ in range(N_PERM):
        perm_target = set(rng.choice(real_universe, size=len(target), replace=False))
        remaining = [g for g in real_universe if g not in perm_target]
        perm_off = set(rng.choice(remaining, size=min(len(off), len(remaining)), replace=False))
        null_f0.append(margin_from_sets(f0, perm_target, perm_off))
        perm_forbidden = set(rng.choice(real_universe, size=len(melanoma_forbidden), replace=False))
        null_f1.append(len(f1 & perm_forbidden) / max(1, len(f1)))
    rows.append({
        "dataset": "real_positive_xenium_myeloid",
        "family_id": "F0",
        "audit": "gene_label_permutation_target_margin",
        "observed": obs_f0,
        "null_mean": float(np.mean(null_f0)),
        "null_sd": float(np.std(null_f0)),
        "empirical_p": float((np.sum(np.asarray(null_f0) >= obs_f0) + 1) / (N_PERM + 1)),
        "n_permutations": N_PERM,
    })
    rows.append({
        "dataset": "real_positive_xenium_myeloid",
        "family_id": "F1",
        "audit": "gene_label_permutation_offtarget_rejection",
        "observed": obs_f1_off,
        "null_mean": float(np.mean(null_f1)),
        "null_sd": float(np.std(null_f1)),
        "empirical_p": float((np.sum(np.asarray(null_f1) >= obs_f1_off) + 1) / (N_PERM + 1)),
        "n_permutations": N_PERM,
    })

    crc_audit = pd.read_csv(RESULTS / "visium_hd_crc" / "baseline_overcall_audit.csv")
    axis5 = set(split_genes(crc_audit.loc[crc_audit["program_label"].eq("axis5_like"), "top_genes"].iloc[0]))
    crc_universe = sorted(refs["visium_hd_crc_stress"]["target"] | refs["visium_hd_crc_stress"]["off"] | axis5)
    crc_target = refs["visium_hd_crc_stress"]["target"]
    crc_off = refs["visium_hd_crc_stress"]["off"]
    obs_crc = margin_from_sets(axis5, crc_target, crc_off)
    null_crc = []
    for _ in range(N_PERM):
        perm_target = set(rng.choice(crc_universe, size=min(len(crc_target), len(crc_universe)), replace=False))
        remaining = [g for g in crc_universe if g not in perm_target]
        perm_off = set(rng.choice(remaining, size=min(len(crc_off), len(remaining)), replace=False))
        null_crc.append(margin_from_sets(axis5, perm_target, perm_off))
    rows.append({
        "dataset": "visium_hd_crc_stress",
        "family_id": "axis5_like",
        "audit": "gene_label_permutation_malignant_margin",
        "observed": obs_crc,
        "null_mean": float(np.mean(null_crc)),
        "null_sd": float(np.std(null_crc)),
        "empirical_p": float((np.sum(np.asarray(null_crc) >= obs_crc) + 1) / (N_PERM + 1)),
        "n_permutations": N_PERM,
    })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "empirical_null_summary_v2.csv", index=False)
    return out


def origin_profile_permutation_audit() -> pd.DataFrame:
    rng = np.random.default_rng(991)
    rows = []
    datasets = {
        "controlled_positive": RESULTS / "controlled_positive",
        "pseudo_st_balanced": RESULTS / "pseudo_st_positive" / "balanced",
        "pseudo_st_low_purity": RESULTS / "pseudo_st_positive" / "low_purity",
        "real_positive_xenium_myeloid": RESULTS / "real_positive_xenium_melanoma",
        "visium_hd_crc_stress": RESULTS / "visium_hd_crc",
    }
    for dataset, base in datasets.items():
        edge_path = base / "candidate_origin_graph_edges.csv"
        if not edge_path.exists():
            continue
        edges = pd.read_csv(edge_path)
        axes = sorted(set(edges["axis_i"].astype(str)) | set(edges["axis_j"].astype(str)))
        origin_by_axis: dict[str, str] = {}
        for row in edges.itertuples(index=False):
            origin_by_axis[str(row.axis_i)] = str(row.origin_i)
            origin_by_axis[str(row.axis_j)] = str(row.origin_j)
        origins = np.array([origin_by_axis[a] for a in axes], dtype=object)
        axis_to_pos = {a: i for i, a in enumerate(axes)}
        expr = edges["expression_edge"].astype(bool).to_numpy()
        pruned = expr & (~edges["origin_constrained_edge"].astype(bool).to_numpy())
        observed = int(pruned.sum())
        null_counts = []
        for _ in range(N_PERM):
            shuffled = origins.copy()
            rng.shuffle(shuffled)
            conflict = []
            for row in edges.itertuples(index=False):
                oi = shuffled[axis_to_pos[str(row.axis_i)]]
                oj = shuffled[axis_to_pos[str(row.axis_j)]]
                conflict.append(oi != oj)
            conflict = np.asarray(conflict, dtype=bool)
            null_counts.append(int((expr & conflict).sum()))
        null = np.asarray(null_counts)
        rows.append({
            "dataset": dataset,
            "audit": "origin_profile_permutation_conflict_edges",
            "observed_pruned_expression_edges": observed,
            "null_mean_conflict_edges": float(null.mean()),
            "null_sd_conflict_edges": float(null.std()),
            "empirical_p_ge_observed": float((np.sum(null >= observed) + 1) / (N_PERM + 1)),
            "n_permutations": N_PERM,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "origin_profile_permutation_summary_v2.csv", index=False)
    return out


def adaptive_label_consistency(null: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in null.itertuples(index=False):
        if row.family_id == "F0":
            adaptive = "target_axis" if row.observed > 0 and row.empirical_p < 0.05 else "not_target_axis"
            fixed = "target_axis"
        elif row.family_id == "F1":
            adaptive = "off_target" if row.observed > 0 and row.empirical_p < 0.05 else "non_identifiable"
            fixed = "off_target"
        else:
            adaptive = "not_target_axis" if row.observed <= 0 or row.empirical_p >= 0.05 else "target_axis"
            fixed = "not_target_axis"
        rows.append({
            "dataset": row.dataset,
            "family_id": row.family_id,
            "audit": row.audit,
            "fixed_call": fixed,
            "empirical_null_call": adaptive,
            "consistent": fixed == adaptive,
            "empirical_p": row.empirical_p,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "adaptive_label_consistency_v2.csv", index=False)
    return out


def write_compact_summary(
    baseline: pd.DataFrame,
    split: pd.DataFrame,
    perturb: pd.DataFrame,
    null: pd.DataFrame,
    profile_null: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    def table_text(df: pd.DataFrame) -> str:
        return df.to_csv(index=False).strip()

    bsum = baseline.groupby("baseline", as_index=False).agg(
        target_calls=("target_calls", "sum"),
        truth_supported_calls=("truth_supported_calls", "sum"),
        unsupported_promotions=("unsupported_promotions", "sum"),
        offtarget_rejected=("offtarget_rejected", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()),
    )
    split_target = split["f0_strict_target_retained"].mean()
    split_off = split["f1_forbidden_rejection_retained"].mean()
    pert = perturb.groupby(["audit", "fraction"], as_index=False).agg(
        target_retention=("f0_strict_target_retained", "mean"),
        off_target_rejection=("f1_forbidden_rejection_retained", "mean"),
        false_target_promotion=("f1_false_target_promotion", "mean"),
    )
    lines = [
        "# ORIGIN-Ax v2 Interpretation Audits",
        "",
        "These audits do not change ORIGIN-Ax scoring or thresholds.",
        "",
        "## Baseline comparison",
        "",
        table_text(bsum),
        "",
        "## Split stability",
        "",
        f"- F0 strict target retention across {N_SPLIT} anchor/held-out splits: {split_target:.2f}.",
        f"- F1 forbidden off-target rejection across {N_SPLIT} splits: {split_off:.2f}.",
        "",
        "## Reference perturbation",
        "",
        table_text(pert),
        "",
        "## Empirical null",
        "",
        table_text(null),
        "",
        "## Origin-profile permutation",
        "",
        table_text(profile_null),
        "",
        "## Fixed-rule versus empirical-null consistency",
        "",
        table_text(consistency),
        "",
    ]
    (RESULTS / "INTERPRETATION_AUDITS_V2_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    baseline = baseline_comparison_v2()
    split = split_stability_audit()
    perturb = reference_perturbation_audit()
    null = empirical_null_audit()
    profile_null = origin_profile_permutation_audit()
    consistency = adaptive_label_consistency(null)
    write_compact_summary(baseline, split, perturb, null, profile_null, consistency)
    print(baseline.groupby("baseline")["unsupported_promotions"].sum().to_string())
    print(split[["f0_strict_target_retained", "f1_forbidden_rejection_retained"]].mean().to_string())
    print(null[["dataset", "family_id", "audit", "observed", "empirical_p"]].to_string(index=False))
    print(profile_null[["dataset", "observed_pruned_expression_edges", "empirical_p_ge_observed"]].to_string(index=False))
    print(consistency[["dataset", "family_id", "fixed_call", "empirical_null_call", "consistent"]].to_string(index=False))


if __name__ == "__main__":
    main()
