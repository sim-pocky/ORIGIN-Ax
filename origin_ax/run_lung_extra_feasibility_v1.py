"""Extra real-dataset feasibility audit on Xenium lung.

This is intentionally supplemental. It does not change ORIGIN-Ax scoring,
thresholds, graph construction, or the main manuscript claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core import OriginAxConfig
from run_common import run_origin_ax
from run_real_positive_v1 import (
    TARGET_CONFIGS,
    _max_offtarget_source,
    build_real_neighborhoods,
    strict_real_positive_summary,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_v1" / "extra_lung_feasibility"


LYMPHOID_TARGET = [
    "CD3D", "CD3E", "TRAC", "CD8A", "NKG7", "GZMB", "PRF1", "MS4A1", "CD79A", "MZB1",
]
MYELOID_OFF = ["AIF1", "CD68", "CD14", "CXCL10", "FCGR3A"]
STROMAL_OFF = ["COL5A2", "PDGFRA", "PDGFRB", "SFRP2", "ACTA2"]
ENDOTHELIAL_OFF = ["PECAM1", "VWF"]
LUNG_EPITHELIAL_OFF = ["EPCAM", "KRT7", "AGER", "FOXJ1"]


def apply_lung_offtarget_report(fam: pd.DataFrame) -> pd.DataFrame:
    categories = {
        "myeloid": MYELOID_OFF,
        "stromal": STROMAL_OFF,
        "endothelial": ENDOTHELIAL_OFF,
        "lung_epithelial": LUNG_EPITHELIAL_OFF,
    }
    forbidden = set(categories)
    out = fam.copy()
    sources = [_max_offtarget_source(str(top), categories) for top in out["top_genes"]]
    out["max_offtarget_source"] = sources
    for idx, source in zip(out.index, sources):
        if source in forbidden:
            out.loc[idx, "family_label"] = f"{source}_like"
            out.loc[idx, "final_call"] = "not_target_axis"
            out.loc[idx, "rejection_or_downgrade_reason"] = "lung_extra_offtarget_reference"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/tumor_st/xenium_lung/processed.h5ad")
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--n-neighborhoods", type=int, default=4000)
    parser.add_argument("--neighbor-cells", type=int, default=25)
    parser.add_argument("--axis-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    TARGET_CONFIGS["lymphoid_immune_lung"] = {
        "cell_types": ["T_cell", "NK", "B_cell"],
        "target_genes": LYMPHOID_TARGET,
        "normal_penalty_genes": LUNG_EPITHELIAL_OFF,
        "tme_penalty_genes": MYELOID_OFF + STROMAL_OFF + ENDOTHELIAL_OFF,
        "offtarget_categories": {
            "myeloid": MYELOID_OFF,
            "stromal": STROMAL_OFF,
            "endothelial": ENDOTHELIAL_OFF,
            "lung_epithelial": LUNG_EPITHELIAL_OFF,
        },
        "normal_penalty_weight": 2.0,
        "tme_penalty_weight": 2.0,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    build_args = argparse.Namespace(
        path=args.path,
        label_key=args.label_key,
        target="lymphoid_immune_lung",
        target_cell_types=None,
        n_neighborhoods=args.n_neighborhoods,
        neighbor_cells=args.neighbor_cells,
        seed=args.seed,
    )
    dat = build_real_neighborhoods(build_args)
    dat["neighborhood_meta"].to_csv(OUT / "neighborhood_anchor_table.csv", index=False)
    dat["gene_weight_table"].to_csv(OUT / "gene_weight_anchor.csv", index=False)
    pd.DataFrame([{
        "dataset": "xenium_lung",
        "target": "lymphoid_immune",
        "anchor_genes": ",".join(sorted(dat["anchor_genes"])),
        "heldout_genes": ",".join(sorted(dat["heldout_genes"])),
        "target_cell_types": ",".join(dat["target_cell_types"]),
        "n_neighborhoods": args.n_neighborhoods,
        "neighbor_cells": args.neighbor_cells,
    }]).to_csv(OUT / "design.csv", index=False)

    fam, summary = run_origin_ax(
        dataset="extra_xenium_lung_lymphoid_immune",
        x=dat["x"],
        genes=dat["genes"],
        coords=dat["coords"],
        clone=dat["clone"],
        niche=dat["niche"],
        gene_weights=dat["gene_weights"],
        truth_eval_genes=dat["truth_eval_genes"],
        out_dir=OUT,
        config=OriginAxConfig(k=args.axis_k),
    )
    fam = apply_lung_offtarget_report(fam)
    fam.to_csv(OUT / "axis_family_table.csv", index=False)
    strict = strict_real_positive_summary(fam, {
        "myeloid": MYELOID_OFF,
        "stromal": STROMAL_OFF,
        "endothelial": ENDOTHELIAL_OFF,
        "lung_epithelial": LUNG_EPITHELIAL_OFF,
    })
    strict.to_csv(OUT / "strict_summary.csv", index=False)

    top_target = fam[fam["final_call"].eq("target_axis")].copy()
    if not top_target.empty:
        top_target = top_target.sort_values(["heldout_truth_overlap", "origin_core_score"], ascending=False).head(1)
        main_family = str(top_target.iloc[0]["family_id"])
        main_genes = str(top_target.iloc[0]["top_genes"])
    else:
        main_family = ""
        main_genes = ""
    n_strict = int(strict["strict_positive"].astype(bool).sum())
    n_target = int(fam["final_call"].eq("target_axis").sum())
    legacy_drop = {
        "dataset",
        "n_target_families",
        "n_truth_target_families",
        "n_passed_truth_mal_axis",
        "n_false_target_calls",
    }
    compact = {
        **{k: v for k, v in summary.items() if k not in legacy_drop},
        "dataset": "extra_xenium_lung_lymphoid_immune",
        "target": "lymphoid_immune",
        "n_target_families": n_target,
        "n_strict_positive_families": n_strict,
        "n_non_strict_target_families": int(n_target - n_strict),
        "n_false_target_calls_under_lung_strict_rule": int(strict["false_target_boundary"].astype(bool).sum()),
        "n_forbidden_offtarget_rejected": int(fam["max_offtarget_source"].astype(str).ne("none").sum()),
        "main_positive_family": main_family,
        "main_positive_top_genes": main_genes,
        "final_use": "supplement_candidate" if n_strict >= 1 else "not_used_in_main_claim",
    }
    pd.DataFrame([compact]).to_csv(OUT / "extra_lung_feasibility_summary.csv", index=False)
    lines = [
        "# Extra Xenium Lung Feasibility",
        "",
        "This supplemental run does not change ORIGIN-Ax scoring or manuscript claims.",
        "",
        f"- Target: lymphoid immune ({compact['target']}).",
        f"- Candidate axes: {compact['n_candidates']}.",
        f"- Axis families: {compact['n_axis_families']}.",
        f"- Target families: {compact['n_target_families']}.",
        f"- Strict positives: {compact['n_strict_positive_families']}.",
        f"- Forbidden off-target rejections: {compact['n_forbidden_offtarget_rejected']}.",
        f"- Main positive family: {main_family or 'none'}.",
        f"- Final use: {compact['final_use']}.",
        "",
    ]
    (OUT / "EXTRA_LUNG_FEASIBILITY_SUMMARY.md").write_text("\n".join(lines))
    print(pd.DataFrame([compact]).to_string(index=False))
    print(strict.to_string(index=False))


if __name__ == "__main__":
    main()
