"""Final ORIGIN-Ax v1 result checks.

This script verifies paper-level claims from saved CSV outputs only. It does not
run decompositions, change scoring, or expand any dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results_v1"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AssertionError(f"missing required file: {path}")
    return pd.read_csv(path)


def _one(df: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = df.loc[df[column].astype(str) == value]
    if len(rows) != 1:
        raise AssertionError(f"expected exactly one row where {column}={value!r}, found {len(rows)}")
    return rows.iloc[0]


def main() -> None:
    real_strict = _read(RESULTS / "real_positive_xenium_melanoma" / "strict_real_positive_summary.csv")
    if int(real_strict["strict_positive"].astype(bool).sum()) < 1:
        raise AssertionError("real-positive experiment must contain at least one strict positive family")
    forbidden = {"stromal", "fibroblast", "ECM", "endothelial", "malignant_melanoma"}
    forbidden_rejected = real_strict.loc[
        real_strict["max_offtarget_source"].astype(str).isin(forbidden)
        & real_strict["final_call"].astype(str).eq("not_target_axis")
    ]
    if forbidden_rejected.empty:
        raise AssertionError("real-positive experiment must reject at least one forbidden off-target family")

    crc_audit = _read(RESULTS / "visium_hd_crc" / "baseline_overcall_audit.csv")
    axis5 = _one(crc_audit, "program_label", "axis5_like")
    if str(axis5["v1_final_call"]) not in {"not_target_axis", "off_target_axis"}:
        raise AssertionError("CRC Axis-5-like final_call must not be target_axis")

    crc_summary = _read(RESULTS / "visium_hd_crc" / "summary.csv")
    if int(crc_summary["n_target_families"].iloc[0]) != 0:
        raise AssertionError("CRC target families must be 0")

    controlled = _read(RESULTS / "controlled_positive" / "summary.csv")
    if int(controlled["n_target_families"].iloc[0]) <= 0:
        raise AssertionError("controlled positive must have target families > 0")

    pseudo = _read(RESULTS / "pseudo_st_positive" / "summary.csv")
    for condition in ["balanced", "low_purity"]:
        row = pseudo.loc[pseudo["condition"].astype(str) == condition]
        if row.empty:
            raise AssertionError(f"missing pseudo-ST condition: {condition}")
        if int(row["n_target_families"].iloc[0]) <= 0:
            raise AssertionError(f"pseudo-ST {condition} must have target families > 0")

    bench = _read(RESULTS / "benchmark_refinement_summary.csv")
    bench_pseudo = bench.loc[bench["dataset"].astype(str) == "pseudo_ST_positive"]
    if bench_pseudo.empty or (bench_pseudo["origin_ax_target_calls"].astype(int) <= 0).any():
        raise AssertionError("benchmark pseudo-ST refinement must retain target calls for every benchmark method")
    if (bench_pseudo["origin_ax_false_target_calls"].astype(int) > bench_pseudo["benchmark_false_target_calls"].astype(int)).any():
        raise AssertionError("benchmark pseudo-ST refinement must not increase false target calls")

    bench_real = bench.loc[bench["dataset"].astype(str) == "real_positive_xenium_myeloid"]
    if bench_real.empty or not bench_real["strict_positive_found"].astype(bool).any():
        raise AssertionError("benchmark real-positive refinement must find a strict positive for at least one input method")
    for method in ["raw_ICA_only", "NMF_only", "cNMF_like_repeated_NMF"]:
        row = bench_real.loc[bench_real["benchmark_method"].astype(str) == method]
        if not bool(row["off_target_rejected"].iloc[0]):
            raise AssertionError(f"benchmark real-positive {method} must reject an off-target family")

    bench_crc = bench.loc[bench["dataset"].astype(str) == "visium_hd_crc_stress"]
    if bench_crc.empty:
        raise AssertionError("missing benchmark CRC stress rows")
    if (bench_crc["origin_ax_target_calls"].astype(int) != 0).any():
        raise AssertionError("benchmark CRC stress must have 0 origin_ax_target_calls after off-target report")
    if (~bench_crc["off_target_rejected"].astype(bool)).any():
        raise AssertionError("benchmark CRC stress must reject Axis-5-like/off-target families")

    swap = _read(RESULTS / "target_swap_generalization_summary.csv")
    f1_swap = _one(swap, "family_id", "myeloid_F1_like")
    f2_swap = _one(swap, "family_id", "stromal_F2_like")
    if str(f1_swap["call_under_stromal_target"]) == "target_axis":
        raise AssertionError("target-swap F1-like myeloid family must not be a stromal target")
    if str(f2_swap["call_under_myeloid_target"]) == "target_axis":
        raise AssertionError("target-swap F2-like stromal family must not be a myeloid target")
    if str(f2_swap["call_under_stromal_target"]) != "target_axis":
        raise AssertionError("target-swap F2-like stromal family must be recovered as stromal target")

    out = RESULTS / "final_check_passed.txt"
    out.write_text(
        "Final ORIGIN-Ax v1 checks passed.\n"
        "Verified: real strict positive family, real forbidden off-target rejection, "
        "CRC Axis-5-like not-target, CRC target families = 0, "
        "controlled and pseudo-ST positives recover target families, "
        "benchmark refinement retains positives and rejects CRC off-target calls, "
        "target-swap recovers the stromal F2-like family only under the stromal target definition.\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
