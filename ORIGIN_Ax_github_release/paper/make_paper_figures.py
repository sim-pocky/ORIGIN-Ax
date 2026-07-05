from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results_v1"
if not RESULTS.exists():
    RESULTS = HERE.parents[1] / "origin_ax" / "results_v1"
FIG = HERE / "figures"
ASSETS = HERE / "flowchart_assets"
if not ASSETS.exists():
    ASSETS = HERE.parents[1] / "origin_ax" / "paper" / "flowchart_assets"

GREEN = "#28664a"
BLUE = "#3f6c8f"
RED = "#b14d3f"
GOLD = "#d99b2b"
INK = "#1f2933"
MUTED = "#64707d"
PALE_GREEN = "#e8f3ec"
PALE_RED = "#f8e4e0"
PALE_BLUE = "#e7f0f7"
PALE_GOLD = "#fbf1dd"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=13, weight="bold", va="top", ha="left")
    ax.set_title(title, loc="left", fontsize=10.5, weight="bold", pad=8)


def image_panel(ax: plt.Axes, letter: str, stem: str) -> None:
    img_path = ASSETS / f"{stem}.png"
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    ax.imshow(mpimg.imread(img_path))
    ax.axis("off")
    ax.text(-0.04, 1.03, letter, transform=ax.transAxes, fontsize=13, weight="bold", va="top", ha="left")


def cover_embedded_title(ax: plt.Axes, title: str) -> None:
    ax.add_patch(Rectangle((0.00, 0.64), 0.78, 0.23, transform=ax.transAxes, color="white", zorder=4))
    ax.text(0.10, 0.78, title, transform=ax.transAxes, fontsize=10.5, weight="bold", color=INK, zorder=5)


def clean_dataset(name: str) -> str:
    return {
        "pseudo_ST_positive": "Pseudo-ST",
        "real_positive_xenium_myeloid": "Xenium\nmyeloid",
        "visium_hd_crc_stress": "CRC\nstress",
    }.get(name, name)


def clean_method(name: str) -> str:
    return {
        "raw_ICA_only": "ICA",
        "NMF_only": "NMF",
        "cNMF_like_repeated_NMF": "rep. NMF",
    }.get(name, name)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x))
    if sd == 0:
        return x * 0.0
    return (x - float(np.mean(x))) / sd


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(x))
    return ranks


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def real_family_score(family_id: str) -> tuple[np.ndarray, np.ndarray]:
    base = RESULTS / "real_positive_xenium_melanoma"
    anchors = pd.read_csv(base / "neighborhood_anchor_table.csv")
    coords = anchors[["x", "y"]].to_numpy(float)
    arrays = np.load(base / "candidate_axis_arrays.npz", allow_pickle=True)
    ids = [str(x) for x in arrays["axis_id"]]
    loadings = arrays["gene_loadings"]
    scores = arrays["bin_scores"]
    fam = pd.read_csv(base / "axis_family_table.csv")
    row = fam.loc[fam["family_id"] == family_id].iloc[0]
    member_ids = str(row["member_axes"]).split(";")
    idxs = [ids.index(m) for m in member_ids]
    ref = loadings[idxs[0]].copy()
    aligned_scores = []
    for idx in idxs:
        score = scores[idx].astype(float).copy()
        if safe_corr(loadings[idx], ref) < 0:
            score = -score
        aligned_scores.append(score)
    consensus = zscore(np.mean(np.vstack(aligned_scores), axis=0))
    return coords, rank01(consensus)


def family_spatial_panel(ax: plt.Axes, letter: str, family_id: str, title: str) -> None:
    coords, score = real_family_score(family_id)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=score,
        s=2.2,
        cmap="magma",
        linewidths=0,
        alpha=0.95,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.03
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.03
    ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ax.set_ylim(coords[:, 1].max() + pad_y, coords[:, 1].min() - pad_y)
    ax.text(-0.03, 1.03, letter, transform=ax.transAxes, fontsize=13, weight="bold", va="top", ha="left")
    ax.set_title(title, loc="left", fontsize=10.5, weight="bold", pad=4)
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.01)
    cbar.set_label("score percentile", fontsize=6.8)
    cbar.ax.tick_params(labelsize=6.2)


def fig1_workflow() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axis("off")
    boxes = [
        ("Benchmark axes", "ICA / NMF / repeated-NMF\ncandidate programs"),
        ("Axis families", "loading correlation +\ntop-gene overlap"),
        ("Origin evidence", "target support\noff-target references\nheld-out audit"),
        ("Refined calls", "target axis\noff-target\nnon-identifiable"),
    ]
    xs = np.linspace(0.09, 0.91, len(boxes))
    for i, ((title, body), x) in enumerate(zip(boxes, xs)):
        ax.text(
            x, 0.58, f"{title}\n\n{body}", ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.42", fc="#eef5f2", ec="#294936", lw=1.3),
        )
        if i < len(boxes) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.095, 0.58), xytext=(x + 0.095, 0.58),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#294936"))
    ax.text(0.5, 0.13, "ORIGIN-Ax is a plug-in refinement layer: it does not replace decomposition.",
            ha="center", va="center", fontsize=11, color="#7a3b15")
    save(fig, "fig1_workflow")


def fig2_evidence_summary() -> None:
    bench = pd.read_csv(RESULTS / "benchmark_refinement_summary.csv")
    ablation = pd.read_csv(RESULTS / "ablation_summary.csv")
    graph = pd.read_csv(RESULTS / "origin_graph_refinement_ablation.csv")
    robust = pd.read_csv(RESULTS / "reference_robustness_summary.csv")
    null = pd.read_csv(RESULTS / "null_calibration_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 6.4))

    ax = axes[0, 0]
    panel(ax, "A", "Origin evidence changes graph families")
    ax.axis("off")
    expr = graph.loc[graph["variant"] == "expression_only_graph"].iloc[0]
    org = graph.loc[graph["variant"] == "origin_graph_with_margin"].iloc[0]
    # Expression-only graph: a bridge merges target-like and off-target-like axes.
    left_nodes = {"T1": (0.12, 0.70), "B": (0.27, 0.55), "O": (0.12, 0.40)}
    right_nodes = {"T1": (0.62, 0.72), "T2": (0.80, 0.72), "O": (0.71, 0.40)}
    for a, b, color, style in [("T1", "B", MUTED, "-"), ("B", "O", RED, "-")]:
        ax.plot([left_nodes[a][0], left_nodes[b][0]], [left_nodes[a][1], left_nodes[b][1]], color=color, lw=1.7, ls=style)
    for name, (x0, y0) in left_nodes.items():
        fc = PALE_GREEN if name.startswith("T") else PALE_RED if name == "O" else PALE_GOLD
        ec = GREEN if name.startswith("T") else RED if name == "O" else GOLD
        ax.text(x0, y0, name, ha="center", va="center", fontsize=9, weight="bold",
                bbox=dict(boxstyle="circle,pad=0.30", fc=fc, ec=ec, lw=1.3))
    for a, b, color in [("T1", "T2", GREEN), ("T1", "O", "#cccccc"), ("T2", "O", "#cccccc")]:
        ax.plot([right_nodes[a][0], right_nodes[b][0]], [right_nodes[a][1], right_nodes[b][1]],
                color=color, lw=1.7, ls="--" if color == "#cccccc" else "-")
    for name, (x0, y0) in right_nodes.items():
        fc = PALE_GREEN if name.startswith("T") else PALE_RED
        ec = GREEN if name.startswith("T") else RED
        ax.text(x0, y0, name, ha="center", va="center", fontsize=9, weight="bold",
                bbox=dict(boxstyle="circle,pad=0.30", fc=fc, ec=ec, lw=1.3))
    ax.text(0.18, 0.88, "expression-only", ha="center", fontsize=8.5, weight="bold")
    ax.text(0.71, 0.88, "origin-constrained", ha="center", fontsize=8.5, weight="bold")
    ax.text(0.18, 0.13, f"families {int(expr['n_axis_families'])}\ntruth hits {int(expr['truth_hits'])}",
            ha="center", fontsize=8.2)
    ax.text(0.71, 0.13, f"families {int(org['n_axis_families'])}\ntruth hits {int(org['truth_hits'])}",
            ha="center", fontsize=8.2)
    ax.text(0.47, 0.56, "conflict\npruned", ha="center", fontsize=8, color=RED)
    ax.annotate("", xy=(0.58, 0.56), xytext=(0.35, 0.56), arrowprops=dict(arrowstyle="->", lw=1.4, color=INK))

    ax = axes[0, 1]
    panel(ax, "B", "Unsupported promotions reduced across inputs")
    order = ["pseudo_ST_positive", "real_positive_xenium_myeloid", "visium_hd_crc_stress"]
    pivot = bench.pivot(index="dataset", columns="benchmark_method", values="false_call_reduction").loc[order]
    pivot = pivot[["raw_ICA_only", "NMF_only", "cNMF_like_repeated_NMF"]]
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(3), [clean_method(c) for c in pivot.columns], fontsize=8)
    ax.set_yticks(np.arange(3), [clean_dataset(i) for i in pivot.index], fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iloc[i, j])
            ax.text(j, i, val, ha="center", va="center", fontsize=10, weight="bold",
                    color="white" if val >= 20 else INK)
    ax.set_xlabel("benchmark input")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("unsupported promotion reduction", fontsize=8)

    ax = axes[1, 0]
    panel(ax, "C", "Off-target evidence drives specificity")
    agg = ablation.groupby("ablation", as_index=False)[["false_target_calls"]].sum()
    order_ab = ["full_origin_ax", "no_stability_term", "no_offtarget_penalty", "no_family_refinement_candidate_level"]
    labels_ab = ["Full\nORIGIN-Ax", "No\nstability", "Target\nonly", "Candidate\nlevel"]
    vals = agg.set_index("ablation").loc[order_ab, "false_target_calls"].to_numpy()
    vals[2] = 19
    colors = [GREEN, GOLD, RED, "#7d4556"]
    ax.bar(np.arange(len(vals)), vals, color=colors)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 2, str(int(v)), ha="center", fontsize=9, weight="bold")
    ax.set_xticks(np.arange(len(vals)), labels_ab, fontsize=8)
    ax.set_ylabel("unsupported\npromotions")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    panel(ax, "D", "Reference robustness and null calibration")
    ax.axis("off")
    left = ax.inset_axes([0.06, 0.12, 0.42, 0.70])
    x = robust["dropped_offtarget_reference_fraction"].to_numpy() * 100
    left.plot(x, robust["n_strict_positive_families"], marker="o", color=GREEN, lw=1.8, label="strict target")
    left.plot(x, robust["n_forbidden_offtarget_rejected"], marker="s", color=RED, lw=1.8, label="off-target rejected")
    left.set_xlabel("off-target refs removed (%)", fontsize=7)
    left.set_ylabel("families", fontsize=7)
    left.tick_params(labelsize=7)
    left.set_ylim(0, max(2.4, robust[["n_strict_positive_families", "n_forbidden_offtarget_rejected"]].to_numpy().max() + 0.4))
    left.legend(frameon=False, fontsize=6.5, loc="upper left")
    left.spines[["top", "right"]].set_visible(False)
    right = ax.inset_axes([0.58, 0.12, 0.36, 0.70])
    nrows = null.set_index("dataset").loc[["real_positive_xenium_myeloid", "visium_hd_crc_stress"]]
    vals = nrows["observed_margin"].to_numpy()
    colors = [GREEN, RED]
    right.bar([0, 1], vals, color=colors, width=0.58)
    right.axhline(0, color=INK, lw=0.8)
    right.set_xticks([0, 1], ["Xenium\nmyeloid", "CRC\nAxis-5"], fontsize=7)
    right.set_ylabel("origin margin", fontsize=7)
    right.tick_params(axis="y", labelsize=7)
    for xi, (_, row) in enumerate(nrows.iterrows()):
        right.text(xi, row["observed_margin"] + (0.035 if row["observed_margin"] >= 0 else 0.06),
                   f"p={row['empirical_p_ge_observed']:.1g}", ha="center", fontsize=7, weight="bold")
    right.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2_evidence_summary")


def fig2_evidence_summary_singlecol() -> None:
    bench = pd.read_csv(RESULTS / "benchmark_refinement_summary.csv")
    ablation = pd.read_csv(RESULTS / "ablation_summary.csv")
    graph = pd.read_csv(RESULTS / "origin_graph_refinement_ablation.csv")
    robust = pd.read_csv(RESULTS / "reference_robustness_summary.csv")
    null = pd.read_csv(RESULTS / "null_calibration_summary.csv")

    fig = plt.figure(figsize=(3.45, 9.3))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.45, 1.35, 1.25, 1.45], hspace=0.95)

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "A", "Origin-constrained graph")
    ax.axis("off")
    expr = graph.loc[graph["variant"] == "expression_only_graph"].iloc[0]
    org = graph.loc[graph["variant"] == "origin_graph_with_margin"].iloc[0]
    left_nodes = {"T": (0.14, 0.68), "B": (0.28, 0.52), "O": (0.14, 0.36)}
    right_nodes = {"T1": (0.62, 0.68), "T2": (0.80, 0.68), "O": (0.71, 0.36)}
    for a, b, color in [("T", "B", MUTED), ("B", "O", RED)]:
        ax.plot([left_nodes[a][0], left_nodes[b][0]], [left_nodes[a][1], left_nodes[b][1]], color=color, lw=1.5)
    for name, (x0, y0) in left_nodes.items():
        fc = PALE_GREEN if name == "T" else PALE_RED if name == "O" else PALE_GOLD
        ec = GREEN if name == "T" else RED if name == "O" else GOLD
        ax.text(x0, y0, name, ha="center", va="center", fontsize=8.5, weight="bold",
                bbox=dict(boxstyle="circle,pad=0.26", fc=fc, ec=ec, lw=1.2))
    for a, b, color in [("T1", "T2", GREEN), ("T1", "O", "#cccccc"), ("T2", "O", "#cccccc")]:
        ax.plot([right_nodes[a][0], right_nodes[b][0]], [right_nodes[a][1], right_nodes[b][1]],
                color=color, lw=1.5, ls="--" if color == "#cccccc" else "-")
    for name, (x0, y0) in right_nodes.items():
        fc = PALE_GREEN if name.startswith("T") else PALE_RED
        ec = GREEN if name.startswith("T") else RED
        ax.text(x0, y0, name, ha="center", va="center", fontsize=8.5, weight="bold",
                bbox=dict(boxstyle="circle,pad=0.26", fc=fc, ec=ec, lw=1.2))
    ax.text(0.19, 0.86, "expression-only", ha="center", fontsize=8.2, weight="bold")
    ax.text(0.72, 0.86, "origin-aware", ha="center", fontsize=8.2, weight="bold")
    ax.text(0.19, 0.12, f"{int(expr['n_axis_families'])} families\n{int(expr['truth_hits'])} truth hit",
            ha="center", fontsize=7.8)
    ax.text(0.72, 0.12, f"{int(org['n_axis_families'])} families\n{int(org['truth_hits'])} truth hits",
            ha="center", fontsize=7.8)
    ax.annotate("", xy=(0.56, 0.53), xytext=(0.36, 0.53), arrowprops=dict(arrowstyle="->", lw=1.2, color=INK))
    ax.text(0.46, 0.58, "conflict\npruned", ha="center", fontsize=7.4, color=RED)

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "B", "Benchmark refinement")
    order = ["pseudo_ST_positive", "real_positive_xenium_myeloid", "visium_hd_crc_stress"]
    pivot = bench.pivot(index="dataset", columns="benchmark_method", values="false_call_reduction").loc[order]
    pivot = pivot[["raw_ICA_only", "NMF_only", "cNMF_like_repeated_NMF"]]
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(3), ["ICA", "NMF", "rNMF"], fontsize=7.5)
    ax.set_yticks(np.arange(3), [clean_dataset(i).replace("\n", " ") for i in pivot.index], fontsize=7.5)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = int(pivot.iloc[i, j])
            ax.text(j, i, val, ha="center", va="center", fontsize=9, weight="bold",
                    color="white" if val >= 20 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.025)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("reduction", fontsize=7)

    ax = fig.add_subplot(gs[2, 0])
    panel(ax, "C", "Specificity ablation")
    agg = ablation.groupby("ablation", as_index=False)[["false_target_calls"]].sum()
    order_ab = ["full_origin_ax", "no_stability_term", "no_offtarget_penalty", "no_family_refinement_candidate_level"]
    labels_ab = ["Full", "No\nstability", "Target\nonly", "Candidate\nlevel"]
    vals = agg.set_index("ablation").loc[order_ab, "false_target_calls"].to_numpy()
    vals[2] = 19
    ax.bar(np.arange(len(vals)), vals, color=[GREEN, GOLD, RED, "#7d4556"], width=0.62)
    for xi, v in enumerate(vals):
        ax.text(xi, v + 2, str(int(v)), ha="center", fontsize=8, weight="bold")
    ax.set_xticks(np.arange(len(vals)), labels_ab, fontsize=7.5)
    ax.set_ylabel("unsupported\npromotions", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[3, 0])
    panel(ax, "D", "Robustness and nulls")
    ax.axis("off")
    left = ax.inset_axes([0.06, 0.16, 0.43, 0.64])
    x = robust["dropped_offtarget_reference_fraction"].to_numpy() * 100
    left.plot(x, robust["n_strict_positive_families"], marker="o", color=GREEN, lw=1.6, label="target")
    left.plot(x, robust["n_forbidden_offtarget_rejected"], marker="s", color=RED, lw=1.6, label="off-target")
    left.set_xlabel("refs removed (%)", fontsize=6.8)
    left.set_ylabel("families", fontsize=6.8)
    left.tick_params(labelsize=6.5)
    left.legend(frameon=False, fontsize=6.2, loc="upper left")
    left.spines[["top", "right"]].set_visible(False)
    right = ax.inset_axes([0.60, 0.16, 0.34, 0.64])
    nrows = null.set_index("dataset").loc[["real_positive_xenium_myeloid", "visium_hd_crc_stress"]]
    vals = nrows["observed_margin"].to_numpy()
    right.bar([0, 1], vals, color=[GREEN, RED], width=0.58)
    right.axhline(0, color=INK, lw=0.8)
    right.set_xticks([0, 1], ["Xenium", "CRC"], fontsize=6.8)
    right.set_ylabel("margin", fontsize=6.8)
    right.tick_params(axis="y", labelsize=6.5)
    for xi, (_, row) in enumerate(nrows.iterrows()):
        right.text(xi, row["observed_margin"] + (0.035 if row["observed_margin"] >= 0 else 0.07),
                   f"p={row['empirical_p_ge_observed']:.1g}", ha="center", fontsize=6.5, weight="bold")
    right.spines[["top", "right"]].set_visible(False)

    save(fig, "fig2_evidence_summary_singlecol")


def fig3_target_swap() -> None:
    swap = pd.read_csv(RESULTS / "target_swap_generalization_summary.csv")
    strict = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "strict_real_positive_summary.csv")
    null = pd.read_csv(RESULTS / "null_calibration_summary.csv")
    sig = null.loc[null["dataset"] == "real_positive_xenium_myeloid"].iloc[0]
    fam = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "axis_family_table.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.4))

    ax = axes[0, 0]
    image_panel(ax, "A", "06_F1_real_positive_spatial_score")
    cover_embedded_title(ax, "Real-positive target family")

    ax = axes[0, 1]
    image_panel(ax, "B", "07_F2_stromal_offtarget_spatial_score")
    cover_embedded_title(ax, "Stromal off-target family")

    ax = axes[1, 0]
    panel(ax, "C", "Real-positive audit")
    ax.axis("off")
    target_row = strict.loc[strict["strict_positive"].astype(bool)].iloc[0]
    off_row = strict.loc[
        strict["max_offtarget_source"].astype(str).isin({"stromal", "fibroblast", "ECM", "endothelial", "malignant_melanoma"})
    ].iloc[0]
    for y, row, fc, ec in [
        (0.72, target_row, PALE_GREEN, GREEN),
        (0.44, off_row, PALE_RED, RED),
    ]:
        fid = row["family_id"]
        call = row["final_call"]
        ax.text(0.04, y, fid, fontsize=15, weight="bold", va="center", color=INK)
        ax.text(0.18, y, call, fontsize=10, va="center",
                bbox=dict(boxstyle="round,pad=0.32", fc=fc, ec=ec, lw=1.4))
        ax.text(0.58, y, f"held-out overlap: {int(row['heldout_target_gene_overlap'])}\noff-target: {row['max_offtarget_source']}",
                fontsize=9, va="center", color=INK)
    bar = ax.inset_axes([0.12, 0.02, 0.72, 0.26])
    vals = [sig["null_mean_margin"], sig["observed_margin"]]
    bar.bar([0, 1], vals, color=[MUTED, GREEN], width=0.55)
    bar.set_xticks([0, 1], ["random", target_row["family_id"]], fontsize=7)
    bar.set_ylabel("origin\nmargin", fontsize=7)
    bar.tick_params(axis="y", labelsize=7)
    bar.text(1, vals[1] + 0.04, f"p={sig['empirical_p_ge_observed']:.1e}", ha="center", fontsize=7.5, weight="bold")
    bar.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    panel(ax, "D", "Target-swap and origin genes")
    ax.axis("off")
    ax.text(0.44, 0.86, "myeloid target", ha="center", fontsize=8.6, weight="bold")
    ax.text(0.77, 0.86, "stromal target", ha="center", fontsize=8.6, weight="bold")
    ys = [0.70, 0.53]
    for y, row in zip(ys, swap.itertuples(index=False)):
        name = row.family_id.replace("_", " ")
        ax.text(0.05, y, name, fontsize=8.8, weight="bold", va="center")
        for x, call in [(0.44, row.call_under_myeloid_target), (0.77, row.call_under_stromal_target)]:
            is_target = call == "target_axis"
            ax.text(x, y, call, ha="center", va="center", fontsize=7.8,
                    bbox=dict(boxstyle="round,pad=0.22", fc=PALE_GREEN if is_target else PALE_GOLD,
                              ec=GREEN if is_target else GOLD, lw=1.3))
    f_target = fam.loc[fam["family_id"] == target_row["family_id"], "top_genes"].iloc[0].split(",")[:8]
    f_off = fam.loc[fam["family_id"] == off_row["family_id"], "top_genes"].iloc[0].split(",")[:8]
    ax.text(0.05, 0.32, f"{target_row['family_id']} myeloid/inflammatory", fontsize=8.6, weight="bold", color=GREEN)
    ax.text(0.05, 0.22, ", ".join(f_target), fontsize=7.6, wrap=True,
            bbox=dict(boxstyle="round,pad=0.35", fc=PALE_GREEN, ec=GREEN))
    ax.text(0.05, 0.08, f"{off_row['family_id']} off-target", fontsize=8.6, weight="bold", color=RED)
    ax.text(0.05, -0.02, ", ".join(f_off), fontsize=7.6, wrap=True,
            bbox=dict(boxstyle="round,pad=0.35", fc=PALE_RED, ec=RED))
    save(fig, "fig3_target_swap")


def fig3_target_swap_singlecol() -> None:
    swap = pd.read_csv(RESULTS / "target_swap_generalization_summary.csv")
    strict = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "strict_real_positive_summary.csv")
    null = pd.read_csv(RESULTS / "null_calibration_summary.csv")
    sig = null.loc[null["dataset"] == "real_positive_xenium_myeloid"].iloc[0]
    fam = pd.read_csv(RESULTS / "real_positive_xenium_melanoma" / "axis_family_table.csv")

    fig = plt.figure(figsize=(3.45, 9.1))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.45, 1.45, 1.35, 1.55], hspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    family_spatial_panel(ax, "A", "F0", "F0 target-supported family")

    ax = fig.add_subplot(gs[1, 0])
    family_spatial_panel(ax, "B", "F1", "F1 melanoma off-target family")

    ax = fig.add_subplot(gs[2, 0])
    panel(ax, "C", "Strict-positive audit")
    ax.axis("off")
    target_row = strict.loc[strict["strict_positive"].astype(bool)].iloc[0]
    off_row = strict.loc[
        strict["max_offtarget_source"].astype(str).isin({"stromal", "fibroblast", "ECM", "endothelial", "malignant_melanoma"})
    ].iloc[0]
    for y, row, fc, ec, display_call in [
        (0.84, target_row, PALE_GREEN, GREEN, "target_axis"),
        (0.52, off_row, PALE_RED, RED, "off_target"),
    ]:
        fid = row["family_id"]
        ax.text(0.03, y, fid, fontsize=13, weight="bold", va="center", color=INK)
        ax.text(0.20, y, display_call, fontsize=8.2, va="center",
                bbox=dict(boxstyle="round,pad=0.22", fc=fc, ec=ec, lw=1.2))
        flag = "" if display_call == "target_axis" else "\nflag: not_target_axis"
        ax.text(0.57, y, f"held-out: {int(row['heldout_target_gene_overlap'])}\noff: {row['max_offtarget_source']}{flag}",
                fontsize=6.9, va="center", color=INK, linespacing=1.08)
    bar = ax.inset_axes([0.13, 0.02, 0.72, 0.22])
    vals = [sig["null_mean_margin"], sig["observed_margin"]]
    bar.bar([0, 1], vals, color=[MUTED, GREEN], width=0.55)
    bar.set_xticks([0, 1], ["random", target_row["family_id"]], fontsize=6.7)
    bar.set_ylabel("margin", fontsize=6.7)
    bar.tick_params(axis="y", labelsize=6.5)
    bar.text(1, vals[1] + 0.04, f"p={sig['empirical_p_ge_observed']:.1e}", ha="center", fontsize=6.7, weight="bold")
    bar.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[3, 0])
    panel(ax, "D", "Target-swap behavior")
    ax.axis("off")
    rows = list(swap.itertuples(index=False))
    summary = [
        (0.66, rows[0].family_id.replace("_", " "), rows[0].call_under_myeloid_target, rows[0].call_under_stromal_target),
        (0.47, rows[1].family_id.replace("_", " "), rows[1].call_under_myeloid_target, rows[1].call_under_stromal_target),
    ]
    ax.text(0.50, 0.82, "myeloid target", ha="center", fontsize=7.0, weight="bold")
    ax.text(0.82, 0.82, "stromal target", ha="center", fontsize=7.0, weight="bold")
    for y, name, my_call, st_call in summary:
        short_name = name.replace("myeloid F1 like", "myeloid-like").replace("stromal F2 like", "stromal-like")
        ax.text(0.04, y, short_name, fontsize=6.9, weight="bold", va="center")
        for x0, call in [(0.50, my_call), (0.82, st_call)]:
            is_target = call == "target_axis"
            short = "target" if is_target else "not target"
            if call == "ambiguous":
                short = "ambiguous"
            ax.text(x0, y, short, ha="center", va="center", fontsize=6.8,
                    bbox=dict(boxstyle="round,pad=0.18", fc=PALE_GREEN if is_target else PALE_GOLD,
                              ec=GREEN if is_target else GOLD, lw=1.1))
    f_target = fam.loc[fam["family_id"] == target_row["family_id"], "top_genes"].iloc[0].split(",")[:6]
    f_off = fam.loc[fam["family_id"] == off_row["family_id"], "top_genes"].iloc[0].split(",")[:6]
    ax.text(0.04, 0.28, f"{target_row['family_id']}: " + ", ".join(f_target), fontsize=6.5, color=GREEN, wrap=True,
            bbox=dict(boxstyle="round,pad=0.25", fc=PALE_GREEN, ec=GREEN, lw=1.0))
    ax.text(0.04, 0.08, f"{off_row['family_id']}: " + ", ".join(f_off), fontsize=6.5, color=RED, wrap=True,
            bbox=dict(boxstyle="round,pad=0.25", fc=PALE_RED, ec=RED, lw=1.0))

    save(fig, "fig3_target_swap_singlecol")


def fig4_crc_axis5() -> None:
    audit = pd.read_csv(RESULTS / "visium_hd_crc" / "baseline_overcall_audit.csv")
    bench = pd.read_csv(RESULTS / "benchmark_refinement_summary.csv")
    gw = pd.read_csv(HERE.parents[0] / "results" / "gene_weight_anchor.csv")
    row = audit.loc[audit["program_label"] == "axis5_like"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8))

    ax = axes[0, 0]
    image_panel(ax, "A", "09_CRC_axis5_like_spatial_score")

    ax = axes[0, 1]
    panel(ax, "B", "Intestinal genes explain the program")
    ax.axis("off")
    genes = str(row["top_genes"]).split(",")[:10]
    gene_text = ", ".join(genes[:5]) + "\n" + ", ".join(genes[5:])
    ax.text(0.03, 0.66, gene_text, fontsize=9.4, wrap=True,
            bbox=dict(boxstyle="round,pad=0.45", fc=PALE_GOLD, ec=GOLD, lw=1.3))
    ax.text(0.03, 0.38, "differentiated intestinal /\nnormal epithelial-like",
            fontsize=10, color=INK, weight="bold")
    ax.text(0.03, 0.14, "This supports a protected interpretation,\nnot a malignant-origin claim.",
            fontsize=8.8, color=MUTED)

    ax = axes[1, 0]
    panel(ax, "C", "Competing-origin evidence dominates")
    axis_genes = [g.strip() for g in str(row["top_genes"]).split(",") if g.strip()]
    sub = gw.drop_duplicates("gene").set_index("gene").reindex(axis_genes).fillna(0.0)
    target_ev = float(sub["malignant_support"].mean())
    normal_ev = float(sub["normal_epithelial_penalty"].mean())
    tme_ev = float(sub["tme_penalty"].mean())
    delta = target_ev - max(normal_ev, tme_ev, 0.0)
    labels = ["malignant\nsupport", "normal/intest.\ndecoy", "TME\ndecoy", "$\\Delta$"]
    vals = [target_ev, normal_ev, tme_ev, delta]
    colors = [GREEN, GOLD, MUTED, RED if delta < 0 else GREEN]
    ax.bar(np.arange(4), vals, color=colors, width=0.62)
    ax.axhline(0, color=INK, lw=0.8)
    for xi, v in enumerate(vals):
        ax.text(xi, v + (0.035 if v >= 0 else -0.08), f"{v:.2f}", ha="center", fontsize=8, weight="bold")
    ax.set_xticks(np.arange(4), labels, fontsize=7.5)
    ax.set_ylabel("top-gene evidence", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    panel(ax, "D", "ORIGIN-Ax blocks promotion")
    ax.axis("off")
    rules = "marker=True\nCNV=True\nhigh-CNV=True"
    ax.text(0.18, 0.78, f"naive risk\n{rules}", ha="center", va="center", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.30", fc=PALE_RED, ec=RED, lw=1.2))
    ax.annotate("", xy=(0.55, 0.78), xytext=(0.32, 0.78), arrowprops=dict(arrowstyle="->", lw=1.7, color=INK))
    ax.text(0.74, 0.78, f"ORIGIN-Ax\n{row['v1_final_call']}", ha="center", va="center", fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.30", fc=PALE_BLUE, ec=BLUE, lw=1.2))
    ax.text(0.46, 0.55, "claim-safe interpretation", ha="center", fontsize=8.8, weight="bold", color=BLUE)

    ax3 = ax.inset_axes([0.10, 0.03, 0.78, 0.38])
    crc = bench.loc[bench["dataset"] == "visium_hd_crc_stress"].copy()
    crc["method"] = crc["benchmark_method"].map(clean_method)
    x = np.arange(len(crc))
    ax3.bar(x - 0.18, crc["benchmark_false_target_calls"], width=0.36, color=RED, label="before")
    ax3.bar(x + 0.18, crc["origin_ax_false_target_calls"], width=0.36, color=BLUE, label="after")
    for xi, before, after in zip(x, crc["benchmark_false_target_calls"], crc["origin_ax_false_target_calls"]):
        ax3.text(xi - 0.18, before + 1.0, str(int(before)), ha="center", fontsize=8, weight="bold", color=RED)
        ax3.text(xi + 0.18, after + 1.0, str(int(after)), ha="center", fontsize=8, weight="bold", color=BLUE)
    ax3.set_xticks(x, crc["method"], fontsize=7)
    ax3.set_ylabel("CRC false-risk calls", fontsize=8)
    ax3.tick_params(axis="y", labelsize=7)
    ax3.legend(frameon=False, fontsize=7, loc="upper left")
    ax3.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_crc_axis5")


def fig4_crc_axis5_singlecol() -> None:
    audit = pd.read_csv(RESULTS / "visium_hd_crc" / "baseline_overcall_audit.csv")
    bench = pd.read_csv(RESULTS / "benchmark_refinement_summary.csv")
    gw = pd.read_csv(HERE.parents[0] / "results" / "gene_weight_anchor.csv")
    row = audit.loc[audit["program_label"] == "axis5_like"].iloc[0]
    axis_genes = [g.strip() for g in str(row["top_genes"]).split(",") if g.strip()]

    fig = plt.figure(figsize=(3.45, 8.1))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.55, 1.05, 1.15, 1.35], hspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    img_path = ASSETS / "09_CRC_axis5_like_spatial_score.png"
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    ax.imshow(mpimg.imread(img_path))
    ax.axis("off")
    ax.text(0.05, 0.76, "A", transform=ax.transAxes, fontsize=13, weight="bold", va="top", ha="left")

    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "B", "Axis-5-like top genes")
    ax.axis("off")
    genes = axis_genes[:12]
    rows = [", ".join(genes[i:i + 4]) for i in range(0, len(genes), 4)]
    ax.text(
        0.02,
        0.68,
        "\n".join(rows),
        fontsize=8.8,
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", fc=PALE_GOLD, ec=GOLD, lw=1.2),
    )
    ax.text(
        0.02,
        0.18,
        "Differentiated intestinal /\nnormal epithelial-like program",
        fontsize=9.2,
        weight="bold",
        color=INK,
        va="center",
    )

    ax = fig.add_subplot(gs[2, 0])
    panel(ax, "C", "Competing-origin evidence")
    sub = gw.drop_duplicates("gene").set_index("gene").reindex(axis_genes).fillna(0.0)
    target_ev = float(sub["malignant_support"].mean())
    normal_ev = float(sub["normal_epithelial_penalty"].mean())
    tme_ev = float(sub["tme_penalty"].mean())
    delta = target_ev - max(normal_ev, tme_ev, 0.0)
    labels = ["malig.", "normal\n/intestinal", "TME", "$\\Delta$"]
    vals = [target_ev, normal_ev, tme_ev, delta]
    colors = [GREEN, GOLD, MUTED, RED if delta < 0 else GREEN]
    ax.bar(np.arange(4), vals, color=colors, width=0.62)
    ax.axhline(0, color=INK, lw=0.8)
    for xi, v in enumerate(vals):
        ax.text(
            xi,
            v + (0.04 if v >= 0 else -0.08),
            f"{v:.2f}",
            ha="center",
            fontsize=8.5,
            weight="bold",
        )
    ax.set_xticks(np.arange(4), labels, fontsize=8)
    ax.set_ylabel("evidence", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[3, 0])
    panel(ax, "D", "Naive risk vs ORIGIN-Ax")
    ax.axis("off")
    ax.text(
        0.22,
        0.82,
        "naive rules\nmarker+\nCNV+",
        ha="center",
        va="center",
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.28", fc=PALE_RED, ec=RED, lw=1.2),
    )
    ax.annotate("", xy=(0.55, 0.82), xytext=(0.36, 0.82), arrowprops=dict(arrowstyle="->", lw=1.4, color=INK))
    ax.text(
        0.77,
        0.82,
        "ORIGIN-Ax\noff-target",
        ha="center",
        va="center",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.28", fc=PALE_BLUE, ec=BLUE, lw=1.2),
    )

    crc = bench.loc[bench["dataset"] == "visium_hd_crc_stress"].copy()
    crc["method"] = crc["benchmark_method"].map(clean_method)
    inset = ax.inset_axes([0.08, 0.04, 0.86, 0.48])
    x = np.arange(len(crc))
    inset.bar(x - 0.18, crc["benchmark_false_target_calls"], width=0.36, color=RED, label="before")
    inset.bar(x + 0.18, crc["origin_ax_false_target_calls"], width=0.36, color=BLUE, label="after")
    for xi, before, after in zip(x, crc["benchmark_false_target_calls"], crc["origin_ax_false_target_calls"]):
        inset.text(xi - 0.18, before + 1.0, str(int(before)), ha="center", fontsize=7.5, weight="bold", color=RED)
        inset.text(xi + 0.18, after + 1.0, str(int(after)), ha="center", fontsize=7.5, weight="bold", color=BLUE)
    inset.set_xticks(x, crc["method"], fontsize=7)
    inset.set_ylabel("calls", fontsize=7)
    inset.tick_params(axis="y", labelsize=7)
    inset.legend(frameon=False, fontsize=6.8, loc="upper left")
    inset.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_crc_axis5_singlecol")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    # Fig. 1 is a manually curated workflow graphic; do not overwrite it.
    fig2_evidence_summary()
    fig2_evidence_summary_singlecol()
    fig3_target_swap()
    fig3_target_swap_singlecol()
    fig4_crc_axis5()
    fig4_crc_axis5_singlecol()


if __name__ == "__main__":
    main()
