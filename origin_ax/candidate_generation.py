"""Candidate axis generation for ORIGIN-Ax v1."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, NMF, PCA

from core import safe_corr, target_support_score, top_genes_from_loading, zscore


@dataclass
class CandidateAxis:
    axis_id: str
    source_method: str
    seed: int
    k: int
    loading: np.ndarray
    score: np.ndarray
    top_genes: list[str]
    v0_gate_result: str = "not_evaluated"
    orientation_flipped: bool = False


def _orient_to_target(loading: np.ndarray, score: np.ndarray, target_support: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    if np.sum(np.abs(target_support)) < 1e-9:
        return loading, score, False
    signed = float(np.dot(loading, target_support))
    if signed < 0:
        return -loading, -score, True
    return loading, score, False


def _ica(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_pc = min(40, x.shape[0] - 1, x.shape[1] - 1)
    z = PCA(n_pc, random_state=seed).fit_transform(zscore(x))
    scores = FastICA(k, random_state=seed, max_iter=1000, tol=1e-3).fit_transform(z)
    loadings = (zscore(x).T @ zscore(scores)) / x.shape[0]
    return zscore(scores), loadings.T


def _nmf(x: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.log1p(np.clip(x, 0, None))
    x0 = x0 - x0.min()
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=700)
    scores = model.fit_transform(x0)
    loadings = model.components_
    return zscore(scores), loadings


def _make_candidates(
    x: np.ndarray,
    genes: np.ndarray,
    gene_weights: dict[str, np.ndarray],
    k: int,
    seeds: tuple[int, ...],
    prefix: str,
    subset_idx: np.ndarray | None = None,
) -> list[CandidateAxis]:
    candidates: list[CandidateAxis] = []
    x_fit = x if subset_idx is None else x[subset_idx]
    for seed in seeds:
        for method, fn in [("ICA", _ica), ("NMF", _nmf)]:
            try:
                scores_fit, loadings = fn(x_fit, k, seed)
                # Project all bins onto the learned loading to keep scores comparable.
                scores_all = zscore(x) @ zscore(loadings).T / x.shape[1]
                for j in range(k):
                    loading, score, flipped = _orient_to_target(
                        loadings[j].astype(float),
                        scores_all[:, j].astype(float),
                        gene_weights["target_support"],
                    )
                    src = f"{prefix}_{method}" if prefix else method
                    candidates.append(CandidateAxis(
                        axis_id=f"{src}_seed{seed}_axis{j}",
                        source_method=src,
                        seed=seed,
                        k=k,
                        loading=loading,
                        score=score,
                        top_genes=top_genes_from_loading(genes, loading, n=30),
                        orientation_flipped=flipped,
                    ))
            except Exception as exc:
                print(f"[warn] candidate generation failed method={method} prefix={prefix} seed={seed}: {exc}", flush=True)
    return candidates


def generate_candidates(
    x: np.ndarray,
    genes: np.ndarray,
    gene_weights: dict[str, np.ndarray],
    k: int = 6,
    seeds: tuple[int, ...] = (0, 1),
) -> tuple[list[CandidateAxis], pd.DataFrame]:
    candidates = _make_candidates(x, genes, gene_weights, k, seeds, prefix="")
    support_score, diag = target_support_score(x, gene_weights)
    rows = [diag.copy()]
    if support_score is not None:
        thr = np.quantile(support_score, 0.70)
        subset = np.where(support_score >= thr)[0]
        rows[0]["subset_size"] = int(len(subset))
        rows[0]["score_quantile_threshold"] = float(thr)
        candidates += _make_candidates(x, genes, gene_weights, k, seeds, prefix="high_target_subset", subset_idx=subset)
    return candidates, pd.DataFrame(rows)


def candidate_table(candidates: list[CandidateAxis]) -> pd.DataFrame:
    return pd.DataFrame([{
        "axis_id": c.axis_id,
        "source_method": c.source_method,
        "seed": c.seed,
        "k": c.k,
        "top_genes": ",".join(c.top_genes[:12]),
        "orientation_flipped": c.orientation_flipped,
        "v0_gate_result": c.v0_gate_result,
    } for c in candidates])
