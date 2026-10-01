"""Bootstrap confidence intervals + confusion-matrix export for change metrics.

Builds a raw confusion matrix once from a reference/prediction raster pair (reusing
the same intersection-window alignment as ``compute_final_metrics``), then bootstraps
the CM cell counts via the multinomial trick to get percentile CIs on macro IoU/F1.
O(num_classes^2) per iteration -> 1000 iterations is trivial on CPU.

The macro IoU/F1 derived here are verified against the torchmetrics values in the
existing CSVs before any CI is trusted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def load_aligned_flat(
    reference_file_path: Path,
    prediction_file_path: Path,
    ignore_index: int = 255,
    binary: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the spatial intersection of two single-band rasters as aligned, valid,
    flattened int arrays. Mirrors compute_final_metrics' alignment logic."""
    with rasterio.open(reference_file_path) as m, rasterio.open(prediction_file_path) as p:
        bm, bp = m.bounds, p.bounds
        inter = rasterio.coords.BoundingBox(
            left=max(bm.left, bp.left), bottom=max(bm.bottom, bp.bottom),
            right=min(bm.right, bp.right), top=min(bm.top, bp.top),
        )
        if inter.left >= inter.right or inter.bottom >= inter.top:
            raise ValueError("No overlapping area between rasters.")
        ref = m.read(1, window=m.window(*inter))
        if m.res == p.res:
            pred = p.read(1, window=p.window(*inter))
        else:
            raise ValueError("Resolutions differ; resample before CI computation.")

    # crop to common shape (rounding can differ by 1 px)
    h = min(ref.shape[0], pred.shape[0])
    w = min(ref.shape[1], pred.shape[1])
    ref = ref[:h, :w].ravel()
    pred = pred[:h, :w].ravel()

    if binary:
        ref = np.where((ref >= 1) & (ref <= 8), 1, ref)
        pred = np.where((pred >= 1) & (pred <= 8), 1, pred)

    valid = (ref != ignore_index) & (pred != ignore_index)
    return ref[valid].astype(np.int64), pred[valid].astype(np.int64)


def load_aligned_2d(
    reference_file_path: Path, prediction_file_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Read the spatial intersection of two rasters as aligned 2-D int arrays
    (no flatten, no ignore-filter) for block/tile-based resampling."""
    with rasterio.open(reference_file_path) as m, rasterio.open(prediction_file_path) as p:
        bm, bp = m.bounds, p.bounds
        inter = rasterio.coords.BoundingBox(
            left=max(bm.left, bp.left), bottom=max(bm.bottom, bp.bottom),
            right=min(bm.right, bp.right), top=min(bm.top, bp.top),
        )
        if inter.left >= inter.right or inter.bottom >= inter.top:
            raise ValueError("No overlapping area between rasters.")
        ref = m.read(1, window=m.window(*inter))
        if m.res != p.res:
            raise ValueError("Resolutions differ; resample before CI computation.")
        pred = p.read(1, window=p.window(*inter))
    h = min(ref.shape[0], pred.shape[0])
    w = min(ref.shape[1], pred.shape[1])
    return ref[:h, :w].astype(np.int16), pred[:h, :w].astype(np.int16)


def tile_confusion_matrices(
    ref2d: np.ndarray, pred2d: np.ndarray, num_classes: int,
    block: int, ignore_index: int = 255,
) -> np.ndarray:
    """Per-tile raw confusion matrices over a regular block grid.

    Returns array (n_nonempty_tiles, num_classes, num_classes). Tiles are the
    spatial resampling unit for a block bootstrap (accounts for autocorrelation).
    """
    h, w = ref2d.shape
    cms = []
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            r = ref2d[r0:r0 + block, c0:c0 + block].ravel()
            p = pred2d[r0:r0 + block, c0:c0 + block].ravel()
            valid = (r != ignore_index) & (p != ignore_index)
            if not valid.any():
                continue
            r, p = r[valid].astype(np.int32), p[valid].astype(np.int32)
            cm = np.bincount(r * num_classes + p, minlength=num_classes * num_classes)
            cms.append(cm.reshape(num_classes, num_classes).astype(np.int64))
    return np.stack(cms)


def collapse_binary(cm: np.ndarray) -> np.ndarray:
    """Collapse an N-class change CM to 2x2 (no-change=0 vs any change 1..N-1)."""
    b = np.zeros((2, 2), dtype=cm.dtype)
    b[0, 0] = cm[0, 0]
    b[0, 1] = cm[0, 1:].sum()
    b[1, 0] = cm[1:, 0].sum()
    b[1, 1] = cm[1:, 1:].sum()
    return b


def block_bootstrap_ci(
    tile_cms: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
    binary: bool = False,
) -> dict[str, float]:
    """Percentile CIs for macro IoU/F1 by resampling spatial tiles with replacement."""
    rng = np.random.default_rng(seed)
    n = tile_cms.shape[0]
    ious = np.empty(n_boot)
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cm = tile_cms[idx].sum(axis=0)
        if binary:
            cm = collapse_binary(cm)
        ious[b], f1s[b] = macro_iou_f1(cm)
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        "IoU_ci_low": float(np.percentile(ious, lo)),
        "IoU_ci_high": float(np.percentile(ious, hi)),
        "F1_ci_low": float(np.percentile(f1s, lo)),
        "F1_ci_high": float(np.percentile(f1s, hi)),
        "n_tiles": n,
    }


def confusion_matrix(ref: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Raw-count confusion matrix, rows = true (ref), cols = predicted."""
    idx = ref * num_classes + pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def macro_iou_f1(cm: np.ndarray) -> tuple[float, float]:
    """Macro IoU and macro F1 from a confusion matrix (nanmean over supported classes)."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    return float(np.nanmean(iou)), float(np.nanmean(f1))


def bootstrap_ci(
    cm: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Percentile CIs for macro IoU/F1 via multinomial resampling of CM cell counts."""
    rng = np.random.default_rng(seed)
    flat = cm.ravel().astype(np.float64)
    total = int(flat.sum())
    probs = flat / total
    k = cm.shape[0]
    ious = np.empty(n_boot)
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        resampled = rng.multinomial(total, probs).reshape(k, k)
        ious[b], f1s[b] = macro_iou_f1(resampled)
    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    return {
        "IoU_ci_low": float(np.percentile(ious, lo)),
        "IoU_ci_high": float(np.percentile(ious, hi)),
        "F1_ci_low": float(np.percentile(f1s, lo)),
        "F1_ci_high": float(np.percentile(f1s, hi)),
    }


def cm_and_ci(
    reference_file_path: Path,
    prediction_file_path: Path,
    num_classes: int,
    ignore_index: int = 255,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, float, float, dict[str, float]]:
    """Convenience: returns (raw_cm, macro_iou, macro_f1, ci_dict)."""
    binary = num_classes == 2
    ref, pred = load_aligned_flat(reference_file_path, prediction_file_path, ignore_index, binary)
    cm = confusion_matrix(ref, pred, num_classes)
    iou, f1 = macro_iou_f1(cm)
    ci = bootstrap_ci(cm, n_boot=n_boot, seed=seed)
    return cm, iou, f1, ci
