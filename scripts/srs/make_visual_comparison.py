"""Cross-temporal visual comparison figure of the SRS manuscript.

Columns: RGB 2013 | RGB 2020 | reference change | Clay v1.0 pixel prediction |
post-processed polygon output. Two example windows are selected automatically
from the 2013-2020 reference change map as the most class-diverse windows
inside the four LiDAR-covered test patches.

Needs HABITALP_DATA (reference change map, orthophotos, Clay v1.0 outputs before
and after post-processing). Output: outputs/srs/figures/.
"""
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

D = Path(os.environ["HABITALP_DATA"])
OUT = Path(__file__).resolve().parents[2] / "outputs" / "srs" / "figures"
REF = D / "habitalp_change/habitalp_change_v4_2013-2020.tif"
PRED = D / "model_output/clay-v1-rgb-nir-ndsm-phaseE/change_map_2013-2020.tif"
POST = (D / "model_output/clay-v1-rgb-nir-ndsm-phaseE/"
        "constraint_resolution_+_polygonfix_cc_updated/change_map_2013-2020.tif")
RGB13 = D / "processed/orthophoto_gis_stmk/flug_2013_2015_rgb.tif"
RGB20 = D / "processed/orthophoto_gis_stmk/flug_2019_2021_rgb.tif"

# Change-class ids follow src/trainers/utils.py::get_change_array
CLASSES = [
    (0, "No change", "#ffffff"),
    (1, "Mature tree density loss", "#3ef2a0"),
    (2, "Old-growth density loss", "#8e44c8"),
    (3, "Forest setback / young loss", "#d9f542"),
    (4, "Forest stage progression", "#137b93"),
    (5, "Forest density gain", "#ff7f0e"),
    (6, "Early forest establishment", "#a9a95e"),
    (7, "Clearcut loss", "#66b447"),
    (8, "Other transition", "#d17ba4"),
]
LUT = np.ones((256, 4))
for cid, _, col in CLASSES:
    LUT[cid] = mpl.colors.to_rgba(col)
LUT[255] = (0.85, 0.85, 0.85, 1.0)  # nodata grey

WIN_M = 300.0        # window edge length in metres
COARSE = 20          # decimation factor for the window search (4 m pixels)
N_WIN = 2


def pick_windows(ref_path):
    """Return N_WIN non-overlapping (minx, miny, maxx, maxy) windows in the
    reference map that contain the most balanced mix of change classes."""
    with rasterio.open(ref_path) as s:
        arr = s.read(1, out_shape=(s.height // COARSE, s.width // COARSE))
        tr = s.transform * s.transform.scale(COARSE, COARSE)
    n = int(WIN_M / (tr.a))
    scores = []
    step = n // 2
    for r in range(0, arr.shape[0] - n, step):
        for c in range(0, arr.shape[1] - n, step):
            w = arr[r:r + n, c:c + n]
            if (w == 255).mean() > 0.02:
                continue
            frac = np.bincount(w.ravel(), minlength=9)[:9] / w.size
            change = 1 - frac[0]
            if change < 0.25 or change > 0.75:
                continue
            present = (frac[1:] > 0.03).sum()
            ent = -(frac[frac > 0] * np.log(frac[frac > 0])).sum()
            scores.append((present + ent, r, c))
    scores.sort(reverse=True)
    chosen = []
    for _, r, c in scores:
        if all(abs(r - r2) >= n and abs(c - c2) >= n for _, r2, c2 in chosen) or not chosen:
            if all(abs(r - r2) >= n or abs(c - c2) >= n for _, r2, c2 in chosen):
                chosen.append((None, r, c))
        if len(chosen) == N_WIN:
            break
    boxes = []
    for _, r, c in chosen:
        x0, y0 = tr * (c, r + n)
        x1, y1 = tr * (c + n, r)
        boxes.append((x0, y0, x1, y1))
    return boxes


def read_box(path, box, out_px):
    with rasterio.open(path) as s:
        win = from_bounds(*box, transform=s.transform)
        shape = (s.count, out_px, out_px)
        a = s.read(out_shape=shape, window=win, resampling=rasterio.enums.Resampling.nearest)
    return a


def main():
    boxes = pick_windows(REF)
    out_px = 750
    titles = ["RGB 2013", "RGB 2020", "Reference change\n(2013–2020)",
              "Clay v1.0 prediction", "Post-processed\npolygon output"]
    fig, axes = plt.subplots(len(boxes), 5, figsize=(7.2, 1.55 * len(boxes) + 0.9))
    for i, box in enumerate(boxes):
        rgb13 = np.moveaxis(read_box(RGB13, box, out_px), 0, -1)
        rgb20 = np.moveaxis(read_box(RGB20, box, out_px), 0, -1)
        ref = read_box(REF, box, out_px)[0]
        pred = read_box(PRED, box, out_px)[0]
        post = read_box(POST, box, out_px)[0]
        panels = [rgb13, rgb20, LUT[ref], LUT[pred], LUT[post]]
        for j, (ax, img) in enumerate(zip(axes[i], panels)):
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(titles[j], fontsize=7.5, pad=3)
        axes[i][0].set_ylabel(f"({'ab'[i]})", fontsize=8, rotation=0, labelpad=10, va="center")
        print(f"window {i}: {box}")
    handles = [Patch(facecolor=c, edgecolor="#555555", linewidth=0.4, label=n) for _, n, c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handlelength=1.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.90, bottom=0.20, wspace=0.05, hspace=0.06)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Fig_6_Visual_Comparison_cross-temporal.png", dpi=600, facecolor="white")
    print("wrote Fig_6_Visual_Comparison_cross-temporal.png")


if __name__ == "__main__":
    main()
