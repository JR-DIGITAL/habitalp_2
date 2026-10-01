"""Fig 1 of the MDPI manuscript: study area map with UTM grid, scale bar,
north arrow and an Austria inset. Output: outputs/mdpi/figures/Fig_1.png (600 dpi).

Environment variables:
  HABITALP_DATA   HabitAlp 2.0 dataset root (splits/outlines.gpkg, 2019 orthophoto)
  HABITALP_ORTHO  optional: other basemap than data_2020/aerial_rgb_2019_2021.tif
  NE_DIR          folder with the Natural Earth admin-1 10m shapefile
                  (ne_10m_admin_1_states_provinces.shp, https://www.naturalearthdata.com/)
"""
import os
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrow
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

D = Path(os.environ["HABITALP_DATA"])
NE_DIR = Path(os.environ["NE_DIR"])
OUT = Path(__file__).resolve().parents[2] / "outputs" / "mdpi" / "figures" / "Fig_1.png"

ORTHO = Path(os.environ.get("HABITALP_ORTHO", D / "data_2020/aerial_rgb_2019_2021.tif"))
OUTLINES = D / "splits/outlines.gpkg"

mpl.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

outlines = gpd.read_file(OUTLINES).to_crs(32633)
hab = outlines[outlines["year"] == 2013]
test = outlines[outlines["year"] == 2020]

pad = 1000
minx, miny, maxx, maxy = hab.total_bounds
minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad

# --- basemap: orthophoto at reduced resolution, clipped to study area extent ---
with rasterio.open(ORTHO) as src:
    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    scale = 1800 / win.width
    out_shape = (3, int(win.height * scale), int(win.width * scale))
    rgb = src.read(out_shape=out_shape, window=win, boundless=True, fill_value=0,
                   resampling=rasterio.enums.Resampling.average)
rgb = np.moveaxis(rgb, 0, -1).astype(float)
nodata = rgb.sum(axis=-1) < 40
# mild brightening for print
rgb = np.clip(rgb * 1.15 + 10, 0, 255) / 255.0
rgb[nodata] = 0.93

fig = plt.figure(figsize=(7.0, 6.3))
ax = fig.add_axes([0.09, 0.15, 0.89, 0.84])
ax.imshow(rgb, extent=(minx, maxx, miny, maxy), interpolation="bilinear", zorder=0)

hab.boundary.plot(ax=ax, color="#d7191c", linewidth=1.6, zorder=4)
test.boundary.plot(ax=ax, color="#ff8c00", linewidth=1.6, zorder=5)

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_aspect("equal")

# --- UTM grid ---
xt = np.arange(np.ceil(minx / 5000) * 5000, maxx, 5000)
yt = np.arange(np.ceil(miny / 5000) * 5000, maxy, 5000)
ax.set_xticks(xt)
ax.set_yticks(yt)
ax.set_xticklabels([f"{int(x/1000)}" for x in xt])
ax.set_yticklabels([f"{int(y/1000)}" for y in yt], rotation=90, va="center")
ax.set_xlabel("Easting (km), UTM zone 33N (EPSG:32633)")
ax.set_ylabel("Northing (km)")
ax.grid(True, color="white", alpha=0.45, linewidth=0.5, zorder=2)
ax.tick_params(length=3)

# --- scale bar (5 km) ---
sb_x0 = minx + 1200
sb_y0 = miny + 1200
for i, col in enumerate(["black", "white", "black", "white", "black"]):
    ax.add_patch(mpl.patches.Rectangle((sb_x0 + i * 1000, sb_y0), 1000, 250,
                                       facecolor=col, edgecolor="black", linewidth=0.6, zorder=6))
for i in (0, 2.5, 5):
    ax.text(sb_x0 + i * 1000, sb_y0 + 380, f"{i:g}", ha="center", va="bottom", fontsize=8,
            zorder=6, bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"))
ax.text(sb_x0 + 5000 + 250, sb_y0 + 380, "km", ha="left", va="bottom", fontsize=8, zorder=6,
        bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"))

# --- north arrow ---
nx, ny = maxx - 1400, maxy - 2600
ax.add_patch(FancyArrow(nx, ny, 0, 1400, width=350, head_width=900, head_length=700,
                        length_includes_head=True, color="black", zorder=6))
ax.text(nx, ny - 250, "N", ha="center", va="top", fontsize=9, fontweight="bold", zorder=6,
        bbox=dict(facecolor="white", alpha=0.7, pad=0.8, edgecolor="none"))

# --- legend ---
handles = [
    Line2D([0], [0], color="#d7191c", lw=1.6, label="Study area (HabitAlp reference mapping)"),
    Line2D([0], [0], color="#ff8c00", lw=1.6, label="Cross-temporal test patches (2020 LiDAR)"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.535, 0.0), ncol=2,
           fontsize=8.5, frameon=False, handlelength=2.2)

# --- inset: Austria ---
adm1 = gpd.read_file(NE_DIR / "ne_10m_admin_1_states_provinces.shp")
adm1 = adm1[adm1["adm0_a3"] == "AUT"].to_crs(32633)
iax = inset_axes(ax, width="27%", height="18%", loc="upper left", borderpad=0.6)
adm1.plot(ax=iax, facecolor="#e6e6e6", edgecolor="white", linewidth=0.6)
adm1.boundary.plot(ax=iax, color="#888888", linewidth=0.4)
hab.plot(ax=iax, facecolor="#d7191c", edgecolor="#d7191c", linewidth=1.5)
iax.set_xticks([]); iax.set_yticks([])
iax.set_xlabel(""); iax.set_ylabel("")
iax.set_aspect("equal")
for sp in iax.spines.values():
    sp.set_linewidth(0.6)
iax.set_facecolor("white")
iax.set_title("Austria", fontsize=7, pad=2)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=600, facecolor="white")
print("wrote", OUT, "extent", (minx, miny, maxx, maxy))
