"""B10: model-efficiency appendix table (parameters, GFLOPs, inference throughput, training cost).

Usage (repo root, .venv314, LD_PRELOAD and OMP_NUM_THREADS set):
    python scripts/b10_efficiency.py cpu     # parameters, GFLOPs, input size, tiling density (CPU only)
    python scripts/b10_efficiency.py gpu     # inference throughput, peak memory, train-step peak memory
    python scripts/b10_efficiency.py wandb   # training wall time and device memory from W&B
    python scripts/b10_efficiency.py table   # merge into outputs/r1/b10_efficiency.csv

Each stage caches its results in models/b10/<stage>.json (outside git), so stages can be rerun alone.
Models: the seed-0 revision run of each model, best val checkpoint (same rule as
scripts/make_r1_infer_configs.py best_checkpoint), loaded as src/inference/inference.py does.

Conventions:
- FLOPs from torch.utils.flop_counter.FlopCounterMode (one multiply-add = 2 FLOPs), batch 1,
  5 x 256 x 256 input (RGB+NIR+nDSM), forward pass of task.model in eval mode.
- "Used" parameters: those that receive a gradient in one forward/backward pass; the rest are in
  the module tree but not in the segmentation forward path (e.g. a pretraining decoder).
- Inference benchmark: synthetic input, batch 4 (the inference batch size), fp32 (default matmul
  precision, no TF32, no autocast), torch.inference_mode, task.predict_step as in
  src/inference/eval.py (logits + argmax); warm-up, then the median of timed repeats, each
  bracketed by torch.cuda.synchronize. Peak memory = torch.cuda.max_memory_allocated with the
  weights on the GPU.
- Train-step peak memory: batch 4, 16-mixed autocast as in the configs, forward + loss + backward +
  AdamW step on synthetic data (a measured lower bound of training memory; data loading not included).
- Tiling: patch 256 px, overlap 64 px per side, so stride 128 px = 64 m at 0.5 m
  (src/inference/eval.py infer_on_whole_image -> GridGeoSampler). Patches per km2 are counted with
  the same sampler on a 1 km x 1 km ROI inside the 2020 imagery.
"""
import argparse
import json
import re
import statistics
import subprocess
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "models" / "b10"
OUT = ROOT / "outputs" / "r1"
WANDB_PROJECT = "university-of-graz-geo/semantic-segmentation-terratorch-extended"

# run name, label, W&B run ID, training machine
RUNS = [
    ("r1-unet-mit-b2-masked-s0", "MiT-B2 U-Net", "3k3z40s0", "rtx4000"),
    ("r1-unet-resnet50-masked-s0", "ResNet-50 U-Net", "65ns3a5l", "rtx4000"),
    ("r1-clay-v1-masked-s0", "Clay v1.0 (FCN)", "gq21z1md", "h200"),
    ("r1-clay-v1.5-masked-s0", "Clay v1.5 (FCN)", "sk6qkuja", "h200"),
    ("r1-prithvi-v2-300-masked-s0", "Prithvi-EO-2.0 300M (FCN)", "ubldwbno", "h200"),
    ("r1-dofa-base-masked-s0", "DOFA base (U-Net dec.)", "j5vt5sme", "rtx4000"),
    ("r1-terramind-masked-s0", "TerraMind base (U-Net dec.)", "sc3l6dpy", "rtx4000"),
    ("r1-clay-v1-unetdec-masked-s0", "Clay v1.0 (U-Net dec.)", "z24ujt6s", "rtx4000"),
    ("r1-prithvi-v2-300-unetdec-masked-s0", "Prithvi-EO-2.0 300M (U-Net dec.)", "odjm7wpy", "h200"),
]

# peak training GPU memory (GB) of the C5 smoke tests (2 epochs, batch 4,
# 16-mixed; Clay v1.5 on rtx4000)
C5_PEAK_GB = {
    "r1-unet-mit-b2-masked-s0": ("1.8", "rtx4000"),
    "r1-unet-resnet50-masked-s0": ("1.8", "rtx4000"),
    "r1-clay-v1-masked-s0": ("13.8", "h200"),
    "r1-clay-v1.5-masked-s0": ("20.6", "h200"),
    "r1-prithvi-v2-300-masked-s0": ("10.3 (h200: 10.9)", "rtx4000"),
    "r1-dofa-base-masked-s0": ("3.4", "rtx4000"),
    "r1-terramind-masked-s0": ("3.2", "rtx4000"),
    "r1-clay-v1-unetdec-masked-s0": ("6.0", "rtx4000"),
    "r1-prithvi-v2-300-unetdec-masked-s0": ("8.1", "rtx4000"),
}

# pretraining data and backbone (paper Table 3 and Section 2 of main.tex; ResNet-50: ImageNet)
NOTES = {
    "r1-unet-mit-b2-masked-s0": "MiT-B2 (SegFormer), ImageNet",
    "r1-unet-resnet50-masked-s0": "ResNet-50, ImageNet",
    "r1-clay-v1-masked-s0": "ViT-B, satellite + sub-metre aerial (NAIP, LINZ)",
    "r1-clay-v1.5-masked-s0": "ViT-L, satellite + sub-metre aerial (NAIP, LINZ)",
    "r1-prithvi-v2-300-masked-s0": "ViT-L, 4.2M HLS samples (30 m)",
    "r1-dofa-base-masked-s0": "ViT-B, five sensors incl. NAIP (~1 m)",
    "r1-terramind-masked-s0": "ViT-B, generative multimodal (incl. S2, 30 m DEM)",
    "r1-clay-v1-unetdec-masked-s0": "ViT-B, satellite + sub-metre aerial (NAIP, LINZ)",
    "r1-prithvi-v2-300-unetdec-masked-s0": "ViT-L, 4.2M HLS samples (30 m)",
}

PATCH, OVERLAP, BATCH, BANDS = 256, 64, 4, 5


def best_checkpoint(run_dir):
    """Same rule as scripts/make_r1_infer_configs.py: highest val/mIoU in the file name."""
    ckpts = [(float(m.group(1)), c) for c in run_dir.rglob("*.ckpt")
             if (m := re.search(r"mIoU=([0-9]+\.[0-9]+)(-v[0-9]+)?\.ckpt$", c.name))]
    return max(ckpts)[1] if ckpts else None


def load(run, device):
    from terratorch.tasks import SemanticSegmentationTask
    ckpt = best_checkpoint(ROOT / "models" / run)
    if ckpt is None:
        raise FileNotFoundError(f"no best checkpoint under models/{run}/ (copy it from the share with cp)")
    task = SemanticSegmentationTask.load_from_checkpoint(ckpt, map_location=device)
    task.eval()
    return task, ckpt


def parts(model):
    """(encoder, decoder incl. neck and head) module lists for the two wrapper types."""
    if hasattr(model, "smp_model"):
        m = model.smp_model
        return [m.encoder], [m.decoder, m.segmentation_head]
    dec = [getattr(model, k) for k in ("neck", "decoder", "head", "aux_heads") if isinstance(getattr(model, k, None), torch.nn.Module)]
    return [model.encoder], dec


def n_params(mods, pred=lambda p: True):
    seen, n = set(), 0
    for m in mods:
        for p in m.parameters():
            if id(p) not in seen and pred(p):
                seen.add(id(p))
                n += p.numel()
    return n


def load_cache(stage):
    f = CACHE / f"{stage}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_cache(stage, data):
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{stage}.json").write_text(json.dumps(data, indent=1))


def patches_per_km2():
    """Count the inference patches of a 1 km x 1 km ROI with the sampler of infer_on_whole_image."""
    from shapely.geometry import box
    from torchgeo.samplers import GridGeoSampler
    from src.data.datamodules.habitalp import HabitAlp2DataModule
    d = Path(os.environ.get("HABITALP_DATA_ROOT", ROOT / "data" / "Originaldaten"))
    inp = {"rgb": d / "processed/orthophoto_gis_stmk/flug_2019_2021_rgb.tif",
           "cir": d / "processed/orthophoto_gis_stmk/falschfarben_2019_2021.tif",
           "ndsm": d / "processed/elevation_fefimapla_2020/ndsm.tif"}
    mask = str(d / "processed/mask/classes_v4_2020.tif")
    dm = HabitAlp2DataModule(image_paths=inp, mask_path=mask, roi_shape_path=str(d / "roi/training_rois_2020.gpkg"),
                             n_classes=23, batch_size=1, patch_size=(PATCH, PATCH), num_workers=0, res=0.5,
                             image_paths_pred=inp, prediction_mask_path=mask)
    stride = PATCH - 2 * OVERLAP
    dm.setup(stage="predict", stride=stride)
    n_test = len(dm.predict_sampler)
    b = dm.roi.bounds
    x0, y0 = b[0] + 1000, b[1] + 1000
    n_km2 = len(GridGeoSampler(dm.predict_dataset, size=PATCH, stride=stride, roi=box(x0, y0, x0 + 1000, y0 + 1000)))
    return {"stride_px": stride, "patches_1km2": n_km2, "patches_per_km2_asymptotic": (1000 / (stride * 0.5)) ** 2,
            "patches_2020_test": n_test, "roi_2020_area_km2": dm.roi.area / 1e6,
            "roi_2020_bbox_km2": (b[2] - b[0]) * (b[3] - b[1]) / 1e6}


def _sdpa_cpu_flops(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs):
    from torch.utils.flop_counter import sdpa_flop_count
    return sdpa_flop_count(query_shape, key_shape, value_shape)


# on CPU, scaled_dot_product_attention dispatches to an op that FlopCounterMode does not count
CPU_SDPA = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_cpu_flops}


def stage_cpu(runs):
    from torch.utils.flop_counter import FlopCounterMode
    res = load_cache("cpu")
    if "tiling" not in res:
        res["tiling"] = patches_per_km2()
        save_cache("cpu", res)
    for run in runs:
        if run in res:
            continue
        task, ckpt = load(run, "cpu")
        model = task.model
        enc, dec = parts(model)
        shapes = {}
        def first_tensor(a):
            while isinstance(a, (list, tuple)):
                a = a[0]
            return a

        def hook(m, a):  # must return None, or the hook replaces the encoder input
            shapes.setdefault("encoder_input", list(first_tensor(a).shape))

        h = enc[0].register_forward_pre_hook(hook)
        x = torch.randn(1, BANDS, PATCH, PATCH)
        with torch.no_grad(), FlopCounterMode(display=False, custom_mapping=CPU_SDPA) as fc:
            out = model(x)
        h.remove()
        y = out.output if hasattr(out, "output") else out
        by_op = {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}
        # parameters on the forward path: one backward pass on a batch of 1
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(torch.randn(1, BANDS, PATCH, PATCH))
        (out.output if hasattr(out, "output") else out).float().mean().backward()
        used = n_params([model], lambda p: p.grad is not None)
        model.zero_grad(set_to_none=True)
        res[run] = {
            "checkpoint": str(ckpt.relative_to(ROOT)),
            "model_class": type(model).__name__,
            "params_total": n_params([model]),
            "params_encoder": n_params(enc),
            "params_decoder_head": n_params(dec),
            "params_trainable": n_params([model], lambda p: p.requires_grad),
            "params_used": used,
            "freeze_backbone": bool(task.hparams.get("freeze_backbone", False)),
            "gflops": fc.get_total_flops() / 1e9,
            "flops_by_op": by_op,
            "encoder_input": shapes.get("encoder_input"),
            "output_shape": list(y.shape),
            "tiled_inference": task.hparams.get("tiled_inference_parameters"),
        }
        print(run, {k: v for k, v in res[run].items() if k != "flops_by_op"}, flush=True)
        save_cache("cpu", res)
        del task, model
    return res


def gpu_free(max_mib=1024):
    """GPU idle apart from this process: memory of other processes below max_mib and 0 % utilization."""
    import os
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=True).stdout.split("\n")[0]
    used, util = (int(v) for v in q.split(","))
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.split("\n")
    own = sum(int(a.split(",")[1]) for a in apps if a.strip() and int(a.split(",")[0]) == os.getpid())
    return used - own < max_mib and util == 0, used - own, util


def wait_gpu(timeout=3600, poll=180):
    """Wait until other processes use less than 1 GB of GPU memory and utilization is 0 %."""
    t0 = time.time()
    while True:
        ok, used, util = gpu_free()
        if ok:
            return True
        if time.time() - t0 > timeout:
            print(f"GPU busy for {timeout}s (others use {used} MiB, util {util}%), giving up", flush=True)
            return False
        print(f"GPU busy (others use {used} MiB, util {util}%), waiting {poll}s", flush=True)
        time.sleep(poll)


def stage_gpu(runs, warmup=10, iters=20, repeats=7):
    res = load_cache("gpu")
    dev = torch.device("cuda")
    for run in runs:
        if run in res:
            continue
        if not wait_gpu():
            break
        task, _ = load(run, "cpu")
        task.to(dev)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        weights_mib = torch.cuda.memory_allocated() / 2**20
        x = torch.randn(BATCH, BANDS, PATCH, PATCH, device=dev)
        # re-check right before timing: only our own process may be on the card
        if not wait_gpu():
            break
        with torch.inference_mode():
            for _ in range(warmup):
                task.predict_step({"image": x}, 0)
            torch.cuda.synchronize()
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                for _ in range(iters):
                    task.predict_step({"image": x}, 0)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        peak_infer = torch.cuda.max_memory_allocated() / 2**20
        sec_per_batch = statistics.median(times) / iters
        # the same forward with autograd on, as src/inference/eval.py runs it (no no_grad there)
        torch.cuda.reset_peak_memory_stats()
        task.predict_step({"image": x}, 0)
        torch.cuda.synchronize()
        peak_infer_grad = torch.cuda.max_memory_allocated() / 2**20
        # one training step, 16-mixed as in the configs
        peak_train = None
        try:
            task.train()
            opt = torch.optim.AdamW([p for p in task.model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.05)
            yb = torch.randint(1, 24, (BATCH, PATCH, PATCH), device=dev)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = task.model(x)
                    logits = out.output if hasattr(out, "output") else out
                    loss = torch.nn.functional.cross_entropy(logits.float(), yb, ignore_index=0)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak_train = torch.cuda.max_memory_allocated() / 2**20
            del opt
        except torch.OutOfMemoryError:
            peak_train = "OOM"
        res[run] = {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "weights_mib": weights_mib,
            "sec_per_batch": sec_per_batch,
            "patches_per_s": BATCH / sec_per_batch,
            "times_s": times,
            "iters_per_repeat": iters,
            "peak_infer_mib": peak_infer,
            "peak_infer_autograd_mib": peak_infer_grad,
            "peak_train_step_mib": peak_train,
        }
        print(run, {k: v for k, v in res[run].items() if k != "times_s"}, flush=True)
        save_cache("gpu", res)
        del task, x
        torch.cuda.empty_cache()
    return res


def stage_wandb(runs):
    import wandb
    api = wandb.Api(timeout=60)
    res = load_cache("wandb")
    for run, _, rid, machine in RUNS:
        if run not in runs or run in res:
            continue
        r = api.run(f"{WANDB_PROJECT}/{rid}")
        sm = r.history(stream="systemMetrics", samples=5000, pandas=True)
        col = "system.gpu.0.memoryAllocatedBytes"
        res[run] = {
            "wandb_id": rid, "wandb_name": r.name, "state": r.state,
            "host": r.metadata.get("host"), "gpu": r.metadata.get("gpu"),
            "runtime_s": r.summary.get("_runtime"), "epochs": r.summary.get("epoch"),
            "device_mem_max_gib": float(sm[col].max()) / 2**30 if col in sm else None,
        }
        print(run, res[run], flush=True)
        save_cache("wandb", res)
    return res


def stage_table():
    import csv
    cpu, gpu, wb = load_cache("cpu"), load_cache("gpu"), load_cache("wandb")
    tiling = cpu["tiling"]
    ppk = tiling["patches_1km2"]
    rows = []
    for run, label, rid, machine in RUNS:
        c, g, w = cpu.get(run, {}), gpu.get(run, {}), wb.get(run, {})
        row = {
            "model": label, "run": run, "checkpoint": c.get("checkpoint"),
            "params_total_M": round(c["params_total"] / 1e6, 1) if c else None,
            "params_encoder_M": round(c["params_encoder"] / 1e6, 1) if c else None,
            "params_decoder_head_M": round(c["params_decoder_head"] / 1e6, 1) if c else None,
            "params_trainable_M": round(c["params_trainable"] / 1e6, 1) if c else None,
            "params_used_M": round(c["params_used"] / 1e6, 1) if c else None,
            "input_px": c["encoder_input"][-1] if c else None,
            "gflops_per_patch": round(c["gflops"], 1) if c else None,
            "infer_patches_per_s": round(g["patches_per_s"], 1) if g else None,
            "infer_peak_mem_GiB": round(g["peak_infer_mib"] / 1024, 2) if g else None,
            "infer_peak_mem_autograd_GiB": round(g["peak_infer_autograd_mib"] / 1024, 2) if g else None,
            "infer_s_per_km2": round(ppk / g["patches_per_s"], 1) if g else None,
            "train_step_peak_mem_GiB": (round(g["peak_train_step_mib"] / 1024, 2)
                                        if g and isinstance(g["peak_train_step_mib"], float) else g.get("peak_train_step_mib")),
            "train_machine": machine,
            "train_gpu": w.get("gpu"),
            "train_wall_h": round(w["runtime_s"] / 3600, 1) if w.get("runtime_s") else None,
            "train_epochs": w.get("epochs"),
            "train_peak_mem_GB_C5": C5_PEAK_GB[run][0],
            "train_peak_mem_C5_machine": C5_PEAK_GB[run][1],
            "wandb_device_mem_max_GiB": round(w["device_mem_max_gib"], 1) if w.get("device_mem_max_gib") else None,
            "wandb_id": rid,
            "pretraining": NOTES[run],
        }
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "b10_efficiency.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps(tiling, indent=1))
    for r in rows:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cpu", "gpu", "wandb", "table"])
    ap.add_argument("--runs", nargs="*", default=[r[0] for r in RUNS])
    a = ap.parse_args()
    if a.stage == "cpu":
        torch.set_num_threads(3)
        stage_cpu(a.runs)
    elif a.stage == "gpu":
        torch.set_num_threads(3)
        stage_gpu(a.runs)
    elif a.stage == "wandb":
        stage_wandb(a.runs)
    else:
        stage_table()


if __name__ == "__main__":
    main()
