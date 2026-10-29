#!/usr/bin/env python
"""Learning-rate range test (Smith 2017; Lightning Tuner.lr_find) for a revision config.

Builds datamodule and task exactly as src/trainers/train_terratorch.py does for the given config
(masked split, batch 4, AdamW wd 0.05, class weights, seed), but replaces the training run with
Tuner.lr_find: 250 steps from 1e-7 to 1, early_stop_threshold 4.
No W&B, no checkpoints. Writes the loss curve and the suggestion to models/r1_lr_find/<run>.{csv,png,txt}.
Optional: --steps N (default 250) and --tag T (output name suffix); the raw per-step training loss is
saved next to Lightning's smoothed curve (<run>_raw.csv), because with batch 4 the smoothed curve of
a short test is dominated by noise.

    .venv314/bin/python scripts/r1_lr_find.py src/trainers/terratorch/revision/config-r1-unet-resnet50-masked-s0.yaml
    .venv314/bin/python scripts/r1_lr_find.py <config> --steps 2000 --tag 2000
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from lightning.pytorch import Trainer  # noqa: E402
from lightning.pytorch.tuner import Tuner  # noqa: E402

import src.trainers.train_terratorch as tt  # noqa: E402

OUT = BASE / "models" / "r1_lr_find"
NUM_TRAINING, MIN_LR, MAX_LR, EARLY_STOP = 250, 1e-7, 1.0, 4.0


class LRFindTrainer:
    """Stands in for the Trainer of setup_training: fit() runs the range test, test() is skipped."""

    def __init__(self, name: str, gpu_id: int, steps: int):
        self.name, self.steps = name, steps
        self.trainer = Trainer(accelerator="gpu", devices=[gpu_id], precision="16-mixed", logger=False,
                               enable_checkpointing=False, default_root_dir=OUT / name, max_epochs=-1)

    def fit(self, model, datamodule, ckpt_path=None):
        raw = []
        step = model.training_step

        def logged_step(*a, **k):
            out = step(*a, **k)
            loss = out["loss"] if isinstance(out, dict) else out
            raw.append(float(loss.detach()))
            return out

        model.training_step = logged_step
        res = Tuner(self.trainer).lr_find(model, datamodule=datamodule, num_training=self.steps,
                                          min_lr=MIN_LR, max_lr=MAX_LR, early_stop_threshold=EARLY_STOP)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(res.results).to_csv(OUT / f"{self.name}.csv", index=False)
        n = min(len(raw), len(res.results["lr"]))
        pd.DataFrame({"lr": res.results["lr"][:n], "loss_raw": raw[:n]}).to_csv(OUT / f"{self.name}_raw.csv", index=False)
        res.plot(suggest=True).savefig(OUT / f"{self.name}.png", dpi=120)
        sugg = res.suggestion()
        (OUT / f"{self.name}.txt").write_text(
            f"run {self.name}\nsuggestion {sugg:.3g}\nconfigured lr {model.hparams.get('lr')}\n"
            f"steps {self.steps}, range {MIN_LR:g} to {MAX_LR:g}, early_stop_threshold {EARLY_STOP}\n")
        print(f"LR_FIND {self.name}: suggestion {sugg:.3g} (configured {model.hparams.get('lr')})")

    def test(self, *a, **k):
        pass


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=NUM_TRAINING)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = Path(a.config)
    name = cfg.stem.removeprefix("config-") + (f"_{a.tag}" if a.tag else "")
    tt.setup_training = lambda **kw: LRFindTrainer(name, kw.get("gpu_id", 0), a.steps)
    tt.wandb.finish = lambda *a, **k: None
    sys.argv = [sys.argv[0], "--config", str(cfg)]
    torch.backends.cudnn.benchmark = False
    tt.main()


if __name__ == "__main__":
    main()
