"""Train the trained-WM baseline on collected rollouts (T1 model #2).

NOT YET IMPLEMENTED — this is a stub showing the rune wiring shape. The
model itself (skill_wm/models/trained_wm.py) doesn't exist yet; that's the
``skill-wm-trained-wm`` todo. Once the model exists, replace the body of
``train_wm`` with the real torch training loop. The rune envelope is
already correct.

Local sanity:

    python experiments/train_wm/config.py --local

Cluster submit:

    RUNE_NAME=skill-wm-train-001 \\
    SKILL_WM_DATA_RUN=skill-wm-collect-001 \\
    python experiments/train_wm/config.py
"""

from __future__ import annotations

import argparse
import os

import rune

NAME = os.environ.get("RUNE_NAME", "skill-wm-train-smoke")
DATA_RUN = os.environ.get("SKILL_WM_DATA_RUN", "skill-wm-collect-smoke")
EPOCHS = int(os.environ.get("SKILL_WM_EPOCHS", "20"))
BATCH = int(os.environ.get("SKILL_WM_BATCH", "256"))
LR = float(os.environ.get("SKILL_WM_LR", "3e-4"))
TEAM = os.environ.get("RUNE_TEAM", "experimental")
PRESET = os.environ.get("RUNE_PRESET") or None

RUNTIME_PIP = [
    "torch>=2.4,<3",
    "numpy>=2.0,<3",
    "tqdm>=4.66",
    "pyyaml>=6",
]


@rune.train(
    name=NAME,
    gpus=1,
    workers=1,
    team=TEAM,
    preset=PRESET,
    extra_manifest={
        "runtime": {"pip": RUNTIME_PIP},
        "skill_wm": {
            "data_run": DATA_RUN,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "lr": LR,
        },
    },
)
def train_wm(ctx):
    """Train the small (semantic-crop, inventory, action) -> outcome model.

    TODO(skill-wm-trained-wm): implement. Expected shape:
      1. Load all .npz files under {data_root}/{data_run}/rank-*/ep_*.npz.
      2. Build (state, action) -> (success, inv_delta, vit_delta) dataset.
      3. Train a small transformer / MLP for ``epochs`` epochs.
      4. Save the checkpoint to {ckpt_root}/{ctx.name}/wm.pt.
    """
    cfg = ctx.manifest["skill_wm"]
    from pathlib import Path

    rollouts_dir = Path(ctx.durable_datasets_dir) / "skill-wm" / "rollouts" / cfg["data_run"]
    ckpt_dir = Path(ctx.durable_checkpoints_dir) / "skill-wm" / ctx.name
    print(f"[train_wm STUB] would train on {rollouts_dir}")
    print(f"[train_wm STUB] would write to {ckpt_dir}/wm.pt")
    print(f"[train_wm STUB] hyperparams: epochs={cfg['epochs']} batch={cfg['batch_size']} lr={cfg['lr']}")
    raise NotImplementedError(
        "train_wm body not implemented yet — see skill-wm-trained-wm todo"
    )


def main():
    parser = argparse.ArgumentParser(description="Skill-WM trained-WM training")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.local:
        train_wm()
    else:
        train_wm.submit(dry_run="client" if args.dry_run else None)


if __name__ == "__main__":
    main()
