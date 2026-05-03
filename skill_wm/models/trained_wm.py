"""Small trained world model: (crop, inventory, action) -> P(success).

Architecture (deliberately small, ~100-300K params; we want this to look
like a *plausible* trained baseline, not a frontier model — and we want
training to fit on a laptop CPU in minutes):

  semantic_crop (15x15 ints in 0..18)
    -> tile embedding (19, D=16)               # learned per-tile vector
    -> spatial conv (a 2-layer CNN, D=16->32)  # local geometry features
    -> global pool                             # 32-d state vector

  inventory (16 ints, mostly 0..9)
    -> small MLP                               # 16 -> 32

  action (one of 17)
    -> action embedding (17, 32)               # learned per-action vector

  concat (96-d) -> 2-layer MLP -> success_logit (1)

We train with binary cross-entropy on the `success` label of each
ScoringRow. The output of `predict()` is the sigmoid of the logit, so it
plugs straight into the `Predictor` protocol used by the eval harness.

Why this size:
- The LLM sees ~250 tokens of state. A trained model with a few hundred
  thousand parameters and the same view is the comparison the thesis
  cares about: "you don't need a frontier model to score actions."
- 17 actions * 19 tiles * 16 invslots is a small input space; bigger
  models will overfit a 50-episode dataset.
- We verify the harness comparison with this; the actual T1 number
  comes from a slightly bigger version trained on the cluster collect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from skill_wm.data.schema import ACTION_NAMES
from skill_wm.eval.dataset import INVENTORY_KEYS, ScoringRow

log = logging.getLogger(__name__)

NUM_TILES = 19  # 0..18 inclusive (see TILE_LEGEND in baselines.py)
NUM_ACTIONS = len(ACTION_NAMES)
INV_SIZE = len(INVENTORY_KEYS)
CROP_SIZE = 15


def _row_to_tensors(row: ScoringRow) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a ScoringRow's features into model-ready tensors (no batch dim)."""
    crop = torch.from_numpy(row.semantic_crop_before.astype(np.int64))
    inv = torch.from_numpy(row.inventory_before.astype(np.float32))
    action = torch.tensor(row.action, dtype=torch.long)
    return crop, inv, action


class _RowDataset(Dataset):
    def __init__(self, rows: list[ScoringRow]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        crop, inv, action = _row_to_tensors(r)
        label = torch.tensor(float(r.success), dtype=torch.float32)
        return crop, inv, action, label


class TrainedWMNet(nn.Module):
    """The actual nn.Module. Returns success logits (raw, pre-sigmoid)."""

    def __init__(
        self,
        tile_dim: int = 16,
        conv_dim: int = 32,
        inv_dim: int = 32,
        action_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.tile_emb = nn.Embedding(NUM_TILES, tile_dim)
        self.action_emb = nn.Embedding(NUM_ACTIONS, action_dim)

        # 2-layer CNN. Input is (B, tile_dim, 15, 15) after the embedding
        # is permuted into channel-first form.
        self.conv = nn.Sequential(
            nn.Conv2d(tile_dim, conv_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # global pool -> (B, conv_dim, 1, 1)
            nn.Flatten(),
        )

        self.inv_mlp = nn.Sequential(
            nn.Linear(INV_SIZE, inv_dim),
            nn.ReLU(),
        )

        feat_dim = conv_dim + inv_dim + action_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, crop: torch.Tensor, inv: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # crop: (B, 15, 15) int64 -> (B, 15, 15, tile_dim) -> (B, tile_dim, 15, 15)
        embedded = self.tile_emb(crop).permute(0, 3, 1, 2)
        spatial = self.conv(embedded)  # (B, conv_dim)
        inv_feat = self.inv_mlp(inv)  # (B, inv_dim)
        act_feat = self.action_emb(action)  # (B, action_dim)
        feat = torch.cat([spatial, inv_feat, act_feat], dim=1)
        return self.head(feat).squeeze(-1)  # (B,)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.1
    device: str = "cpu"  # MPS is slower than CPU for this size on a laptop
    seed: int = 0
    early_stop_patience: int = 5  # epochs with no val Brier improvement


class TrainedWM:
    """Predictor wrapper: fit() trains the net, predict() returns probabilities.

    Implements the same `Predictor` protocol as RandomPredictor /
    MarginalPredictor / PreconditionPredictor / LLMWorldModel, so it
    drops into `skill_wm.eval.run` with a single dispatch line.
    """

    name = "trained_wm"

    def __init__(self, config: TrainConfig | None = None):
        self.config = config or TrainConfig()
        self.net: TrainedWMNet | None = None
        self.history: list[dict] = []

    def fit(self, train_rows: list[ScoringRow]) -> None:
        """Train on rows. We hold out a small val slice for early stopping."""
        if not train_rows:
            raise ValueError("train_rows is empty")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Internal val split for early stopping. This is *separate* from
        # the eval-harness train/eval split (which is seed-disjoint) — here
        # we just want a stop signal during training. Sample uniformly so
        # the val distribution matches the train distribution.
        n = len(train_rows)
        n_val = max(1, int(round(n * self.config.val_frac)))
        rng = np.random.default_rng(self.config.seed)
        perm = rng.permutation(n)
        val_idx = set(perm[:n_val].tolist())
        tr_rows = [r for i, r in enumerate(train_rows) if i not in val_idx]
        va_rows = [r for i, r in enumerate(train_rows) if i in val_idx]

        device = torch.device(self.config.device)
        self.net = TrainedWMNet().to(device)
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            _RowDataset(tr_rows),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_val_brier = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        for epoch in range(self.config.epochs):
            self.net.train()
            running = 0.0
            seen = 0
            for crop, inv, action, label in train_loader:
                crop = crop.to(device)
                inv = inv.to(device)
                action = action.to(device)
                label = label.to(device)
                opt.zero_grad()
                logit = self.net(crop, inv, action)
                loss = loss_fn(logit, label)
                loss.backward()
                opt.step()
                running += float(loss.detach()) * label.size(0)
                seen += label.size(0)
            train_loss = running / max(seen, 1)
            val_brier = self._brier_on(va_rows)
            self.history.append({"epoch": epoch, "train_loss": train_loss, "val_brier": val_brier})
            log.info("epoch %d  train_loss=%.4f  val_brier=%.4f", epoch, train_loss, val_brier)
            if val_brier + 1e-6 < best_val_brier:
                best_val_brier = val_brier
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.early_stop_patience:
                    log.info("early stop at epoch %d (best val_brier=%.4f)", epoch, best_val_brier)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    @torch.no_grad()
    def _brier_on(self, rows: list[ScoringRow]) -> float:
        if not rows:
            return float("nan")
        probs = self.predict(rows)
        labels = np.array([r.success for r in rows], dtype=np.float64)
        return float(np.mean((probs - labels) ** 2))

    @torch.no_grad()
    def predict(self, eval_rows: list[ScoringRow]) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("call fit() before predict()")
        device = next(self.net.parameters()).device
        self.net.eval()
        loader = DataLoader(
            _RowDataset(eval_rows),
            batch_size=max(self.config.batch_size, 256),
            shuffle=False,
            drop_last=False,
        )
        out: list[np.ndarray] = []
        for crop, inv, action, _label in loader:
            crop = crop.to(device)
            inv = inv.to(device)
            action = action.to(device)
            logit = self.net(crop, inv, action)
            out.append(torch.sigmoid(logit).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    def save(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("nothing to save; fit() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load(self, path: Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
        self.net = TrainedWMNet().to(self.config.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.history = ckpt.get("history", [])
