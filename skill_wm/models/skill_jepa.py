"""MiniHack Skill-JEPA prototype.

This is the first small gate for the LeCun-latest-20 adaptation: learn a
latent outcome model over MiniHack transitions, then use prediction error as a
surprise signal. It is intentionally compact and CPU-friendly; the goal is to
validate the shape before building a larger benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

TEXT_VOCAB_SIZE = 512
TEXT_TOKENS = 16
BLSTATS_DIM = 27


@dataclass(frozen=True)
class MiniHackJEPARow:
    seed: int
    episode: int
    step: int
    action: int
    action_name: str
    glyph_before: np.ndarray
    glyph_after: np.ndarray
    blstats_before: np.ndarray
    blstats_after: np.ndarray
    message_before: str
    message_after: str
    inventory_before: str
    inventory_after: str
    success: bool
    done: bool
    reward: float


@dataclass(frozen=True)
class MiniHackJEPAVocab:
    glyph_to_idx: dict[int, int]
    action_to_idx: dict[str, int]

    @classmethod
    def from_rows(cls, rows: list[MiniHackJEPARow]) -> MiniHackJEPAVocab:
        glyphs: set[int] = set()
        actions: set[str] = set()
        for row in rows:
            glyphs.update(int(x) for x in row.glyph_before.ravel())
            glyphs.update(int(x) for x in row.glyph_after.ravel())
            actions.add(row.action_name)
        return cls(
            glyph_to_idx={glyph: i + 1 for i, glyph in enumerate(sorted(glyphs))},
            action_to_idx={name: i for i, name in enumerate(sorted(actions))},
        )

    @property
    def glyph_vocab_size(self) -> int:
        return len(self.glyph_to_idx) + 1

    @property
    def action_vocab_size(self) -> int:
        return len(self.action_to_idx)

    def encode_glyphs(self, glyphs: np.ndarray) -> np.ndarray:
        out = np.zeros(glyphs.shape, dtype=np.int64)
        for raw, idx in self.glyph_to_idx.items():
            out[glyphs == raw] = idx
        return out

    def encode_action(self, action_name: str) -> int:
        return self.action_to_idx[action_name]


def _hash_token(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % (TEXT_VOCAB_SIZE - 1) + 1


def _encode_text(value: str, max_tokens: int = TEXT_TOKENS) -> np.ndarray:
    tokens = [tok for tok in value.replace("|", " ").split() if tok]
    out = np.zeros(max_tokens, dtype=np.int64)
    for i, token in enumerate(tokens[:max_tokens]):
        out[i] = _hash_token(token.lower())
    return out


def _pad_blstats(value: np.ndarray) -> np.ndarray:
    out = np.zeros(BLSTATS_DIM, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).ravel()[:BLSTATS_DIM]
    out[: arr.shape[0]] = arr
    return out / 100.0


def load_minihack_jepa_shard(path: Path) -> list[MiniHackJEPARow]:
    f = np.load(path, allow_pickle=True)
    n = int(f["action"].shape[0])
    rows: list[MiniHackJEPARow] = []
    for i in range(n):
        rows.append(
            MiniHackJEPARow(
                seed=int(f["seed"][i]),
                episode=int(f["episode"][i]),
                step=int(f["step"][i]),
                action=int(f["action"][i]),
                action_name=str(f["action_name"][i]),
                glyph_before=np.asarray(f["glyph_crop_before"][i]),
                glyph_after=np.asarray(f["glyph_crop_after"][i]),
                blstats_before=np.asarray(f["blstats_before"][i]),
                blstats_after=np.asarray(f["blstats_after"][i]),
                message_before=str(f["message_before"][i]),
                message_after=str(f["message_after"][i]),
                inventory_before=str(f["inventory_before"][i]),
                inventory_after=str(f["inventory_after"][i]),
                success=bool(f["success"][i]),
                done=bool(f["done"][i]),
                reward=float(f["reward"][i]),
            )
        )
    return rows


def load_minihack_jepa_dir(root: Path) -> list[MiniHackJEPARow]:
    paths = sorted(Path(root).rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no MiniHack npz files under {root}")
    rows: list[MiniHackJEPARow] = []
    for path in paths:
        rows.extend(load_minihack_jepa_shard(path))
    return rows


class _MiniHackJEPADataset(Dataset):
    def __init__(self, rows: list[MiniHackJEPARow], vocab: MiniHackJEPAVocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        return {
            "glyph_before": torch.from_numpy(self.vocab.encode_glyphs(row.glyph_before)),
            "glyph_after": torch.from_numpy(self.vocab.encode_glyphs(row.glyph_after)),
            "blstats_before": torch.from_numpy(_pad_blstats(row.blstats_before)),
            "blstats_after": torch.from_numpy(_pad_blstats(row.blstats_after)),
            "message_before": torch.from_numpy(_encode_text(row.message_before)),
            "message_after": torch.from_numpy(_encode_text(row.message_after)),
            "inventory_before": torch.from_numpy(_encode_text(row.inventory_before)),
            "inventory_after": torch.from_numpy(_encode_text(row.inventory_after)),
            "action": torch.tensor(self.vocab.encode_action(row.action_name), dtype=torch.long),
            "success": torch.tensor(float(row.success), dtype=torch.float32),
        }


class MiniHackStateEncoder(nn.Module):
    def __init__(
        self,
        glyph_vocab_size: int,
        text_vocab_size: int = TEXT_VOCAB_SIZE,
        token_dim: int = 32,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.glyph_emb = nn.Embedding(glyph_vocab_size, token_dim, padding_idx=0)
        self.text_emb = nn.Embedding(text_vocab_size, token_dim, padding_idx=0)
        self.glyph_conv = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.blstats_mlp = nn.Sequential(nn.Linear(BLSTATS_DIM, token_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(token_dim * 4, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def _mean_text(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.text_emb(tokens)
        mask = (tokens != 0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        return (emb * mask).sum(dim=1) / denom

    def forward(
        self,
        glyphs: torch.Tensor,
        blstats: torch.Tensor,
        message: torch.Tensor,
        inventory: torch.Tensor,
    ) -> torch.Tensor:
        glyph_feat = self.glyph_emb(glyphs).permute(0, 3, 1, 2)
        glyph_feat = self.glyph_conv(glyph_feat).flatten(1)
        msg_feat = self._mean_text(message)
        inv_feat = self._mean_text(inventory)
        bl_feat = self.blstats_mlp(blstats)
        return self.head(torch.cat([glyph_feat, msg_feat, inv_feat, bl_feat], dim=1))


class MiniHackSkillJEPANet(nn.Module):
    def __init__(self, glyph_vocab_size: int, action_vocab_size: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = MiniHackStateEncoder(glyph_vocab_size, latent_dim=latent_dim)
        self.action_emb = nn.Embedding(action_vocab_size, latent_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.inverse = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, action_vocab_size),
        )

    def encode_batch(self, batch: dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
        return self.encoder(
            batch[f"glyph_{suffix}"],
            batch[f"blstats_{suffix}"],
            batch[f"message_{suffix}"],
            batch[f"inventory_{suffix}"],
        )

    def predict_next(self, z_before: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([z_before, self.action_emb(action)], dim=1))


def anti_collapse_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Small VICReg-style variance/covariance guard for prototype training."""

    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std))
    cov = centered.T @ centered / (z.shape[0] - 1)
    cov = cov - torch.diag(torch.diag(cov))
    cov_loss = cov.pow(2).sum() / z.shape[1]
    return var_loss + 0.05 * cov_loss


@dataclass
class SkillJEPAConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    pred_weight: float = 1.0
    reg_weight: float = 0.05
    inverse_weight: float = 0.1
    seed: int = 0
    device: str = "cpu"


class MiniHackSkillJEPA:
    def __init__(self, vocab: MiniHackJEPAVocab, config: SkillJEPAConfig | None = None):
        self.vocab = vocab
        self.config = config or SkillJEPAConfig()
        self.net = MiniHackSkillJEPANet(vocab.glyph_vocab_size, vocab.action_vocab_size)
        self.history: list[dict[str, float]] = []

    def fit(self, rows: list[MiniHackJEPARow]) -> None:
        if not rows:
            raise ValueError("rows is empty")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        device = torch.device(self.config.device)
        self.net.to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.config.lr)
        loader = DataLoader(
            _MiniHackJEPADataset(rows, self.vocab),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        for epoch in range(self.config.epochs):
            self.net.train()
            total = 0.0
            seen = 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                z_before = self.net.encode_batch(batch, "before")
                z_after = self.net.encode_batch(batch, "after")
                pred = self.net.predict_next(z_before, batch["action"])
                pred_loss = F.mse_loss(pred, z_after.detach())
                inv_logits = self.net.inverse(
                    torch.cat([z_before.detach(), z_after.detach()], dim=1)
                )
                inv_loss = F.cross_entropy(inv_logits, batch["action"])
                reg_loss = anti_collapse_loss(torch.cat([z_before, z_after], dim=0))
                loss = (
                    self.config.pred_weight * pred_loss
                    + self.config.inverse_weight * inv_loss
                    + self.config.reg_weight * reg_loss
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * batch["action"].shape[0]
                seen += int(batch["action"].shape[0])
            self.history.append({"epoch": float(epoch), "loss": total / max(seen, 1)})

    @torch.no_grad()
    def surprise(self, rows: list[MiniHackJEPARow]) -> np.ndarray:
        if not rows:
            return np.array([], dtype=np.float64)
        device = next(self.net.parameters()).device
        self.net.eval()
        loader = DataLoader(
            _MiniHackJEPADataset(rows, self.vocab),
            batch_size=max(self.config.batch_size, 64),
            shuffle=False,
            drop_last=False,
        )
        scores: list[np.ndarray] = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            z_before = self.net.encode_batch(batch, "before")
            z_after = self.net.encode_batch(batch, "after")
            pred = self.net.predict_next(z_before, batch["action"])
            score = (pred - z_after).pow(2).mean(dim=1)
            scores.append(score.cpu().numpy())
        return np.concatenate(scores).astype(np.float64)


def split_rows_by_seed(
    rows: list[MiniHackJEPARow], train_frac: float, seed: int
) -> tuple[list[MiniHackJEPARow], list[MiniHackJEPARow]]:
    seeds = sorted({r.seed for r in rows})
    if len(seeds) < 2:
        raise ValueError(f"need at least two seeds for split, got {seeds}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(seeds))
    n_train = min(max(1, int(round(len(seeds) * train_frac))), len(seeds) - 1)
    train_seeds = {seeds[i] for i in perm[:n_train]}
    return [r for r in rows if r.seed in train_seeds], [
        r for r in rows if r.seed not in train_seeds
    ]


def summarize_surprise(rows: list[MiniHackJEPARow], surprise: np.ndarray) -> dict[str, float | int]:
    labels = np.array([r.success for r in rows], dtype=bool)
    out: dict[str, float | int] = {
        "rows": len(rows),
        "successes": int(labels.sum()),
        "mean_surprise": float(np.mean(surprise)) if len(surprise) else float("nan"),
    }
    if labels.any():
        out["success_surprise"] = float(np.mean(surprise[labels]))
    if (~labels).any():
        out["non_success_surprise"] = float(np.mean(surprise[~labels]))
    return out


def train_eval_summary(
    rows: list[MiniHackJEPARow],
    config: SkillJEPAConfig,
    train_frac: float = 0.6,
) -> dict[str, object]:
    train, evalu = split_rows_by_seed(rows, train_frac=train_frac, seed=config.seed)
    vocab = MiniHackJEPAVocab.from_rows(rows)
    model = MiniHackSkillJEPA(vocab, config)
    model.fit(train)
    train_scores = model.surprise(train)
    eval_scores = model.surprise(evalu)
    train_seeds = sorted({r.seed for r in train})
    eval_seeds = sorted({r.seed for r in evalu})
    return {
        "rows": len(rows),
        "seed_split": {
            "seed": config.seed,
            "train_frac": train_frac,
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
        },
        "vocab": {
            "glyph_vocab_size": vocab.glyph_vocab_size,
            "action_vocab_size": vocab.action_vocab_size,
        },
        "config": asdict(config),
        "final_loss": model.history[-1]["loss"],
        "history": model.history,
        "train": summarize_surprise(train, train_scores),
        "eval": summarize_surprise(evalu, eval_scores),
    }


def _print_summary(summary: dict[str, object]) -> None:
    split = summary["seed_split"]
    vocab = summary["vocab"]
    train = summary["train"]
    evalu = summary["eval"]
    assert isinstance(split, dict)
    assert isinstance(vocab, dict)
    print("MiniHack Skill-JEPA prototype")
    print(f"  train seeds: {split['train_seeds']}  eval seeds: {split['eval_seeds']}")
    print(f"  train rows: {train['rows']}  eval rows: {evalu['rows']}")
    print(f"  glyph vocab: {vocab['glyph_vocab_size']}  action vocab: {vocab['action_vocab_size']}")
    print(f"  final loss: {float(summary['final_loss']):.6f}")
    print("  train surprise:", train)
    print("  eval surprise:", evalu)


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny MiniHack Skill-JEPA prototype.")
    p.add_argument("--data", type=Path, default=Path("data/rollouts/minihack-smoke"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="Optional JSON summary output path.")
    args = p.parse_args()

    rows = load_minihack_jepa_dir(args.data)
    summary = train_eval_summary(
        rows,
        SkillJEPAConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed),
        train_frac=args.train_frac,
    )
    _print_summary(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"  wrote: {args.out}")


if __name__ == "__main__":
    main()
