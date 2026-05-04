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

from skill_wm.envs.minihack_tasks import get_minihack_task_spec

TEXT_VOCAB_SIZE = 512
TEXT_TOKENS = 16
BLSTATS_DIM = 27
OBJECT_TILES: tuple[str, ...] = (" ", ".", ">", "L", "#", "+", "?")
OBJECT_TILE_TO_IDX: dict[str, int] = {tile: i for i, tile in enumerate(OBJECT_TILES)}

_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}


@dataclass(frozen=True)
class MiniHackJEPARow:
    seed: int
    env_id: str
    policy_name: str
    episode: int
    step: int
    action: int
    action_name: str
    logical_pos_before: tuple[int, int]
    logical_pos_after: tuple[int, int]
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
    object_signatures: frozenset[str] = frozenset()

    @classmethod
    def from_rows(cls, rows: list[MiniHackJEPARow]) -> MiniHackJEPAVocab:
        glyphs: set[int] = set()
        actions: set[str] = set()
        object_signatures: set[str] = set()
        for row in rows:
            glyphs.update(int(x) for x in row.glyph_before.ravel())
            glyphs.update(int(x) for x in row.glyph_after.ravel())
            actions.add(row.action_name)
            object_signatures.add(object_signature(row))
        return cls(
            glyph_to_idx={glyph: i + 1 for i, glyph in enumerate(sorted(glyphs))},
            action_to_idx={name: i + 1 for i, name in enumerate(sorted(actions))},
            object_signatures=frozenset(object_signatures),
        )

    @property
    def glyph_vocab_size(self) -> int:
        return len(self.glyph_to_idx) + 1

    @property
    def action_vocab_size(self) -> int:
        return len(self.action_to_idx) + 1

    @property
    def object_vocab_size(self) -> int:
        return len(OBJECT_TILES)

    def encode_glyphs(self, glyphs: np.ndarray) -> np.ndarray:
        out = np.zeros(glyphs.shape, dtype=np.int64)
        for raw, idx in self.glyph_to_idx.items():
            out[glyphs == raw] = idx
        return out

    def encode_action(self, action_name: str) -> int:
        return self.action_to_idx.get(action_name, 0)


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


def _object_tile_idx(tile: str) -> int:
    return OBJECT_TILE_TO_IDX.get(tile, OBJECT_TILE_TO_IDX["?"])


def load_minihack_jepa_shard(path: Path) -> list[MiniHackJEPARow]:
    f = np.load(path, allow_pickle=True)
    n = int(f["action"].shape[0])
    env_ids = f["env_id"] if "env_id" in f.files else np.array([_infer_env_id(path)] * n)
    policy_names = (
        f["policy_name"] if "policy_name" in f.files else np.array([_infer_policy_name(path)] * n)
    )
    logical_pos_before = (
        f["logical_pos_before"]
        if "logical_pos_before" in f.files
        else np.zeros((n, 2), dtype=np.int32)
    )
    logical_pos_after = (
        f["logical_pos_after"]
        if "logical_pos_after" in f.files
        else np.zeros((n, 2), dtype=np.int32)
    )
    rows: list[MiniHackJEPARow] = []
    for i in range(n):
        rows.append(
            MiniHackJEPARow(
                seed=int(f["seed"][i]),
                env_id=str(env_ids[i]),
                policy_name=str(policy_names[i]),
                episode=int(f["episode"][i]),
                step=int(f["step"][i]),
                action=int(f["action"][i]),
                action_name=str(f["action_name"][i]),
                logical_pos_before=tuple(int(x) for x in logical_pos_before[i]),
                logical_pos_after=tuple(int(x) for x in logical_pos_after[i]),
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


def _infer_env_id(path: Path) -> str:
    name = path.parent.name
    if name == "room-goal":
        return "skillwm-room-goal"
    if name == "lava-detour":
        return "skillwm-lava-detour"
    return name


def _infer_policy_name(path: Path) -> str:
    name = path.parent.name
    if "lava-probe" in name:
        return "lava_probe"
    if "noisy" in name:
        return "scripted_nav_noisy"
    return "scripted_nav"


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
        _, target_tile, after_tile = object_tiles(row)
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
            "target_tile": torch.tensor(_object_tile_idx(target_tile), dtype=torch.long),
            "after_tile": torch.tensor(_object_tile_idx(after_tile), dtype=torch.long),
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
    def __init__(
        self,
        glyph_vocab_size: int,
        action_vocab_size: int,
        object_vocab_size: int = len(OBJECT_TILES),
        latent_dim: int = 64,
    ):
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
        self.target_object = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, object_vocab_size),
        )
        self.after_object = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, object_vocab_size),
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

    def object_logits(
        self,
        z_before: torch.Tensor,
        action: torch.Tensor,
        pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_emb = self.action_emb(action)
        target_logits = self.target_object(torch.cat([z_before, action_emb], dim=1))
        after_logits = self.after_object(pred)
        return target_logits, after_logits


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
    object_aux_weight: float = 0.0
    seed: int = 0
    device: str = "cpu"


class MiniHackSkillJEPA:
    def __init__(self, vocab: MiniHackJEPAVocab, config: SkillJEPAConfig | None = None):
        self.vocab = vocab
        self.config = config or SkillJEPAConfig()
        self.net = MiniHackSkillJEPANet(
            vocab.glyph_vocab_size,
            vocab.action_vocab_size,
            vocab.object_vocab_size,
        )
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
                target_logits, after_logits = self.net.object_logits(
                    z_before,
                    batch["action"],
                    pred,
                )
                object_loss = F.cross_entropy(
                    target_logits,
                    batch["target_tile"],
                ) + F.cross_entropy(after_logits, batch["after_tile"])
                loss = (
                    self.config.pred_weight * pred_loss
                    + self.config.inverse_weight * inv_loss
                    + self.config.reg_weight * reg_loss
                    + self.config.object_aux_weight * object_loss
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * batch["action"].shape[0]
                seen += int(batch["action"].shape[0])
            self.history.append({"epoch": float(epoch), "loss": total / max(seen, 1)})

    @torch.no_grad()
    def score_rows(self, rows: list[MiniHackJEPARow]) -> dict[str, np.ndarray]:
        if not rows:
            empty = np.array([], dtype=np.float64)
            return {
                "latent_surprise": empty,
                "target_object_nll": empty,
                "after_object_nll": empty,
                "object_nll": empty,
            }
        device = next(self.net.parameters()).device
        self.net.eval()
        loader = DataLoader(
            _MiniHackJEPADataset(rows, self.vocab),
            batch_size=max(self.config.batch_size, 64),
            shuffle=False,
            drop_last=False,
        )
        latent_scores: list[np.ndarray] = []
        target_nll: list[np.ndarray] = []
        after_nll: list[np.ndarray] = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            z_before = self.net.encode_batch(batch, "before")
            z_after = self.net.encode_batch(batch, "after")
            pred = self.net.predict_next(z_before, batch["action"])
            score = (pred - z_after).pow(2).mean(dim=1)
            target_logits, after_logits = self.net.object_logits(z_before, batch["action"], pred)
            target_nll.append(
                F.cross_entropy(
                    target_logits,
                    batch["target_tile"],
                    reduction="none",
                )
                .cpu()
                .numpy()
            )
            after_nll.append(
                F.cross_entropy(after_logits, batch["after_tile"], reduction="none").cpu().numpy()
            )
            latent_scores.append(score.cpu().numpy())
        target_arr = np.concatenate(target_nll).astype(np.float64)
        after_arr = np.concatenate(after_nll).astype(np.float64)
        return {
            "latent_surprise": np.concatenate(latent_scores).astype(np.float64),
            "target_object_nll": target_arr,
            "after_object_nll": after_arr,
            "object_nll": target_arr + after_arr,
        }

    @torch.no_grad()
    def surprise(self, rows: list[MiniHackJEPARow]) -> np.ndarray:
        return self.score_rows(rows)["latent_surprise"]


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


def split_rows_by_env(
    rows: list[MiniHackJEPARow], eval_env_id: str
) -> tuple[list[MiniHackJEPARow], list[MiniHackJEPARow]]:
    train = [r for r in rows if r.env_id != eval_env_id]
    evalu = [r for r in rows if r.env_id == eval_env_id]
    if not train:
        raise ValueError(f"no train rows after holding out env_id={eval_env_id!r}")
    if not evalu:
        env_ids = sorted({r.env_id for r in rows})
        raise ValueError(f"no eval rows for env_id={eval_env_id!r}; available={env_ids}")
    return train, evalu


def split_rows_by_policy(
    rows: list[MiniHackJEPARow], eval_policy_name: str
) -> tuple[list[MiniHackJEPARow], list[MiniHackJEPARow]]:
    train = [r for r in rows if r.policy_name != eval_policy_name]
    evalu = [r for r in rows if r.policy_name == eval_policy_name]
    if not train:
        raise ValueError(f"no train rows after holding out policy_name={eval_policy_name!r}")
    if not evalu:
        policies = sorted({r.policy_name for r in rows})
        raise ValueError(f"no eval rows for policy_name={eval_policy_name!r}; available={policies}")
    return train, evalu


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


def _tile_at(env_id: str, pos: tuple[int, int]) -> str:
    spec = get_minihack_task_spec(env_id)
    if spec is None:
        return "?"
    x, y = pos
    if y < 0 or y >= len(spec.map_lines):
        return " "
    row = spec.map_lines[y]
    if x < 0 or x >= len(row):
        return " "
    return row[x]


def object_tiles(row: MiniHackJEPARow) -> tuple[str, str, str]:
    dx, dy = _ACTION_DELTAS.get(row.action_name, (0, 0))
    target = (row.logical_pos_before[0] + dx, row.logical_pos_before[1] + dy)
    before_tile = _tile_at(row.env_id, row.logical_pos_before)
    target_tile = _tile_at(row.env_id, target)
    after_tile = _tile_at(row.env_id, row.logical_pos_after)
    return before_tile, target_tile, after_tile


def object_signature(row: MiniHackJEPARow) -> str:
    before_tile, target_tile, after_tile = object_tiles(row)
    return f"{row.action_name}|{before_tile}->{target_tile}->{after_tile}"


def coverage_summary(
    rows: list[MiniHackJEPARow],
    vocab: MiniHackJEPAVocab,
    surprise: np.ndarray | None = None,
) -> dict[str, object]:
    """Summarize train-vocab coverage for explaining task-shift surprise."""

    if not rows:
        return {
            "rows": 0,
            "action_oov_rate": float("nan"),
            "glyph_before_oov_rate": float("nan"),
            "glyph_after_oov_rate": float("nan"),
        }

    train_glyphs = set(vocab.glyph_to_idx)
    action_oov = np.array([r.action_name not in vocab.action_to_idx for r in rows], dtype=bool)
    lava_message = np.array(
        ["lava" in f"{r.message_before} {r.message_after}".lower() for r in rows],
        dtype=bool,
    )
    lava_probe = np.array(
        [
            r.env_id == "skillwm-lava-detour"
            and r.logical_pos_before == (3, 2)
            and r.action_name == "east"
            for r in rows
        ],
        dtype=bool,
    )
    object_signatures = [object_signature(r) for r in rows]
    object_oov = np.array(
        [sig not in vocab.object_signatures for sig in object_signatures],
        dtype=bool,
    )
    glyph_before_oov: list[bool] = []
    glyph_after_oov: list[bool] = []
    glyph_before_unknown = 0
    glyph_after_unknown = 0
    glyph_before_total = 0
    glyph_after_total = 0
    unknown_glyphs: set[int] = set()
    for row in rows:
        before_values = np.asarray(row.glyph_before).ravel()
        after_values = np.asarray(row.glyph_after).ravel()
        before_unknown = [int(x) for x in before_values if int(x) not in train_glyphs]
        after_unknown = [int(x) for x in after_values if int(x) not in train_glyphs]
        unknown_glyphs.update(before_unknown)
        unknown_glyphs.update(after_unknown)
        glyph_before_unknown += len(before_unknown)
        glyph_after_unknown += len(after_unknown)
        glyph_before_total += int(before_values.shape[0])
        glyph_after_total += int(after_values.shape[0])
        glyph_before_oov.append(bool(before_unknown))
        glyph_after_oov.append(bool(after_unknown))

    glyph_transition_oov = np.array(
        [before or after for before, after in zip(glyph_before_oov, glyph_after_oov, strict=True)],
        dtype=bool,
    )
    out: dict[str, object] = {
        "rows": len(rows),
        "env_ids": sorted({r.env_id for r in rows}),
        "action_names": sorted({r.action_name for r in rows}),
        "policy_names": sorted({r.policy_name for r in rows}),
        "action_oov_names": sorted(
            {r.action_name for r in rows if r.action_name not in vocab.action_to_idx}
        ),
        "action_oov_rate": float(action_oov.mean()),
        "object_signatures": sorted(set(object_signatures)),
        "object_oov_signatures": sorted(
            {sig for sig in object_signatures if sig not in vocab.object_signatures}
        ),
        "object_signature_oov_rate": float(object_oov.mean()),
        "done_rate": float(np.mean([r.done for r in rows])),
        "mean_reward": float(np.mean([r.reward for r in rows])),
        "lava_message_rate": float(lava_message.mean()),
        "lava_probe_rate": float(lava_probe.mean()),
        "glyph_oov_values": sorted(unknown_glyphs),
        "glyph_before_oov_rate": float(glyph_before_unknown / max(glyph_before_total, 1)),
        "glyph_after_oov_rate": float(glyph_after_unknown / max(glyph_after_total, 1)),
        "glyph_transition_oov_rate": float(glyph_transition_oov.mean()),
    }
    if surprise is not None and len(surprise):
        if action_oov.any():
            out["action_oov_surprise"] = float(np.mean(surprise[action_oov]))
        if (~action_oov).any():
            out["action_known_surprise"] = float(np.mean(surprise[~action_oov]))
        if glyph_transition_oov.any():
            out["glyph_oov_surprise"] = float(np.mean(surprise[glyph_transition_oov]))
        if (~glyph_transition_oov).any():
            out["glyph_known_surprise"] = float(np.mean(surprise[~glyph_transition_oov]))
        if lava_message.any():
            out["lava_message_surprise"] = float(np.mean(surprise[lava_message]))
        if lava_probe.any():
            out["lava_probe_surprise"] = float(np.mean(surprise[lava_probe]))
        if object_oov.any():
            out["object_oov_surprise"] = float(np.mean(surprise[object_oov]))
        if (~object_oov).any():
            out["object_known_surprise"] = float(np.mean(surprise[~object_oov]))
    return out


def summarize_object_aux(
    rows: list[MiniHackJEPARow],
    vocab: MiniHackJEPAVocab,
    scores: dict[str, np.ndarray],
) -> dict[str, float | int]:
    object_nll = scores["object_nll"]
    target_nll = scores["target_object_nll"]
    after_nll = scores["after_object_nll"]
    object_oov = np.array(
        [object_signature(r) not in vocab.object_signatures for r in rows],
        dtype=bool,
    )
    out: dict[str, float | int] = {
        "rows": len(rows),
        "mean_object_nll": float(np.mean(object_nll)) if len(object_nll) else float("nan"),
        "mean_target_object_nll": float(np.mean(target_nll)) if len(target_nll) else float("nan"),
        "mean_after_object_nll": float(np.mean(after_nll)) if len(after_nll) else float("nan"),
        "object_signature_oov_rate": float(object_oov.mean()) if len(object_oov) else float("nan"),
    }
    if object_oov.any():
        out["object_oov_nll"] = float(np.mean(object_nll[object_oov]))
        out["target_object_oov_nll"] = float(np.mean(target_nll[object_oov]))
        out["after_object_oov_nll"] = float(np.mean(after_nll[object_oov]))
    if (~object_oov).any():
        out["object_known_nll"] = float(np.mean(object_nll[~object_oov]))
        out["target_object_known_nll"] = float(np.mean(target_nll[~object_oov]))
        out["after_object_known_nll"] = float(np.mean(after_nll[~object_oov]))
    return out


def train_eval_summary(
    rows: list[MiniHackJEPARow],
    config: SkillJEPAConfig,
    train_frac: float = 0.6,
    split: str = "seed",
    eval_env_id: str | None = None,
    eval_policy_name: str | None = None,
) -> dict[str, object]:
    if split == "seed":
        train, evalu = split_rows_by_seed(rows, train_frac=train_frac, seed=config.seed)
    elif split == "task":
        if eval_env_id is None:
            raise ValueError("eval_env_id is required for task split")
        train, evalu = split_rows_by_env(rows, eval_env_id=eval_env_id)
    elif split == "policy":
        if eval_policy_name is None:
            raise ValueError("eval_policy_name is required for policy split")
        train, evalu = split_rows_by_policy(rows, eval_policy_name=eval_policy_name)
    else:
        raise ValueError(f"unknown split: {split}")
    vocab = MiniHackJEPAVocab.from_rows(train)
    model = MiniHackSkillJEPA(vocab, config)
    model.fit(train)
    train_scores = model.score_rows(train)
    eval_scores = model.score_rows(evalu)
    train_seeds = sorted({r.seed for r in train})
    eval_seeds = sorted({r.seed for r in evalu})
    split_summary = {
        "mode": split,
        "seed": config.seed,
        "train_frac": train_frac,
        "eval_env_id": eval_env_id,
        "eval_policy_name": eval_policy_name,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "train_env_ids": sorted({r.env_id for r in train}),
        "eval_env_ids": sorted({r.env_id for r in evalu}),
        "train_policy_names": sorted({r.policy_name for r in train}),
        "eval_policy_names": sorted({r.policy_name for r in evalu}),
    }
    summary = {
        "rows": len(rows),
        "split": split_summary,
        "seed_split": split_summary,
        "vocab": {
            "glyph_vocab_size": vocab.glyph_vocab_size,
            "action_vocab_size": vocab.action_vocab_size,
            "action_names": sorted(vocab.action_to_idx),
        },
        "coverage": {
            "train": coverage_summary(train, vocab, train_scores["latent_surprise"]),
            "eval": coverage_summary(evalu, vocab, eval_scores["latent_surprise"]),
        },
        "config": asdict(config),
        "final_loss": model.history[-1]["loss"],
        "history": model.history,
        "train": summarize_surprise(train, train_scores["latent_surprise"]),
        "eval": summarize_surprise(evalu, eval_scores["latent_surprise"]),
    }
    if config.object_aux_weight > 0:
        summary["object_aux"] = {
            "train": summarize_object_aux(train, vocab, train_scores),
            "eval": summarize_object_aux(evalu, vocab, eval_scores),
        }
    return summary


def _print_summary(summary: dict[str, object]) -> None:
    split = summary["split"]
    vocab = summary["vocab"]
    train = summary["train"]
    evalu = summary["eval"]
    assert isinstance(split, dict)
    assert isinstance(vocab, dict)
    print("MiniHack Skill-JEPA prototype")
    print(f"  split: {split['mode']}")
    print(f"  train envs: {split['train_env_ids']}  eval envs: {split['eval_env_ids']}")
    print(
        f"  train policies: {split['train_policy_names']}  "
        f"eval policies: {split['eval_policy_names']}"
    )
    print(f"  train seeds: {split['train_seeds']}  eval seeds: {split['eval_seeds']}")
    print(f"  train rows: {train['rows']}  eval rows: {evalu['rows']}")
    print(f"  glyph vocab: {vocab['glyph_vocab_size']}  action vocab: {vocab['action_vocab_size']}")
    print(f"  final loss: {float(summary['final_loss']):.6f}")
    print("  train surprise:", train)
    print("  eval surprise:", evalu)
    print("  eval coverage:", summary["coverage"]["eval"])
    if "object_aux" in summary:
        print("  object aux eval:", summary["object_aux"]["eval"])


def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny MiniHack Skill-JEPA prototype.")
    p.add_argument("--data", type=Path, default=Path("data/rollouts/minihack-smoke"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--object-aux-weight", type=float, default=0.0)
    p.add_argument("--split", choices=("seed", "task", "policy"), default="seed")
    p.add_argument("--eval-env-id", help="MiniHack env_id to hold out when --split=task.")
    p.add_argument("--eval-policy", help="Policy name to hold out when --split=policy.")
    p.add_argument("--out", type=Path, help="Optional JSON summary output path.")
    args = p.parse_args()

    rows = load_minihack_jepa_dir(args.data)
    summary = train_eval_summary(
        rows,
        SkillJEPAConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            object_aux_weight=args.object_aux_weight,
        ),
        train_frac=args.train_frac,
        split=args.split,
        eval_env_id=args.eval_env_id,
        eval_policy_name=args.eval_policy,
    )
    _print_summary(summary)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"  wrote: {args.out}")


if __name__ == "__main__":
    main()
