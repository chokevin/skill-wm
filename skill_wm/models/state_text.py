"""Render a ScoringRow as a textual prompt for the LLM-as-WM baseline.

Design constraints:

- The LLM and the trained WM must see *the same information*. We give the
  LLM the local 15×15 semantic crop (as ASCII), inventory + vitals as
  JSON, the action being asked about, and the player's facing direction.
  We do **not** give the LLM derived "is_at_table=true" booleans — those
  encode knowledge that should be the model's job to infer. Same view we'd
  hand the trained WM.
- The crop is rendered with one ASCII character per tile (legend in the
  prompt) so token cost stays bounded — full word labels would blow up
  the context window.
- The prompt is written for a single-token "Yes"/"No" completion so we
  can read graded probability from `top_logprobs` (T1 headline metric is
  calibration; hard yes/no would throw away that signal).
"""

from __future__ import annotations

import json

import numpy as np

from skill_wm.eval.dataset import ScoringRow
from skill_wm.models.baselines import TILE_LEGEND

# One char per tile-id. Order MUST match TILE_LEGEND keys.
TILE_GLYPH: dict[int, str] = {
    0: " ",  # void / out of world
    1: "~",  # water
    2: ".",  # grass
    3: "#",  # stone
    4: "_",  # path
    5: ":",  # sand
    6: "T",  # tree
    7: "L",  # lava
    8: "c",  # coal
    9: "i",  # iron
    10: "d",  # diamond
    11: "B",  # table (bench)
    12: "F",  # furnace
    13: "@",  # player
    14: "C",  # cow
    15: "Z",  # zombie
    16: "S",  # skeleton
    17: ">",  # arrow
    18: "P",  # plant
    19: "?",  # masked: outside the predictor's observation budget
}

FACING_NAME: dict[tuple[int, int], str] = {
    (-1, 0): "left",
    (1, 0): "right",
    (0, -1): "up",
    (0, 1): "down",
}


def render_crop(crop: np.ndarray) -> str:
    """Render the 15×15 semantic crop as ASCII with the player marker.

    Crop is world-aligned: ``crop[half+dx, half+dy]`` is the tile at
    ``(player_pos + (dx, dy))``. To make it human-readable, we display
    rows top-to-bottom in screen-y order (so "up" is on top, "down" is
    on bottom) and columns left-to-right in screen-x order.

    In Crafter's coordinate convention: dy = -1 is "up", dy = +1 is
    "down", dx = -1 is "left", dx = +1 is "right". So we transpose the
    crop for display: y becomes the row axis and x becomes the column
    axis. This is purely a display convention; the underlying data
    stays world-aligned.
    """
    h, w = crop.shape
    # Display: rows = y axis (top..bottom = -half..+half), cols = x axis.
    lines = []
    for j in range(w):  # j is the y-axis index (display row)
        row_chars = []
        for i in range(h):  # i is the x-axis index (display col)
            row_chars.append(TILE_GLYPH.get(int(crop[i, j]), "?"))
        lines.append("".join(row_chars))
    return "\n".join(lines)


def render_legend() -> str:
    """Single-line legend mapping glyph -> tile name."""
    parts = []
    for tid, name in TILE_LEGEND.items():
        glyph = TILE_GLYPH.get(tid, "?")
        if name in ("void",):
            continue  # space is self-explanatory
        parts.append(f"{glyph}={name}")
    return ", ".join(parts)


def render_inventory(row: ScoringRow) -> str:
    """Inventory as compact JSON, dropping zero entries to save tokens."""
    inv = row.inventory_dict()
    nonzero = {k: v for k, v in inv.items() if v != 0}
    return json.dumps(nonzero, separators=(",", ":"))


SYSTEM_PROMPT = (
    "You are a world-model oracle for the Crafter environment. Given the "
    "player's local view, inventory, and facing direction, you predict "
    "whether a single action will SUCCEED on the next step. Crafter "
    "actions and success rules:\n"
    "- move_{left,right,up,down}: succeeds iff the target cell is walkable "
    "(grass, path, sand).\n"
    "- do: interacts with the cell the player is facing. Collects wood "
    "(tree), stone (needs wood pickaxe), coal (needs wood pickaxe), "
    "iron (needs stone pickaxe), diamond (needs iron pickaxe); drinks "
    "water; eats plant/cow; attacks zombie/skeleton (needs sword to "
    "reliably hit).\n"
    "- sleep: succeeds iff player enters sleep state (no enemy adjacent).\n"
    "- place_stone/table/furnace: needs the resource and the front cell "
    "to be a valid placement tile.\n"
    "- place_plant: needs sapling and front cell = grass.\n"
    "- make_*: needs ingredients AND adjacent table (and furnace for iron).\n"
    "Answer with a SINGLE TOKEN: Yes or No."
)


def render_user_prompt(row: ScoringRow) -> str:
    """The user message: state description + the action to predict."""
    facing_name = FACING_NAME.get(row.facing_before, str(row.facing_before))
    sleeping_str = "yes" if row.sleeping_before else "no"
    return (
        f"Local view (15x15, player @ center, facing {facing_name}):\n"
        f"```\n{render_crop(row.semantic_crop_before)}\n```\n"
        f"Legend: {render_legend()}\n"
        f"Inventory (nonzero): {render_inventory(row)}\n"
        f"Sleeping: {sleeping_str}\n"
        f"Action: {row.action_name}\n"
        f"Will this action succeed? Answer Yes or No."
    )
