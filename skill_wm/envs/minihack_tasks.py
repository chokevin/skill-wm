"""Controlled MiniHack tasks for the second-environment benchmark."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

MINIHACK_CARDINAL_ACTION_NAMES: tuple[str, ...] = ("north", "east", "south", "west")

_DIRS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}

_WALKABLE_MAP_TILES = {".", "#", "+", ">"}


@dataclass(frozen=True)
class MiniHackTaskSpec:
    """Small custom task definition we can make into a MiniHack env."""

    env_id: str
    description: str
    map_lines: tuple[str, ...]
    start_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    max_episode_steps: int = 80
    action_names: tuple[str, ...] = MINIHACK_CARDINAL_ACTION_NAMES
    hazard_tiles: tuple[str, ...] = ()

    @property
    def des_file(self) -> str:
        return _des_for_map(self.map_lines, self.start_pos, self.goal_pos)

    @property
    def walkable(self) -> frozenset[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for y, row in enumerate(self.map_lines):
            for x, tile in enumerate(row):
                if tile in _WALKABLE_MAP_TILES:
                    cells.add((x, y))
        cells.add(self.start_pos)
        cells.add(self.goal_pos)
        return frozenset(cells)

    def next_action_toward_goal(
        self,
        pos: tuple[int, int],
        coord_offset: tuple[int, int] = (0, 0),
    ) -> str | None:
        """Return the first BFS move from ``pos`` to the task goal."""

        ox, oy = coord_offset
        goal_pos = (self.goal_pos[0] + ox, self.goal_pos[1] + oy)
        walkable = frozenset((x + ox, y + oy) for x, y in self.walkable)

        if pos == goal_pos:
            return None

        q: deque[tuple[int, int]] = deque([pos])
        parent: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {pos: None}

        while q:
            cur = q.popleft()
            if cur == goal_pos:
                break
            for action_name, (dx, dy) in _DIRS.items():
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in parent or nxt not in walkable:
                    continue
                parent[nxt] = (cur, action_name)
                q.append(nxt)

        if goal_pos not in parent:
            return None

        node = goal_pos
        prev = parent[node]
        while prev is not None and prev[0] != pos:
            node = prev[0]
            prev = parent[node]
        return prev[1] if prev is not None else None


def _des_for_map(
    map_lines: tuple[str, ...],
    start_pos: tuple[int, int],
    goal_pos: tuple[int, int],
) -> str:
    width = max(len(row) for row in map_lines)
    height = len(map_lines)
    rows = [row.ljust(width) for row in map_lines]

    sx, sy = start_pos
    gx, gy = goal_pos
    # The destination side of BRANCH just has to differ from the source area.
    bx, by = (0, 0) if (sx, sy) != (0, 0) else (1, 1)

    return "\n".join(
        [
            "MAZE: \"mylevel\", ' '",
            "FLAGS:hardfloor,premapped",
            "INIT_MAP: solidfill,' '",
            "GEOMETRY:center,center",
            "MAP",
            *rows,
            "ENDMAP",
            f'REGION:(0,0,{width - 1},{height - 1}),lit,"ordinary"',
            f"BRANCH:({sx},{sy},{sx},{sy}),({bx},{by},{bx},{by})",
            f"STAIR:({gx},{gy}),down",
            "",
        ]
    )


SKILLWM_MINIHACK_TASKS: dict[str, MiniHackTaskSpec] = {
    "skillwm-room-goal": MiniHackTaskSpec(
        env_id="skillwm-room-goal",
        description="Open room: reach the visible staircase.",
        map_lines=(
            ".........",
            ".........",
            ".........",
            ".........",
            ".........",
        ),
        start_pos=(1, 2),
        goal_pos=(7, 2),
        max_episode_steps=30,
    ),
    "skillwm-lava-detour": MiniHackTaskSpec(
        env_id="skillwm-lava-detour",
        description="Reach the staircase while detouring around a lava column.",
        map_lines=(
            ".........",
            "....L....",
            "....L....",
            "....L....",
            ".........",
        ),
        start_pos=(1, 2),
        goal_pos=(7, 2),
        max_episode_steps=50,
        hazard_tiles=("L",),
    ),
    "skillwm-water-detour": MiniHackTaskSpec(
        env_id="skillwm-water-detour",
        description="Reach the staircase while detouring around a water column.",
        map_lines=(
            ".........",
            "....}....",
            "....}....",
            "....}....",
            ".........",
        ),
        start_pos=(1, 2),
        goal_pos=(7, 2),
        max_episode_steps=50,
        hazard_tiles=("}",),
    ),
}


def registered_minihack_task_names() -> tuple[str, ...]:
    return tuple(SKILLWM_MINIHACK_TASKS)


def get_minihack_task_spec(env_id: str) -> MiniHackTaskSpec | None:
    return SKILLWM_MINIHACK_TASKS.get(env_id)


def make_skillwm_minihack_env(env_id: str) -> Any:
    """Create one of the custom skill-WM MiniHack environments."""

    spec = get_minihack_task_spec(env_id)
    if spec is None:
        raise KeyError(f"unknown skill-WM MiniHack task: {env_id}")

    try:
        from minihack import MiniHack
        from minihack.reward_manager import RewardManager
        from nle import nethack
    except ImportError as e:
        raise RuntimeError(
            "MiniHack support is optional. Install it with "
            "`uv sync --extra minihack` or run `make minihack-smoke`."
        ) from e

    action_by_name = {
        "north": nethack.CompassDirection.N,
        "east": nethack.CompassDirection.E,
        "south": nethack.CompassDirection.S,
        "west": nethack.CompassDirection.W,
    }

    reward_manager = RewardManager()
    reward_manager.add_coordinate_event(
        spec.goal_pos,
        reward=1,
        terminal_required=True,
        terminal_sufficient=True,
    )

    return MiniHack(
        des_file=spec.des_file,
        reward_manager=reward_manager,
        actions=tuple(action_by_name[name] for name in spec.action_names),
        observation_keys=[
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "inv_letters",
            "inv_strs",
        ],
        max_episode_steps=spec.max_episode_steps,
        autopickup=True,
        pet=False,
    )
