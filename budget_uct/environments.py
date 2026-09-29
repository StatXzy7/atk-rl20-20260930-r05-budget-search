"""Hand-built and parametric deterministic environments."""

from __future__ import annotations

import random as _random
from typing import Dict, Tuple

from .env import GraphEnv, GraphSpec


def demo_env() -> GraphEnv:
    """Small environment with all requested pathologies.

    * transposition: s2 and s3 both reach s5 ("merge");
    * trap branch:   s2 -> t9 -> t10 (dead end, cheap-looking first step);
    * varied-cost cycles on s5: a cost-3 self loop and a cost-1 self loop;
    * a detour s1 -> s6 -> s5 costs more than the direct s1 -> s2 path but
      demonstrates multiple routes into the same state.

    Optimal successful plan within the designed budget:
        s0 -> s2 ("to_s2", 4), s2 -> s5 ("merge", 1),
        s5 -> g ("goal", 1)                       total cost 6.
    """
    edges: Dict[str, Dict[str, Tuple[str, int]]] = {
        "s0": {"to_s1": ("s1", 2), "to_s2": ("s2", 4)},
        "s1": {"detour": ("s6", 3), "back": ("s0", 2)},
        "s6": {"to_s5": ("s5", 4)},
        "s2": {"trap": ("t9", 1), "merge": ("s5", 1)},
        "t9": {"deeper": ("t10", 1)},
        "t10": {"loop": ("t10", 1)},
        "s3": {"merge": ("s5", 2)},
        "s5": {
            "loop_cheap": ("s5", 1),
            "loop_pricey": ("s5", 3),
            "to_s3": ("s3", 2),
            "goal": ("g", 1),
        },
    }
    return GraphEnv(GraphSpec(start="s0", edges=edges, goals={"g"}))


def dedup_counterexample_env() -> GraphEnv:
    """Environment in which state-only de-duplication loses the feasible plan.

    From s0, the expensive-first order explores "expensive" first:
        s0 --expensive(12)--> X --go(3)--> G
    Under budget 15 that reaches X with only 3 remaining: go is exactly
    payable here in the corrected (state, remaining) search, so this route is
    actually optimal. The naive bug uses an even tighter budget below.

    With budget 12 the expensive route reaches X with 0 remaining and cannot
    pay X->G (3); the state-only memo then marks X as "dead". The cheap route
        s0 --cheap(4)--> X --go(3)--> G     total 7 <= 12
    then finds X already marked bad and wrongly reports infeasible.
    """
    edges: Dict[str, Dict[str, Tuple[str, int]]] = {
        "s0": {"expensive": ("X", 12), "cheap": ("X", 4)},
        "X": {"go": ("G", 3)},
    }
    return GraphEnv(GraphSpec(start="s0", edges=edges, goals={"G"}))


def layered_env(
    layers: int = 5,
    width: int = 3,
    *,
    seed: int = 20260930,
    max_cost: int = 6,
) -> GraphEnv:
    """Layered finite environment used for the search-budget curves.

    States are ``(layer, column)``; layer 0 holds only the start (0, 0); the
    final layer contains a single guaranteed goal ``(layers-1, 0)`` while its
    other columns are dead ends. Edges mostly advance one layer; a fixed
    fraction add cheap backward/cycle edges. Action costs are positive ints.
    Built with a local seeded RNG so instances are reproducible.
    """
    rng = _random.Random(seed)
    edges: Dict[Tuple[int, int], Dict[str, Tuple[Tuple[int, int], int]]] = {}

    for layer in range(layers - 1):
        columns = 1 if layer == 0 else width
        for col in range(columns):
            state = (layer, col)
            outs: Dict[str, Tuple[Tuple[int, int], int]] = {}

            # Guaranteed forward edges, column 0 always has a cheap edge to
            # the next layer's column 0, ensuring the goal is reachable.
            n_forward = max(1, width)
            for target_col in range(min(width, n_forward)):
                if layer == 0 and target_col == 0:
                    cost = 1
                else:
                    cost = rng.randint(1, max_cost)
                outs[f"f{target_col}"] = ((layer + 1, target_col), cost)

            # Cheaper-looking backward/cycle edges (distinct-cost cycles).
            if layer >= 2 and rng.random() < 0.6:
                back_layer = rng.randint(0, layer - 1)
                back_col = rng.randrange(width)
                outs["back"] = ((back_layer, back_col), rng.randint(1, 3))
            if layer >= 1 and rng.random() < 0.5:
                outs["self"] = (state, rng.randint(1, 2))

            edges[state] = outs

    # Final layer: only column 0 is the goal; other columns are dead ends.
    assert (layers - 1, 0) not in edges
    return GraphEnv(
        GraphSpec(start=(0, 0), edges=edges, goals={(layers - 1, 0)})
    )
