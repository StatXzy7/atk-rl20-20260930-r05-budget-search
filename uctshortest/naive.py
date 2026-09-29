"""Deliberately buggy UCT variant: nodes are keyed by environment state only.

This file exists to demonstrate the exact failure the task warns about:
"reaching the same environment state with different remaining execution
budgets must not be unconditionally merged into one search node".

Here every environment state maps to a single node, whose remaining budget is
frozen at the *first* arrival.  Later arrivals (reachable with a different
cumulative cost) silently inherit that frozen budget.  Two failure modes are
exercised in the tests:

* **false negative** -- the first (expensive) arrival cannot afford the
  continuation, so the merged node is marked hopeless; a later cheaper arrival
  that *can* afford it is missed;
* **false positive** -- the first (cheap) arrival can afford the continuation;
  a later expensive arrival certifies a completion it cannot pay for.
"""

from __future__ import annotations

from typing import Any, Hashable

from .environment import EnvironmentView
from .uct import UCTAgent, UCTConfig


class NaiveStateOnlyUCTAgent(UCTAgent):
    def __init__(
        self,
        view: EnvironmentView,
        budget: int,
        config: UCTConfig | None = None,
        seed: int = 0,
    ) -> None:
        self._state_to_key: dict[Hashable, tuple[Hashable, int]] = {
            view.initial_state: (view.initial_state, budget)
        }
        super().__init__(view, budget, config=config, seed=seed)

    def _key_for(self, state: Hashable, remaining: int) -> tuple[Hashable, int]:
        existing = self._state_to_key.get(state)
        if existing is not None:
            return existing
        key = (state, remaining)
        self._state_to_key[state] = key
        return key

    def _lookup_child(self, key: tuple[Hashable, int]) -> "Node | None":
        state, _remaining = key
        frozen = self._state_to_key.get(state)
        if frozen is not None:
            return self.nodes.get(frozen)
        return None

    def _make_node(self, key: tuple[Hashable, int]) -> "Node":
        state, _remaining = key
        existing = self._state_to_key.get(state)
        if existing is not None and existing in self.nodes:
            return self.nodes[existing]
        self._state_to_key[state] = key
        return super()._make_node(key)

    def plan_for_prefix(self, prefix: tuple | list[Any]) -> tuple | None:
        """Buggy version: completion uses the merged node's frozen budget."""
        state = self.view.initial_state
        for action in prefix:
            if not self.view.has_cached_step(state, action):
                return None
            state, _cost = self.view.step(state, action)
        merged = self._state_to_key.get(state)
        if merged is None:
            return None
        frozen_remaining = merged[1]
        suffix = self._completion_from(state, frozen_remaining, frozenset(), ())
        return None if suffix is None else tuple(prefix) + suffix
