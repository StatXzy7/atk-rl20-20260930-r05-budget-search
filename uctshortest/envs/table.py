"""Small deterministic table-driven environments for the experiments."""

from __future__ import annotations

from typing import Any, Hashable, Mapping, Sequence

from ..environment import Environment, StepOutcome


class TableEnvironment(Environment):
    """Environment defined by explicit transition and success tables.

    ``transitions[state][action] = (next_state, cost)``.  Costs are validated
    (strictly positive ints) on first use through :class:`EnvironmentView`.
    """

    def __init__(
        self,
        transitions: Mapping[Hashable, Mapping[Any, tuple[Hashable, int]]],
        success: set[Hashable],
        initial_state: Hashable,
    ) -> None:
        self._transitions = {s: dict(m) for s, m in transitions.items()}
        self._success = set(success)
        self.initial_state = initial_state
        for table in self._transitions.values():
            for next_state, cost in table.values():
                self._check_contract(next_state, cost)

    def legal_actions(self, state: Hashable) -> Sequence[Any]:
        table = self._transitions.get(state)
        if not table:
            return ()
        return tuple(table.keys())

    def step(self, state: Hashable, action: Any) -> StepOutcome:
        next_state, cost = self._transitions[state][action]
        return StepOutcome(next_state, cost)

    def is_success(self, state: Hashable) -> bool:
        return state in self._success
