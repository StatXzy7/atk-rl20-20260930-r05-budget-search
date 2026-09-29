"""Independent exact shortest-cost solver (Dijkstra).

The solver shares no code or state with the UCT agent other than the
environment contract.  Costs are strictly positive integers, so Dijkstra over
environment states (edges expanded lazily through the same call-counted view)
returns the exact minimum cumulative cost, or proves no successful state is
reachable within ``budget``.

Calls to ``env.step`` are charged through the provided :class:`EnvironmentView`
so experiments can report the oracle's own transition-call usage.  A safety
cap ``max_states`` bounds the oracle on large instances; when the cap is hit
before exhausting the frontier the result is explicitly *not* a proof.
"""

from __future__ import annotations

import heapq
from typing import Any, Hashable, NamedTuple

from .environment import EnvironmentView


class ExactResult(NamedTuple):
    feasible: bool
    best_actions: tuple | None
    best_cost: int | None
    proven: bool
    expanded_states: int
    billed_calls: int
    stop_reason: str


class ShortestCostSolver:
    def __init__(self, view: EnvironmentView, budget: int, max_states: int = 1_000_000):
        self.view = view
        self.budget = budget
        self.max_states = max_states

    def solve(self) -> ExactResult:
        start = self.view.initial_state
        dist: dict[Hashable, int] = {start: 0}
        prev: dict[Hashable, tuple[Hashable, Any]] = {}
        settled: set[Hashable] = set()
        frontier: list[tuple[int, int, Hashable]] = [(0, 0, start)]
        tie = 1
        capped = False

        while frontier:
            cost, _order, state = heapq.heappop(frontier)
            if state in settled:
                continue
            if self.view.is_success(state):
                plan = self._reconstruct(prev, start, state)
                return ExactResult(
                    feasible=True,
                    best_actions=plan,
                    best_cost=cost,
                    proven=True,
                    expanded_states=len(settled),
                    billed_calls=self.view.billed_calls,
                    stop_reason="optimal_found",
                )
            settled.add(state)
            if len(settled) >= self.max_states:
                capped = True
                break
            for action in self.view.legal_actions(state):
                outcome = self.view.step(state, action)
                next_cost = cost + outcome.cost
                if next_cost > self.budget:
                    continue
                nxt = outcome.state
                if next_cost < dist.get(nxt, self.budget + 1):
                    dist[nxt] = next_cost
                    prev[nxt] = (state, action)
                    heapq.heappush(frontier, (next_cost, tie, nxt))
                    tie += 1

        if capped:
            return ExactResult(
                feasible=False,
                best_actions=None,
                best_cost=None,
                proven=False,
                expanded_states=len(settled),
                billed_calls=self.view.billed_calls,
                stop_reason="state_cap_reached",
            )
        return ExactResult(
            feasible=False,
            best_actions=None,
            best_cost=None,
            proven=True,
            expanded_states=len(settled),
            billed_calls=self.view.billed_calls,
            stop_reason="frontier_exhausted",
        )

    @staticmethod
    def _reconstruct(
        prev: dict, start: Hashable, target: Hashable
    ) -> tuple:
        actions: list[Any] = []
        state = target
        while state != start:
            parent, action = prev[state]
            actions.append(action)
            state = parent
        actions.reverse()
        return tuple(actions)
