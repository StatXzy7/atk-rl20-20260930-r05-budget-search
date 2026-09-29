"""Independent exact minimum-cost solver.

This solver is deliberately *independent* of the UCT machinery: it uses
Dijkstra's algorithm over the state graph with the positive action costs as
edge weights, restricted to cumulative cost <= budget. Because every edge
weight is a strictly positive integer, Dijkstra both:

* finds a minimum-cost successful plan when one exists within the budget, and
* proves none exists within the budget after settling every state reachable
  under the budget (a success is never settled => infeasible).

It also records transition calls through :class:`InstrumentedEnv`, so the
exact oracle's own cost (real environment transitions vs. cache hits) is
measured under the same accounting rules as UCT.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .env import Action, Environment, InstrumentedEnv, State

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass
class ExactResult:
    status: str
    plan: Optional[List[Action]]
    cost: Optional[int]
    optimal_cost: Optional[int]
    real_calls: int
    cache_hits: int
    states_settled: int
    proven: bool

    @property
    def found(self) -> bool:
        return self.status == OPTIMAL


def solve_exact(
    env: Environment,
    budget: int,
    *,
    call_budget: Optional[int] = None,
    start: Optional[State] = None,
) -> ExactResult:
    """Return a minimum-cost valid plan within ``budget``.

    ``call_budget`` (number of real transition calls) is optional. If supplied
    and exhausted before the search settles, the result is
    ``budget_exhausted`` and must not be read as an infeasibility proof.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")

    ienv = env if isinstance(env, InstrumentedEnv) else InstrumentedEnv(env)
    root = start if start is not None else ienv.initial_state()

    # dist[state] = cheapest known cost to reach state.
    dist: Dict[State, int] = {root: 0}
    prev: Dict[State, Tuple[State, Action]] = {}
    # Heap entries are tie-broken by a monotone counter for determinism.
    counter = 0
    heap: List[Tuple[int, int, State]] = [(0, counter, root)]
    settled: Dict[State, int] = {}
    goal: Optional[State] = None

    def can_call() -> bool:
        return call_budget is None or ienv.real_calls < call_budget

    while heap:
        path_cost, _, state = heapq.heappop(heap)
        if path_cost != dist.get(state):
            continue  # stale heap entry
        if state in settled:
            continue
        settled[state] = path_cost
        if ienv.is_success(state):
            goal = state
            break

        for action in ienv.actions(state):
            if not can_call():
                heap = []  # force stop; result below is non-proof
                break
            result = ienv.step(state, action)
            new_cost = path_cost + result.cost
            if new_cost > budget:
                continue
            nxt = result.next_state
            if new_cost < dist.get(nxt, budget + 1):
                dist[nxt] = new_cost
                prev[nxt] = (state, action)
                counter += 1
                heapq.heappush(heap, (new_cost, counter, nxt))

    real_calls = ienv.real_calls
    cache_hits = ienv.cache_hits
    n_settled = len(settled)

    if goal is not None:
        optimal_cost = settled[goal]
        plan = _reconstruct(prev, root, goal)
        return ExactResult(
            status=OPTIMAL,
            plan=plan,
            cost=optimal_cost,
            optimal_cost=optimal_cost,
            real_calls=real_calls,
            cache_hits=cache_hits,
            states_settled=n_settled,
            proven=True,
        )

    if not can_call():
        return ExactResult(
            status=BUDGET_EXHAUSTED,
            plan=None,
            cost=None,
            optimal_cost=None,
            real_calls=real_calls,
            cache_hits=cache_hits,
            states_settled=n_settled,
            proven=False,
        )

    # Heap fully drained under the budget without settling a success.
    return ExactResult(
        status=INFEASIBLE,
        plan=None,
        cost=None,
        optimal_cost=None,
        real_calls=real_calls,
        cache_hits=cache_hits,
        states_settled=n_settled,
        proven=True,
    )


def _reconstruct(
    prev: Dict[State, Tuple[State, Action]], root: State, goal: State
) -> List[Action]:
    plan: List[Action] = []
    state = goal
    while state != root:
        parent, action = prev[state]
        plan.append(action)
        state = parent
    plan.reverse()
    return plan
