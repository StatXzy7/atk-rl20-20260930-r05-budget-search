"""Deterministic memoized depth-first search used to expose the failure mode
of de-duplicating search nodes by environment state alone.

Two variants are provided:

* ``STATE_ONLY``    - memoizes failure per environment state. This is unsound
                      under an execution budget: a state first reached *after
                      spending a lot* can be marked "cannot reach the goal",
                      although the same state reached cheaply could.
* ``STATE_REMAIN``  - memoizes per ``(state, remaining_budget)``, which is
                      sound for Markov states with positive action costs.

The DFS is fully deterministic (action order fixed, most expensive first) so
the counterexample reproduces without randomness. Transition calls are metered
through ``InstrumentedEnv`` like every other algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .env import Action, Environment, InstrumentedEnv, State

STATE_ONLY = "state_only"
STATE_REMAIN = "state_remain"


@dataclass
class DFSResult:
    found: bool
    plan: Optional[List[Action]]
    cost: Optional[int]
    real_calls: int
    cache_hits: int
    dedup: str


def memo_dfs(
    env: Environment,
    budget: int,
    *,
    dedup: str = STATE_ONLY,
    expensive_first: bool = True,
) -> DFSResult:
    if dedup not in (STATE_ONLY, STATE_REMAIN):
        raise ValueError("dedup must be STATE_ONLY or STATE_REMAIN")
    ienv = env if isinstance(env, InstrumentedEnv) else InstrumentedEnv(env)
    start = ienv.initial_state()
    bad: Set[object] = set()
    on_stack: Set[object] = set()

    def key(state: State, remaining: int):
        return state if dedup == STATE_ONLY else (state, remaining)

    def order(state: State) -> List[Action]:
        actions = list(ienv.actions(state))
        if expensive_first:
            # Probe costs (charging the instrumented env) and sort descending.
            priced = []
            for action in actions:
                result = ienv.step(state, action)
                priced.append((result.cost, action))
            priced.sort(key=lambda ca: (-ca[0], str(ca[1])))
            return [a for _, a in priced]
        return actions

    def dfs(state: State, remaining: int) -> Optional[List[Action]]:
        if ienv.is_success(state):
            return []
        k = key(state, remaining)
        if k in bad or k in on_stack:
            return None
        on_stack.add(k)
        plan: Optional[List[Action]] = None
        for action in order(state):
            result = ienv.step(state, action)
            if result.cost > remaining:
                continue
            sub = dfs(result.next_state, remaining - result.cost)
            if sub is not None:
                plan = [action] + sub
                break
        on_stack.discard(k)
        if plan is None:
            bad.add(k)  # STATE_ONLY: poisoned for EVERY remaining budget
        return plan

    plan_actions = dfs(start, budget)
    cost: Optional[int] = None
    if plan_actions is not None:
        state = start
        total = 0
        for action in plan_actions:
            result = ienv.step(state, action)
            state = result.next_state
            total += result.cost
        cost = total
    return DFSResult(
        found=plan_actions is not None,
        plan=plan_actions,
        cost=cost,
        real_calls=ienv.real_calls,
        cache_hits=ienv.cache_hits,
        dedup=dedup,
    )
