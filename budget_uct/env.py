"""Deterministic, budgeted tool environments.

Contract
--------
* A state is a hashable Markov state: the future depends only on the state.
* ``actions(state)`` returns the legal actions at that state.
* ``step(state, action)`` is deterministic and returns ``(next_state, cost)``
  where ``cost`` is a strictly positive integer.
* ``is_success(state)`` decides whether a state is a terminal success.

The environment is allowed to contain transpositions (many action paths to the
same state) and cycles. A plan is valid only if (a) every action is legal in
sequence and (b) the accumulated integer cost does not exceed the execution
budget and (c) the final state is a success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, runtime_checkable
from typing import Protocol


Action = Hashable
State = Hashable
Transition = Tuple[State, int]


@dataclass(frozen=True)
class StepResult:
    next_state: State
    cost: int


@runtime_checkable
class Environment(Protocol):
    def initial_state(self) -> State: ...

    def actions(self, state: State) -> Sequence[Action]: ...

    def step(self, state: State, action: Action) -> StepResult: ...

    def is_success(self, state: State) -> bool: ...


@dataclass
class GraphSpec:
    """Explicit finite transition table.

    ``edges[state][action] = (next_state, cost)``. States without an entry are
    terminal (dead ends or goals). Costs must be strictly positive integers.
    """

    start: State
    edges: Dict[State, Dict[Action, Transition]]
    goals: set

    def validate(self) -> None:
        if self.start not in self.edges and self.start not in self.goals:
            raise ValueError(f"start state {self.start!r} has no transitions and is not a goal")
        for state, outs in self.edges.items():
            if not outs:
                raise ValueError(f"state {state!r} declares an empty action set")
            for action, (_, cost) in outs.items():
                if not isinstance(cost, int) or isinstance(cost, bool) or cost <= 0:
                    raise ValueError(
                        f"cost must be a strictly positive int, got {cost!r} on "
                        f"{state!r} --{action!r}-->"
                    )


class GraphEnv:
    """Environment backed by an explicit :class:`GraphSpec`."""

    def __init__(self, spec: GraphSpec, *, validate: bool = True):
        if validate:
            spec.validate()
        self.spec = spec

    def initial_state(self) -> State:
        return self.spec.start

    def actions(self, state: State) -> List[Action]:
        # Dict insertion order gives deterministic, reproducible action order.
        return list(self.spec.edges.get(state, {}).keys())

    def step(self, state: State, action: Action) -> StepResult:
        nxt, cost = self.spec.edges[state][action]
        return StepResult(nxt, cost)

    def is_success(self, state: State) -> bool:
        return state in self.spec.goals


class InstrumentedEnv:
    """Wraps an environment and meters real transition calls.

    ``step`` is backed by a per-state cache: the first query for a
    ``(state, action)`` pair is a *real* environment transition and consumes one
    unit of the search call budget; repeat queries are cache hits and never
    consume budget. This is what enforces "缓存命中不冒充新调用".
    """

    def __init__(self, env: Environment):
        self.env = env
        self._cache: Dict[Tuple[State, Action], StepResult] = {}
        self.real_calls = 0
        self.cache_hits = 0

    def initial_state(self) -> State:
        return self.env.initial_state()

    def actions(self, state: State) -> Sequence[Action]:
        # Listing legal actions is free; only step() is metered.
        return self.env.actions(state)

    def step(self, state: State, action: Action) -> StepResult:
        key = (state, action)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        result = self.env.step(state, action)
        if result.cost <= 0:
            raise ValueError("environment returned a non-positive cost")
        self._cache[key] = result
        self.real_calls += 1
        return result

    def is_success(self, state: State) -> bool:
        return self.env.is_success(state)

    def cached_step(self, state: State, action: Action):
        """Return cached (result, True) or (None, False); counts cache hits."""
        cached = self._cache.get((state, action))
        if cached is not None:
            self.cache_hits += 1
        return cached

    @property
    def calls(self) -> int:
        return self.real_calls


def execute_plan(
    env: Environment, start: State, plan: Sequence[Action]
) -> Tuple[State, int]:
    """Run a plan deterministically; raises on illegal actions.

    Returns ``(final_state, total_cost)``.
    """
    state = start
    total = 0
    for action in plan:
        legal = list(env.actions(state))
        if action not in legal:
            raise ValueError(f"illegal action {action!r} in state {state!r}")
        result = env.step(state, action)
        state = result.next_state
        total += result.cost
    return state, total
