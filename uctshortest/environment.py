"""Environment contract and a call-counted, caching environment view.

The environment is deterministic and exposes:

* hashable Markov states;
* ``legal_actions(state)``;
* deterministic ``step(state, action) -> (next_state, cost)`` with strictly
  positive integer costs;
* ``is_success(state)``.

Search budgets are charged in *real* transition calls only.  A cached
transition must never be billed again, and roll-out transitions are billed
exactly the same way as tree-expansion transitions.
"""

from __future__ import annotations

from typing import Any, Hashable, NamedTuple, Sequence


class StepOutcome(NamedTuple):
    state: Hashable
    cost: int


class PlanError(Exception):
    """Raised when an alleged plan is not executable or exceeds the budget."""


class Environment:
    """Abstract deterministic tool environment.

    Subclasses implement :meth:`legal_actions`, :meth:`step`,
    :meth:`is_success` and :attr:`initial_state`.  Costs returned by
    :meth:`step` must be strictly positive integers.
    """

    initial_state: Hashable

    def legal_actions(self, state: Hashable) -> Sequence[Any]:
        raise NotImplementedError

    def step(self, state: Hashable, action: Any) -> StepOutcome:
        raise NotImplementedError

    def is_success(self, state: Hashable) -> bool:
        raise NotImplementedError

    def _check_contract(self, next_state: Hashable, cost: int) -> None:
        if not isinstance(cost, int) or isinstance(cost, bool) or cost <= 0:
            raise ValueError(f"cost must be a strictly positive int, got {cost!r}")
        try:
            hash(next_state)
        except TypeError as exc:
            raise TypeError("environment states must be hashable") from exc


class EnvironmentView:
    """Wraps an environment with call accounting and transition caching.

    Attributes:
        billed_calls: number of real ``env.step`` calls performed through
            :meth:`step` since construction (or the last :meth:`restore_state`).
        cache_hits: number of :meth:`step` requests served from the cache.
        cache_enabled: when ``False`` every request hits the real environment
            (used by tests proving that simulation is billed).
    """

    def __init__(
        self,
        env: Environment,
        cache_enabled: bool = True,
        billed_calls: int = 0,
        cache_hits: int = 0,
        transition_cache: dict | None = None,
        legal_cache: dict | None = None,
        success_cache: dict | None = None,
    ) -> None:
        self.env = env
        self.cache_enabled = cache_enabled
        self.billed_calls = billed_calls
        self.cache_hits = cache_hits
        self._transition_cache: dict = (
            transition_cache if transition_cache is not None else {}
        )
        self._legal_cache: dict = legal_cache if legal_cache is not None else {}
        self._success_cache: dict = success_cache if success_cache is not None else {}

    @property
    def initial_state(self) -> Hashable:
        return self.env.initial_state

    def legal_actions(self, state: Hashable) -> Sequence[Any]:
        if state not in self._legal_cache:
            actions = tuple(self.env.legal_actions(state))
            self._legal_cache[state] = actions
            return actions
        return self._legal_cache[state]

    def is_success(self, state: Hashable) -> bool:
        cached = self._success_cache.get(state)
        if cached is None:
            cached = bool(self.env.is_success(state))
            self._success_cache[state] = cached
        return cached

    def step(self, state: Hashable, action: Any) -> StepOutcome:
        key = (state, action)
        if self.cache_enabled and key in self._transition_cache:
            self.cache_hits += 1
            return self._transition_cache[key]
        outcome = StepOutcome(*self.env.step(state, action))
        self.env._check_contract(outcome.state, outcome.cost)
        self._transition_cache[key] = outcome
        self.billed_calls += 1
        return outcome

    def has_cached_step(self, state: Hashable, action: Any) -> bool:
        return (state, action) in self._transition_cache

    def validate_plan(
        self,
        actions: Sequence[Any],
        budget: int,
        billed_calls_before: int | None = None,
        use_cache: bool = False,
    ) -> int:
        """Re-execute ``actions`` against the real environment.

        Returns the total cost.  Raises :class:`PlanError` if an action is not
        legal, a transition fails, the terminal state is not successful or the
        cumulative cost exceeds ``budget``.
        """
        state = self.env.initial_state
        total = 0
        for index, action in enumerate(actions):
            if action not in tuple(self.env.legal_actions(state)):
                raise PlanError(f"action {action!r} at position {index} is not legal")
            if billed_calls_before is not None:
                before = self.billed_calls
            outcome = self.step(state, action) if use_cache else StepOutcome(
                *self.env.step(state, action)
            )
            if billed_calls_before is not None and not use_cache:
                if self.billed_calls != before + 1:
                    raise PlanError("validation transition was not billed correctly")
            next_state, cost = outcome
            total += cost
            if total > budget:
                raise PlanError(
                    f"plan exceeds budget after {index + 1} actions: {total} > {budget}"
                )
            state = next_state
        if not self.env.is_success(state):
            raise PlanError("plan ends in a non-success state")
        return total

    def snapshot_state(self) -> dict:
        """Return a saveable, restorable state of this view.

        Only stdlib types are used (dict/tuple/int) so the snapshot can be
        pickled or JSON-serialised for environments with primitive states.
        """
        return {
            "billed_calls": self.billed_calls,
            "cache_hits": self.cache_hits,
            "transition_cache": dict(self._transition_cache),
            "legal_cache": dict(self._legal_cache),
            "success_cache": dict(self._success_cache),
        }

    def restore_state(self, snapshot: dict) -> None:
        self.billed_calls = snapshot["billed_calls"]
        self.cache_hits = snapshot["cache_hits"]
        self._transition_cache = dict(snapshot["transition_cache"])
        self._legal_cache = dict(snapshot["legal_cache"])
        self._success_cache = dict(snapshot["success_cache"])
