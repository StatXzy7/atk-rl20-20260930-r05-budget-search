"""Hand-built small environments used as oracles in tests and experiments.

Every environment is deterministic, has strictly positive integer action
costs and hashable states, and may contain transpositions, trap branches and
cycles with differing costs.
"""

from __future__ import annotations

from .table import TableEnvironment


def transpose_env() -> TableEnvironment:
    """Two action paths merge at ``m`` with different prefix costs.

    ``a`` (cost 3) and ``b,d`` (cost 1+2=3) reach ``m``; ``c`` (cost 1)
    reaches ``m`` for total 1, exposing a genuine transposition.  Optimum to
    ``g`` is 4 via ``c,g``.
    """
    return TableEnvironment(
        transitions={
            "s": {"a": ("x", 3), "b": ("y", 1), "c": ("m", 1)},
            "x": {"g": ("g", 2)},
            "y": {"d": ("m", 2)},
            "m": {"g": ("g", 3)},
            "g": {},
        },
        success={"g"},
        initial_state="s",
    )


def trap_env() -> TableEnvironment:
    """Cheap-looking branch leads into a dead trap; honest branch costs 4."""
    return TableEnvironment(
        transitions={
            "s": {"cheap": ("t", 1), "honest": ("h", 2)},
            "t": {"wiggle": ("t", 1)},
            "h": {"finish": ("g", 2)},
            "g": {},
        },
        success={"g"},
        initial_state="s",
    )


def cycles_env() -> TableEnvironment:
    """Two self-loops with different costs plus a real exit.

    Cheap loop ``spin1`` (1) and expensive loop ``spin5`` (5) return to the
    same state; ``go`` reaches the goal for cost 3.  With budget 7 a naive
    agent can burn budget on loops; optimum is 3 via ``go``.
    """
    return TableEnvironment(
        transitions={
            "s": {"spin1": ("s", 1), "spin5": ("s", 5), "go": ("g", 3)},
            "g": {},
        },
        success={"g"},
        initial_state="s",
    )


def combined_env() -> TableEnvironment:
    """Transpositions, a trap branch and unequal-cost cycles together."""
    return TableEnvironment(
        transitions={
            "s": {"a": ("x", 3), "b": ("y", 1), "c": ("m", 1), "trap": ("t", 1)},
            "x": {"g": ("g", 2), "back": ("s", 2)},
            "y": {"d": ("m", 2), "loop": ("y", 1)},
            "m": {"g": ("g", 3), "expensive_loop": ("m", 4)},
            "t": {"wiggle": ("t", 1)},
            "g": {},
        },
        success={"g"},
        initial_state="s",
    )


def infeasible_env() -> TableEnvironment:
    """Goal is reachable only for cost 6; budget 5 makes it infeasible."""
    return TableEnvironment(
        transitions={
            "s": {"a": ("x", 1), "b": ("y", 2)},
            "x": {"g": ("g", 5)},
            "y": {"g": ("g", 4)},
            "g": {},
        },
        success={"g"},
        initial_state="s",
    )


INFEASIBLE_BUDGET = 5


def dedup_false_negative_env() -> TableEnvironment:
    """Counterexample: state-only merging hides the affordable arrival.

    First UCT expansion takes the expensive action ``b`` (cost 5) to ``m``;
    the merged node freezes remaining budget 1 and marks continuation ``g``
    (cost 2) as an over-budget leaf.  The cheaper action ``c`` (cost 2) later
    reaches the same state ``m`` with 4 remaining, which *can* afford ``g``
    (optimum 4), but the merged node never re-opens that edge: the buggy
    search reports no plan even though the optimum is 4.
    """
    return TableEnvironment(
        transitions={
            "s": {"b": ("m", 5), "c": ("m", 2)},
            "m": {"g": ("g", 2)},
            "g": {},
        },
        success={"g"},
        initial_state="s",
    )


def dedup_false_positive_env() -> TableEnvironment:
    """Counterexample: state-only merging certifies an over-budget plan.

    Cheap action ``c`` (cost 2) reaches ``m`` first with 4 remaining; ``g``
    costs 4, so ``c,g`` is feasible (cost 6).  Expensive action ``b`` (cost 5)
    later reaches ``m`` with only 1 remaining and must not be allowed ``g``;
    state-only merging still answers "completion g is known from m", wrongly
    certifying ``b,g`` (cost 9 > budget 6).
    """
    return TableEnvironment(
        transitions={
            "s": {"c": ("m", 2), "b": ("m", 5)},
            "m": {"g": ("g", 4)},
            "g": {},
        },
        success={"g"},
        initial_state="s",
    )
