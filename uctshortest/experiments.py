"""Budget-curve experiments comparing UCT against the exact oracle.

Runs the correct DAG-UCT agent over a sweep of *real transition call* budgets
and many seeds, reporting: fraction of runs that discover any feasible plan,
fraction returning the proven optimum, average optimality gap
(``best_cost - optimal``, undefined when no plan is found), and cache-hit
behaviour.  The exact solver is run independently on a fresh view to obtain
the optimum and its own call count.

Results are written as CSV plus a short Markdown summary; no external models
or plotting dependencies are required.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .environment import Environment, EnvironmentView, PlanError
from .exact import ShortestCostSolver
from .uct import UCTAgent, UCTConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    builder: Callable[[], Environment]
    budget: int
    budget_curve: tuple[int, ...]


def _validate(env: Environment, actions: tuple, budget: int) -> int:
    return EnvironmentView(env, cache_enabled=False).validate_plan(
        actions, budget
    )


def run_curve(
    scenario: Scenario,
    seeds: int = 20,
    c_ucb: float = 1.4142135623730951,
) -> tuple[list[dict], dict]:
    oracle_view = EnvironmentView(scenario.builder())
    oracle = ShortestCostSolver(oracle_view, scenario.budget).solve()
    rows: list[dict] = []
    for call_budget in scenario.budget_curve:
        found_flags: list[int] = []
        optimal_flags: list[int] = []
        miss_flags: list[int] = []
        gaps: list[int] = []
        billed: list[int] = []
        cache_hits: list[int] = []
        for seed in range(seeds):
            env = scenario.builder()
            view = EnvironmentView(env)
            agent = UCTAgent(
                view,
                scenario.budget,
                config=UCTConfig(c_ucb=c_ucb),
                seed=seed,
            )
            result = agent.search(call_budget)
            found_flags.append(int(result.found))
            miss_flags.append(0 if result.found else 1)
            if result.found:
                cost = _validate(env, result.best_actions, scenario.budget)
                if cost != result.best_cost:
                    raise PlanError(
                        f"reported cost {result.best_cost} != executed {cost}"
                    )
                gaps.append(cost - oracle.best_cost)
                optimal_flags.append(int(cost == oracle.best_cost))
            else:
                optimal_flags.append(0)
            billed.append(result.billed_calls)
            cache_hits.append(result.cache_hits)
        rows.append(
            {
                "scenario": scenario.name,
                "call_budget": call_budget,
                "seeds": seeds,
                "find_rate": statistics.mean(found_flags),
                "optimal_rate": statistics.mean(optimal_flags),
                "mean_gap_when_found": (
                    statistics.mean(gaps) if gaps else ""
                ),
                "max_gap": max(gaps) if gaps else "",
                "mean_billed": statistics.mean(billed),
                "mean_cache_hits": statistics.mean(cache_hits),
            }
        )
    summary = {
        "scenario": scenario.name,
        "budget": scenario.budget,
        "oracle_feasible": oracle.feasible,
        "oracle_optimal_cost": oracle.best_cost,
        "oracle_best_actions": oracle.best_actions,
        "oracle_billed_calls": oracle.billed_calls,
        "oracle_proven": oracle.proven,
    }
    return rows, summary


def write_outputs(
    rows: list[dict],
    summaries: list[dict],
    out_dir: str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "call_budget",
        "seeds",
        "find_rate",
        "optimal_rate",
        "mean_gap_when_found",
        "max_gap",
        "mean_billed",
        "mean_cache_hits",
    ]
    with (out / "budget_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Search budget curve (20 seeds per point)", ""]
    for summary in summaries:
        lines.append(f"## {summary['scenario']}")
        lines.append(
            "- oracle: feasible="
            f"{summary['oracle_feasible']}, optimal_cost="
            f"{summary['oracle_optimal_cost']}, plan="
            f"{summary['oracle_best_actions']}, billed_calls="
            f"{summary['oracle_billed_calls']}, proven="
            f"{summary['oracle_proven']}"
        )
        lines.append("")
        lines.append(
            "| call budget | found rate | optimal rate | mean gap "
            "(if found) | max gap | mean billed | mean cache hits |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            if row["scenario"] != summary["scenario"]:
                continue
            lines.append(
                f"| {row['call_budget']} | {row['find_rate']:.2f} "
                f"| {row['optimal_rate']:.2f} "
                f"| {row['mean_gap_when_found']} | {row['max_gap']} "
                f"| {row['mean_billed']:.1f} | {row['mean_cache_hits']:.1f} |"
            )
        lines.append("")
    (out / "budget_curve.md").write_text("\n".join(lines))
