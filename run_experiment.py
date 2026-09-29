"""Entry point: run all budget-curve experiments and write results/."""

from __future__ import annotations

from uctshortest.envs import (
    combined_env,
    cycles_env,
    dedup_false_negative_env,
    dedup_false_positive_env,
    infeasible_env,
    INFEASIBLE_BUDGET,
    transpose_env,
    trap_env,
)
from uctshortest.experiments import Scenario, run_curve, write_outputs

SCENARIOS = [
    Scenario("transpose", transpose_env, 6, (2, 3, 4, 5, 6, 8, 10, 14, 20, 30)),
    Scenario("trap", trap_env, 6, (2, 3, 4, 5, 6, 8, 10, 14, 20, 30)),
    Scenario("cycles", cycles_env, 7, (2, 3, 4, 5, 6, 8, 10, 14, 20, 30)),
    Scenario("combined", combined_env, 8, (2, 3, 4, 5, 6, 8, 10, 14, 20, 30)),
    Scenario("infeasible", infeasible_env, INFEASIBLE_BUDGET, (2, 4, 6, 8, 12, 20)),
    Scenario(
        "dedup_false_negative",
        dedup_false_negative_env,
        6,
        (2, 3, 4, 5, 6, 8, 12),
    ),
    Scenario(
        "dedup_false_positive",
        dedup_false_positive_env,
        6,
        (2, 3, 4, 5, 6, 8, 12),
    ),
]


def main() -> None:
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for scenario in SCENARIOS:
        rows, summary = run_curve(scenario, seeds=20)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{scenario.name}: optimum={summary['oracle_optimal_cost']} "
            f"oracle_calls={summary['oracle_billed_calls']}"
        )
    write_outputs(all_rows, summaries, "results")
    print("wrote results/budget_curve.csv and results/budget_curve.md")


if __name__ == "__main__":
    main()
