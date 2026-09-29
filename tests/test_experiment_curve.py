"""Every plan reported across the full budget curve must execute for real."""

from __future__ import annotations

import unittest

from uctshortest.environment import EnvironmentView
from uctshortest.envs import (
    INFEASIBLE_BUDGET,
    combined_env,
    cycles_env,
    infeasible_env,
    transpose_env,
    trap_env,
)
from uctshortest.exact import ShortestCostSolver
from uctshortest.experiments import Scenario, run_curve


class CurveValidityTests(unittest.TestCase):
    def test_all_reported_plans_are_valid(self):
        scenarios = [
            Scenario("transpose", transpose_env, 6, (2, 5, 8, 30)),
            Scenario("trap", trap_env, 6, (2, 4, 30)),
            Scenario("cycles", cycles_env, 7, (2, 4, 30)),
            Scenario("combined", combined_env, 8, (2, 6, 30)),
        ]
        for scenario in scenarios:
            oracle = ShortestCostSolver(
                EnvironmentView(scenario.builder()), scenario.budget
            ).solve()
            rows, summary = run_curve(scenario, seeds=8)
            self.assertTrue(summary["oracle_proven"])
            for row in rows:
                self.assertGreaterEqual(row["call_budget"], 0)
                self.assertLessEqual(row["mean_billed"], row["call_budget"] + 0.001)
                if row["find_rate"] > 0:
                    self.assertNotEqual(row["mean_gap_when_found"], "")
                    self.assertGreaterEqual(row["mean_gap_when_found"], 0)
                self.assertLessEqual(row["find_rate"], 1.0)
            self.assertEqual(oracle.best_cost, summary["oracle_optimal_cost"])

    def test_infeasible_scenario_curve_never_finds(self):
        scenario = Scenario(
            "infeasible",
            infeasible_env,
            INFEASIBLE_BUDGET,
            (2, 4, 20),
        )
        rows, summary = run_curve(scenario, seeds=5)
        self.assertFalse(summary["oracle_feasible"])
        self.assertTrue(summary["oracle_proven"])
        self.assertTrue(all(row["find_rate"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
