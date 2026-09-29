"""The exact solver must find the proven optimum or prove infeasibility."""

from __future__ import annotations

import unittest

from uctshortest.environment import EnvironmentView
from uctshortest.envs import (
    INFEASIBLE_BUDGET,
    combined_env,
    cycles_env,
    dedup_false_negative_env,
    dedup_false_positive_env,
    infeasible_env,
    transpose_env,
    trap_env,
)
from uctshortest.exact import ShortestCostSolver


class ExactSolverTests(unittest.TestCase):
    def _check_optimal(self, env, budget, expected_actions, expected_cost):
        view = EnvironmentView(env)
        result = ShortestCostSolver(view, budget).solve()
        self.assertTrue(result.feasible)
        self.assertTrue(result.proven)
        self.assertEqual(result.best_cost, expected_cost)
        executed_cost = view.validate_plan(result.best_actions, budget)
        self.assertEqual(executed_cost, expected_cost)
        self.assertEqual(list(result.best_actions), list(expected_actions))

    def test_transpose_optimum(self):
        self._check_optimal(transpose_env(), 6, ["c", "g"], 4)

    def test_trap_optimum(self):
        self._check_optimal(trap_env(), 6, ["honest", "finish"], 4)

    def test_cycles_optimum(self):
        self._check_optimal(cycles_env(), 7, ["go"], 3)

    def test_combined_optimum_uses_transposition(self):
        self._check_optimal(combined_env(), 8, ["c", "g"], 4)

    def test_dedup_scenarios_optimum(self):
        self._check_optimal(dedup_false_negative_env(), 6, ["c", "g"], 4)
        self._check_optimal(dedup_false_positive_env(), 6, ["c", "g"], 6)

    def test_infeasible_proven(self):
        view = EnvironmentView(infeasible_env())
        result = ShortestCostSolver(view, INFEASIBLE_BUDGET).solve()
        self.assertFalse(result.feasible)
        self.assertTrue(result.proven)
        self.assertEqual(result.stop_reason, "frontier_exhausted")

    def test_dijkstra_independent_call_accounting(self):
        view = EnvironmentView(cycles_env())
        before = view.billed_calls
        ShortestCostSolver(view, 7).solve()
        self.assertGreater(view.billed_calls, before)

    def test_state_cap_is_not_a_proof(self):
        view = EnvironmentView(combined_env())
        result = ShortestCostSolver(view, 8, max_states=1).solve()
        self.assertFalse(result.proven)
        self.assertEqual(result.stop_reason, "state_cap_reached")


if __name__ == "__main__":
    unittest.main()
