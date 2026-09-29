"""State-only node merging must demonstrably produce wrong answers."""

from __future__ import annotations

import unittest

from uctshortest.environment import EnvironmentView, PlanError
from uctshortest.envs import (
    dedup_false_negative_env,
    dedup_false_positive_env,
)
from uctshortest.exact import ShortestCostSolver
from uctshortest.naive import NaiveStateOnlyUCTAgent
from uctshortest.uct import UCTAgent


class FalseNegativeTests(unittest.TestCase):
    def test_oracle_optimum_is_four(self):
        oracle = ShortestCostSolver(
            EnvironmentView(dedup_false_negative_env()), 6
        ).solve()
        self.assertTrue(oracle.proven)
        self.assertEqual(oracle.best_cost, 4)
        self.assertEqual(oracle.best_actions, ("c", "g"))

    def test_correct_uct_finds_optimum(self):
        env = dedup_false_negative_env()
        result = UCTAgent(EnvironmentView(env), 6, seed=0).search(20)
        self.assertTrue(result.found)
        self.assertEqual(result.best_cost, 4)

    def test_state_only_merge_reports_no_plan_false_negative(self):
        env = dedup_false_negative_env()
        agent = NaiveStateOnlyUCTAgent(EnvironmentView(env), 6, seed=0)
        result = agent.search(20)
        self.assertFalse(
            result.found,
            "state-only merging missed the affordable c->m arrival and "
            "wrongly reported infeasibility",
        )
        m_entries = [k for k in agent.nodes if k[0] == "m"]
        self.assertEqual(len(m_entries), 1, "states must be merged by the bug")


class FalsePositiveTests(unittest.TestCase):
    def test_correct_uct_rejects_over_budget_prefix(self):
        env = dedup_false_positive_env()
        agent = UCTAgent(EnvironmentView(env), 6, seed=0)
        agent.search(20)
        self.assertIsNone(agent.plan_for_prefix(("b",)))

    def test_state_only_merge_certifies_over_budget_plan(self):
        env = dedup_false_positive_env()
        agent = NaiveStateOnlyUCTAgent(EnvironmentView(env), 6, seed=0)
        agent.search(20)
        bogus = agent.plan_for_prefix(("b",))
        self.assertEqual(
            bogus,
            ("b", "g"),
            "the merged node certifies continuation g from the frozen "
            "cheap arrival although the b arrival cannot afford it",
        )
        with self.assertRaises(PlanError):
            EnvironmentView(env).validate_plan(bogus, 6)

    def test_naive_global_best_remains_feasible(self):
        env = dedup_false_positive_env()
        result = NaiveStateOnlyUCTAgent(EnvironmentView(env), 6, seed=0).search(
            20
        )
        self.assertEqual(result.best_actions, ("c", "g"))
        self.assertEqual(result.best_cost, 6)


if __name__ == "__main__":
    unittest.main()
