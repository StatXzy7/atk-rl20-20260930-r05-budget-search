"""UCT correctness: executable plans, budgets, optimality gap, billing."""

from __future__ import annotations

import unittest

from uctshortest.environment import EnvironmentView, PlanError
from uctshortest.envs import (
    INFEASIBLE_BUDGET,
    combined_env,
    cycles_env,
    infeasible_env,
    transpose_env,
    trap_env,
)
from uctshortest.exact import ShortestCostSolver
from uctshortest.uct import UCTAgent, UCTConfig


def oracle_optimum(env, budget):
    view = EnvironmentView(env)
    result = ShortestCostSolver(view, budget).solve()
    return result.best_cost


class UCTPlanTests(unittest.TestCase):
    def assert_plan_valid(self, env, result, budget):
        self.assertTrue(result.found)
        executed = EnvironmentView(env, cache_enabled=False).validate_plan(
            result.best_actions, budget
        )
        self.assertEqual(executed, result.best_cost)
        self.assertLessEqual(result.best_cost, budget)

    def test_finds_optimum_transpose(self):
        env = transpose_env()
        result = UCTAgent(EnvironmentView(env), 6, seed=0).search(30)
        self.assert_plan_valid(env, result, 6)
        self.assertEqual(result.best_cost, 4)

    def test_finds_optimum_trap(self):
        env = trap_env()
        result = UCTAgent(EnvironmentView(env), 6, seed=0).search(30)
        self.assert_plan_valid(env, result, 6)
        self.assertEqual(result.best_cost, 4)

    def test_combined_optimal_or_feasible_gap_nonnegative(self):
        optimum = oracle_optimum(combined_env(), 8)
        found_optimal = False
        for seed in range(20):
            env = combined_env()
            result = UCTAgent(EnvironmentView(env), 8, seed=seed).search(30)
            if not result.found:
                continue
            self.assert_plan_valid(env, result, 8)
            self.assertGreaterEqual(result.best_cost, optimum)
            found_optimal = found_optimal or result.best_cost == optimum
        self.assertTrue(found_optimal, "no seed found the proven optimum")

    def test_cycles_never_overspends_or_loops(self):
        optimum = oracle_optimum(cycles_env(), 7)
        for seed in range(10):
            env = cycles_env()
            result = UCTAgent(EnvironmentView(env), 7, seed=seed).search(30)
            if result.found:
                self.assert_plan_valid(env, result, 7)
                self.assertEqual(result.best_cost, optimum)

    def test_not_found_is_not_a_proof_of_infeasibility(self):
        env = infeasible_env()
        result = UCTAgent(EnvironmentView(env), INFEASIBLE_BUDGET, seed=0).search(
            4
        )
        self.assertFalse(result.found)
        self.assertFalse(result.proven_optimal)
        oracle = ShortestCostSolver(EnvironmentView(infeasible_env()), 5).solve()
        self.assertTrue(oracle.proven and not oracle.feasible)

    def test_found_but_never_claims_proven_optimal(self):
        env = transpose_env()
        result = UCTAgent(EnvironmentView(env), 6, seed=0).search(30)
        self.assertTrue(result.found)
        self.assertFalse(result.proven_optimal)

    def test_tight_budget_finding_nothing_is_silent(self):
        env = trap_env()
        result = UCTAgent(EnvironmentView(env), 6, seed=0).search(1)
        self.assertFalse(result.found)
        self.assertIsNone(result.best_actions)

    def test_invalid_plan_would_raise(self):
        env = transpose_env()
        with self.assertRaises(PlanError):
            EnvironmentView(env).validate_plan(("a", "nope"), 6)
        with self.assertRaises(PlanError):
            EnvironmentView(env).validate_plan(("a", "g", "g"), 6)


class UCTBudgetTests(unittest.TestCase):
    def test_call_budget_bounds_real_transitions(self):
        env = combined_env()
        view = EnvironmentView(env)
        result = UCTAgent(view, 8, seed=0).search(5)
        self.assertLessEqual(result.billed_calls, 5)

    def test_cache_hits_are_not_billed(self):
        env = transpose_env()
        view = EnvironmentView(env)
        UCTAgent(view, 6, seed=0).search(20)
        self.assertGreater(view.cache_hits, 0)
        self.assertEqual(view.billed_calls, len(view._transition_cache))

    def test_rollout_transitions_are_billed(self):
        cached_view = EnvironmentView(transpose_env())
        UCTAgent(cached_view, 6, seed=0).search(20)
        cached_calls = cached_view.billed_calls

        uncached_view = EnvironmentView(
            transpose_env(), cache_enabled=False
        )
        UCTAgent(uncached_view, 6, seed=0).search(40)
        self.assertGreater(uncached_view.billed_calls, cached_calls)
        self.assertEqual(uncached_view.cache_hits, 0)

    def test_dominance_pruning_preserves_optimum(self):
        env = combined_env()
        result = UCTAgent(
            EnvironmentView(env),
            8,
            config=UCTConfig(use_dominance=True),
            seed=2,
        ).search(30)
        self.assertTrue(result.found)
        self.assertEqual(result.best_cost, 4)


if __name__ == "__main__":
    unittest.main()
