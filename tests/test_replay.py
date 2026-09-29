"""Deterministic replay via explicit, saveable RNG state and call accounting."""

from __future__ import annotations

import copy
import pickle
import unittest

from uctshortest.environment import EnvironmentView
from uctshortest.envs import combined_env, transpose_env
from uctshortest.exact import ShortestCostSolver
from uctshortest.uct import UCTAgent


def run_once(env, budget, call_budget, seed):
    view = EnvironmentView(env)
    agent = UCTAgent(view, budget, seed=seed)
    return agent.search(call_budget), agent.rng_getstate(), view.snapshot_state()


class ReplayTests(unittest.TestCase):
    def test_same_seed_and_budget_replay_bit_for_bit(self):
        first, rng_state, _ = run_once(combined_env(), 8, 12, seed=7)
        env = combined_env()
        view = EnvironmentView(env)
        agent = UCTAgent(view, 8, rng_state=None, seed=7)
        second = agent.search(12)
        self.assertEqual(first, second)
        self.assertEqual(agent.rng_getstate(), rng_state)

    def test_restored_rng_continues_identically(self):
        env = transpose_env()
        view = EnvironmentView(env)
        agent = UCTAgent(view, 6, seed=3)
        part = agent.search(3)
        snapshot = agent.snapshot_state()
        rest1 = agent.search(10)

        view2 = EnvironmentView(transpose_env())
        agent2 = UCTAgent(view2, 6)
        agent2.restore_state(copy.deepcopy(snapshot))
        rest2 = agent2.search(10)
        self.assertEqual(rest1, rest2)

    def test_rng_state_is_picklable(self):
        agent = UCTAgent(EnvironmentView(combined_env()), 8, seed=1)
        agent.search(4)
        blob = pickle.dumps(agent.rng_getstate())
        restored = pickle.loads(blob)
        self.assertEqual(restored, agent.rng_getstate())

    def test_view_snapshot_restores_counters_and_cache(self):
        view = EnvironmentView(transpose_env())
        agent = UCTAgent(view, 6, seed=0)
        agent.search(5)
        snapshot = view.snapshot_state()
        calls_before = view.billed_calls
        agent.search(20)
        self.assertGreater(view.billed_calls, calls_before)
        view.restore_state(snapshot)
        self.assertEqual(view.billed_calls, snapshot["billed_calls"])
        self.assertEqual(view.cache_hits, snapshot["cache_hits"])

    def test_seed_changes_trajectory_but_optimum_is_seed_invariant(self):
        optimum = ShortestCostSolver(
            EnvironmentView(combined_env()), 8
        ).solve().best_cost
        costs = set()
        for seed in range(6):
            result = UCTAgent(
                EnvironmentView(combined_env()), 8, seed=seed
            ).search(10)
            if result.found:
                costs.add(result.best_cost)
                self.assertGreaterEqual(result.best_cost, optimum)
        self.assertTrue(costs)


if __name__ == "__main__":
    unittest.main()
