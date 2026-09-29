import unittest

from budget_uct import environments as E
from budget_uct import execute_plan
from budget_uct.env import InstrumentedEnv
from budget_uct.exact import (
    BUDGET_EXHAUSTED,
    INFEASIBLE,
    OPTIMAL,
    solve_exact,
)


class TestExact(unittest.TestCase):
    def test_demo_optimal_with_transposition_trap_cycles(self):
        env = E.demo_env()
        res = solve_exact(env, budget=15)
        self.assertEqual(res.status, OPTIMAL)
        self.assertEqual(res.plan, ["to_s2", "merge", "goal"])
        self.assertEqual(res.cost, 6)
        self.assertTrue(res.proven)
        # Plan really executes and respects the budget.
        fresh = E.demo_env()
        state, cost = execute_plan(fresh, fresh.initial_state(), res.plan)
        self.assertTrue(fresh.is_success(state))
        self.assertEqual(cost, 6)
        self.assertLessEqual(cost, 15)

    def test_zero_cost_via_empty_plan_at_goal_start(self):
        from budget_uct.env import GraphEnv, GraphSpec

        env = GraphEnv(GraphSpec(start="g", edges={}, goals={"g"}))
        res = solve_exact(env, budget=0)
        self.assertEqual(res.status, OPTIMAL)
        self.assertEqual(res.plan, [])
        self.assertEqual(res.cost, 0)

    def test_infeasibility_proof_below_threshold(self):
        env = E.demo_env()
        res = solve_exact(env, budget=4)
        self.assertEqual(res.status, INFEASIBLE)
        self.assertTrue(res.proven)
        self.assertIsNone(res.plan)

    def test_cycles_never_improve_path(self):
        # Both positive-cost cycles must not make Dijkstra revisit a settled
        # state with a smaller distance.
        env = E.demo_env()
        res = solve_exact(env, budget=100)
        self.assertEqual(res.cost, 6)

    def test_layered_threshold_matches_oracle(self):
        env = E.layered_env()
        full = solve_exact(env, budget=1000)
        self.assertEqual(full.status, OPTIMAL)
        below = solve_exact(E.layered_env(), budget=full.cost - 1)
        self.assertEqual(below.status, INFEASIBLE)
        self.assertTrue(below.proven)
        at = solve_exact(E.layered_env(), budget=full.cost)
        self.assertEqual(at.cost, full.cost)

    def test_call_budget_exhausted_is_not_a_proof(self):
        env = E.layered_env()
        res = solve_exact(env, budget=100, call_budget=2)
        self.assertEqual(res.status, BUDGET_EXHAUSTED)
        self.assertFalse(res.proven)

    def test_cache_hits_tracked_separately(self):
        env = InstrumentedEnv(E.demo_env())
        solve_exact(env, budget=15)
        self.assertGreater(env.real_calls, 0)
        # Deterministic graph: solve again on the same wrapper, all hits.
        calls_before = env.real_calls
        solve_exact(env, budget=15)
        self.assertEqual(env.real_calls, calls_before)
        self.assertGreater(env.cache_hits, 0)


if __name__ == "__main__":
    unittest.main()
