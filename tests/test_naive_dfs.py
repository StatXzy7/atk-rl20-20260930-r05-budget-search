import unittest

from budget_uct import environments as E
from budget_uct import execute_plan, solve_exact
from budget_uct.naive_dfs import STATE_ONLY, STATE_REMAIN, memo_dfs


class TestNaiveDedup(unittest.TestCase):
    def test_state_only_dedup_gives_false_negative(self):
        env = E.dedup_counterexample_env()
        res = memo_dfs(env, budget=12, dedup=STATE_ONLY)
        # The bug: expensive path marks X dead at remaining 0; cheap path then
        # trusts the state-only memo and misses the feasible cost-7 plan.
        self.assertFalse(res.found)
        # Oracle confirms a feasible, optimal plan actually exists.
        truth = solve_exact(E.dedup_counterexample_env(), budget=12)
        self.assertTrue(truth.proven)
        self.assertEqual(truth.cost, 7)

    def test_state_remaining_dedup_is_correct(self):
        env = E.dedup_counterexample_env()
        res = memo_dfs(env, budget=12, dedup=STATE_REMAIN)
        self.assertTrue(res.found)
        self.assertEqual(res.plan, ["cheap", "go"])
        self.assertEqual(res.cost, 7)
        state, cost = execute_plan(
            E.dedup_counterexample_env(), "s0", res.plan
        )
        self.assertEqual(state, "G")
        self.assertEqual(cost, 7)

    def test_budget_below_minimum_both_agree_infeasible(self):
        env = E.dedup_counterexample_env()
        a = memo_dfs(env, budget=6, dedup=STATE_ONLY)
        b = memo_dfs(E.dedup_counterexample_env(), budget=6, dedup=STATE_REMAIN)
        self.assertFalse(a.found)
        self.assertFalse(b.found)

    def test_identical_state_via_distinct_paths_merging_condition(self):
        # At the same remaining budget, merging is sound: the same Markov state
        # and same affordability class have identical futures.
        res = memo_dfs(E.dedup_counterexample_env(), budget=100, dedup=STATE_ONLY)
        self.assertTrue(res.found)  # with slack the stale memo does not bite


if __name__ == "__main__":
    unittest.main()
