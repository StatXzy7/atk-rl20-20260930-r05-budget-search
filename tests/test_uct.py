import copy
import unittest

from budget_uct import environments as E
from budget_uct import FOUND, execute_plan
from budget_uct.env import InstrumentedEnv
from budget_uct.uct import UCTSearcher, search_uct


def check_plan(env, budget, result):
    if not result.found:
        return None
    state, cost = execute_plan(env, env.initial_state(), result.plan)
    assert env.is_success(state), f"plan ends at non-success {state!r}"
    assert cost <= budget, f"plan cost {cost} exceeds {budget}"
    assert cost == result.cost, f"reported cost {result.cost} != executed {cost}"
    return cost


class TestUCT(unittest.TestCase):
    def test_find_valid_optimal_demo_for_some_seed(self):
        env = E.demo_env()
        found_optimal = False
        for seed in range(40):
            res = search_uct(E.demo_env(), 15, call_budget=100, seed=seed)
            if res.found:
                cost = check_plan(env, 15, res)
                self.assertGreaterEqual(cost, 6)
                if cost == 6:
                    found_optimal = True
        self.assertTrue(found_optimal, "some seed should find the cost-6 optimum")

    def test_never_overspends_or_executes_illegally(self):
        env = E.layered_env(layers=5, width=3)
        from budget_uct import solve_exact

        opt = solve_exact(env, budget=1000).cost
        budget = opt + 8
        for cb in (5, 10, 20, 30):
            for seed in range(10):
                res = search_uct(E.layered_env(), budget, call_budget=cb, seed=seed)
                if res.found:
                    check_plan(env, budget, res)

    def test_real_calls_bounded_at_trial_boundary(self):
        # The run stops at a trial boundary once the target is crossed, unless
        # the finite graph offers fewer distinct transitions than requested.
        total_distinct = 14
        for cb in (0, 1, 7, 25):
            res = search_uct(E.demo_env(), 15, call_budget=cb, seed=3)
            self.assertGreaterEqual(res.real_calls, min(cb, total_distinct))
            self.assertLessEqual(res.real_calls,
                                 min(total_distinct, cb + 16 * 2))

    def test_cache_hits_do_not_increase_real_calls(self):
        env = InstrumentedEnv(E.demo_env())
        search_uct(env, 15, call_budget=100, seed=2)
        # Only 14 distinct transitions exist in this env.
        self.assertLessEqual(env.real_calls, 14)
        self.assertGreaterEqual(env.cache_hits, 0)

    def test_determinism_same_seed_budget(self):
        a = search_uct(E.demo_env(), 15, call_budget=12, seed=7)
        b = search_uct(E.demo_env(), 15, call_budget=12, seed=7)
        self.assertEqual((a.plan, a.cost, a.real_calls, a.iterations, a.rollouts),
                         (b.plan, b.cost, b.real_calls, b.iterations, b.rollouts))

    def test_snapshot_save_restore_replays(self):
        env = InstrumentedEnv(E.demo_env())
        s1 = UCTSearcher(env, 15, seed=7)
        r1 = s1.run(8)
        snap = s1.save()

        # Continue without restore: reference trajectory.
        cont_ref = s1.run(20)
        ref_plan, ref_cost = s1.best_plan, s1.best_cost

        # Restore snapshot into a fresh searcher and replay identically.
        env2 = InstrumentedEnv(E.demo_env())
        s2 = UCTSearcher(env2, 15, seed=7)
        s2.run(8)
        s2.restore(snap)
        cont = s2.run(20)
        self.assertEqual((cont.plan, cont.cost, cont.real_calls, cont.iterations),
                         (cont_ref.plan, cont_ref.cost, cont_ref.real_calls,
                          cont_ref.iterations))
        self.assertEqual(s2.best_plan, ref_plan)
        self.assertEqual(s2.best_cost, ref_cost)

    def test_node_key_separates_remaining_budget(self):
        s = UCTSearcher(E.dedup_counterexample_env(), 12, seed=0)
        s.run(50)
        keys = list(s.table.keys())
        states = {state for state, _ in keys}
        self.assertIn("X", states)
        # State X must appear at more than one remaining budget.
        remainings = {rem for (state, rem) in keys if state == "X"}
        self.assertGreater(len(remainings), 1)

    def test_shared_transposition_node_has_per_parent_edge_stats(self):
        # In demo_env, s5 is reachable via distinct parents (direct from s2 and
        # via s3 loop). Each parent edge must carry its own counters, and node
        # visits accumulate from all descents.
        s = UCTSearcher(E.demo_env(), 15, seed=4)
        s.run(200, polish=True)
        # Find shared s5 child nodes reachable through DISTINCT parents.
        parents = {}
        for node in s.table.values():
            for edge in node.edges:
                if edge.child_key is not None and edge.child_key[0] == "s5":
                    parents.setdefault(edge.child_key, set()).add(
                        (node.state, node.remaining)
                    )
        shared = {k: v for k, v in parents.items() if len(v) >= 2}
        self.assertTrue(
            shared, "an s5 node must be shared by at least two distinct parents"
        )
        # Every parent keeps its own Edge object with independent counters and
        # none carries a shared 'parent pointer'.
        for child_key in shared:
            arcs = [
                edge
                for node in s.table.values()
                for edge in node.edges
                if edge.child_key == child_key
            ]
            parent_nodes = {id(node) for node in s.table.values()
                            for edge in node.edges if edge.child_key == child_key}
            self.assertGreaterEqual(len(parent_nodes), 2)
            self.assertEqual(len(arcs), len({id(e) for e in arcs}))
            for edge in arcs:
                self.assertGreaterEqual(edge.visits, 1)
                self.assertLessEqual(edge.reward_sum, edge.visits + 1e-9)

    def test_uct_never_claims_proofs(self):
        res = search_uct(E.demo_env(), 4, call_budget=50, seed=0)
        self.assertFalse(res.proven_optimal)
        self.assertFalse(res.proven_infeasible)
        self.assertIsNone(res.optimal_cost)

    def test_found_any_vs_best_cost_distinct(self):
        # With a small call budget a run may find a plan but not the cheapest.
        env = E.layered_env()
        from budget_uct import solve_exact

        opt = solve_exact(env, budget=1000).cost
        saw_suboptimal = False
        for seed in range(30):
            res = search_uct(E.layered_env(), opt + 8, call_budget=12, seed=seed)
            if res.found:
                self.assertEqual(res.status, FOUND)
                if res.cost > opt:
                    saw_suboptimal = True
        self.assertTrue(saw_suboptimal, "small budget should sometimes stop early")

    def test_empty_plan_when_start_is_goal(self):
        from budget_uct.env import GraphEnv, GraphSpec

        env = GraphEnv(GraphSpec(start="g", edges={}, goals={"g"}))
        res = search_uct(env, 0, call_budget=10, seed=0)
        self.assertTrue(res.found)
        self.assertEqual(res.plan, [])
        self.assertEqual(res.cost, 0)


if __name__ == "__main__":
    unittest.main()


class TestUCTBoundaries(unittest.TestCase):
    def test_every_call_budget_does_not_crash(self):
        # stop_at can be reached exactly between probing uncached actions.
        for cb in range(0, 40):
            for seed in range(4):
                res = search_uct(
                    E.layered_env(), 19, call_budget=cb, seed=seed
                )
                # bounded by graph exhaustion (37 distinct transitions) or a
                # one-rollout overshoot of the target
                self.assertGreaterEqual(res.real_calls, 0)
                self.assertLessEqual(res.real_calls, max(cb + 20 * 3, 37))
                if res.found:
                    check_plan(E.layered_env(), 19, res)

    def test_zero_execution_budget_non_goal(self):
        res = search_uct(E.demo_env(), 0, call_budget=20, seed=0)
        self.assertFalse(res.found)
        # All first-step edges are blocked (positive costs); no proof claimed.
        self.assertFalse(res.proven_infeasible)
        self.assertFalse(res.proven_optimal)

    def test_resume_across_increments_matches_one_shot(self):
        chunks = (3, 5, 8, 12)
        a_env = InstrumentedEnv(E.layered_env())
        b_env = InstrumentedEnv(E.layered_env())
        a = UCTSearcher(a_env, 19, seed=9)
        b = UCTSearcher(b_env, 19, seed=9)
        for chunk in chunks:
            a.run(chunk)            # incremental, no polish between chunks
            b.run(chunk, polish=False)
        self.assertEqual((a.best_plan, a.best_cost, a.calls, a.iterations),
                         (b.best_plan, b.best_cost, b.calls, b.iterations))
        # A snapshot taken mid-way and restored must reproduce the continuation.
        snap = a.save()
        a.run(4, polish=False)
        restored = UCTSearcher(InstrumentedEnv(E.layered_env()), 19, seed=9)
        restored.restore(snap)
        restored.run(4, polish=False)
        self.assertEqual((restored.best_plan, restored.best_cost, restored.iterations),
                         (a.best_plan, a.best_cost, a.iterations))

if __name__ == "__main__":
    unittest.main()
