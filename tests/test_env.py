import unittest

from budget_uct.env import GraphEnv, GraphSpec, InstrumentedEnv, execute_plan


class TestEnvironment(unittest.TestCase):
    def test_positive_cost_required(self):
        spec = GraphSpec(start=0, edges={0: {"a": (1, 0)}}, goals={1})
        with self.assertRaises(ValueError):
            GraphEnv(spec)

    def test_bool_cost_rejected(self):
        spec = GraphSpec(start=0, edges={0: {"a": (1, True)}}, goals={1})
        with self.assertRaises(ValueError):
            GraphEnv(spec)

    def test_missing_start_rejected(self):
        spec = GraphSpec(start=0, edges={1: {"a": (1, 1)}}, goals=set())
        with self.assertRaises(ValueError):
            GraphEnv(spec)

    def test_call_accounting_cache_hits_free(self):
        spec = GraphSpec(start=0, edges={0: {"a": (1, 2), "b": (1, 3)}}, goals={1})
        env = InstrumentedEnv(GraphEnv(spec))
        env.step(0, "a")
        env.step(0, "a")  # cache hit
        env.step(0, "b")
        self.assertEqual(env.real_calls, 2)
        self.assertEqual(env.cache_hits, 1)

    def test_execute_plan_and_illegal_action(self):
        spec = GraphSpec(start=0, edges={0: {"a": (1, 2)}}, goals={1})
        env = GraphEnv(spec)
        final, cost = execute_plan(env, 0, ["a"])
        self.assertEqual(final, 1)
        self.assertEqual(cost, 2)
        with self.assertRaises(ValueError):
            execute_plan(env, 0, ["b"])


if __name__ == "__main__":
    unittest.main()
