import unittest

from budget_uct import environments as E
from budget_uct.uct import UCTSearcher


class TestGraphSemantics(unittest.TestCase):
    def test_transposition_shared_node_independent_parent_edges(self):
        """s5 reached from s2 and (after a cycle) from s3 must share one node
        per (state, remaining) while each incoming edge keeps private stats."""
        s = UCTSearcher(E.demo_env(), 15, seed=5)
        s.run(300, polish=True)

        # Collect edges into every s5 node.
        incoming = {}
        for node in s.table.values():
            for edge in node.edges:
                if edge.child_key is not None and edge.child_key[0] == "s5":
                    incoming.setdefault(edge.child_key, []).append(
                        (node.state, node.remaining, edge)
                    )
        shared = {k: v for k, v in incoming.items() if len({(st, rem) for st, rem, _ in v}) >= 2}
        self.assertTrue(
            shared,
            "at least one s5 node must be entered from two distinct parents",
        )
        for child_key, arcs in shared.items():
            distinct_parents = {(st, rem) for st, rem, _ in arcs}
            self.assertGreaterEqual(len(distinct_parents), 2)
            edge_objects = [edge for _, _, edge in arcs]
            self.assertEqual(len(edge_objects), len({id(e) for e in edge_objects}))
            for edge in edge_objects:
                self.assertGreaterEqual(edge.visits, 1)
                self.assertLessEqual(edge.reward_sum, edge.visits + 1e-9)

    def test_backprop_uses_concrete_path_without_parent_pointer(self):
        # A smoke test that deep search over cyclic edges never raises and
        # never produces inconsistent stats on the shared cycle node s5.
        s = UCTSearcher(E.demo_env(), 15, seed=11)
        s.run(400, polish=True)
        for node in s.table.values():
            self.assertGreaterEqual(node.visits, 1)
            self.assertLessEqual(node.reward_sum, node.visits + 1e-9)
        self.assertIsNotNone(s.best_plan)
        self.assertEqual(s.best_cost, 6)


if __name__ == "__main__":
    unittest.main()
