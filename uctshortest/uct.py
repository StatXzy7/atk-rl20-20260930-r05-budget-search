"""UCT search over the deterministic (state, remaining-budget) graph.

Search-graph design (see DESIGN.md for the full rules):

* A search *node* is identified by ``(env_state, remaining_budget)``.  Because
  every action costs a strictly positive integer, each descent strictly lowers
  the remaining budget, so the search graph is acyclic even when the
  underlying environment contains cycles.  The same environment state reached
  with the same remaining budget is shared (a real transposition); it is never
  merged with a node carrying a different remaining budget.
* Visit statistics live on the *parent edge* (``(parent_key, action)``).  A
  shared node may be evaluated once while every incoming parent edge keeps its
  own visit count and average reward.  Tree-style "increment child.visit on
  back-up" is deliberately not used on this cyclic-capable graph.
* ``node.entries`` counts how many times tree-policy selection entered the
  node and is used to guarantee every untried edge gets explored before
  exploitation begins.
"""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Hashable, NamedTuple

from .environment import EnvironmentView

NodeKey = tuple[Hashable, int]


class Edge(NamedTuple):
    action: Any
    child_key: NodeKey | None
    leaf_kind: str | None = None


@dataclass
class Node:
    key: NodeKey
    legal_actions: tuple = ()
    actions_known: bool = False
    edges: dict[Any, Edge] = field(default_factory=dict)
    edge_visits: dict[Any, int] = field(default_factory=dict)
    edge_value: dict[Any, float] = field(default_factory=dict)
    dead: bool = False
    entries: int = 0
    static_estimate: float | None = None


class Incumbent(NamedTuple):
    actions: tuple
    cost: int


class SearchResult(NamedTuple):
    found: bool
    best_actions: tuple | None
    best_cost: int | None
    iterations: int
    billed_calls: int
    cache_hits: int
    budget_exhausted: bool
    proven_optimal: bool


class UCTConfig(NamedTuple):
    c_ucb: float = 1.4142135623730951
    feasible_base: float = 0.5
    success_reward: float = 1.0
    use_dominance: bool = False


LEAF_OVERSPENT = "overspent"
LEAF_PRUNED = "pruned"
_ABORT = object()


class UCTAgent:
    """DAG-UCT shortest-cost search agent.

    Parameters are fixed by the problem statement (see ``DESIGN.md`` for the
    full reward / expansion / simulation / back-up definitions):

    * **Reward of a simulated path** with total cost ``C`` under execution
      budget ``B``: ``R = 0`` on failure (cycle/dead-end/over-budget); on
      success ``R = feasible_base + (1 - feasible_base) * (B - C) / B`` which
      lies in ``[feasible_base, 1]``, so any ``R >= feasible_base`` certifies
      feasibility and cheaper feasible plans get strictly larger rewards.
    * **Leaf evaluation**: a newly created node is valued by one roll-out;
      terminal success nodes get reward 1 without spending a transition;
      overspent/over-budget leaf edges get value 0.
    * **Expansion**: one new node per iteration; nodes keyed by
      ``(state, remaining_budget)``.
    * **Simulation**: random-uniform legal actions, billed through the same
      view, stopped on budget, repeated state (positive-cost cycle), dead end
      or success.
    * **Back-up**: update the edges on the walked path only,
      ``V <- V + (R - V) / n``; shared incoming edges keep independent
      statistics.
    """

    def __init__(
        self,
        view: EnvironmentView,
        budget: int,
        config: UCTConfig | None = None,
        seed: int = 0,
        rng_state: Any = None,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.view = view
        self.budget = budget
        self.config = config or UCTConfig()
        self.rng = random.Random()
        if rng_state is not None:
            self.rng.setstate(rng_state)
        else:
            self.rng.seed(seed)
        self.nodes: dict[NodeKey, Node] = {}
        self._dominator: dict[Hashable, NodeKey] = {}
        root_key = (view.initial_state, budget)
        self.root = self._make_node(root_key)
        self.incumbent: Incumbent | None = None
        self.iterations = 0

    def _make_node(self, key: NodeKey) -> Node:
        node = self.nodes.get(key)
        if node is None:
            node = Node(key=key)
            self.nodes[key] = node
        return node

    def _key_for(self, state: Hashable, remaining: int) -> NodeKey:
        return (state, remaining)

    def _ensure_actions(self, node: Node) -> None:
        if not node.actions_known:
            node.legal_actions = tuple(self.view.legal_actions(node.key[0]))
            node.actions_known = True
            if not node.legal_actions:
                node.dead = True

    def _select_untried(self, node: Node) -> Any:
        for action in node.legal_actions:
            if action not in node.edges and action not in node.edge_visits:
                return action
        return None

    def _ucb_score(self, node: Node, action: Any) -> float:
        visits = node.edge_visits[action]
        value = node.edge_value[action]
        exploration = self.config.c_ucb * (node.entries / max(1, visits)) ** 0.5
        return value + exploration

    def _select_ucb(self, node: Node) -> Any:
        candidates = [
            action
            for action in node.legal_actions
            if action in node.edge_visits
        ]
        best_score = max(self._ucb_score(node, a) for a in candidates)
        best = [a for a in candidates if self._ucb_score(node, a) == best_score]
        if len(best) == 1:
            return best[0]
        return self.rng.choice(best)

    def _progressed(self, billed_at_start: int, call_budget: int | None) -> bool:
        """Whether another iteration may produce useful evidence."""
        if self.view.billed_calls > billed_at_start:
            return True
        return call_budget is None or self.view.billed_calls < call_budget

    def _iterate(self, call_budget: int | None) -> bool:
        """Run one UCT iteration.

        Returns ``False`` when the iteration consumed no *new* real
        transition call.  Such an iteration may still back up a cached path,
        but the search loop must not start another one, which guarantees
        termination when the call budget is spent (and with cycles).
        """
        billed_at_start = self.view.billed_calls
        node = self.root
        path_edges: list[tuple[Node, Any]] = []
        prefix_actions: list[Any] = []
        visited_keys: set[NodeKey] = {node.key}
        node.entries += 1

        while True:
            state, remaining = node.key
            if self.view.is_success(state):
                if path_edges:
                    self._backup(path_edges, self.config.success_reward)
                self._update_incumbent(prefix_actions)
                return self._progressed(billed_at_start, call_budget)
            self._ensure_actions(node)
            if node.dead:
                if path_edges:
                    self._backup(path_edges, 0.0)
                return self._progressed(billed_at_start, call_budget)

            untried = self._select_untried(node)
            if untried is not None:
                reward = self._expand(
                    node,
                    untried,
                    path_edges,
                    prefix_actions,
                    visited_keys,
                    call_budget,
                )
                if reward is _ABORT:
                    node.entries -= 1
                    return False
                self._backup(path_edges, reward)
                return self._progressed(billed_at_start, call_budget)

            action = self._select_ucb(node)
            edge = node.edges[action]
            node.edge_visits[action] += 1
            path_edges.append((node, action))
            prefix_actions.append(action)
            if edge.leaf_kind is not None:
                self._backup(path_edges, node.edge_value[action])
                return self._progressed(billed_at_start, call_budget)
            if edge.child_key in visited_keys:
                self._backup(path_edges, 0.0)
                return self._progressed(billed_at_start, call_budget)
            child = self.nodes[edge.child_key]
            visited_keys.add(child.key)
            child.entries += 1
            node = child

    def _lookup_child(self, key: NodeKey) -> Node | None:
        return self.nodes.get(key)

    def _expand(
        self,
        node: Node,
        action: Any,
        path_edges: list[tuple[Node, Any]],
        prefix_actions: list[Any],
        visited_keys: set,
        call_budget: int | None,
    ) -> float:
        state, remaining = node.key
        cached = self.view.has_cached_step(state, action)
        budget_hit = (
            call_budget is not None and self.view.billed_calls >= call_budget
        )
        if budget_hit and not cached:
            return _ABORT
        outcome = self.view.step(state, action)
        cost = outcome.cost
        path_edges.append((node, action))
        prefix_actions.append(action)
        if cost > remaining:
            node.edges[action] = Edge(action, None, LEAF_OVERSPENT)
            self._set_visit(node, action)
            return 0.0
        child_key = self._key_for(outcome.state, remaining - cost)
        self._set_visit(node, action)
        child = self._lookup_child(child_key)
        if child is not None:
            node.edges[action] = Edge(action, child.key, None)
            visited_keys.add(child.key)
            child.entries += 1
            return self._evaluate_existing(child, prefix_actions)
        if self._is_pruned(child_key):
            dominator = self.nodes[self._dominator[child_key[0]]]
            node.edges[action] = Edge(action, dominator.key, LEAF_PRUNED)
            estimate = dominator.static_estimate
            return estimate if estimate is not None else 0.0
        child = self._make_node(child_key)
        node.edges[action] = Edge(action, child_key, None)
        visited_keys.add(child.key)
        child.entries += 1
        reward = self._evaluate_child(child, prefix_actions, call_budget)
        if reward is None:
            return _ABORT
        return reward

    @staticmethod
    def _set_visit(node: Node, action: Any) -> None:
        node.edge_visits[action] = 1
        node.edge_value[action] = 0.0

    def _is_pruned(self, child_key: NodeKey) -> bool:
        """Return True when an already kept node dominates ``child_key``.

        Dominance condition: a kept node (s, r') dominates a new node (s, r)
        iff ``r' >= r``.  This is sound because transitions are deterministic
        and Markov, costs are strictly positive integers and success depends
        on state alone, so every completion feasible within r is feasible
        within r' with identical cost (the extra r'-r units of budget are
        simply unused).  Only *already existing* nodes can dominate; the first
        arrival at each state always creates a node, and later arrivals with
        more remaining budget replace the registered dominator.
        """
        if not self.config.use_dominance:
            return False
        state, remaining = child_key
        kept = self._dominator.get(state)
        if kept is None:
            self._dominator[state] = child_key
            return False
        if kept in self.nodes and kept[1] >= remaining:
            return True
        if remaining > kept[1]:
            self._dominator[state] = child_key
        return False

    def _evaluate_existing(self, child: Node, prefix_actions: list[Any]) -> float:
        state, _remaining = child.key
        if self.view.is_success(state):
            self._update_incumbent(prefix_actions)
            return self.config.success_reward
        if child.static_estimate is not None:
            return child.static_estimate
        if child.dead:
            child.static_estimate = 0.0
            return 0.0
        best_action = max(
            child.legal_actions,
            key=lambda a: child.edge_value.get(a, 0.0),
            default=None,
        )
        return child.edge_value.get(best_action, 0.0) if best_action is not None else 0.0

    def _evaluate_child(
        self, child: Node, prefix_actions: list[Any], call_budget: int | None
    ) -> float | None:
        state, remaining = child.key
        if self.view.is_success(state):
            child.static_estimate = self.config.success_reward
            self._update_incumbent(prefix_actions)
            return self.config.success_reward
        self._ensure_actions(child)
        if child.dead:
            child.static_estimate = 0.0
            return 0.0
        reward = self._rollout(state, remaining, prefix_actions, call_budget)
        if reward is not None:
            child.static_estimate = reward
        return reward

    def _rollout(
        self,
        state: Hashable,
        remaining: int,
        prefix_actions: list[Any],
        call_budget: int | None,
    ) -> float | None:
        """One random simulation; every transition is billed via the view.

        Returns ``None`` if the call budget runs out mid-rollout: the whole
        iteration is aborted without back-up so no unpaid evidence pollutes
        edge statistics.
        """
        actions: list[Any] = list(prefix_actions)
        cost_used = self.budget - remaining
        seen: set[Hashable] = {self.view.initial_state, state}
        current = state
        spare = remaining
        while True:
            if self.view.is_success(current):
                self._update_incumbent(actions)
                return self._shaped_reward(cost_used)
            legal = tuple(self.view.legal_actions(current))
            if not legal:
                return 0.0
            action = self.rng.choice(legal)
            if call_budget is not None and self.view.billed_calls >= call_budget:
                return None
            outcome = self.view.step(current, action)
            actions.append(action)
            if outcome.cost > spare:
                return 0.0
            cost_used += outcome.cost
            spare -= outcome.cost
            if outcome.state in seen:
                return 0.0
            seen.add(outcome.state)
            current = outcome.state

    def _shaped_reward(self, total_cost: int) -> float:
        if self.budget == 0:
            return self.config.success_reward
        fraction = total_cost / self.budget
        base = self.config.feasible_base
        return base + (1.0 - base) * (1.0 - fraction)

    def _backup(self, path_edges: list[tuple[Node, Any]], reward: float) -> None:
        for node, action in path_edges:
            visits = node.edge_visits[action]
            old = node.edge_value[action]
            node.edge_value[action] = old + (reward - old) / visits

    def _update_incumbent(self, actions: list[Any] | tuple) -> None:
        plan = tuple(actions)
        total = self._plan_cost_and_terminal(plan)
        if total is None:
            return
        cost, terminal = total
        if cost > self.budget or not self.view.is_success(terminal):
            return
        if self.incumbent is None or cost < self.incumbent.cost:
            self.incumbent = Incumbent(plan, cost)

    def _plan_cost_and_terminal(
        self, plan: tuple | list[Any]
    ) -> tuple[int, Hashable] | None:
        state = self.view.initial_state
        total = 0
        for action in plan:
            cached = self.view.has_cached_step(state, action)
            if not cached:
                return None
            next_state, cost = self.view.step(state, action)
            total += cost
            state = next_state
        return total, state
    def search(self, call_budget: int) -> SearchResult:
        """Run iterations until ``call_budget`` real transitions are billed.

        One final cache-only iteration is allowed after the budget is reached
        so a leaf reached on the last billed transition can be backed up; the
        loop then stops regardless of what cached edges can reach.
        """
        idle_limit = 1 + len(self.nodes)
        idle = 0
        while self.view.billed_calls < call_budget:
            before = self.view.billed_calls
            self.iterations += 1
            progressed = self._iterate(call_budget)
            if self.view.billed_calls == before:
                idle += 1
            else:
                idle = 0
            if not progressed or idle > idle_limit:
                break
            idle_limit = 1 + len(self.nodes)
        return self.result(call_budget)

    def result(self, call_budget: int | None = None) -> SearchResult:
        exhausted = (
            call_budget is not None and self.view.billed_calls >= call_budget
        )
        if self.incumbent is not None:
            return SearchResult(
                found=True,
                best_actions=self.incumbent.actions,
                best_cost=self.incumbent.cost,
                iterations=self.iterations,
                billed_calls=self.view.billed_calls,
                cache_hits=self.view.cache_hits,
                budget_exhausted=exhausted,
                proven_optimal=False,
            )
        return SearchResult(
            found=False,
            best_actions=None,
            best_cost=None,
            iterations=self.iterations,
            billed_calls=self.view.billed_calls,
            cache_hits=self.view.cache_hits,
            budget_exhausted=exhausted,
            proven_optimal=False,
        )

    def plan_for_prefix(self, prefix: tuple | list[Any]) -> tuple | None:
        """Return a full feasible plan with the given root prefix, if known.

        The prefix is executed against its *own* arrival state/remaining
        budget, then a cached in-tree completion is searched.  The correct
        implementation uses the prefix's own remaining budget; the buggy
        state-only variant (see :mod:`uctshortest.naive`) answers from the
        merged node, which can certify an over-budget plan.
        """
        state = self.view.initial_state
        remaining = self.budget
        for action in prefix:
            if not self.view.has_cached_step(state, action):
                return None
            next_state, cost = self.view.step(state, action)
            if cost > remaining:
                return None
            remaining -= cost
            state = next_state
        suffix = self._completion_from(state, remaining, frozenset(), ())
        return None if suffix is None else tuple(prefix) + suffix

    def _completion_from(
        self,
        state: Hashable,
        remaining: int,
        seen: frozenset,
        suffix: tuple,
    ) -> tuple | None:
        if self.view.is_success(state):
            return suffix
        if state in seen:
            return None
        node = self.nodes.get(self._key_for(state, remaining))
        if node is None:
            return None
        ranked = sorted(
            (
                a
                for a in node.legal_actions
                if a in node.edges and node.edges[a].leaf_kind is None
            ),
            key=lambda a: node.edge_value.get(a, 0.0),
            reverse=True,
        )
        for action in ranked:
            next_state, cost = self.view.step(state, action)
            if cost > remaining:
                continue
            found = self._completion_from(
                next_state,
                remaining - cost,
                seen | {state},
                suffix + (action,),
            )
            if found is not None:
                return found
        return None

    def rng_getstate(self) -> Any:
        return self.rng.getstate()

    def snapshot_state(self) -> dict:
        """Capture the full replay state (RNG, search graph, counters).

        The search graph (nodes/edges, per-edge statistics, action-cost cache
        and dominance registry) is included, so an agent can be stopped,
        saved, restored in a fresh process and continue the search from the
        exact same point.  Primitive states/actions make the snapshot
        picklable; ``rng`` uses :meth:`random.Random.getstate`.
        """
        return {
            "rng": self.rng.getstate(),
            "iterations": self.iterations,
            "incumbent": None
            if self.incumbent is None
            else {"actions": self.incumbent.actions, "cost": self.incumbent.cost},
            "nodes": pickle.dumps(self.nodes),
            "cost_cache": pickle.dumps({}),
            "dominator": pickle.dumps(self._dominator),
            "view": self.view.snapshot_state(),
        }

    def restore_state(self, snapshot: dict) -> None:
        self.rng.setstate(snapshot["rng"])
        self.iterations = snapshot["iterations"]
        inc = snapshot["incumbent"]
        self.incumbent = None if inc is None else Incumbent(
            tuple(inc["actions"]), inc["cost"]
        )
        self.nodes = pickle.loads(snapshot["nodes"])
        self._dominator = pickle.loads(snapshot["dominator"])
        root_key = (self.view.initial_state, self.budget)
        self.root = self.nodes[root_key]
        self.view.restore_state(snapshot["view"])
