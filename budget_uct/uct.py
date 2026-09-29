"""UCT (UCB1) search for a minimum-cost plan in a deterministic tool env.

Design summary (see README for the full specification)
-------------------------------------------------------
* Search nodes are keyed by ``(env_state, remaining_budget)``. Two visits to
  the same environment state with different remaining execution budgets are
  *not* merged: from one node some action may still be affordable while from
  the other it is not. See ``naive_dfs`` for a counterexample to state-only
  de-duplication.
* Transposition nodes are shared, but each *incoming parent edge* keeps its own
  visit/reward statistics. Backprop walks the concrete path of
  (node, edge) pairs taken during the current descent, so merging nodes with
  many parents cannot double-count or redirect reward through a single parent.
* Call budget accounting uses real environment transitions only
  (``InstrumentedEnv``). Cache hits are free. Simulations are charged exactly
  like tree transitions.
* Rewards are binary (1.0 success within budget, else 0.0). Separately from
  the reward signal, every successful rollout updates an incumbent that stores
  the *minimum cost* successful plan seen, so "found any plan" and
  "cost of the current best plan" are distinct outputs.
* Randomness comes exclusively from a ``random.Random`` whose state is part of
  the search snapshot; identical (input, seed, call budget) replays exactly.
* UCT never proves optimality and never proves infeasibility; the independent
  exact solver supplies both kinds of proof within its scale.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .env import Action, Environment, InstrumentedEnv, State

FOUND = "found"
NOT_FOUND = "not_found"

_SQRT2 = math.sqrt(2.0)
NodeKey = Tuple[State, int]


@dataclass
class Edge:
    """A parent -> child arc; each parent keeps its own edges and stats."""

    action: Action
    cost: int = 0
    child_key: Optional[NodeKey] = None
    visits: int = 0
    reward_sum: float = 0.0
    blocked: bool = False

    def mean(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0


@dataclass
class UCTNode:
    state: State
    remaining: int
    visits: int = 0
    reward_sum: float = 0.0
    edges: List[Edge] = field(default_factory=list)
    expanded: bool = False


@dataclass
class Snapshot:
    rng_state: object
    real_calls: int
    cache_hits: int
    iterations: int
    rollouts: int
    best_plan: Optional[List[Action]]
    best_cost: Optional[int]
    table: Dict[NodeKey, UCTNode]
    transition_cache: Dict[object, object]
    root_key: NodeKey


@dataclass
class UCTResult:
    status: str
    plan: Optional[List[Action]]
    cost: Optional[int]
    optimal_cost: Optional[int]
    real_calls: int
    cache_hits: int
    iterations: int
    rollouts: int
    proven_optimal: bool = False
    proven_infeasible: bool = False

    @property
    def found(self) -> bool:
        return self.status == FOUND


class UCTSearcher:
    """Incrementally resumable UCT search over a deterministic budgeted env."""

    def __init__(
        self,
        env: Environment,
        budget: int,
        *,
        seed: int = 0,
        exploration: float = _SQRT2,
        start: Optional[State] = None,
    ):
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.ienv = env if isinstance(env, InstrumentedEnv) else InstrumentedEnv(env)
        self.budget = budget
        self.rng = random.Random(seed)
        self.exploration = exploration
        root_state = start if start is not None else self.ienv.initial_state()
        self.table: Dict[NodeKey, UCTNode] = {}
        self.root = self._make_node(root_state, budget)
        self.iterations = 0
        self.rollouts = 0
        self.best_plan: Optional[List[Action]] = None
        self.best_cost: Optional[int] = None
        if self.ienv.is_success(root_state) and budget >= 0:
            self.best_plan = []
            self.best_cost = 0

    # ------------------------------------------------------------- bookkeeping
    def _make_node(self, state: State, remaining: int) -> UCTNode:
        key = (state, remaining)
        node = self.table.get(key)
        if node is None:
            node = UCTNode(state=state, remaining=remaining)
            self.table[key] = node
        return node

    def _ensure_edges(self, node: UCTNode) -> None:
        """Lazily materialize one untried Edge per legal action. The real
        transition cost is learned when the edge is first expanded."""
        if node.expanded:
            return
        node.expanded = True
        for action in self.ienv.actions(node.state):
            node.edges.append(Edge(action=action))

    @property
    def calls(self) -> int:
        return self.ienv.real_calls

    # --------------------------------------------------------- save / restore
    def save(self) -> Snapshot:
        return Snapshot(
            rng_state=self.rng.getstate(),
            real_calls=self.ienv.real_calls,
            cache_hits=self.ienv.cache_hits,
            iterations=self.iterations,
            rollouts=self.rollouts,
            best_plan=None if self.best_plan is None else list(self.best_plan),
            best_cost=self.best_cost,
            table=copy.deepcopy(self.table),
            transition_cache=copy.deepcopy(self.ienv._cache),
            root_key=(self.root.state, self.root.remaining),
        )

    def restore(self, snap: Snapshot) -> None:
        self.rng.setstate(snap.rng_state)
        self.ienv.real_calls = snap.real_calls
        self.ienv.cache_hits = snap.cache_hits
        self.iterations = snap.iterations
        self.rollouts = snap.rollouts
        self.best_plan = None if snap.best_plan is None else list(snap.best_plan)
        self.best_cost = snap.best_cost
        self.table = copy.deepcopy(snap.table)
        self.ienv._cache = copy.deepcopy(snap.transition_cache)
        self.root = self.table[snap.root_key]

    # ------------------------------------------------------------------ run
    def run(self, call_budget: int, *, polish: bool = False) -> UCTResult:
        """Run trials until the NEW real-transition counter reaches the target
        (relative to the current counter). Repeated calls continue the same
        search.

        When ``polish`` is False the method stops as soon as the requested new
        calls are charged (or no further real call is structurally possible),
        so incremental continuation is bit-for-bit identical to one long run.
        When True, a bounded, deterministic window of purely cached rollouts is
        added afterwards to polish the incumbent; this spends RNG draws but
        obtains no new environment information.
        """
        if call_budget < 0:
            raise ValueError("call_budget must be non-negative")
        stop_at = self.ienv.real_calls + call_budget
        # A bounded, deterministic "polish" window of purely cached iterations
        # used after no more real transitions can be obtained: cached rollouts
        # may still improve the incumbent, but no new information is available.
        polish_budget = 128
        polish_count = 0
        safety_cap = 100_000

        while self.iterations < safety_cap:
            # Charging is decided at trial boundaries. A trial always runs to
            # completion, so its RNG stream never depends on where the caller
            # split the budget; this is what makes incremental continuation
            # identical to one long run.
            cache_only = False
            if self.ienv.real_calls >= stop_at:
                if not polish:
                    break  # stop at the first trial boundary past the target
                cache_only = True
                polish_count += 1
                if polish_count >= polish_budget or self._saturated():
                    break
            charged = self._iterate(cache_only=cache_only)
            self.iterations += 1
            if charged:
                polish_count = 0
            if (
                self.ienv.real_calls < stop_at
                and not charged
                and not self._new_call_possible()
            ):
                # Finite graph fully cached before the target was met.
                if not polish:
                    break
                polish_count += 1
                if polish_count >= polish_budget or self._saturated():
                    break

        return self._result()

    def _result(self) -> UCTResult:
        return UCTResult(
            status=FOUND if self.best_plan is not None else NOT_FOUND,
            plan=None if self.best_plan is None else list(self.best_plan),
            cost=self.best_cost,
            optimal_cost=None,  # UCT never proves optimality
            real_calls=self.ienv.real_calls,
            cache_hits=self.ienv.cache_hits,
            iterations=self.iterations,
            rollouts=self.rollouts,
        )

    # ------------------------------------------------------------- one trial
    def _iterate(self, *, cache_only: bool = False) -> bool:
        """One UCT trial. Returns True iff at least one real transition call
        was charged during the trial."""
        path_nodes: List[UCTNode] = [self.root]
        path_edges: List[Edge] = []
        charged = False
        node = self.root

        # 1) Selection. Descend while the node is fully tried, choosing an edge
        #    by UCB1 among edges that are payable and that do not point back to
        #    a node already on this concrete descent (cycle guard).
        while True:
            self._ensure_edges(node)
            usable = [e for e in node.edges if not e.blocked]
            untried = [e for e in usable if e.visits == 0]
            if untried:
                break
            if not usable:
                self._backprop(path_nodes, path_edges, 0.0)
                return charged
            on_path = {id(n) for n in path_nodes}
            edge = self._best_ucb(node, usable, on_path)
            if edge is None:
                # Every usable edge closes a cycle in this descent: treat the
                # current node as the evaluation leaf.
                break
            path_edges.append(edge)
            child = self.table[edge.child_key]
            path_nodes.append(child)
            node = child
            if self.ienv.is_success(node.state):
                self._consider_path(path_nodes, path_edges)
                self._backprop(path_nodes, path_edges, 1.0)
                return charged

        # 2) Expansion (one random untried action => one real transition).
        self._ensure_edges(node)
        untried = [e for e in node.edges if not e.blocked and e.visits == 0]
        if untried and not cache_only:
            edge = self.rng.choice(untried)
            result = self.ienv.step(node.state, edge.action)
            charged = True
            edge.cost = result.cost
            if result.cost > node.remaining:
                edge.blocked = True
                self._backprop(path_nodes, path_edges, 0.0)
                return charged
            child = self._make_node(result.next_state, node.remaining - result.cost)
            edge.child_key = (child.state, child.remaining)
            path_edges.append(edge)
            path_nodes.append(child)
            if self.ienv.is_success(child.state):
                self._consider_path(path_nodes, path_edges)
                self._backprop(path_nodes, path_edges, 1.0)
                return charged
            eval_node = child
        else:
            eval_node = node

        # 3) Simulation from eval_node.
        extra_charged = self._rollout(
            eval_node, path_nodes, path_edges, cache_only=cache_only
        )
        return charged or extra_charged

    def _best_ucb(
        self, node: UCTNode, edges: List[Edge], on_path: set
    ) -> Optional[Edge]:
        """UCB1: mean + c * sqrt(ln N(parent) / n(edge)). Edges whose child is
        already on the descent path are skipped. Untried edges are handled by
        the caller. Ties break by action order for strict determinism."""
        best: Optional[Edge] = None
        best_value = -math.inf
        log_n = math.log(max(1, node.visits))
        for edge in edges:
            if edge.visits == 0:
                continue
            child = self.table.get(edge.child_key) if edge.child_key else None
            if child is not None and id(child) in on_path:
                continue
            value = edge.mean() + self.exploration * math.sqrt(
                log_n / edge.visits
            )
            if value > best_value + 1e-15:
                best_value = value
                best = edge
        return best

    def _rollout(
        self,
        node: UCTNode,
        path_nodes: List[UCTNode],
        path_edges: List[Edge],
        cache_only: bool = False,
    ) -> bool:
        """Uniform random simulation over all legal actions. One transition call
        per simulation step (cache hit if known, else one real charged call).
        The rollout ends at success (reward 1), a dead end, or a sampled action
        whose cost exceeds the remaining budget (reward 0). Positive integer
        costs make every rollout bounded by the remaining budget. A trial runs
        to completion regardless of the run-level call counter, so its RNG
        stream is independent of how the caller split the call budget."""
        state = node.state
        remaining = node.remaining
        rollout_actions: List[Action] = []
        charged = False
        reward = 0.0

        legal = list(self.ienv.actions(state))
        while not self.ienv.is_success(state) and legal:
            # Uniform random over ALL legal actions (standard rollout policy).
            # One transition call per simulation step: cache hit if known,
            # otherwise exactly one real charged call. Drawing the action
            # before knowing its cost keeps the RNG stream independent of the
            # run-level call budget.
            action = self.rng.choice(legal)
            cached = self.ienv.cached_step(state, action)
            if cached is None:
                if cache_only:
                    break  # polishing must never charge a new transition
                cached = self.ienv.step(state, action)
                charged = True
            if cached.cost > remaining:
                break  # sampled suffix is not payable under the budget
            state = cached.next_state
            remaining -= cached.cost
            rollout_actions.append(action)
            legal = list(self.ienv.actions(state))
        else:
            if self.ienv.is_success(state):
                reward = 1.0

        self.rollouts += 1
        if reward == 1.0:
            plan = [e.action for e in path_edges] + rollout_actions
            self._consider(plan, self.budget - remaining)
        # 4) Backpropagation over the concrete (node, edge) descent path only.
        self._backprop(path_nodes, path_edges, reward)
        return charged

    # ------------------------------------------------------------- incumbent
    def _consider_path(
        self, path_nodes: List[UCTNode], path_edges: List[Edge]
    ) -> None:
        plan = [e.action for e in path_edges]
        cost = self.budget - path_nodes[-1].remaining
        self._consider(plan, cost)

    def _consider(self, plan: List[Action], cost: int) -> None:
        """Accept a candidate only after independently executing its concrete
        action sequence: every action must be legal, the final state must be a
        success and total cost must fit the budget. This guards the incumbent
        against any node-merging confusion; reward propagation and incumbent
        cost are separate mechanisms."""
        if cost < 0 or cost > self.budget:
            return
        state = self.ienv.initial_state()
        total = 0
        for action in plan:
            legal = self.ienv.actions(state)
            if action not in legal:
                return
            result = self.ienv.step(state, action)  # cached after first search
            total += result.cost
            if total > self.budget:
                return
            state = result.next_state
        if not self.ienv.is_success(state):
            return
        if self.best_cost is None or total < self.best_cost:
            self.best_plan = list(plan)
            self.best_cost = total

    # ----------------------------------------------------------- backprop
    def _backprop(
        self, path_nodes: List[UCTNode], path_edges: List[Edge], reward: float
    ) -> None:
        """Update node and per-parent-edge stats along the concrete descent.
        Nodes on the path are updated once even if a transposition repeats; the
        reward is never sent through any 'parent pointer' (shared nodes may
        have several parents), which is the graph-safe adaptation of the tree
        rule."""
        for node in path_nodes:
            node.visits += 1
            node.reward_sum += reward
        for edge in path_edges:
            edge.visits += 1
            edge.reward_sum += reward

    def _new_call_possible(self) -> bool:
        """BFS over tried, payable tree edges. True iff some reachable node has
        an untried, non-blocked edge whose (state, action) transition is not yet
        cached, i.e. a further real environment call is structurally possible.
        """
        seen = {id(self.root)}
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            self._ensure_edges(node)
            for edge in node.edges:
                if edge.blocked:
                    continue
                if edge.visits == 0 or edge.child_key is None:
                    if (node.state, edge.action) not in self.ienv._cache:
                        return True
                    continue
                child = self.table.get(edge.child_key)
                if child is None or id(child) in seen:
                    continue
                seen.add(id(child))
                queue.append(child)
        return False

    # ----------------------------------------------------------- saturation
    def _saturated(self) -> bool:
        """True iff every edge reachable from the root through payable,
        within-budget tree edges has been tried at least once. With positive
        integer costs and the ``(state, remaining)`` key the reachable graph is
        finite, so this is a well-defined stopping test."""
        seen = {id(self.root)}
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            self._ensure_edges(node)
            for edge in node.edges:
                if edge.blocked:
                    continue
                if edge.child_key is None or edge.visits == 0:
                    return False  # a payable edge has never been expanded
                child = self.table.get(edge.child_key)
                if child is None or id(child) in seen:
                    continue
                seen.add(id(child))
                queue.append(child)
        return True


def search_uct(
    env: Environment,
    budget: int,
    call_budget: int,
    *,
    seed: int = 0,
    exploration: float = _SQRT2,
) -> UCTResult:
    """One-shot UCT search. Returns found/not-found plus the best (lowest
    observed cost) valid plan. Never asserts optimality or infeasibility."""
    searcher = UCTSearcher(env, budget, seed=seed, exploration=exploration)
    return searcher.run(call_budget, polish=True)
