"""Deterministic shortest-cost action search: UCT vs an exact oracle.

This package compares:

* :class:`uctshortest.uct.UCTAgent` -- a UCT/Monte-Carlo tree-search agent over
  the (state, remaining-budget) acyclic search graph;
* :class:`uctshortest.exact.ShortestCostSolver` -- an independent exact solver
  (Dijkstra) that proves optimality or the absence of any feasible plan.

No external models are used.
"""

from .environment import Environment, EnvironmentView, PlanError

__all__ = ["Environment", "EnvironmentView", "PlanError"]
