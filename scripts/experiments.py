"""Run the comparison experiments and emit reports under ``reports/``.

Outputs:
  reports/budget_curve.csv     per (call_budget, seed) UCT rows
  reports/summary.txt          human-readable summary incl. oracle + gaps

Run:  python3 scripts/experiments.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from budget_uct import (  # noqa: E402
    environments as E,
    execute_plan,
    memo_dfs,
    search_uct,
    solve_exact,
    STATE_ONLY,
    STATE_REMAIN,
)
from budget_uct.env import InstrumentedEnv  # noqa: E402

REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
SEEDS = list(range(20))


@dataclass
class Validity:
    ok: bool
    final_state: object
    cost: Optional[int]
    reason: str


def validate_plan(env, budget: int, plan) -> Validity:
    if plan is None:
        return Validity(False, None, None, "no plan returned")
    try:
        state, cost = execute_plan(env, env.initial_state(), plan)
    except ValueError as exc:
        return Validity(False, None, None, f"illegal execution: {exc}")
    if cost > budget:
        return Validity(False, state, cost, f"cost {cost} exceeds budget {budget}")
    if not env.is_success(state):
        return Validity(False, state, cost, f"ends at non-success {state!r}")
    return Validity(True, state, cost, "")


def curve_for(env, budget: int, call_budgets: List[int], optimal: Optional[int]):
    rows = []
    for call_budget in call_budgets:
        for seed in SEEDS:
            fresh = type(env)(env.spec)
            result = search_uct(fresh, budget, call_budget=call_budget, seed=seed)
            check = validate_plan(env, budget, result.plan) if result.found else None
            gap = None
            if result.found and optimal is not None:
                gap = result.cost - optimal
            rows.append(
                {
                    "call_budget": call_budget,
                    "seed": seed,
                    "found": int(result.found),
                    "cost": result.cost if result.found else "",
                    "gap_to_optimal": gap if gap is not None else "",
                    "real_calls": result.real_calls,
                    "cache_hits": result.cache_hits,
                    "iterations": result.iterations,
                    "rollouts": result.rollouts,
                    "valid": "" if check is None else int(check.ok),
                    "invalid_reason": "" if check is None or check.ok else check.reason,
                }
            )
    return rows


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["call_budget"]].append(row)
    summary = []
    for cb in sorted(grouped):
        bucket = grouped[cb]
        found = [r for r in bucket if r["found"]]
        optimal = [r for r in found if r["gap_to_optimal"] == 0]
        summary.append(
            {
                "call_budget": cb,
                "n": len(bucket),
                "found_rate": round(len(found) / len(bucket), 3),
                "optimal_rate": round(len(optimal) / len(bucket), 3),
                "mean_cost": round(sum(r["cost"] for r in found) / len(found), 3)
                if found
                else "",
                "worst_gap": max((r["gap_to_optimal"] for r in found), default=""),
                "mean_calls": round(sum(r["real_calls"] for r in bucket) / len(bucket), 2),
            }
        )
    return summary


def main() -> int:
    os.makedirs(REPORTS, exist_ok=True)
    lines: List[str] = []

    def emit(text=""):
        print(text)
        lines.append(text)

    # ---------------- Demo environment: transposition / trap / cost cycles
    demo = E.demo_env()
    emit("== Demo environment ==")
    opt15 = solve_exact(type(demo)(demo.spec), budget=15)
    emit(
        f"exact budget=15: status={opt15.status} plan={opt15.plan} "
        f"cost={opt15.cost} proven={opt15.proven} "
        f"real_calls={opt15.real_calls} cache_hits={opt15.cache_hits}"
    )
    opt4 = solve_exact(type(demo)(demo.spec), budget=4)
    emit(
        f"exact budget=4 : status={opt4.status} proven={opt4.proven} "
        f"real_calls={opt4.real_calls} (infeasibility proof)"
    )

    # ---------------- Layered environment for the budget curve
    layer = E.layered_env(layers=6, width=4)
    big_budget = 100
    oracle = solve_exact(type(layer)(layer.spec), budget=big_budget)
    emit("")
    emit("== Layered environment (layers=6, width=4) ==")
    emit(
        f"oracle budget={big_budget}: status={oracle.status} "
        f"optimal_cost={oracle.cost} plan={oracle.plan} "
        f"real_calls={oracle.real_calls} states_settled={oracle.states_settled}"
    )
    execution_budget = oracle.cost + 8
    call_budgets = [8, 16, 24, 32, 44, 56, 74]
    rows = curve_for(layer, execution_budget, call_budgets, oracle.cost)

    csv_path = os.path.join(REPORTS, "budget_curve.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    emit(f"execution budget={execution_budget}; seeds={len(SEEDS)}")
    emit("call_budget | found | optimal | mean_cost | worst_gap | mean_real_calls")
    for stat in aggregate(rows):
        emit(
            f"{stat['call_budget']:>11} | {stat['found_rate']:5.3f} | "
            f"{stat['optimal_rate']:7.3f} | {str(stat['mean_cost']):>9} | "
            f"{str(stat['worst_gap']):>9} | {stat['mean_calls']}"
        )

    invalid = [r for r in rows if r["found"] and not r["valid"]]
    emit(f"invalid plans returned by UCT: {len(invalid)} (must be 0)")

    # ---------------- Exact oracle calls under small execution budgets
    emit("")
    emit("== Exact solver as oracle across execution budgets (layered) ==")
    for b in range(oracle.cost - 2, oracle.cost + 4):
        res = solve_exact(type(layer)(layer.spec), budget=max(0, b))
        emit(
            f"budget={max(0, b):>3}: {res.status:>10} cost={res.cost} "
            f"calls={res.real_calls} settled={res.states_settled} proven={res.proven}"
        )

    # ---------------- Counterexample to state-only de-duplication
    ce = E.dedup_counterexample_env()
    emit("")
    emit("== Counterexample: de-duplicating by environment state alone ==")
    bad = memo_dfs(type(ce)(ce.spec), budget=12, dedup=STATE_ONLY)
    good = memo_dfs(type(ce)(ce.spec), budget=12, dedup=STATE_REMAIN)
    truth = solve_exact(type(ce)(ce.spec), budget=12)
    emit(f"state-only dedup : found={bad.found} plan={bad.plan} (WRONG: feasible plan exists)")
    emit(f"(state,remaining): found={good.found} plan={good.plan} cost={good.cost}")
    emit(f"exact oracle     : status={truth.status} plan={truth.plan} cost={truth.cost}")

    with open(os.path.join(REPORTS, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    emit("")
    emit(f"wrote {csv_path} and reports/summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
