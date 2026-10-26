"""Reproduce a capacity counterexample to printed equations (19)–(26)."""
from __future__ import annotations

import json
from pathlib import Path

from .data import Settings, read_instance
from .models import SolveOptions, _callback, build_model


def audit() -> dict:
    import gurobipy as gp
    p = read_instance(Path(__file__).with_name("cases") / "compact_counterexample.txt",
                      Settings(gamma_q=1, gamma_t=0, sigma_q=0.3, sigma_t=0))
    route = [0, 1, 2, 4, 3, 5, 6, 7]
    report = dict(**p.metadata(), route=route, nominal_onboard_after_node3=9,
                  worst_onboard_after_node3=11, methods={})
    try:
        p.validate([route])
    except ValueError as error:
        report["replay_error"] = str(error)
    else:
        raise AssertionError("counterexample unexpectedly became feasible")
    with gp.Env(params={"OutputFlag": 0}) as env:
        for method in ("compact", "compact-guarded", "branch-cut", "onboard-compact"):
            built = build_model(p, method, SolveOptions(seconds=10, warm_start=False), env=env)
            with built.model as model:
                chosen = set(zip(route, route[1:]))
                for arc, variable in built.x.items():
                    variable.LB = variable.UB = int(arc in chosen)
                if method in ("compact-guarded", "branch-cut"):
                    model.optimize(lambda m, where: _callback(built, m, where))
                    if built.callback_error:
                        raise RuntimeError("counterexample callback failed") from built.callback_error
                else:
                    model.optimize()
                status = "Optimal" if model.Status == gp.GRB.OPTIMAL else "Infeasible" if model.Status == gp.GRB.INFEASIBLE else str(model.Status)
                entry = dict(status=status, cuts=built.cut_counts)
                if model.SolCount:
                    entry["load_variables"] = {v.VarName: v.X for v in model.getVars()
                                               if v.VarName.startswith(("Q[", "W[")) and v.X}
                report["methods"][method] = entry
    assert report["methods"]["compact"]["status"] == "Optimal"
    assert all(r["status"] == "Infeasible" for name, r in report["methods"].items() if name != "compact")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
