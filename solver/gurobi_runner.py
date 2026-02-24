from __future__ import annotations

class SolveError(RuntimeError):
    pass

def solve_model(model, log_path: str | None = None, mip_gap: float = 1e-3, time_limit: float = 120):
    """Solve a gurobipy.Model with standard parameters and consistent error handling."""
    try:
        from gurobipy import GRB
    except Exception as e:
        raise SolveError("gurobipy is not available in this environment.") from e

    if log_path:
        model.setParam("LogFile", log_path)
    model.setParam("MIPGap", mip_gap)
    model.setParam("TimeLimit", time_limit)

    model.optimize()

    # ===== add here: early dump for debugging =====
    try:
        model.write("debug_model.lp")
    except Exception:
        pass

    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return model

    # INF_OR_UNBD: rerun with DualReductions=0 to disambiguate
    if model.Status == getattr(GRB, "INF_OR_UNBD", -999):
        try:
            model.setParam("DualReductions", 0)
            model.optimize()
        except Exception:
            pass

    if model.Status == GRB.INFEASIBLE:
        try:
            model.computeIIS()
            model.write("debug_infeasible.ilp")
        except Exception:
            pass
        raise SolveError("Model infeasible (IIS written to debug_infeasible.ilp).")

    if model.Status == GRB.UNBOUNDED:
        raise SolveError("Model unbounded.")

    if model.Status == GRB.TIME_LIMIT:
        raise SolveError("Time limit reached.")

    raise SolveError(f"Gurobi ended with status={model.Status}.")

