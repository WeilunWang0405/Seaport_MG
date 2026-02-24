from __future__ import annotations

from typing import Any, Dict, Optional

def solve_gurobi(model, time_limit: Optional[float] = None,
                 mip_gap: Optional[float] = None,
                 threads: Optional[int] = None,
                 verbose: bool = True) -> Dict[str, Any]:
    """
    Solve a gurobipy.Model and return meta info.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(
            "gurobipy is not available. Please install Gurobi Python package and make sure license is configured."
        ) from e

    # basic params
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)
    if mip_gap is not None:
        model.Params.MIPGap = float(mip_gap)
    if threads is not None:
        model.Params.Threads = int(threads)

    model.optimize()

    # ===== add here: status diagnostics/export (right after optimize) =====
    status = int(model.Status)
    if status in (GRB.INFEASIBLE, getattr(GRB, "INF_OR_UNBD", -999)):
        # 可选：导出模型方便排查（路径你也可以改成项目 debug 目录）
        try:
            model.write("debug_model.lp")
        except Exception:
            pass
        # IIS
        if status == GRB.INFEASIBLE:
            try:
                model.computeIIS()
                model.write("debug_infeasible.ilp")
            except Exception:
                pass

    meta = {
        "status": status,
        "obj_val": None,
        "runtime_sec": getattr(model, "Runtime", None),
        "mip_gap": getattr(model, "MIPGap", None),
        "message": "",
    }

    if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        try:
            meta["obj_val"] = float(model.ObjVal)
        except Exception:
            meta["obj_val"] = None
        meta["message"] = "Solved."
    elif status == GRB.INFEASIBLE:
        meta["message"] = "Infeasible."
    elif status == GRB.UNBOUNDED:
        meta["message"] = "Unbounded."
    else:
        meta["message"] = f"Finished with status={status}."

    return meta
