from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from data.schemas import SystemData
from model.aclin_opt_relaxed_v2 import build_aclin_optimization, ACLinParams
from results.postprocess import extract_results
from engine.gurobi_runner import solve_gurobi


def run_optimization(system: SystemData, data_dir: Path | None = None) -> Tuple[object, Dict]:
    """
    Called by GUI controller. Returns (ResultData, meta).
    """
    # Default price if not provided
    grid_price = getattr(system, "price", None)
    if grid_price is None:
        system.price = np.ones(system.T, dtype=float)
    else:
        system.price = np.array(grid_price, dtype=float).reshape(-1)

    # Params from sys_params.yaml (optional)
    p = getattr(system, "params", {}) or {}
    params = ACLinParams(
        s_base_kva=float(p.get("s_base_kva", 10000.0)),
        v_min=float(p.get("v_min", 0.9)),
        v_max=float(p.get("v_max", 1.1)),
        temp_out_c=float(p.get("temp_out_c", 25.0)),
        q_over_p_reefer=float(p.get("q_over_p_reefer", 0.8)),
        q_over_p_ci=float(p.get("q_over_p_ci", 0.9)),
        allow_export=bool(p.get("allow_export", True)),
        export_price=float(p.get("export_price", 0.0)),
    )

    # Build + solve
    m, idx, var = build_aclin_optimization(system, params=params)
    solve_meta = solve_gurobi(m)

    # Extract results into the object TopologyView already consumes
    result = extract_results(system, idx, var)

    meta = {"engine": "gurobi", "message": solve_meta.get("message", ""), **solve_meta}
    return result, meta
