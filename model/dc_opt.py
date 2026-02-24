from __future__ import annotations
import numpy as np
from data.indexer import build_indexer
from .network_dc import add_dc_network
from .devices import add_ess, add_cold_ironing

def build_dc_optimization(system, grid_price: np.ndarray | None = None, no_simultaneous_ess: bool = True, ess_cycle_penalty: float = 0.0):
    import gurobipy as gp
    from gurobipy import GRB

    T = system.T
    dt = system.dt_hours
    buses = system.buses
    lines = system.lines

    indexer = build_indexer(buses, lines)
    nb = len(indexer.bus_ids)

    slack_candidates = [b.bus_id for b in buses if int(getattr(b, "is_slack", 0)) == 1]
    if not slack_candidates:
        raise ValueError("No slack bus found in sys_bus.csv (is_slack=1).")
    slack_bus_id = slack_candidates[0]
    slack_idx = indexer.bus_to_idx[slack_bus_id]

    m = gp.Model("port_ops_dc")

    theta, P_line = add_dc_network(m, indexer, lines, T, slack_bus_id)

    p_grid_pos = m.addMVar(T, lb=0.0, name="p_grid_pos")
    p_grid_neg = m.addMVar(T, lb=0.0, name="p_grid_neg")
    p_grid = p_grid_pos - p_grid_neg

    P_ess, SOC_start, p_ch, p_dis = add_ess(m, T, dt, system.ess, no_simultaneous=no_simultaneous_ess)
    ess_bus_idx = indexer.bus_to_idx[system.ess.node_number]

    p_cold_task = add_cold_ironing(m, T, dt, system.cold_ironing)
    K = 0 if p_cold_task is None else p_cold_task.shape[1]

    pv = system.pv_kw.copy()
    load = system.load_kw.copy()

    cold_fix = np.zeros((nb, T), dtype=float)
    if system.cold_ironing:
        for tk in system.cold_ironing:
            bi = indexer.bus_to_idx[tk.node_number]
            cold_fix[bi, tk.start_time:tk.end_time] += tk.demand_fix_kw

    # MVP: reefer treated as 0 fixed load here
    reefer_fix = np.zeros((nb, T), dtype=float)

    for t in range(T):
        for bi in range(nb):
            inj = float(pv[bi, t]) - float(load[bi, t]) - float(cold_fix[bi, t]) - float(reefer_fix[bi, t])

            if p_cold_task is not None and K > 0:
                cols = [k for k, tk in enumerate(system.cold_ironing) if tk.node_number == indexer.bus_ids[bi]]
                if cols:
                    inj -= p_cold_task[t, cols].sum()

            if bi == ess_bus_idx:
                inj += P_ess[t]
            if bi == slack_idx:
                inj += p_grid[t]

            expr = inj
            for e, ln in enumerate(lines):
                frm = indexer.bus_to_idx[ln.from_bus]
                to  = indexer.bus_to_idx[ln.to_bus]
                if bi == to:
                    expr += P_line[t, e]
                if bi == frm:
                    expr -= P_line[t, e]
            m.addConstr(expr == 0.0, name=f"p_balance[t={t},bus={indexer.bus_ids[bi]}]")

    if grid_price is None:
        grid_price = np.ones(T, dtype=float)
    grid_price = np.asarray(grid_price, dtype=float).reshape(-1)
    if grid_price.size != T:
        raise ValueError("grid_price length must equal T.")

    obj = grid_price @ p_grid_pos
    if ess_cycle_penalty > 0:
        obj += ess_cycle_penalty * (p_ch.sum() + p_dis.sum())

    m.setObjective(obj, GRB.MINIMIZE)

    var_dict = {
        "theta": theta,
        "P_line": P_line,
        "P_ess": P_ess,
        "SOC": SOC_start,
        "p_cold_task": p_cold_task,
        "p_grid_pos": p_grid_pos,
        "p_grid_neg": p_grid_neg,
    }
    return m, indexer, var_dict
