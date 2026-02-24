from __future__ import annotations
import numpy as np

def add_ess(m, T: int, dt_hours: float, ess, no_simultaneous: bool = True):
    from gurobipy import GRB

    p_ch = m.addMVar(T, lb=0.0, ub=ess.max_charging_power_kw, name="p_ch")
    p_dis = m.addMVar(T, lb=0.0, ub=ess.max_discharging_power_kw, name="p_dis")

    if no_simultaneous:
        y = m.addMVar(T, vtype=GRB.BINARY, name="ess_mode")  # 1=charge
        m.addConstr(p_ch <= ess.max_charging_power_kw * y, name="ess_ch_mode")
        m.addConstr(p_dis <= ess.max_discharging_power_kw * (1 - y), name="ess_dis_mode")

    soc = m.addMVar(T + 1, lb=ess.min_soc, ub=ess.max_soc, name="soc")
    m.addConstr(soc[0] == ess.initial_soc, name="soc_init")
    m.addConstr(soc[T] == ess.end_soc, name="soc_end")

    cap = ess.capacity_kwh
    for t in range(T):
        delta = (ess.eta_ch * p_ch[t] * dt_hours - (p_dis[t] * dt_hours) / ess.eta_dis) / cap
        m.addConstr(soc[t + 1] == soc[t] + delta, name=f"soc_dyn[{t}]")

    P_ess = p_dis - p_ch
    SOC_start = soc[0:T]
    return P_ess, SOC_start, p_ch, p_dis

def add_cold_ironing(m, T: int, dt_hours: float, tasks):
    K = len(tasks)
    if K == 0:
        return None

    lb = np.zeros((T, K), dtype=float)
    ub = np.zeros((T, K), dtype=float)
    for k, tk in enumerate(tasks):
        lb[tk.start_time:tk.end_time, k] = tk.min_charging_power_kw
        ub[tk.start_time:tk.end_time, k] = tk.max_charging_power_kw

    p_task = m.addMVar((T, K), lb=lb, ub=ub, name="p_cold_task")
    for k, tk in enumerate(tasks):
        m.addConstr(p_task[tk.start_time:tk.end_time, k].sum() * dt_hours == tk.battery_demand_kwh,
                    name=f"cold_energy[{tk.task_number}]")
    return p_task
