from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.indexer import build_indexer
from model.devices import add_ess


@dataclass(frozen=True)
class ACLinParams:
    s_base_kva: float = 10000.0
    v_min: float = 0.9
    v_max: float = 1.1
    rho_v: float = 1e5  # penalty weight for voltage magnitude relaxation
    rho_temp: float = 1e4  # penalty weight for reefer temperature bound relaxation (degC)
    transformer_capacity_kva: float = 0.0  # if >0, compute grid/transformer loading based on p_grid,q_grid
    temp_out_c: float = 25.0
    q_over_p_reefer: float = 0.8   # 你的 Matlab 里用 q_R = eta_R * p_R
    q_over_p_ci: float = 0.9       # 你的 Matlab 里用 q_CI = eta_CI * p_CI
    allow_export: bool = False     # 对齐你 Matlab 的 p_MG >= 0（不允许反送）
    export_price: float = 0.0


def _build_ybus_from_lines(bus_ids: list[int], lines) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Ybus (no shunts) from series impedance lines (r,x).
    Assume r/x are in consistent units (e.g. per-unit).
    Return (G, B) such that Ybus = G + jB.
    """
    n = len(bus_ids)
    bus_to_i = {bid: k for k, bid in enumerate(bus_ids)}
    Y = np.zeros((n, n), dtype=complex)

    for ln in lines:
        i = bus_to_i[int(ln.from_bus)]
        j = bus_to_i[int(ln.to_bus)]
        r = float(ln.r)
        x = float(ln.x)
        z = complex(r, x)
        if abs(z) <= 1e-12:
            raise ValueError(f"Line {ln.line_id} has near-zero impedance.")
        y = 1.0 / z
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y

    return Y.real, Y.imag


def build_aclin_optimization(system, params: ACLinParams):
    """
    Deterministic single-scenario MILP:
      - Linearized AC nodal injection equations (G,B)*V,theta
      - ESS SOC dynamics + charge/discharge split + optional binary
      - Reefers on/off power bounds + linear thermal dynamics (implicit -> linear equality)
      - Line active flow definition + thermal limit
    """
    import gurobipy as gp
    from gurobipy import GRB

    T = int(system.T)
    dt_hours = float(system.dt_hours)

    idx = build_indexer(system.buses, system.lines)
    nb = len(idx.bus_ids)
    nL = len(idx.edge_keys)

    # Slack bus
    slack_list = [b.bus_id for b in system.buses if int(getattr(b, "is_slack", 0)) == 1]
    slack_bus = slack_list[0] if slack_list else idx.bus_ids[0]
    slack_i = idx.bus_to_idx[slack_bus]

    # Ybus (G,B)
    G, B = _build_ybus_from_lines(idx.bus_ids, system.lines)

    m = gp.Model("port_ops_aclin")

    # ---------------- Variables ----------------
    # Voltage magnitude variables (relaxed bounds via slack variables below)
    V = m.addMVar((nb, T), lb=0.0, ub=GRB.INFINITY, name="V")
    # NOTE: V_slack was unused; remove to avoid confusion.
    theta = m.addMVar((nb, T), lb=-GRB.INFINITY, name="theta")

    # Voltage magnitude relaxation (guarantee feasibility)
    sV_under = m.addMVar((nb, T), lb=0.0, name="sV_under")
    sV_over  = m.addMVar((nb, T), lb=0.0, name="sV_over")
    m.addConstr(V >= params.v_min - sV_under, name="Vmin_relaxed")
    m.addConstr(V <= params.v_max + sV_over,  name="Vmax_relaxed")

    # Slack fix
    m.addConstr(V[slack_i, :] == 1.0, name="slack_V")
    m.addConstr(theta[slack_i, :] == 0.0, name="slack_theta")

    # Grid import/export split (to optionally allow export)
    p_grid_pos = m.addMVar((T,), lb=0.0, name="p_grid_pos")  # import
    p_grid_neg = m.addMVar((T,), lb=0.0, name="p_grid_neg")  # export
    if not params.allow_export:
        m.addConstr(p_grid_neg == 0.0, name="no_export")
    p_grid = p_grid_pos - p_grid_neg

    q_grid = m.addMVar((T,), lb=-GRB.INFINITY, name="q_grid")

     # ESS (devices.add_ess returns: (something, SOC, p_ch, p_dis))
    _ignored, SOC_start, p_ch, p_dis = add_ess(m, T, dt_hours, system.ess, no_simultaneous=True)

    # IMPORTANT: make net ESS power a real variable (avoid MLinExpr in var_dict)
    P_ess = m.addMVar((T,), lb=-GRB.INFINITY, name="P_ess")
    m.addConstr(P_ess == (p_dis - p_ch), name="P_ess_def")
    # reactive from ESS (free)
    q_ess = m.addMVar((T,), lb=-GRB.INFINITY, name="q_ess")

    tugboat_demands = list(getattr(system, "tugboat_energy_demands", []) or [])
    tugboat_groups = sorted({str(td.group) for td in tugboat_demands})
    nTg = len(tugboat_groups)
    if nTg > 0:
        p_tugboat = m.addMVar((nTg, T), lb=0.0, name="p_tugboat")
        tugboat_group_to_idx = {g: i for i, g in enumerate(tugboat_groups)}
        for td_idx, td in enumerate(tugboat_demands):
            g_idx = tugboat_group_to_idx[str(td.group)]
            st = int(td.start_time)
            en = int(td.end_time)
            st_emin = int(td.start_time_emin)
            en_emin = int(td.end_time_emin)
            m.addConstr(
                p_tugboat[g_idx, st:en].sum() * dt_hours == float(td.E),
                name=f"tugboat_E[{td_idx}]",
            )
            m.addConstr(
                p_tugboat[g_idx, st_emin:en_emin].sum() * dt_hours >= float(td.Emin),
                name=f"tugboat_Emin[{td_idx}]",
            )
    else:
        p_tugboat = None

    # Reefers
    nR = len(system.reefers)
    if nR > 0:
        p_rf = m.addMVar((nR, T), lb=0.0, name="p_rf")
        u_rf = m.addMVar((nR, T), vtype=GRB.BINARY, name="u_rf")
        Temp = m.addMVar((nR, T), lb=-GRB.INFINITY, name="Temp")
        sT_under = m.addMVar((nR, T), lb=0.0, name="sT_under")
        sT_over  = m.addMVar((nR, T), lb=0.0, name="sT_over")

        # Collect per-reefer params
        area = np.zeros(nR, dtype=float)
        h = np.zeros(nR, dtype=float)
        c = np.zeros(nR, dtype=float)
        mass = np.zeros(nR, dtype=float)
        tmin = np.zeros(nR, dtype=float)
        tmax = np.zeros(nR, dtype=float)
        tini = np.zeros(nR, dtype=float)
        pmin = np.zeros(nR, dtype=float)
        pmax = np.zeros(nR, dtype=float)

        for r, rf in enumerate(system.reefers):
            ct = system.content_types[rf.content_type]
            rt = system.reefer_types[rf.reefer_type]
            area[r] = float(rt.area_m2)
            h[r] = float(rt.heat_transition_coefficient_w_per_m2k)
            c[r] = float(ct.specific_heat_capacity_j_per_kgk)
            mass[r] = float(rf.mass_kg)
            tmin[r] = float(ct.min_temp)
            tmax[r] = float(ct.max_temp)
            tini[r] = float(rf.ini_temp)
            pmin[r] = float(rt.power_min_kw)
            pmax[r] = float(rt.power_max_kw)

        # alpha/beta
        k = (area * h * dt_hours * 3600.0) / (mass * c + 1e-12)
        alpha = 1.0 - np.exp(-k)
        beta = (1000.0 * dt_hours * 3600.0) / (mass * c + 1e-12)

        # Power bounds with on/off
        for r in range(nR):
            m.addConstr(p_rf[r, :] >= u_rf[r, :] * pmin[r], name=f"rf_pmin[{r}]")
            m.addConstr(p_rf[r, :] <= u_rf[r, :] * pmax[r], name=f"rf_pmax[{r}]")

        # Linear thermal dynamics (implicit -> linear)
        Tout = float(params.temp_out_c)
        for r in range(nR):
            a = float(alpha[r])
            b = float(beta[r])
            # t=0
            m.addConstr((1.0 + a) * Temp[r, 0] + b * p_rf[r, 0] == float(tini[r]) + a * Tout,
                        name=f"rf_temp_init[{r}]")
            # t>=1
            for t in range(1, T):
                m.addConstr((1.0 + a) * Temp[r, t] - Temp[r, t - 1] + b * p_rf[r, t] == a * Tout,
                            name=f"rf_temp[{r},{t}]")            # Temperature bounds with relaxation
            for t in range(T):
                m.addConstr(Temp[r, t] >= float(tmin[r]) - sT_under[r, t], name=f"rf_tmin[{r},{t}]")
                m.addConstr(Temp[r, t] <= float(tmax[r]) + sT_over[r, t], name=f"rf_tmax[{r},{t}]")
    else:
        p_rf = None
        u_rf = None
        Temp = None
        sT_under = None
        sT_over = None

    # ---------------- Aggregation by bus ----------------
    reefers_at_bus: list[list[int]] = [[] for _ in range(nb)]
    if nR > 0:
        for r, rf in enumerate(system.reefers):
            bi = idx.bus_to_idx[int(rf.node_number)]
            reefers_at_bus[bi].append(r)

    tugboat_groups_at_bus: list[list[int]] = [[] for _ in range(nb)]
    if nTg > 0:
        tugboat_group_to_bus = dict(getattr(system, "tugboat_group_to_node", {}) or {})
        for g in tugboat_groups:
            if str(g) not in tugboat_group_to_bus:
                raise ValueError(f"Missing tugboat group->node mapping for group: {g}")
            bi_tg = int(tugboat_group_to_bus[str(g)])
            if bi_tg not in idx.bus_to_idx:
                raise ValueError(f"Tugboat group {g} maps to unknown bus: {bi_tg}")
            tugboat_groups_at_bus[idx.bus_to_idx[bi_tg]].append(tugboat_group_to_idx[g])

    load_kw = np.array(system.load_kw, dtype=float)  # (nb,T)
    pv_kw = np.array(system.pv_kw, dtype=float)      # (nb,T)

    cold_fix_bus = np.zeros((nb, T), dtype=float)

    ess_i = idx.bus_to_idx[int(system.ess.node_number)]
    S = float(params.s_base_kva)

    # ---------------- AC linear injection equations ----------------
    for t in range(T):
        for i in range(nb):
            # Cold ironing at bus i
            p_cold_var = 0.0
            p_cold_total = p_cold_var + float(cold_fix_bus[i, t])

            p_tugboat_i = 0.0
            if nTg > 0 and tugboat_groups_at_bus[i]:
                p_tugboat_i = gp.quicksum(p_tugboat[g_idx, t] for g_idx in tugboat_groups_at_bus[i])

            # Reefer at bus i
            p_rf_i = 0.0
            if nR > 0 and reefers_at_bus[i]:
                p_rf_i = gp.quicksum(p_rf[r, t] for r in reefers_at_bus[i])

            # Active injection (positive = inject to grid)
            P_inj = float(pv_kw[i, t]) - float(load_kw[i, t]) - p_cold_total - p_rf_i - p_tugboat_i
            if i == slack_i:
                P_inj = P_inj + p_grid[t]
            if i == ess_i:
                P_inj = P_inj + P_ess[t]

            rhs_p = gp.quicksum(G[i, j] * V[j, t] for j in range(nb)) - gp.quicksum(
                B[i, j] * theta[j, t] for j in range(nb)
            )
            m.addConstr(P_inj / S == rhs_p, name=f"ac_pinj[{i},{t}]")

            # Reactive (no base load Q in current sysdata; if you have Q later再加)
            q_cold = params.q_over_p_ci * (p_cold_total + p_tugboat_i)
            q_rf_ = 0.0
            if nR > 0 and reefers_at_bus[i]:
                q_rf_ = gp.quicksum(params.q_over_p_reefer * p_rf[r, t] for r in reefers_at_bus[i])

            Q_inj = -q_cold - q_rf_
            if i == slack_i:
                Q_inj = Q_inj + q_grid[t]
            if i == ess_i:
                Q_inj = Q_inj - q_ess[t]

            rhs_q = -gp.quicksum(B[i, j] * V[j, t] for j in range(nb)) - gp.quicksum(
                G[i, j] * theta[j, t] for j in range(nb)
            )
            m.addConstr(Q_inj / S == rhs_q, name=f"ac_qinj[{i},{t}]")

    # ---------------- Line flows for limits + GUI display ----------------
    P_line = m.addMVar((T, nL), lb=-GRB.INFINITY, name="P_line")
    Q_line = m.addMVar((T, nL), lb=-GRB.INFINITY, name="Q_line")
    for e, ln in enumerate(system.lines):
        i = idx.bus_to_idx[int(ln.from_bus)]
        j = idx.bus_to_idx[int(ln.to_bus)]
        z = complex(float(ln.r), float(ln.x))
        y = 1.0 / z
        g = float(y.real)
        b = float(y.imag)

        m.addConstr(
            P_line[:, e] == (g * (V[i, :] - V[j, :]) - b * (theta[i, :] - theta[j, :])) * S,
            name=f"pline_def[{e}]",
        )
        m.addConstr(
            Q_line[:, e] == (-b * (V[i, :] - V[j, :]) - g * (theta[i, :] - theta[j, :])) * S,
            name=f"qline_def[{e}]",
        )
        rate = float(getattr(ln, "rate_kw", 1e18))
        m.addConstr(P_line[:, e] <= rate, name=f"pline_ub[{e}]")
        m.addConstr(P_line[:, e] >= -rate, name=f"pline_lb[{e}]")

    # ---------------- Objective ----------------
    # If you later add sys_price.csv into SystemData as system.price, this will use it automatically.
    price = getattr(system, "price", None)
    if price is None:
        price = np.ones(T, dtype=float)
    else:
        price = np.array(price, dtype=float).reshape(-1)
        if price.size != T:
            raise ValueError(f"price length must be T={T}, got {price.size}")

    obj = gp.quicksum(price[t] * p_grid_pos[t] - params.export_price * p_grid_neg[t] for t in range(T))
    # Penalize voltage relaxation to keep voltages within limits when feasible
    obj += params.rho_v * gp.quicksum(sV_under[i, t] + sV_over[i, t] for i in range(nb) for t in range(T))
    if nR > 0:
        obj += params.rho_temp * gp.quicksum(sT_under[r, t] + sT_over[r, t] for r in range(nR) for t in range(T))
    m.setObjective(obj, GRB.MINIMIZE)

    var = {
        "V": V,
        "theta": theta,
        "P_line": P_line,
        "Q_line": Q_line,
        "P_ess": P_ess,
        "SOC": SOC_start,
        "p_grid_pos": p_grid_pos,
        "p_grid_neg": p_grid_neg,
        "p_grid": p_grid,
        "q_grid": q_grid,
        "sV_under": sV_under,
        "sV_over": sV_over,
    }
    if p_tugboat is not None:
        var["p_tugboat"] = p_tugboat
    if nR > 0:
        var["p_rf"] = p_rf
        var["Temp"] = Temp
        var["u_rf"] = u_rf

    return m, idx, var
