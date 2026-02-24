from __future__ import annotations
import numpy as np
from data.indexer import build_indexer
from results.types import ResultData

def run_mock_dc(system) -> ResultData:
    """Fallback: no optimization, but produces consistent DC power-flow outputs."""
    T = system.T
    dt = system.dt_hours
    indexer = build_indexer(system.buses, system.lines)
    nb = len(indexer.bus_ids)
    ne = len(indexer.edge_keys)

    slack_candidates = [b.bus_id for b in system.buses if int(getattr(b, "is_slack", 0)) == 1]
    if not slack_candidates:
        raise ValueError("No slack bus found (is_slack=1).")
    slack_bus = slack_candidates[0]
    slack_idx = indexer.bus_to_idx[slack_bus]

    # Cold-ironing variable charging (uniform energy over window)
    K = len(system.cold_ironing)
    p_task = np.zeros((T, K), dtype=float)
    for k, tk in enumerate(system.cold_ironing):
        window = tk.end_time - tk.start_time
        if window <= 0:
            continue
        avg = tk.battery_demand_kwh / (window * dt)
        avg = float(np.clip(avg, tk.min_charging_power_kw, tk.max_charging_power_kw))
        p_task[tk.start_time:tk.end_time, k] = avg

    # Fixed cold demand by bus/time
    cold_fix = np.zeros((nb, T), dtype=float)
    for tk in system.cold_ironing:
        bi = indexer.bus_to_idx[tk.node_number]
        cold_fix[bi, tk.start_time:tk.end_time] += tk.demand_fix_kw

    # Aggregate cold var by bus/time
    cold_var = np.zeros((nb, T), dtype=float)
    for k, tk in enumerate(system.cold_ironing):
        bi = indexer.bus_to_idx[tk.node_number]
        cold_var[bi, :] += p_task[:, k]

    # ESS inactive
    P_ess = np.zeros(T, dtype=float)
    SOC = np.full(T, system.ess.initial_soc, dtype=float)

    # Net injection excluding grid at slack
    inj = system.pv_kw - system.load_kw - cold_fix - cold_var  # (nb, T)

    # Grid balances at slack
    inj_slack = -inj.sum(axis=0)
    inj[slack_idx, :] += inj_slack

    # Build susceptance matrix
    B = np.zeros((nb, nb), dtype=float)
    for ln in system.lines:
        i = indexer.bus_to_idx[ln.from_bus]
        j = indexer.bus_to_idx[ln.to_bus]
        if abs(ln.x) < 1e-9:
            continue
        bij = 1.0 / ln.x
        B[i, i] += bij
        B[j, j] += bij
        B[i, j] -= bij
        B[j, i] -= bij

    theta = np.zeros((T, nb), dtype=float)
    keep = [i for i in range(nb) if i != slack_idx]
    Brr = B[np.ix_(keep, keep)]
    for tt in range(T):
        Pr = inj[keep, tt]
        try:
            theta_r = np.linalg.solve(Brr, Pr)
        except np.linalg.LinAlgError:
            theta_r = np.linalg.lstsq(Brr, Pr, rcond=None)[0]
        theta[tt, keep] = theta_r
        theta[tt, slack_idx] = 0.0

    P_line = np.zeros((T, ne), dtype=float)
    for e, ln in enumerate(system.lines):
        i = indexer.bus_to_idx[ln.from_bus]
        j = indexer.bus_to_idx[ln.to_bus]
        P_line[:, e] = (theta[:, i] - theta[:, j]) / ln.x

    V = np.ones((T, nb), dtype=float)
    return ResultData(
        T=T,
        bus_ids=indexer.bus_ids,
        edge_keys=indexer.edge_keys,
        V_bus=V,
        theta_bus=theta,
        P_line=P_line,
        P_ess=P_ess,
        SOC=SOC,
        P_cold_task=p_task
    )
