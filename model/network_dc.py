from __future__ import annotations

def add_dc_network(m, indexer, lines, T: int, slack_bus_id: int):
    """Add DC network constraints to a gurobipy.Model."""
    from gurobipy import GRB

    nb = len(indexer.bus_ids)
    ne = len(indexer.edge_keys)

    theta = m.addMVar((T, nb), lb=-GRB.INFINITY, name="theta")
    P_line = m.addMVar((T, ne), lb=-GRB.INFINITY, name="P_line")

    slack_idx = indexer.bus_to_idx[slack_bus_id]
    for t in range(T):
        m.addConstr(theta[t, slack_idx] == 0.0, name=f"slack_theta[{t}]")

    for e, ln in enumerate(lines):
        if abs(ln.x) < 1e-9:
            raise ValueError(f"Line {ln.line_id} has x≈0, DC flow undefined.")
        i = indexer.bus_to_idx[ln.from_bus]
        j = indexer.bus_to_idx[ln.to_bus]
        m.addConstr(P_line[:, e] == (theta[:, i] - theta[:, j]) / ln.x, name=f"dcflow[{ln.line_id}]")
        m.addConstr(P_line[:, e] <= ln.rate_kw, name=f"line_ub[{ln.line_id}]")
        m.addConstr(P_line[:, e] >= -ln.rate_kw, name=f"line_lb[{ln.line_id}]")

    return theta, P_line
