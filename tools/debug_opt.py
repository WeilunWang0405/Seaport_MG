from __future__ import annotations

import sys
from pathlib import Path

# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path

import numpy as np

from data.io import load_system
from model.aclin_opt_relaxed_v2 import build_aclin_optimization, ACLinParams


def _model_copy(system, **updates):
    """pydantic v2: model_copy; v1: copy(update=...)"""
    if hasattr(system, "model_copy"):
        return system.model_copy(update=updates)
    return system.copy(update=updates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Path to sysdata folder")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--dt", type=float, default=0.5, help="hours")
    ap.add_argument("--no-reefers", action="store_true", help="Disable reefers (set reefers=[])")
    ap.add_argument("--no-ci", action="store_true", help="Disable cold-ironing tasks (set cold_ironing=[])")
    ap.add_argument("--freeze-ess", action="store_true", help="Freeze ESS: set charge/discharge power to 0")
    ap.add_argument("--allow-export", action="store_true", help="Allow export to HV grid (p_grid can be negative)")
    ap.add_argument("--write-lp", action="store_true", help="Write model LP file")
    ap.add_argument("--iis", action="store_true", help="Compute IIS if infeasible")
    ap.add_argument("--out", type=str, default="debug_out", help="Output folder")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load system (same as GUI)
    system = load_system(data_dir, T=args.T, dt_hours=args.dt)

    # 2) optional switches to isolate infeasibility source
    if args.no_reefers:
        system = _model_copy(system, reefers=[])
        print("[DEBUG] Reefers disabled.")

    if args.no_ci:
        system = _model_copy(system, cold_ironing=[])
        print("[DEBUG] Cold-ironing disabled.")

    if args.freeze_ess:
        ess = system.ess
        # best-effort update (adapt field names if your ESS schema differs)
        ess2 = ess
        if hasattr(ess, "model_copy"):
            ess2 = ess.model_copy(update={"max_charging_power_kw": 0.0, "max_discharging_power_kw": 0.0})
        else:
            ess2 = ess.copy(update={"max_charging_power_kw": 0.0, "max_discharging_power_kw": 0.0})
        system = _model_copy(system, ess=ess2)
        print("[DEBUG] ESS frozen (Pch=Pdis=0).")

    # 3) params
    p = getattr(system, "params", {}) or {}
    params = ACLinParams(
        s_base_kva=float(p.get("s_base_kva", 10000.0)),
        v_min=float(p.get("v_min", 0.9)),
        v_max=float(p.get("v_max", 1.1)),
        temp_out_c=float(p.get("temp_out_c", 25.0)),
        q_over_p_reefer=float(p.get("q_over_p_reefer", 0.8)),
        q_over_p_ci=float(p.get("q_over_p_ci", 0.9)),
        allow_export=bool(args.allow_export),
        export_price=float(p.get("export_price", 0.0)),
    )

    # 4) build model
    m, idx, var = build_aclin_optimization(system, params=params)

    # 5) solver params for debugging
    m.Params.OutputFlag = 1
    m.Params.InfUnbdInfo = 1
    m.Params.DualReductions = 0
    m.Params.NumericFocus = 2  # can try 3 if numeric issues
    # m.Params.FeasibilityTol = 1e-6

    # 6) export LP (before solve)
    if args.write_lp:
        lp_path = out_dir / "model.lp"
        m.write(str(lp_path))
        print(f"[DEBUG] Wrote LP to: {lp_path}")

    # 7) solve
    m.optimize()

    status = int(m.Status)
    print(f"[DEBUG] Gurobi status = {status}")

    # 8) infeasible diagnosis
    # GRB.INFEASIBLE = 3
    if args.iis and status == 3:
        print("[DEBUG] Model infeasible. Computing IIS...")
        m.computeIIS()

        ilp_path = out_dir / "model_iis.ilp"
        m.write(str(ilp_path))
        print(f"[DEBUG] Wrote IIS to: {ilp_path}")

        # print a short list of IIS constraints
        bad = []
        for c in m.getConstrs():
            if getattr(c, "IISConstr", 0) == 1:
                bad.append(c.ConstrName)
        print(f"[DEBUG] IIS constraints count: {len(bad)}")
        for name in bad[:50]:
            print("  -", name)

        # also dump bounds IIS
        vb = []
        for v in m.getVars():
            if getattr(v, "IISLB", 0) == 1 or getattr(v, "IISUB", 0) == 1:
                vb.append(v.VarName)
        if vb:
            print(f"[DEBUG] IIS variable bounds count: {len(vb)}")
            for name in vb[:50]:
                print("  -", name)

    # 9) optimal info
    if status in (2, 9):  # OPTIMAL=2, SUBOPTIMAL=9
        print(f"[DEBUG] Obj = {m.ObjVal}")
        sol_path = out_dir / "solution.sol"
        m.write(str(sol_path))
        print(f"[DEBUG] Wrote solution to: {sol_path}")


if __name__ == "__main__":
    main()
