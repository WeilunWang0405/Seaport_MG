# Port Operations GUI (Python) — Template Project

This is a runnable starter project for a port microgrid/operations GUI with:
- Automatic network topology rendering from `sysdata/sys_line.csv` + `sysdata/sys_bus.csv`
- Time slider to inspect results at each time step
- Modular layers: data → engine → model/solver → results → UI
- Default optimization engine:
  - Uses Gurobi (`gurobipy`) if available
  - Falls back to a deterministic DC power-flow mock runner if Gurobi is not installed

## Quick start

### 1) Create/activate an environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> If you have Gurobi, also install `gurobipy` and ensure your license works.

### 3) Run
```bash
python main.py
```

## Data files (sysdata/)
- `sys_bus.csv`: bus metadata and optional layout coordinates
- `sys_line.csv`: network lines (from_bus→to_bus defines reference direction for the arrow)
- `sys_load.csv`: load matrix (bus_id + columns t0..t(T-1))
- `sys_pv.csv`: PV matrix (bus_id + columns t0..t(T-1))
- `sys_ess.yaml`: ESS parameters
- `sys_cold_ironing.csv`: cold-ironing tasks
- `sys_reefer*.csv`: reefer inventory and type tables (loaded and validated; not yet optimized in the MVP)

## Where to extend
- Upgrade network constraints: implement LinDistFlow in `model/` and return `V_bus` not equal to 1.0
- Add reefer thermal dynamics and decision variables in `model/` and `results/`
- Add price input `sys_price.csv` and objective refinements in `engine/runner.py`
