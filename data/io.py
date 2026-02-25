from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import yaml

from .schemas import (
    Bus,
    Line,
    ESS,
    ColdIroningTask,
    ReeferItem,
    ContentType,
    ReeferType,
    SystemData,
    TugboatEnergyDemand,
)


def _read_matrix_by_bus(csv_path: Path, bus_ids: list[int], T: int) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "bus_id" not in df.columns:
        raise ValueError(f"{csv_path.name} missing 'bus_id' column.")

    # time columns
    cols = [f"t{i}" for i in range(T)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing time columns: {missing[:5]} ...")

    # ---- protect against duplicate bus_id labels (root cause of reindex error) ----
    if df["bus_id"].duplicated().any():
        # 方案1（更严格）：直接报错并指出重复的 bus_id
        # raise ValueError(
        #     f"{csv_path.name}: duplicated bus_id labels found: {dup_ids[:20]} "
        #     f"(total {len(dup_ids)}). Please ensure each bus_id appears once."
        # )

        # 方案2（更实用）：允许重复行，自动按 bus_id 聚合（sum）
        df = df.groupby("bus_id", as_index=False)[cols].sum()

    # align to bus order
    df = df.set_index("bus_id").reindex(bus_ids)

    mat = df[cols].to_numpy(dtype=float)
    if np.isnan(mat).any():
        # 这里通常是：sys_load/sys_pv 缺少某些 bus_id 行
        missing_bus = df.index[df[cols].isna().any(axis=1)].tolist()
        raise ValueError(
            f"{csv_path.name}: missing values after reindex. "
            f"Check bus_id coverage. Missing/NaN buses (sample): {missing_bus[:20]}"
        )
    return mat


def _time_to_index(ts: datetime, t0: datetime, dt_hours: float) -> int:
    dt_seconds = dt_hours * 3600.0
    if dt_seconds <= 0:
        raise ValueError("dt_hours must be positive")
    return int(math.floor((ts - t0).total_seconds() / dt_seconds + 1e-9))


def _normalize_tugboat_group(raw) -> str:
    g = str(raw).strip()
    if g.endswith(".0"):
        g = g[:-2]
    digits = "".join(ch for ch in g if ch.isdigit())
    if digits == "":
        raise ValueError(f"Invalid tugboat_charger_group/group: {raw}")
    return digits


def _load_tugboat_demands(
    data_dir: Path, T: int, dt_hours: float
) -> list[TugboatEnergyDemand]:
    demand_path = data_dir / "tugboat_E_demand.json"
    emin_path = data_dir / "tugboat_Emin.json"
    if not demand_path.exists() or not emin_path.exists():
        return []

    demand_rows = json.loads(demand_path.read_text(encoding="utf-8"))
    emin_rows = json.loads(emin_path.read_text(encoding="utf-8"))

    if not isinstance(demand_rows, list) or not isinstance(emin_rows, list):
        raise ValueError("tugboat JSON files must contain a list of records.")
    if len(demand_rows) != len(emin_rows):
        raise ValueError(
            "tugboat_E_demand.json and tugboat_Emin.json must have same length."
        )

    parsed_demand = []
    parsed_emin = []
    for idx, row in enumerate(demand_rows):
        if not isinstance(row, dict):
            raise ValueError(f"tugboat_E_demand.json row {idx} must be object.")
        row = dict(row)
        row["tstart"] = datetime.fromisoformat(str(row["tstart"]))
        row["tend"] = datetime.fromisoformat(str(row["tend"]))
        parsed_demand.append(row)

    for idx, row in enumerate(emin_rows):
        if not isinstance(row, dict):
            raise ValueError(f"tugboat_Emin.json row {idx} must be object.")
        row = dict(row)
        row["tstart"] = datetime.fromisoformat(str(row["tstart"]))
        row["tend"] = datetime.fromisoformat(str(row["tend"]))
        parsed_emin.append(row)

    t0 = min(r["tstart"] for r in parsed_demand + parsed_emin)
    out: list[TugboatEnergyDemand] = []

    for idx, (drow, erow) in enumerate(zip(parsed_demand, parsed_emin)):
        g_d = _normalize_tugboat_group(drow.get("group"))
        g_e = _normalize_tugboat_group(erow.get("group"))
        if g_d != g_e:
            raise ValueError(
                f"Mismatch between tugboat JSON files at row {idx}: group {g_d} != {g_e}"
            )

        start_time = _time_to_index(drow["tstart"], t0=t0, dt_hours=dt_hours)
        end_time = _time_to_index(drow["tend"], t0=t0, dt_hours=dt_hours)
        start_time_emin = _time_to_index(erow["tstart"], t0=t0, dt_hours=dt_hours)
        end_time_emin = _time_to_index(erow["tend"], t0=t0, dt_hours=dt_hours)
        if end_time <= start_time:
            raise ValueError(
                f"Invalid tugboat E window at row {idx}: tend must be after tstart."
            )
        if end_time_emin <= start_time_emin:
            raise ValueError(
                f"Invalid tugboat Emin window at row {idx}: tend must be after tstart."
            )

        out.append(
            TugboatEnergyDemand(
                group=g_d,
                tstart=drow["tstart"],
                tend=drow["tend"],
                tstart_emin=erow["tstart"],
                tend_emin=erow["tend"],
                start_time=start_time,
                end_time=end_time,
                start_time_emin=start_time_emin,
                end_time_emin=end_time_emin,
                E=float(drow["E"]),
                Emin=float(erow["Emin"]),
            )
        )

    for td in out:
        if td.end_time > T:
            raise ValueError(
                f"Tugboat E window exceeds horizon T={T}: {td.group} [{td.start_time}, {td.end_time})"
            )
        if td.end_time_emin > T:
            raise ValueError(
                f"Tugboat Emin window exceeds horizon T={T}: {td.group} [{td.start_time_emin}, {td.end_time_emin})"
            )

    return out



def _load_tugboat_group_to_node(data_dir: Path) -> dict[str, int]:
    map_path = data_dir / "sys.tug_charger.csv"
    if not map_path.exists():
        return {}

    df = pd.read_csv(map_path)
    expected = {"tugboat_charger_group", "mg_node"}
    if not expected.issubset(set(df.columns)):
        raise ValueError("sys.tug_charger.csv must contain columns: tugboat_charger_group, mg_node")

    out: dict[str, int] = {}
    for _, row in df.iterrows():
        gid = _normalize_tugboat_group(row["tugboat_charger_group"])
        node = int(row["mg_node"])
        out[gid] = node
    return out

def load_system(data_dir: Path, T: int, dt_hours: float) -> SystemData:
    data_dir = Path(data_dir)

    bus_df = pd.read_csv(data_dir / "sys_bus.csv")
    buses = [Bus(**row) for row in bus_df.to_dict(orient="records")]
    buses_sorted = sorted(buses, key=lambda b: b.bus_id)
    bus_ids = [b.bus_id for b in buses_sorted]
    # ensure bus_ids are unique (otherwise reindex will fail)
    if len(set(bus_ids)) != len(bus_ids):
        dup = pd.Series(bus_ids)[pd.Series(bus_ids).duplicated()].unique().tolist()
        raise ValueError(f"sys_bus.csv has duplicated bus_id: {dup[:20]}")

    line_df = pd.read_csv(data_dir / "sys_line.csv")
    lines = [Line(**row) for row in line_df.to_dict(orient="records")]

    load_kw = _read_matrix_by_bus(data_dir / "sys_load.csv", bus_ids, T)
    pv_kw = _read_matrix_by_bus(data_dir / "sys_pv.csv", bus_ids, T)

    # Optional: sys_price.csv
    price = None
    price_path = data_dir / "sys_price.csv"
    if price_path.exists():
        dfp = pd.read_csv(price_path)
        if "price" in dfp.columns:
            price = dfp["price"].to_numpy(dtype=float)
        else:
            price = dfp.iloc[:, 0].to_numpy(dtype=float)

    # Optional: sys_params.yaml
    params = {}
    params_path = data_dir / "sys_params.yaml"
    if params_path.exists():
        params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}

    ess_dict = yaml.safe_load((data_dir / "sys_ess.yaml").read_text(encoding="utf-8"))
    ess = ESS(**ess_dict)

    ci_df = pd.read_csv(data_dir / "sys_cold_ironing.csv")
    cold_ironing = [ColdIroningTask(**row) for row in ci_df.to_dict(orient="records")]

    rf_df = pd.read_csv(data_dir / "sys_reefer.csv")
    reefers = [ReeferItem(**row) for row in rf_df.to_dict(orient="records")]

    ct_df = pd.read_csv(data_dir / "sys_content_type.csv")
    content_types = {
        row["content_type"]: ContentType(**row)
        for row in ct_df.to_dict(orient="records")
    }

    rt_df = pd.read_csv(data_dir / "sys_reefer_type.csv")
    reefer_types = {
        row["reefer_type"]: ReeferType(**row) for row in rt_df.to_dict(orient="records")
    }

    tugboat_energy_demands = _load_tugboat_demands(
        data_dir=data_dir, T=T, dt_hours=dt_hours
    )
    tugboat_group_to_node = _load_tugboat_group_to_node(data_dir)

    return SystemData(
        T=T,
        dt_hours=dt_hours,
        buses=buses_sorted,
        lines=lines,
        load_kw=load_kw,
        pv_kw=pv_kw,
        ess=ess,
        cold_ironing=cold_ironing,
        reefers=reefers,
        content_types=content_types,
        reefer_types=reefer_types,
        tugboat_energy_demands=tugboat_energy_demands,
        tugboat_group_to_node=tugboat_group_to_node,
        price=price,
        params=params,
    )
