from __future__ import annotations

import json
from pathlib import Path

from data.io import load_system


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_tugboat_json(tmp_path: Path):
    _write_text(
        tmp_path / "sys_bus.csv",
        "bus_id,bus_name,is_slack,base_kv\n1,GRID,1,10\n2,BUS2,0,10\n",
    )
    _write_text(
        tmp_path / "sys_line.csv",
        "line_id,from_bus,to_bus,r,x,rate_kw\nL1,1,2,0.01,0.01,1000\n",
    )
    _write_text(
        tmp_path / "sys_load.csv",
        "bus_id,t0,t1,t2,t3\n1,10,10,10,10\n2,20,20,20,20\n",
    )
    _write_text(
        tmp_path / "sys_pv.csv",
        "bus_id,t0,t1,t2,t3\n1,0,0,0,0\n2,1,1,1,1\n",
    )
    _write_text(
        tmp_path / "sys_ess.yaml",
        """
node_number: 2
capacity_kwh: 100
max_soc: 0.9
min_soc: 0.1
max_charging_power_kw: 50
max_discharging_power_kw: 50
initial_soc: 0.5
end_soc: 0.5
eta_ch: 1.0
eta_dis: 1.0
""".strip(),
    )
    _write_text(
        tmp_path / "sys_cold_ironing.csv",
        "task_number,node_number,start_time,end_time,demand_fix_kw,battery_demand_kwh,max_charging_power_kw,min_charging_power_kw\n"
        "1,2,0,1,0,0,100,0\n",
    )
    _write_text(
        tmp_path / "sys_reefer.csv",
        "reefer_id,node_number,content_type,ini_temp,reefer_type,mass_kg\nR1,2,food,5,rt1,1000\n",
    )
    _write_text(
        tmp_path / "sys_content_type.csv",
        "content_type,max_temp,min_temp,specific_heat_capacity_j_per_kgk\nfood,10,-5,3800\n",
    )
    _write_text(
        tmp_path / "sys_reefer_type.csv",
        "reefer_type,area_m2,power_max_kw,power_min_kw,heat_transition_coefficient_w_per_m2k\n"
        "rt1,10,20,1,15\n",
    )

    _write_text(
        tmp_path / "sys.tug_charger.csv",
        "tugboat_charger_group,mg_node\n1,2\n2,1\n",
    )

    demand = [
        {
            "group": "1",
            "tstart": "2026-02-24T08:00:00+08:00",
            "tend": "2026-02-24T10:00:00+08:00",
            "E": 186.7,
        }
    ]
    emin = [
        {
            "group": "1",
            "tstart": "2026-02-24T08:00:00+08:00",
            "tend": "2026-02-24T11:00:00+08:00",
            "Emin": 120.0,
        }
    ]
    (tmp_path / "tugboat_E_demand.json").write_text(
        json.dumps(demand), encoding="utf-8"
    )
    (tmp_path / "tugboat_Emin.json").write_text(json.dumps(emin), encoding="utf-8")

    sysdata = load_system(tmp_path, T=4, dt_hours=1.0)
    assert len(sysdata.tugboat_energy_demands) == 1
    td = sysdata.tugboat_energy_demands[0]
    assert td.group == "1"
    assert td.start_time == 0
    assert td.end_time == 2
    assert td.E == 186.7
    assert td.Emin == 120.0
    assert td.start_time_emin == 0
    assert td.end_time_emin == 3
    assert sysdata.tugboat_group_to_node["1"] == 2
