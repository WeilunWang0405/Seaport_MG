from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class Line(BaseModel):
    line_id: str
    from_bus: int
    to_bus: int
    r: float
    x: float
    rate_kw: float


class Bus(BaseModel):
    bus_id: int
    bus_name: str = ""
    is_slack: int = 0
    base_kv: float = 10.0
    x: Optional[float] = None
    y: Optional[float] = None


class ESS(BaseModel):
    node_number: int
    capacity_kwh: float
    max_soc: float
    min_soc: float
    max_charging_power_kw: float
    max_discharging_power_kw: float
    initial_soc: float
    end_soc: float
    eta_ch: float = 1.0
    eta_dis: float = 1.0

    @model_validator(mode="after")
    def check_soc(self):
        if not (0 <= self.min_soc <= self.initial_soc <= self.max_soc <= 1):
            raise ValueError(
                "SOC bounds invalid: require 0 <= min <= initial <= max <= 1"
            )
        if not (0 <= self.end_soc <= 1):
            raise ValueError("end_soc must be in [0,1]")
        return self


class ColdIroningTask(BaseModel):
    task_number: int
    node_number: int
    start_time: int
    end_time: int

    demand_fix_kw: float = Field(
        validation_alias=AliasChoices("demand_fix_kw", "demand_fix")
    )
    battery_demand_kwh: float = Field(
        validation_alias=AliasChoices("battery_demand_kwh", "battery_demand")
    )
    max_charging_power_kw: float = Field(
        validation_alias=AliasChoices(
            "max_charging_power_kw", "max_charging_power", "max_charg"
        )
    )
    min_charging_power_kw: float = Field(
        validation_alias=AliasChoices("min_charging_power_kw", "min_charging_power")
    )


class ReeferItem(BaseModel):
    reefer_id: str
    node_number: int
    content_type: str
    ini_temp: float
    reefer_type: str
    mass_kg: float


class ContentType(BaseModel):
    content_type: str
    max_temp: float
    min_temp: float
    specific_heat_capacity_j_per_kgk: float


class ReeferType(BaseModel):
    reefer_type: str
    area_m2: float
    power_max_kw: float
    power_min_kw: float
    heat_transition_coefficient_w_per_m2k: float


class TugboatEnergyDemand(BaseModel):
    group: str
    tstart: datetime
    tend: datetime
    tstart_emin: datetime
    tend_emin: datetime
    start_time: int
    end_time: int
    start_time_emin: int
    end_time_emin: int
    E: float
    Emin: float


class SystemData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    T: int
    dt_hours: float
    buses: List[Bus]
    lines: List[Line]
    load_kw: np.ndarray  # (nb, T)
    pv_kw: np.ndarray  # (nb, T)
    ess: ESS
    cold_ironing: List[ColdIroningTask]
    reefers: List[ReeferItem]
    content_types: Dict[str, ContentType]
    reefer_types: Dict[str, ReeferType]
    tugboat_energy_demands: List[TugboatEnergyDemand] = Field(default_factory=list)
    tugboat_group_to_node: Dict[str, int] = Field(default_factory=dict)
    price: Optional[np.ndarray] = None
    params: Dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dims_and_refs(self):
        nb = len(self.buses)
        if self.load_kw.shape != (nb, self.T):
            raise ValueError(
                f"load_kw must be shape (nb,T)=({nb},{self.T}), got {self.load_kw.shape}"
            )
        if self.pv_kw.shape != (nb, self.T):
            raise ValueError(
                f"pv_kw must be shape (nb,T)=({nb},{self.T}), got {self.pv_kw.shape}"
            )

        bus_ids = {b.bus_id for b in self.buses}
        if self.ess.node_number not in bus_ids:
            raise ValueError("ESS node_number not in bus set.")

        for ln in self.lines:
            if ln.from_bus not in bus_ids or ln.to_bus not in bus_ids:
                raise ValueError(f"Line {ln.line_id} uses unknown bus id.")

        for tk in self.cold_ironing:
            if tk.node_number not in bus_ids:
                raise ValueError("Cold-ironing task uses unknown node_number.")
            if not (0 <= tk.start_time < tk.end_time <= self.T):
                raise ValueError("Cold-ironing start/end_time out of range.")

        for rf in self.reefers:
            if rf.node_number not in bus_ids:
                raise ValueError("Reefer item uses unknown node_number.")
            if rf.content_type not in self.content_types:
                raise ValueError(f"Unknown content_type: {rf.content_type}")
            if rf.reefer_type not in self.reefer_types:
                raise ValueError(f"Unknown reefer_type: {rf.reefer_type}")

        for td in self.tugboat_energy_demands:
            if not (0 <= td.start_time < td.end_time <= self.T):
                raise ValueError("Tugboat demand start/end_time out of range.")
            if not (0 <= td.start_time_emin < td.end_time_emin <= self.T):
                raise ValueError("Tugboat Emin start/end_time out of range.")
            if td.Emin < 0 or td.E < 0:
                raise ValueError("Tugboat demand energy must be non-negative.")
            if td.Emin > td.E:
                raise ValueError("Tugboat demand requires Emin <= E.")

        for g, node in self.tugboat_group_to_node.items():
            if int(node) not in bus_ids:
                raise ValueError(f"Tugboat group {g} maps to unknown node: {node}")

        return self
