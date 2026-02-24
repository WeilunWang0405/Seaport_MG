from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class Indexer:
    bus_ids: List[int]
    bus_to_idx: Dict[int, int]
    edge_keys: List[Tuple[int, int]]          # aligned with sys_line order
    edge_to_idx: Dict[Tuple[int, int], int]

def build_indexer(buses, lines) -> Indexer:
    bus_ids = [b.bus_id for b in buses]
    bus_to_idx = {bid: i for i, bid in enumerate(bus_ids)}
    edge_keys = [(ln.from_bus, ln.to_bus) for ln in lines]
    edge_to_idx = {ek: i for i, ek in enumerate(edge_keys)}
    return Indexer(bus_ids, bus_to_idx, edge_keys, edge_to_idx)
