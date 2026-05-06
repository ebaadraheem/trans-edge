from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Network transfer energy coefficients
# ---------------------------------------------------------------------------
TRANSFER_ENERGY_KWH_PER_GB = {
        "4g_lte": 0.14,
        "5g": 0.08,
        "wifi": 0.08,
        "fiber": 0.04,
        "default": 0.05,
    }

# ---------------------------------------------------------------------------
# Electricity Maps API
# ---------------------------------------------------------------------------
_API_BASE = "https://api.electricitymap.org/v3/carbon-intensity/latest"

ZONE_CODES: Dict[str, str] = {
    "node-uk":  "GB",   
    "node-usa": "US-CAL-CISO",  
    "node-se":  "SE",    
}

STATIC_FALLBACK_CI: Dict[str, float] = {
    "node-uk":  233.0,   
    "node-usa": 386.0,   
    "node-se":   13.0,   
}



# ---------------------------------------------------------------------------
# Per-node energy / carbon state
# ---------------------------------------------------------------------------

@dataclass
class NodeCarbonState:
    node_name:                  str
    carbon_intensity_gco2_kwh:  float          
    avg_power_w:                float          
    network_type:               str = "Fiber"
    pue:                        float = 1.0

    # Cumulative accumulators (updated by record_inference)
    total_energy_kwh:           float = field(default=0.0, repr=False)
    total_carbon_gco2:          float = field(default=0.0, repr=False)
    inference_count:            int   = field(default=0,   repr=False)
    _last_update_ts:            float = field(default_factory=time.monotonic,
                                              repr=False)

    avg_exec_time_ms:           float = 200.0

    def record_inference(self, exec_time_ms: float) -> float:
        
        energy_kwh = self._compute_energy(exec_time_ms)
        carbon_gco2 = energy_kwh * self.carbon_intensity_gco2_kwh * self.pue

        self.total_energy_kwh  += energy_kwh
        self.total_carbon_gco2 += carbon_gco2
        self.inference_count   += 1

        alpha = 0.2
        self.avg_exec_time_ms = (
            alpha * exec_time_ms + (1 - alpha) * self.avg_exec_time_ms
        )
        self._last_update_ts = time.monotonic()
        return carbon_gco2

    def estimate_inference_energy(self, exec_time_ms: Optional[float] = None) -> float:
        
        t = exec_time_ms if exec_time_ms is not None else self.avg_exec_time_ms
        return self._compute_energy(t)
    
    def transfer_carbon(self, size_mb: float) -> float:
        
        size_gb = size_mb / 1024.0
        coeff   = TRANSFER_ENERGY_KWH_PER_GB.get(
            self.network_type,
            TRANSFER_ENERGY_KWH_PER_GB["default"],
        )
        e_trans = size_gb * coeff                           # kWh
        return e_trans * self.carbon_intensity_gco2_kwh     # gCO₂

    def snapshot(self) -> Dict:
      
        return {
            "node":                      self.node_name,
            "ci_gco2_kwh":               self.carbon_intensity_gco2_kwh,
            "total_energy_kwh":          round(self.total_energy_kwh, 8),
            "total_carbon_gco2":         round(self.total_carbon_gco2, 8),
            "inference_count":           self.inference_count,
            "avg_exec_time_ms":          round(self.avg_exec_time_ms, 2),
            "network_type":              self.network_type,
        }


    def _compute_energy(self, exec_time_ms: float) -> float:
        return self.avg_power_w * exec_time_ms / 3_600_000_000.0


# ---------------------------------------------------------------------------
# Carbon Monitor  
# ---------------------------------------------------------------------------

class CarbonMonitor:

    def __init__(
        self,
        node_profiles,
        api_key: Optional[str] = None,
        refresh_interval_s: float = 300.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ELECTRICITY_MAPS_API_KEY", "")
        self._refresh_interval = refresh_interval_s
        self._last_refresh: float = 0.0

        self.nodes: Dict[str, NodeCarbonState] = {}
        for np in node_profiles:
            name = np.name if hasattr(np, "name") else np["name"]
            self.nodes[name] = NodeCarbonState(
                node_name=name,
                carbon_intensity_gco2_kwh=self._initial_ci(name, np),
                avg_power_w=getattr(np, "avg_power_w", np.get("avg_power_w", 15.0))
                            if not hasattr(np, "avg_power_w") else np.avg_power_w,
                network_type=getattr(np, "network_type",
                                     np.get("network_type", "Fiber"))
                             if not hasattr(np, "network_type") else np.network_type,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, force: bool = False) -> None:
      
        now = time.monotonic()
        if not force and (now - self._last_refresh) < self._refresh_interval:
            return
        self._last_refresh = now

        if not self._api_key:
            log.debug("[carbon_monitor] No API key – using static CI values.")
            return

        for node_name, state in self.nodes.items():
            zone = ZONE_CODES.get(node_name)
            if not zone:
                log.warning("[carbon_monitor] No zone code for %s; skipping.", node_name)
                continue
            ci = self._fetch_ci(zone)
            if ci is not None:
                state.carbon_intensity_gco2_kwh = ci
                log.info("[carbon_monitor] %s CI updated: %.1f gCO₂/kWh", node_name, ci)

    def record(self, node_name: str, exec_time_ms: float) -> float:
       
        self._maybe_refresh()
        return self.nodes[node_name].record_inference(exec_time_ms)

    def estimate_compute_carbon(self, node_name: str,
                                exec_time_ms: Optional[float] = None) -> float:
        
        return (self.nodes[node_name].estimate_inference_energy(exec_time_ms)
                * self.nodes[node_name].carbon_intensity_gco2_kwh)

    def estimate_transfer_carbon(self, src_node: str, size_mb: float) -> float:
        
        return self.nodes[src_node].transfer_carbon(size_mb)

    def cluster_snapshot(self) -> Dict:
        
        total_energy = sum(s.total_energy_kwh  for s in self.nodes.values())
        total_carbon = sum(s.total_carbon_gco2 for s in self.nodes.values())
        total_infs   = sum(s.inference_count    for s in self.nodes.values())
        carbon_eff   = total_infs / total_carbon if total_carbon > 0 else 0.0
        return {
            "total_energy_kwh":   round(total_energy, 8),
            "total_carbon_gco2":  round(total_carbon, 8),
            "total_inferences":   total_infs,
            "carbon_efficiency":  round(carbon_eff, 2), 
            "nodes":              [s.snapshot() for s in self.nodes.values()],
        }

    def _initial_ci(self, node_name: str, profile) -> float:
        explicit = (
            getattr(profile, "carbon_intensity_gco2_kwh", None)
            or (profile.get("carbon_intensity_gco2_kwh") if isinstance(profile, dict) else None)
        )
        if explicit is not None:
            return float(explicit)

        if self._api_key:
            zone = ZONE_CODES.get(node_name)
            if zone:
                ci = self._fetch_ci(zone)
                if ci is not None:
                    return ci

        return STATIC_FALLBACK_CI.get(node_name, 400.0)

    def _maybe_refresh(self) -> None:
        if self._refresh_interval > 0:
            self.refresh()

    def _fetch_ci(self, zone: str) -> Optional[float]:
       
        try:
            resp = requests.get(
                _API_BASE,
                params={"zone": zone},
                headers={"auth-token": self._api_key},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                ci   = data.get("carbonIntensity")
                if ci is not None:
                    return float(ci)
                log.warning("[carbon_monitor] Unexpected payload for zone %s: %s",
                            zone, data)
            elif resp.status_code == 401:
                log.error("[carbon_monitor] Invalid API key – switching to static values.")
                self._api_key = ""   # disable further attempts this session
            else:
                log.warning("[carbon_monitor] API status %d for zone %s",
                            resp.status_code, zone)
        except requests.exceptions.RequestException as exc:
            log.warning("[carbon_monitor] API unreachable (%s) – using static CI.", exc)
        return None


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from src.core.model_partitioner import build_default_nodes

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s – %(message)s")

    nodes   = build_default_nodes()
    monitor = CarbonMonitor(node_profiles=nodes, refresh_interval_s=0)

    for i in range(10):
        node_name = nodes[i % len(nodes)].name
        exec_ms   = 250.0 + i * 5
        carbon    = monitor.record(node_name, exec_ms)
        print(f"  inf {i+1:02d} → {node_name}: {carbon*1000:.4f} mgCO₂")

    print("\nCluster snapshot:")
    print(json.dumps(monitor.cluster_snapshot(), indent=2))