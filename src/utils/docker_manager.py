
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.core.model_partitioner import Partition

log = logging.getLogger(__name__)

try:
    import docker
    from docker.errors import  NotFound, APIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Container configuration
# ---------------------------------------------------------------------------

EDGE_NODE_IMAGE   = os.environ.get("CARBONEDGE_IMAGE", "carbonedge-node:latest")
INFERENCE_PORT    = 5000           
CPU_PERIOD_US     = 100_000       
STATS_TIMEOUT_S   = 2.0

@dataclass
class ContainerConfig:
    """Maps a NodeProfile to Docker resource constraints."""
    node_name:    str
    cpu_cores:    float         
    ram_mb:       int            
    host_port:    int            
    network_type: str = "Fiber"  

    @property
    def cpu_quota(self) -> int:
        return int(self.cpu_cores * CPU_PERIOD_US)

    @property
    def mem_limit_str(self) -> str:
        return f"{self.ram_mb}m"


def node_profile_to_config(np, base_port: int = 5100) -> ContainerConfig:
   
    name     = getattr(np, "name",      np["name"])      if not hasattr(np, "name")     else np.name
    cpu      = getattr(np, "cpu_cores", np["cpu_cores"]) if not hasattr(np, "cpu_cores") else np.cpu_cores
    ram_gb   = getattr(np, "ram_gb",    np["ram_gb"])    if not hasattr(np, "ram_gb")    else np.ram_gb
    net_type = getattr(np, "network_type", "Fiber")

    import re
    idx_match = re.search(r"\d+$", name)
    idx       = int(idx_match.group()) if idx_match else 0
    host_port = base_port + idx

    return ContainerConfig(
        node_name=name,
        cpu_cores=cpu,
        ram_mb=int(ram_gb * 1024),
        host_port=host_port,
        network_type=net_type,
    )


# ---------------------------------------------------------------------------
# Docker Node Manager
# ---------------------------------------------------------------------------

class DockerNodeManager:

    def __init__(
        self,
        configs: List[ContainerConfig],
        image: str = EDGE_NODE_IMAGE,
        network_name: str = "carbonedge-net",
        dry_run: bool = False,
    ) -> None:
        self._configs      = {c.node_name: c for c in configs}
        self._image        = image
        self._network_name = network_name
        self._dry_run      = dry_run or not DOCKER_AVAILABLE
        self._containers: Dict[str, object] = {}

        if not self._dry_run:
            try:
                self._client = docker.from_env()
                self._client.ping()
                log.info("[docker] Connected to Docker daemon.")
            except Exception as exc:
                log.warning("[docker] Docker unavailable (%s); switching to dry-run.", exc)
                self._dry_run = True
        else:
            self._client = None
            if not DOCKER_AVAILABLE:
                log.info("[docker] docker-py not installed; running in dry-run mode.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_all(self) -> None:
        if not self._dry_run:
            self._ensure_network()

        for node_name, cfg in self._configs.items():
            self._start_container(node_name, cfg)

        if not self._dry_run:
            self._wait_for_healthy()

    def stop_all(self) -> None:
        for node_name in list(self._containers.keys()):
            self._stop_container(node_name)

    def restart_node(self, node_name: str) -> None:
        cfg = self._configs.get(node_name)
        if cfg is None:
            raise KeyError(f"Unknown node: {node_name!r}")
        self._stop_container(node_name)
        self._start_container(node_name, cfg)
        if not self._dry_run:
            self._wait_for_healthy(nodes=[node_name])

    # ------------------------------------------------------------------
    # Inference proxy
    # ------------------------------------------------------------------

    def run_inference(self, node_name: str, partition: Partition) -> Dict:
       
        if self._dry_run:
            return self._mock_inference(
            node_name,
            partition.partition_id,
            partition.compute_cost_mflops)

        cfg = self._configs[node_name]
        url = f"http://localhost:{cfg.host_port}/infer"
        payload = {
            "partition_id": partition.partition_id,
            "layer_start":  partition.layer_ids[0],
            "layer_end":    partition.layer_ids[-1],
            "input_shape":  partition.input_shape
        }
        t0 = time.perf_counter()
        try:
            resp = _requests.post(url, json=payload, timeout=30)
            latency_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()
            data["latency_ms"] = latency_ms
            data["success"]    = True
            return data
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            log.error("[docker] Inference failed on %s: %s", node_name, exc)
            return {
                "node":         node_name,
                "partition_id": partition.partition_id,
                "latency_ms":   latency_ms,
                "success":      False,
                "error":        str(exc),
            }

    # ------------------------------------------------------------------
    # Resource stats
    # ------------------------------------------------------------------

    def get_stats(self, node_name: str) -> Dict:
        
        if self._dry_run or node_name not in self._containers:
            return self._mock_stats(node_name)

        try:
            container = self._containers[node_name]
            raw = container.stats(stream=False)
            return self._parse_stats(node_name, raw)
        except Exception as exc:
            log.warning("[docker] Stats failed for %s: %s", node_name, exc)
            return self._mock_stats(node_name)

    def get_all_stats(self) -> Dict[str, Dict]:
        return {n: self.get_stats(n) for n in self._configs}

    def _ensure_network(self) -> None:
        try:
            self._client.networks.get(self._network_name)
            log.debug("[docker] Network %r already exists.", self._network_name)
        except NotFound:
            self._client.networks.create(self._network_name, driver="bridge")
            log.info("[docker] Created bridge network %r.", self._network_name)

    def _start_container(self, node_name: str, cfg: ContainerConfig) -> None:
        if self._dry_run:
            log.info("[docker:dry-run] Would start container %r.", node_name)
            self._containers[node_name] = f"dry-run-{node_name}"
            return

        try:
            old = self._client.containers.get(node_name)
            old.remove(force=True)
            log.debug("[docker] Removed stale container %r.", node_name)
        except NotFound:
            pass

        try:
            container = self._client.containers.run(
                image=self._image,
                name=node_name,
                detach=True,
                network=self._network_name,
                cpu_period=CPU_PERIOD_US,
                cpu_quota=cfg.cpu_quota,
                mem_limit=cfg.mem_limit_str,
                ports={f"{INFERENCE_PORT}/tcp": cfg.host_port},
                environment={
                    "NODE_NAME": node_name,
                    "INFERENCE_PORT": str(INFERENCE_PORT),
                },
                remove=False,
            )
            self._containers[node_name] = container
            log.info(
                "[docker] Started %r  cpu_quota=%d  mem=%s  host_port=%d",
                node_name, cfg.cpu_quota, cfg.mem_limit_str, cfg.host_port,
            )
        except APIError as exc:
            log.error("[docker] Failed to start %r: %s", node_name, exc)
            raise

    def _stop_container(self, node_name: str) -> None:
        if self._dry_run:
            log.info("[docker:dry-run] Would stop container %r.", node_name)
            self._containers.pop(node_name, None)
            return

        container = self._containers.pop(node_name, None)
        if container is None:
            return
        try:
            container.stop(timeout=5)
            container.remove()
            log.info("[docker] Stopped and removed %r.", node_name)
        except Exception as exc:
            log.warning("[docker] Error stopping %r: %s", node_name, exc)

    def _wait_for_healthy(
        self,
        nodes: Optional[List[str]] = None,
        timeout_s: float = 30.0,
        poll_s: float = 0.5,
    ) -> None:
        if not REQUESTS_AVAILABLE:
            return
        targets = nodes or list(self._configs.keys())
        deadline = time.monotonic() + timeout_s
        pending  = set(targets)

        while pending and time.monotonic() < deadline:
            for node_name in list(pending):
                cfg = self._configs[node_name]
                try:
                    r = _requests.get(
                        f"http://localhost:{cfg.host_port}/health",
                        timeout=1,
                    )
                    if r.status_code == 200:
                        log.info("[docker] %r is healthy.", node_name)
                        pending.discard(node_name)
                except Exception:
                    pass
            if pending:
                time.sleep(poll_s)

        if pending:
            log.warning("[docker] Timed out waiting for: %s", pending)


    @staticmethod
    def _parse_stats(node_name: str, raw: Dict) -> Dict:
        cpu_delta    = (raw["cpu_stats"]["cpu_usage"]["total_usage"]
                        - raw["precpu_stats"]["cpu_usage"]["total_usage"])
        system_delta = (raw["cpu_stats"]["system_cpu_usage"]
                        - raw["precpu_stats"]["system_cpu_usage"])
        n_cpus       = raw["cpu_stats"].get("online_cpus", 1)
        cpu_pct      = (cpu_delta / system_delta * n_cpus * 100.0
                        if system_delta > 0 else 0.0)

        mem_usage_mb = raw["memory_stats"]["usage"] / (1024 ** 2)
        mem_limit_mb = raw["memory_stats"]["limit"] / (1024 ** 2)

        net_rx = net_tx = 0
        for iface_stats in raw.get("networks", {}).values():
            net_rx += iface_stats.get("rx_bytes", 0)
            net_tx += iface_stats.get("tx_bytes", 0)

        return {
            "node":          node_name,
            "cpu_pct":       round(cpu_pct, 3),
            "mem_usage_mb":  round(mem_usage_mb, 2),
            "mem_limit_mb":  round(mem_limit_mb, 2),
            "mem_pct":       round(mem_usage_mb / mem_limit_mb * 100, 2)
                             if mem_limit_mb > 0 else 0.0,
            "net_rx_bytes":  net_rx,
            "net_tx_bytes":  net_tx,
        }

    @staticmethod
    def _mock_inference(
        node_name: str,
        partition_id: int,
        compute_cost_mflops: float,
    ) -> Dict:
        simulated_ms = compute_cost_mflops / 4.0 + 10.0
        time.sleep(simulated_ms / 1000.0)
        return {
            "node":         node_name,
            "partition_id": partition_id,
            "latency_ms":   round(simulated_ms, 2),
            "success":      True,
            "mode":         "dry-run",
        }

    @staticmethod
    def _mock_stats(node_name: str) -> Dict:
        import random
        rng = random.Random(hash(node_name) % 999)
        return {
            "node":         node_name,
            "cpu_pct":      round(rng.uniform(5, 40), 2),
            "mem_usage_mb": round(rng.uniform(50, 300), 2),
            "mem_limit_mb": 512.0,
            "mem_pct":      round(rng.uniform(10, 60), 2),
            "net_rx_bytes": rng.randint(1000, 100_000),
            "net_tx_bytes": rng.randint(1000, 50_000),
        }


# ---------------------------------------------------------------------------
# Quick self-test 
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s – %(message)s")

    from src.core.model_partitioner import build_default_nodes

    nodes   = build_default_nodes()
    configs = [node_profile_to_config(n) for n in nodes]
    manager = DockerNodeManager(configs, dry_run=True)

    manager.start_all()

    print("\nInference results:")
    for cfg in configs:
        result = manager.run_inference(cfg.node_name, 0, compute_cost_mflops=500)
        print(f"  {result}")

    print("\nStats:")
    for cfg in configs:
        print(f"  {manager.get_stats(cfg.node_name)}")

    manager.stop_all()
    print("\nAll containers stopped.")