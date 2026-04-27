"""
src/utils/metrics_logger.py
----------------------------
CSV-based metrics logger for CarbonEdge simulation runs.

Records one row per inference event:
  timestamp_s, run_id, mode, inference_id, node, partition_id,
  latency_ms, throughput_rps, energy_kwh, carbon_gco2,
  compute_carbon_gco2, transfer_carbon_gco2,
  transfer_size_mb, network_type, ci_gco2_kwh,
  scheduling_overhead_ms, cumulative_carbon_gco2

Also writes a per-run summary row to a separate *_summary.csv* file.

Design notes
------------
* Thread-safe via a lock; safe to call from SimPy processes and the
  Flask health thread simultaneously.
* The logger flushes after every N rows (flush_interval) so data is
  not lost on abrupt termination.
* The run_id defaults to an ISO-8601 timestamp so multiple experiment
  runs never overwrite each other.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InferenceRecord:
    """One inference event across one partition."""
    timestamp_s:             float
    run_id:                  str
    mode:                    str         # scheduling mode label
    inference_id:            int         # global counter for this run
    node:                    str
    partition_id:            int
    latency_ms:              float
    throughput_rps:          float       # 1000 / latency_ms
    energy_kwh:              float       # compute energy for this partition
    carbon_gco2:             float       # total (compute + transfer)
    compute_carbon_gco2:     float
    transfer_carbon_gco2:    float
    transfer_size_mb:        float
    network_type:            str
    ci_gco2_kwh:             float
    scheduling_overhead_ms:  float
    cumulative_carbon_gco2:  float       # running total for the run


@dataclass
class RunSummary:
    """Aggregated stats written once when a simulation run ends."""
    run_id:                  str
    mode:                    str
    total_inferences:        int
    total_latency_ms:        float
    avg_latency_ms:          float
    p95_latency_ms:          float
    max_latency_ms:          float
    total_throughput_rps:    float
    avg_throughput_rps:      float
    total_energy_kwh:        float
    total_carbon_gco2:       float
    carbon_efficiency:       float       # inferences / gCO₂
    avg_scheduling_oh_ms:    float
    duration_s:              float
    nodes_used:              str         # comma-separated list


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class MetricsLogger:
    """
    Thread-safe CSV logger.

    Parameters
    ----------
    output_dir    : Directory to write CSV files into.
    run_id        : Identifier for this experiment run.
    mode          : Scheduling mode label (e.g. "tans_green").
    flush_interval: Flush to disk every N records.
    """

    DETAIL_SUFFIX  = "_detail.csv"
    SUMMARY_SUFFIX = "_summary.csv"

    def __init__(
        self,
        output_dir: Path | str = Path("results"),
        run_id: Optional[str] = None,
        mode: str = "tans_green",
        flush_interval: int = 10,
    ) -> None:
        self._dir    = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._mode   = mode
        self._flush  = flush_interval

        # Running accumulators
        self._inf_id: int   = 0
        self._cum_carbon: float = 0.0
        self._latencies: List[float] = []
        self._throughputs: List[float] = []
        self._overheads: List[float]   = []
        self._total_energy: float = 0.0
        self._nodes_used: set  = set()
        self._start_ts: float  = time.monotonic()

        # File handles
        detail_path  = self._dir / f"{self._run_id}{self.DETAIL_SUFFIX}"
        self._detail_path = detail_path

        self._lock   = threading.Lock()
        self._buffer: List[InferenceRecord] = []
        self._detail_file  = open(detail_path, "w", newline="", encoding="utf-8")
        self._detail_writer: Optional[csv.DictWriter] = None
        self._req_latencies: Dict[int, float] = {}
        
        log.info("[metrics] Run %r → %s", self._run_id, detail_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_inference(
        self,
        *,
        node: str,
        partition_id: int,
        latency_ms: float,
        request_id: int,
        energy_kwh: float,
        compute_carbon_gco2: float,
        transfer_carbon_gco2: float,
        transfer_size_mb: float,
        network_type: str,
        ci_gco2_kwh: float,
        scheduling_overhead_ms: float = 0.0,
    ) -> InferenceRecord:
        """
        Record one inference event and return the InferenceRecord for callers
        that need the cumulative carbon value.
        """
        total_carbon = compute_carbon_gco2 + transfer_carbon_gco2

        with self._lock:
            if request_id not in self._req_latencies:
                self._req_latencies[request_id] = 0.0
            self._req_latencies[request_id] += latency_ms
            self._inf_id       += 1
            self._cum_carbon   += total_carbon
            self._total_energy += energy_kwh
            self._nodes_used.add(node)
            tput = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            self._throughputs.append(tput)
            self._overheads.append(scheduling_overhead_ms)
            

            rec = InferenceRecord(
                timestamp_s=          time.time(),
                run_id=               self._run_id,
                mode=                 self._mode,
                inference_id=         self._inf_id,
                node=                 node,
                partition_id=         partition_id,
                latency_ms=           round(latency_ms, 4),
                throughput_rps=       round(tput, 4),
                energy_kwh=           round(energy_kwh, 10),
                carbon_gco2=          round(total_carbon, 10),
                compute_carbon_gco2=  round(compute_carbon_gco2, 10),
                transfer_carbon_gco2= round(transfer_carbon_gco2, 10),
                transfer_size_mb=     round(transfer_size_mb, 6),
                network_type=         network_type,
                ci_gco2_kwh=          ci_gco2_kwh,
                scheduling_overhead_ms= round(scheduling_overhead_ms, 4),
                cumulative_carbon_gco2= round(self._cum_carbon, 10),
            )

            self._buffer.append(rec)
            self._flush_if_needed()

        return rec

    def finalize(self, sim_time_ms: float = 0.0) -> RunSummary:
        """
        Flush remaining records, write the summary row, close file handles.
        """
        with self._lock:
            self._flush_buffer()
            self._detail_file.close()

        summary = self._build_summary(sim_time_ms)
        self._write_summary(summary)
        log.info(
            "[metrics] Run %r done — %d inferences, %.6f gCO₂ total, "
            "%.2f inf/gCO₂ efficiency",
            self._run_id, summary.total_inferences,
            summary.total_carbon_gco2, summary.carbon_efficiency,
        )
        return summary

    def current_stats(self) -> Dict:
        """Live snapshot (no lock needed for reads on CPython GIL)."""
        n = len(self._latencies)
        if n == 0:
            return {"inference_count": 0}
        avg_lat = sum(self._latencies) / n
        return {
            "inference_count":       n,
            "avg_latency_ms":        round(avg_lat, 2),
            "total_carbon_gco2":     round(self._cum_carbon, 8),
            "total_energy_kwh":      round(self._total_energy, 10),
            "carbon_efficiency":     round(n / self._cum_carbon, 2)
                                     if self._cum_carbon > 0 else 0,
            "nodes_used":            sorted(self._nodes_used),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush_if_needed(self) -> None:
        if len(self._buffer) >= self._flush:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return

        if self._detail_writer is None:
            # Initialise writer on first flush so we have a record to inspect
            field_names = [f.name for f in fields(InferenceRecord)]
            self._detail_writer = csv.DictWriter(
                self._detail_file, fieldnames=field_names
            )
            self._detail_writer.writeheader()

        for rec in self._buffer:
            self._detail_writer.writerow(asdict(rec))
        self._detail_file.flush()
        self._buffer.clear()

    def _build_summary(self, sim_time_ms: float = 0.0) -> RunSummary:
        true_request_latencies = list(self._req_latencies.values())
        total_requests = len(true_request_latencies)
        if total_requests == 0:
            return RunSummary(
                run_id=self._run_id, mode=self._mode,
                total_inferences=0, total_latency_ms=0, avg_latency_ms=0,
                p95_latency_ms=0, max_latency_ms=0, total_throughput_rps=0,
                avg_throughput_rps=0, total_energy_kwh=0, total_carbon_gco2=0,
                carbon_efficiency=0, avg_scheduling_oh_ms=0,
                duration_s=time.monotonic() - self._start_ts,
                nodes_used="",
            )

        sorted_lat = sorted(true_request_latencies)
        p95_idx    = max(0, int(0.95 * total_requests) - 1)
        
        # Global throughput calculation
        if sim_time_ms > 0:
            sim_time_seconds = sim_time_ms / 1000.0
            true_throughput_rps = total_requests / sim_time_seconds
        else:
            true_throughput_rps = 0.0 # Handled by the engine fix we discussed

        return RunSummary(
            run_id=               self._run_id,
            mode=                 self._mode,
            total_inferences=     total_requests,   # FIXED: Now represents true full inferences
            total_latency_ms=     round(sum(true_request_latencies), 4),
            avg_latency_ms=       round(sum(true_request_latencies) / total_requests, 4), # FIXED
            p95_latency_ms=       round(sorted_lat[p95_idx], 4), # FIXED
            max_latency_ms=       round(sorted_lat[-1], 4),      # FIXED
            total_throughput_rps= round(true_throughput_rps, 4),
            avg_throughput_rps=   round(true_throughput_rps, 4),
            total_energy_kwh=     round(self._total_energy, 10),
            total_carbon_gco2=    round(self._cum_carbon, 10),
            carbon_efficiency=    round(total_requests / self._cum_carbon, 4) if self._cum_carbon > 0 else 0.0, # FIXED
            avg_scheduling_oh_ms= round(sum(self._overheads) / len(self._overheads), 4) if self._overheads else 0.0,
            duration_s=           round(time.monotonic() - self._start_ts, 3),
            nodes_used=           ",".join(sorted(self._nodes_used)),
        )

    def _write_summary(self, summary: RunSummary) -> None:
        summary_path = self._dir / f"{self._run_id}{self.SUMMARY_SUFFIX}"
        field_names  = [f.name for f in fields(RunSummary)]
        write_header = not summary_path.exists()

        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(summary))

        log.info("[metrics] Summary → %s", summary_path)


# ---------------------------------------------------------------------------
# Convenience: multi-run comparison helper
# ---------------------------------------------------------------------------

def compare_runs(result_dir: Path | str) -> None:
    """
    Print a quick ASCII comparison table from all *_summary.csv files in
    *result_dir*.  Useful after running multiple scheduling modes.
    """
    result_dir = Path(result_dir)
    summaries: List[Dict] = []
    for p in sorted(result_dir.glob("*_summary.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            summaries.extend(csv.DictReader(f))

    if not summaries:
        print("No summary files found.")
        return

    cols = ["run_id", "mode", "total_inferences", "avg_latency_ms",
            "total_carbon_gco2", "carbon_efficiency", "duration_s"]
    col_w = [max(len(c), max(len(str(r.get(c, ""))) for r in summaries)) + 2
             for c in cols]

    header = "".join(c.ljust(w) for c, w in zip(cols, col_w))
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in summaries:
        print("".join(str(row.get(c, "")).ljust(w) for c, w in zip(cols, col_w)))
    print(sep)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import tempfile

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s – %(message)s")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(output_dir=tmpdir, mode="tans_green")

        rng = random.Random(42)
        nodes = ["node-uk", "node-usa", "node-se"]
        cis   = {"node-uk": 233, "node-usa": 386, "node-se": 13}

        for i in range(30):
            node  = nodes[i % 3]
            lat   = rng.uniform(50, 400)
            e     = rng.uniform(1e-6, 1e-4)
            c_comp = e * cis[node]
            c_xfer = rng.uniform(1e-7, 1e-5)
            logger.log_inference(
                node=node,
                partition_id=i % 3,
                latency_ms=lat,
                energy_kwh=e,
                compute_carbon_gco2=c_comp,
                transfer_carbon_gco2=c_xfer,
                transfer_size_mb=rng.uniform(0.1, 2.0),
                network_type="Fiber" if node != "node-usa" else "5G",
                ci_gco2_kwh=cis[node],
                scheduling_overhead_ms=rng.uniform(0.02, 0.1),
            )

        summary = logger.finalize()
        print(f"\nSummary: {summary}")

        print("\nLive stats at end:")
        import json
        print(json.dumps(logger.current_stats(), indent=2))

        compare_runs(tmpdir)