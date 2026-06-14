# TRACE

### Transfer-Aware Carbon-Efficient Scheduling for Partitioned Deep Neural Network Inference at the Edge

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red)
![Docker](https://img.shields.io/badge/Docker-7.0+-2496ED)
![License](https://img.shields.io/badge/License-MIT-green)

TRACE is a research-oriented simulation framework for **carbon-efficient distributed inference at the edge**.

Traditional edge inference systems often optimize only compute execution cost while ignoring the hidden environmental and performance impact caused by **moving intermediate tensors across networks**.

TRACE introduces a **Transfer-Aware scheduling strategy** that jointly optimizes:

* Compute carbon emissions
* Network transfer carbon emissions
* Resource utilization
* End-to-end inference latency

The framework partitions deep neural networks across heterogeneous edge devices and dynamically schedules execution using multiple optimization strategies.

---

# Motivation

As AI workloads move toward edge environments, sustainability becomes increasingly important.

Existing approaches typically optimize:

* Compute utilization
* Node-level carbon intensity
* Execution throughput

However, large intermediate tensor transfers across networks (4G, 5G, WiFi, Fiber) can introduce:

* Additional energy consumption
* Higher latency
* Hidden carbon costs

TRACE addresses this gap by explicitly modeling **transfer-aware carbon scheduling**.

---

# Features

## Transfer-Aware Carbon Modeling

Estimate total emissions using:

```
Total Carbon =
Compute Carbon
+
Network Transfer Carbon
```

Carbon estimation includes:

* Node power consumption
* Grid carbon intensity
* Data transfer size
* Network energy coefficients
* Infrastructure efficiency (PUE)

---

## Dynamic Scheduling Modes

TRACE supports multiple execution policies.

| Mode          | Goal                                   |
| ------------- | -------------------------------------- |
| `performance` | Minimize latency                       |
| `balanced`    | Balance performance and sustainability |
| `green`       | Optimize compute-side carbon           |
| `tans_green`  | Optimize compute + transfer carbon     |

---

## Heterogeneous Edge Simulation

Simulate distributed environments with:

* Variable CPU capacity
* Variable RAM
* Different network technologies
* Different regional carbon intensities

Supported network types:

* 4G LTE
* 5G
* WiFi
* Fiber

---

## Layer-Level DNN Partitioning

Automatically partition models using:

* Layer FLOPs
* Memory usage
* Tensor transfer size
* Device capability scores

---

## Docker-Based Execution

Optional real container deployment for:

* Multi-node execution
* Distributed inference testing
* Reproducible experiments

---

# Repository Structure

```text
TRACE/
│
├── requirements.txt
│
├── data/
│   └── generate_profile.py
│
├── docker/
│   └── edge-node/
│       ├── Dockerfile
│       ├── model_service.py
│       └── requirements.txt
│
├── experiments/
│   ├── docker_simulation.py
│   ├── rate_sweep.py
│   ├── analyze_node_usage.py
│   ├── plot_all_node_usage.py
│   ├── plot_per_node_usage.py
│   └── plot_sweep.py
│
├── src/
│   ├── core/
│   │   ├── carbon_monitor.py
│   │   ├── execution_engine.py
│   │   ├── model_partitioner.py
│   │   └── task_scheduler.py
│   │
│   └── utils/
│       ├── docker_manager.py
│       └── metrics_logger.py
│
└── paper.pdf
```

---

# Architecture

```text
Model Profile
      ↓
Model Partitioner
      ↓
Scheduler
      ↓
Execution Engine
      ↓
Carbon Monitor
      ↓
Metrics Logger
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/ebaadraheem/TRACE.git
cd TRACE
```

---

## Create Virtual Environment

Linux / macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running Simulations

## Quick Validation Run

```bash
python experiments/docker_simulation.py --quick
```

---

## Standard Simulation

```bash
python experiments/docker_simulation.py
```

---

## Run Specific Scheduling Mode

```bash
python experiments/docker_simulation.py --mode tans_green
```

Available modes:

```text
performance
balanced
green
tans_green
```

---

## Customize Workload

```bash
python experiments/docker_simulation.py \
  --requests 100 \
  --rate 2 \
  --seed 42
```

Parameters:

| Argument     | Description                  |
| ------------ | ---------------------------- |
| `--requests` | Number of inference requests |
| `--rate`     | Arrival rate                 |
| `--mode`     | Scheduling strategy          |
| `--docker`   | Enable container execution   |
| `--results`  | Output directory             |
| `--seed`     | Random seed                  |

---

# Docker Execution

Enable real distributed inference:

```bash
python experiments/docker_simulation.py --docker
```

This launches:

```text
Edge Node Containers
↓
Distributed Requests
↓
Metrics Collection
↓
Carbon Analysis
```

---

# Experiment Workflow

## Run Simulations

```bash
python experiments/docker_simulation.py
```

---

## Analyze Results

```bash
python experiments/analyze_node_usage.py
```

---

## Generate Visualizations

```bash
python experiments/plot_all_node_usage.py
```

or

```bash
python experiments/plot_sweep.py
```

---

# Carbon Model

TRACE estimates emissions using:

```text
Carbon =
(Energy × CarbonIntensity × PUE)
+
(TransferEnergy × CarbonIntensity)
```

Where:

* Energy → compute energy
* CarbonIntensity → gCO₂/kWh
* PUE → infrastructure overhead
* TransferEnergy → network transfer energy

---

# Example Research Questions

* How much carbon is caused by tensor transfers?
* Can lower carbon also reduce latency?
* How should models be partitioned?
* Which network type produces the lowest total emissions?

---

# Results

Expected observations:

* Lower total emissions under `tans_green`
* Reduced unnecessary transfers
* Improved latency for transfer-heavy models
* Better scheduling decisions in heterogeneous environments

---

# Requirements

Core packages:

```text
simpy==4.1.1
docker==7.0.0
requests==2.31.0
pandas==2.2.1
numpy==1.26.4
torch==2.2.1
torchvision==0.17.1
```

---

# Paper

The repository includes:

```text
paper.pdf
```

for implementation and research details.

---


Steps:

1. Fork repository
2. Create feature branch
3. Commit changes
4. Open pull request

---

# License

This project is licensed under the **MIT License**.

---

# Citation

```bibtex
@software{trace2026,
  title={TRACE: Transfer-Aware Carbon-Efficient Scheduling for Partitioned Deep Neural Network Inference at the Edge},
  year={2026}
}
```

---

Built for sustainable and transfer-aware edge AI research.

> [!WARNING]
> **Research Status — Preprint / Simulation Only**
>
> TRACE is currently an **early-stage research prototype evaluated only in simulated environments**.
>
> The current results are based on **simulation experiments and modeled assumptions**, not deployment on physical edge infrastructure or production systems.
>
> The repository is released to:
>
> * Share methodology and implementation details
> * Enable reproducibility and discussion
> * Serve as a starting point for future research
> * Collect feedback before real-world validation
>
> **Please do not use this repository as evidence of production performance, deployment readiness, or validated environmental impact claims.**
>
> The authors currently **do not recommend operational use or citing quantitative results as established benchmarks** until the framework is validated on:
>
> * Real edge hardware
> * Real network environments
> * Physical power measurements
> * End-to-end deployment experiments
>
> At this stage, TRACE should be treated as a:
>
> **Research preprint + simulation framework + exploratory baseline implementation**
>
> rather than a finalized system.

---

# Limitations

Current limitations include:

* Simulation-only evaluation
* Simplified carbon and transfer energy assumptions
* No validation on heterogeneous real-world devices
* No measured hardware power traces
* No production-scale benchmarking
* Network behavior may not reflect real deployment conditions

Future work will focus on:

1. Validation on physical edge devices
2. Real power and energy measurements
3. Deployment across heterogeneous clusters
4. Expanded scheduler evaluation
5. Statistical robustness analysis
6. Comparison against established baselines

---

# Citation Guidance

If you use this repository academically, please cite it as:

> **a simulation framework / exploratory preprint**
>
> and not as validated production evidence.

If building on this work, we recommend independently reproducing and validating all results.
