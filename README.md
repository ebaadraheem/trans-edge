# TANS: Transfer-Aware Node Selection for Sustainable Edge AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-7.0.0-2496ED.svg)](https://www.docker.com/)

## Overview
As artificial intelligence moves to the network edge, the carbon footprint of executing distributed deep learning inference has become a critical sustainability challenge[cite: 3]. While recent frameworks like CarbonEdge optimize for the carbon intensity of compute nodes, they contain a critical blind spot: **they ignore the massive energy and latency costs of moving data across heterogeneous networks (e.g., 4G, 5G, WiFi, Fiber)**[cite: 1, 3, 4].

This repository introduces **TANS (Transfer-Aware Node Selection)**, a novel scheduling framework that jointly optimizes both **compute carbon** and **network transfer carbon** against strict latency constraints. 

By anticipating and avoiding disastrous network transfers, the TANS scheduler proves that sustainability does not always require a performance trade-off. For lightweight models (MobileNetV2) and massive-transfer models (VGG16), `tans_green` achieves up to **50% carbon reduction while simultaneously achieving the lowest system latency**.

## Key Features
* **Transfer-Aware Carbon Modeling:** Calculates total emissions by combining local compute power, grid carbon intensity (gCO2/kWh), and the specific energy coefficient of the network medium (kWh/GB).
* **Heterogeneous Edge Simulation:** Built-in support for simulating varied edge environments (0.4 to 1.0 CPU cores, 4G to Fiber networks, 200 to 620 gCO2/kWh) using Docker and SimPy.
* **Layer-by-Layer Profiling:** Automated PyTorch hooks to extract FLOPs, parameter counts, and intermediate tensor transfer sizes (MB) for any CNN architecture.
* **Multiple Scheduling Baselines:** Compare the proposed `tans_green` mode directly against standard `performance`, `balanced`, and compute-only `green` modes[cite: 1, 4].

## Repository Structure
```text
├── requirements.txt           # Python dependencies (SimPy, Docker, PyTorch, Pandas)
├── data/
│   ├── generate_profile.py    # PyTorch script to profile model layers (FLOPs, MBs)
│   └── resnet50_profile.csv   # Example output profile
├── docker/
│   └── edge-node/             # Dockerized Flask inference server for edge simulation
│       ├── Dockerfile
│       ├── model_service.py
│       └── requirements.txt
├── experiments/
│   └── docker_simulation.py   # Main entry point to run multi-mode simulations
└── src/
    ├── core/
    │   ├── carbon_monitor.py  # Tracks real-time/static grid CI and network transfer costs
    │   ├── execution_engine.py# SimPy engine managing concurrent requests and routing
    │   ├── model_partitioner.py # Layer-wise model splitting algorithm
    │   └── task_scheduler.py  # Min-Max normalized TANS routing algorithm
    └── utils/
        ├── docker_manager.py  # Handles lifecycle of Docker edge containers
        └── metrics_logger.py  # Aggregates latency, throughput, and carbon telemetry


        
