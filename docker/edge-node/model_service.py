import logging
import os
import time
import json
import torch
import torch.nn as nn
import torchvision.models as models
from flask import Flask, jsonify, request
import psutil

NODE_NAME = os.environ.get("NODE_NAME", "edge-node-unknown")
INFERENCE_PORT = int(os.environ.get("INFERENCE_PORT", "5000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("model_service")

app = Flask(__name__)


log.info("Loading PyTorch ResNet-50...")
_full_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# _full_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
_full_model.eval()

_flat_modules = []
for name, module in _full_model.named_modules():
    if len(list(module.children())) == 0:
        _flat_modules.append(module)
log.info(f"Loaded {len(_flat_modules)} leaf modules.")


# Execution Engine
def execute_partition(layer_start: int, layer_end: int, input_shape: list) -> float:
    sub_model = nn.Sequential(*_flat_modules[layer_start : layer_end + 1])
    
    dummy_input = torch.randn(*input_shape)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = sub_model(dummy_input)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return elapsed_ms

@app.post("/infer")
def infer():
    body = request.get_json(force=True, silent=True) or {}
    partition_id = int(body.get("partition_id", 0))
    layer_start  = int(body.get("layer_start", 0))
    layer_end    = int(body.get("layer_end", 0))
    
    shape_str = body.get("input_shape", "[1, 3, 224, 224]")
    input_shape = json.loads(shape_str)

    log.info(f"Executing Partition {partition_id}: Layers {layer_start} -> {layer_end}")

    try:
        exec_ms = execute_partition(layer_start, layer_end, input_shape)
        
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().used / (1024 ** 2)
        
        return jsonify({
            "node": NODE_NAME,
            "partition_id": partition_id,
            "exec_time_ms": round(exec_ms, 3),
            "cpu_pct": cpu,
            "mem_mb": mem,
            "success": True,
        })
    except Exception as exc:
        log.exception("Inference error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500

@app.get("/health")
def health():
    return jsonify({"status": "ok", "node": NODE_NAME})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=INFERENCE_PORT, threaded=True)