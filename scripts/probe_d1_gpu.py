#!/usr/bin/env python3
"""Record the D1 CUDA/package probe, including FP16 and BF16 backward steps."""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path


def dtype_step(torch: object, dtype: object) -> dict[str, object]:
    try:
        left = torch.randn(512, 512, device="cuda", dtype=dtype, requires_grad=True)
        right = torch.randn(512, 512, device="cuda", dtype=dtype, requires_grad=True)
        loss = (left @ right).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        return {"ok": True, "loss": float(loss.item())}
    except Exception as error:  # probe must preserve the concrete runtime failure
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def main() -> int:
    import torch

    packages = ("torch", "transformers", "accelerate", "peft", "bitsandbytes", "datasets")
    properties = torch.cuda.get_device_properties(0)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in packages},
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "fp16_forward_backward": dtype_step(torch, torch.float16),
        "bf16_forward_backward": dtype_step(torch, torch.bfloat16),
    }
    output = Path("artifacts/hardware_probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
