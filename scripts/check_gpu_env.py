from __future__ import annotations

import json
import subprocess
from pathlib import Path


def main() -> None:
    info = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
    except Exception as exc:
        info["torch_error"] = repr(exc)

    try:
        info["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=15)[:4000]
    except Exception as exc:
        info["nvidia_smi_error"] = repr(exc)

    Path("runs").mkdir(exist_ok=True)
    Path("runs/gpu_env.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
