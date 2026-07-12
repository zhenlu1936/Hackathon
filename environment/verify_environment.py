#!/usr/bin/env python3
"""Report environment parity with specification/environments.txt."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from importlib import import_module


EXPECTED = {
    "python": "3.12.3",
    "gcc": "13.3.0",
    "g++": "13.3.0",
    "nvcc_release": "12.8",
    "onnx": "1.22.0",
    "onnxruntime": "1.27.0",
    "torch": "2.13.0+cu130",
    "cupy": "14.1.1",
}


def command_output(*command: str) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def module_version(name: str) -> str | None:
    try:
        return str(import_module(name).__version__)
    except (ImportError, AttributeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        action="store_true",
        help="Return nonzero unless every machine and package requirement matches.",
    )
    args = parser.parse_args()

    actual = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "gcc": (command_output("gcc", "-dumpfullversion") or "missing").splitlines()[0],
        "g++": (command_output("g++", "-dumpfullversion") or "missing").splitlines()[0],
        "nvcc": command_output("nvcc", "--version") or "missing",
        "onnx": module_version("onnx") or "missing",
        "onnxruntime": module_version("onnxruntime") or "missing",
        "torch": module_version("torch") or "missing",
        "cupy": module_version("cupy") or "missing",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "<unset>"),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "<unset>"),
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": os.environ.get(
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "<unset>"
        ),
    }

    checks = {
        "os": platform.system() == "Linux" and platform.release().startswith("6.8.0-110"),
        "machine": platform.machine() == "x86_64",
        "python": actual["python"] == EXPECTED["python"],
        "gcc": actual["gcc"] == EXPECTED["gcc"],
        "g++": actual["g++"] == EXPECTED["g++"],
        "nvcc": f"release {EXPECTED['nvcc_release']}" in actual["nvcc"],
        "onnx": actual["onnx"] == EXPECTED["onnx"],
        "onnxruntime": actual["onnxruntime"] == EXPECTED["onnxruntime"],
        "torch": actual["torch"] == EXPECTED["torch"],
        "cupy": actual["cupy"] == EXPECTED["cupy"],
    }

    for name, value in actual.items():
        mark = "OK" if checks.get(name) else ("MISMATCH" if name in checks else "INFO")
        print(f"{mark:8} {name:42} {value}")

    try:
        import torch

        print(f"INFO     {'torch.cuda':42} {torch.version.cuda}")
        print(f"INFO     {'torch.cuda.is_available':42} {torch.cuda.is_available()}")
    except ImportError:
        pass

    return 1 if args.target and not all(checks.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())

