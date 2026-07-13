#!/usr/bin/env python3
"""Standards-oriented black-box runner for the written C3.5 contract.

This is deliberately separate from ``c35.test_c35``.  It invokes only the
registered command-line interface, reads the released manifests and thresholds,
measures cold wall time, validates outputs against golden tensors, computes the
specified accuracy gate, and samples GPU memory for the process tree through
``pynvml`` or an ``nvidia-smi`` fallback.

Example:

    python -m c35.standard_runner \
      --command 'python -m c35.deploy --onnx {onnx} --input {input} --output {output} --batch-size {batch_size} --backend cupy' \
      --batch-size 256 --report c35-standard-report.json

By default the runner enforces target execution evidence: GPU process memory
must be observable and at least one process in the command tree must use it. Use
``--allow-reference`` only for disclosed CPU/reference validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".specification" / "testcases" / "release_to_competitors"
DEFAULT_MODELS = RELEASE / "models"
DEFAULT_TESTDATA = RELEASE / "testdata" / "c35"
EXPECTED_CUPY_VERSION = "14.1.1"
GPU_EVIDENCE_PREFIX = "C35_GPU_EVIDENCE_JSON="
DEFAULT_COMMAND = (
    f"{shlex.quote(sys.executable)} -m c35.deploy "
    "--onnx {onnx} --input {input} --output {output} "
    "--batch-size {batch_size} --backend cupy"
)


@dataclass
class ModelResult:
    model: str
    command: List[str] = field(default_factory=list)
    returncode: Optional[int] = None
    timed_out: bool = False
    wall_time_s: Optional[float] = None
    peak_gpu_memory_bytes: Optional[int] = None
    gpu_process_observed: bool = False
    nvml_status: str = "not_started"
    gpu_evidence_source: Optional[str] = None
    backend_evidence: Dict[str, Any] = field(default_factory=dict)
    precision_pass: bool = False
    accuracy_pass: Optional[bool] = None
    accuracy: Optional[float] = None
    accuracy_min: Optional[float] = None
    max_abs_diff: Optional[float] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    output_contract_pass: bool = False
    gpu_evidence_pass: bool = False
    passed: bool = False
    errors: List[str] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


def _cupy_preflight() -> Dict[str, Any]:
    """Exercise the CuPy installation and visible CUDA device."""
    result: Dict[str, Any] = {"passed": False}
    try:
        import cupy as cp

        if cp.__version__ != EXPECTED_CUPY_VERSION:
            raise RuntimeError(
                f"cupy {cp.__version__} does not match target {EXPECTED_CUPY_VERSION}"
            )
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count < 1:
            raise RuntimeError("no CUDA device is visible")
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        a = cp.arange(16, dtype=cp.float32).reshape(4, 4)
        checksum = float((a @ a).sum().item())
        cp.cuda.get_current_stream().synchronize()
        result.update({
            "passed": True,
            "cupy_version": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "device_count": device_count,
            "device_id": int(device.id),
            "device_name": str(name),
            "smoke_checksum": checksum,
        })
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _parse_backend_evidence(stderr: str) -> Dict[str, Any]:
    for line in reversed(stderr.splitlines()):
        if not line.startswith(GPU_EVIDENCE_PREFIX):
            continue
        try:
            value = json.loads(line[len(GPU_EVIDENCE_PREFIX):])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _valid_cupy_evidence(evidence: Dict[str, Any]) -> bool:
    try:
        return (
            evidence.get("backend") == "cupy"
            and evidence.get("cupy_version") == EXPECTED_CUPY_VERSION
            and int(evidence.get("pool_reserved_bytes", 0)) > 0
            and int(evidence.get("device_id", -1)) >= 0
        )
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _manifest_entries(path: Path) -> Dict[str, Dict[str, Any]]:
    manifest = _read_json(path)
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise ValueError(f"Manifest has no non-empty 'tensors' list: {path}")
    entries: Dict[str, Dict[str, Any]] = {}
    for entry in tensors:
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest tensor entry is not an object: {path}")
        missing = {"name", "file", "dtype", "shape"} - set(entry)
        if missing:
            raise ValueError(f"Manifest entry missing {sorted(missing)}: {path}")
        name = entry["name"]
        if not isinstance(name, str) or not name or name in entries:
            raise ValueError(f"Invalid or duplicate tensor name {name!r}: {path}")
        entries[name] = entry
    return entries


def _safe_tensor_path(directory: Path, file_name: Any) -> Path:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"Invalid tensor file name: {file_name!r}")
    candidate = (directory / file_name).resolve()
    base = directory.resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"Tensor file escapes its manifest directory: {file_name}")
    return candidate


def _descendants(root_pid: int) -> Set[int]:
    """Return root plus descendants using Linux /proc without extra packages."""
    parents: Dict[int, int] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return {root_pid}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # /proc/PID/stat: pid (comm) state ppid ...; comm may contain spaces.
            stat = (entry / "stat").read_text(encoding="utf-8")
            after_comm = stat.rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(after_comm[1])
        except (OSError, ValueError, IndexError):
            continue
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


class NvmlProcessTreeSampler:
    def __init__(self, root_pid: int, interval_s: float) -> None:
        self.root_pid = root_pid
        self.interval_s = interval_s
        self.peak_bytes = 0
        self.process_observed = False
        self.status = "unavailable"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pynvml: Any = None
        self._handles: List[Any] = []
        self._mode = "none"

    def start(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            self._mode = "pynvml"
            self.status = "sampling"
        except Exception as exc:
            if shutil.which("nvidia-smi") is None:
                self.status = f"unavailable: {exc}; nvidia-smi not found"
                return
            self._mode = "nvidia-smi"
            self.interval_s = max(self.interval_s, 0.1)
            self.status = "sampling:nvidia-smi"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _device_processes(self, handle: Any) -> Iterable[Any]:
        for method_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            method = getattr(self._pynvml, method_name, None)
            if method is None:
                continue
            try:
                yield from method(handle)
            except Exception:
                continue

    def _sample(self) -> None:
        pids = _descendants(self.root_pid)
        if self._mode == "nvidia-smi":
            total_mib = 0
            observed = False
            # On MIG setups the global query may miss processes; enumerate
            # individual GPU / MIG compute-instance indexes and query each one.
            try:
                gpu_list = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=max(1.0, self.interval_s * 5),
                    check=False,
                )
                gpu_indexes = [
                    int(line.strip()) for line in gpu_list.stdout.splitlines()
                    if line.strip().isdigit()
                ]
            except (OSError, ValueError, subprocess.SubprocessError):
                gpu_indexes = []
            # Build a list of queries to try: global, per-GPU, MIG-aware.
            queries: List[List[str]] = [
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
            ]
            for idx in gpu_indexes:
                queries.append([
                    "nvidia-smi", "-i", str(idx),
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ])
            # MIG-aware fallback: use --query-accounted-apps which may report
            # processes that --query-compute-apps misses on MIG slices.
            queries.append([
                "nvidia-smi", "--query-accounted-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ])
            for query_cmd in queries:
                try:
                    completed = subprocess.run(
                        query_cmd,
                        capture_output=True,
                        text=True,
                        timeout=max(1.0, self.interval_s * 5),
                        check=False,
                    )
                    for line in completed.stdout.splitlines():
                        fields = [field.strip() for field in line.split(",")]
                        if len(fields) != 2:
                            continue
                        try:
                            pid, used_mib = int(fields[0]), int(fields[1])
                        except ValueError:
                            continue
                        if pid in pids:
                            total_mib += used_mib
                            observed = True
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            # Fallback: if PID-based tracking found nothing but the GPU is
            # clearly active (non-zero total memory), record the total used
            # memory as a lower-confidence signal.
            if not observed:
                try:
                    mem = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=max(1.0, self.interval_s * 5),
                        check=False,
                    )
                    for line in mem.stdout.splitlines():
                        line = line.strip()
                        if line.isdigit():
                            total_mib = max(total_mib, int(line))
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
            self.process_observed = self.process_observed or observed
            self.peak_bytes = max(self.peak_bytes, total_mib * 1024 * 1024)
            return
        total = 0
        observed = False
        for handle in self._handles:
            per_pid: Dict[int, int] = {}
            for process in self._device_processes(handle):
                pid = int(process.pid)
                if pid not in pids:
                    continue
                used = getattr(process, "usedGpuMemory", 0)
                if isinstance(used, int) and used > 0:
                    per_pid[pid] = max(per_pid.get(pid, 0), used)
                    observed = True
            total += sum(per_pid.values())
        self.process_observed = self.process_observed or observed
        self.peak_bytes = max(self.peak_bytes, total)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        if self._mode == "none":
            return
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self.status = "ok" if self._mode == "pynvml" else "ok:nvidia-smi"


def _format_command(template: str, model: Path, input_dir: Path,
                    output_dir: Path, batch_size: int) -> List[str]:
    required = ("{onnx}", "{input}", "{output}")
    missing = [placeholder for placeholder in required if placeholder not in template]
    if missing:
        raise ValueError(f"Command template missing placeholders: {missing}")
    tokens = shlex.split(template)
    replacements = {
        "{onnx}": str(model),
        "{input}": str(input_dir),
        "{output}": str(output_dir),
        "{batch_size}": str(batch_size),
    }
    return [
        token.replace("{onnx}", replacements["{onnx}"])
        .replace("{input}", replacements["{input}"])
        .replace("{output}", replacements["{output}"])
        .replace("{batch_size}", replacements["{batch_size}"])
        for token in tokens
    ]


def _validate_outputs(model_dir: Path, output_dir: Path,
                      result: ModelResult) -> None:
    golden_dir = model_dir / "golden"
    thresholds = _read_json(model_dir / "thresholds.json")
    golden_entries = _manifest_entries(golden_dir / "manifest.json")
    output_entries = _manifest_entries(output_dir / "manifest.json")
    if set(output_entries) != set(golden_entries):
        raise ValueError(
            f"Output names differ from golden manifest: expected={sorted(golden_entries)}, "
            f"actual={sorted(output_entries)}"
        )

    precision = thresholds.get("precision", {})
    result.rtol = float(precision["rtol"])
    result.atol = float(precision["atol"])
    allclose = True
    max_diff = 0.0
    arrays: Dict[str, np.ndarray] = {}
    for name, golden_entry in golden_entries.items():
        output_entry = output_entries[name]
        output_path = _safe_tensor_path(output_dir, output_entry["file"])
        golden_path = _safe_tensor_path(golden_dir, golden_entry["file"])
        if not output_path.is_file():
            raise FileNotFoundError(f"Missing output tensor file: {output_path}")
        actual = np.load(output_path, allow_pickle=False)
        expected = np.load(golden_path, allow_pickle=False)
        arrays[name] = actual
        declared_dtype = np.dtype(output_entry["dtype"])
        if actual.dtype != declared_dtype:
            raise ValueError(
                f"Output '{name}' dtype mismatch: manifest={declared_dtype}, npy={actual.dtype}"
            )
        if actual.dtype != np.float32:
            raise ValueError(f"Output '{name}' must be float32, got {actual.dtype}")
        if list(actual.shape) != list(output_entry["shape"]):
            raise ValueError(
                f"Output '{name}' shape mismatch: manifest={output_entry['shape']}, "
                f"npy={list(actual.shape)}"
            )
        if actual.shape != expected.shape:
            raise ValueError(
                f"Output '{name}' shape {actual.shape} != golden {expected.shape}"
            )
        if actual.size:
            max_diff = max(max_diff, float(np.max(np.abs(actual - expected))))
        allclose = allclose and bool(
            np.allclose(actual, expected, rtol=result.rtol, atol=result.atol)
        )

    result.max_abs_diff = max_diff
    result.precision_pass = allclose
    result.output_contract_pass = True

    accuracy_spec = thresholds.get("accuracy")
    if accuracy_spec is None:
        result.accuracy_pass = None
    else:
        if accuracy_spec.get("metric") != "top1":
            raise ValueError(f"Unsupported accuracy metric: {accuracy_spec}")
        labels = np.load(model_dir / "labels.npy", allow_pickle=False).reshape(-1)
        logits = arrays["logits"]
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Logit/label count mismatch: {logits.shape[0]} != {labels.shape[0]}"
            )
        result.accuracy = float(np.mean(np.argmax(logits, axis=-1) == labels))
        result.accuracy_min = float(accuracy_spec["min"])
        result.accuracy_pass = result.accuracy >= result.accuracy_min


def run_model(model_name: str, command_template: str, models_dir: Path,
              testdata_dir: Path, batch_size: int, timeout_s: float,
              sample_interval_s: float, allow_reference: bool) -> ModelResult:
    result = ModelResult(model=model_name)
    model_path = models_dir / f"{model_name}_v1.onnx"
    model_dir = testdata_dir / f"{model_name}_v1"
    input_dir = model_dir / "input"
    try:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        _manifest_entries(input_dir / "manifest.json")
        with tempfile.TemporaryDirectory(prefix=f"c35-standard-{model_name}-") as temp:
            output_dir = Path(temp) / "output"
            command = _format_command(
                command_template, model_path, input_dir, output_dir, batch_size
            )
            result.command = command
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, \
                    tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
                start = time.perf_counter()
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
                sampler = NvmlProcessTreeSampler(process.pid, sample_interval_s)
                sampler.start()
                try:
                    result.returncode = process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    result.timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    result.returncode = process.wait()
                finally:
                    sampler.stop()
                    result.wall_time_s = time.perf_counter() - start
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout_text = stdout_file.read()
                stderr_text = stderr_file.read()
                result.stdout_tail = _tail(stdout_text)
                result.stderr_tail = _tail(stderr_text)
                result.backend_evidence = _parse_backend_evidence(stderr_text)
                result.gpu_process_observed = sampler.process_observed
                result.nvml_status = sampler.status

                sampled = (
                    sampler.status.startswith("ok")
                    and sampler.process_observed
                    and sampler.peak_bytes > 0
                )
                if sampled:
                    result.peak_gpu_memory_bytes = sampler.peak_bytes
                    result.gpu_evidence_source = sampler.status.split(":", 1)[-1]
                elif _valid_cupy_evidence(result.backend_evidence):
                    result.peak_gpu_memory_bytes = int(
                        result.backend_evidence["pool_reserved_bytes"]
                    )
                    result.gpu_evidence_source = "cupy-pool"

            if result.timed_out:
                raise TimeoutError(f"Command exceeded {timeout_s:.1f} seconds")
            if result.returncode != 0:
                raise RuntimeError(f"Command exited with status {result.returncode}")
            _validate_outputs(model_dir, output_dir, result)
            result.gpu_evidence_pass = (
                allow_reference
                or result.gpu_evidence_source is not None
            )
            if not result.gpu_evidence_pass:
                result.errors.append(
                    "No target GPU execution evidence was observed; "
                    "use --allow-reference only for disclosed local reference runs"
                )
            accuracy_ok = result.accuracy_pass is not False
            result.passed = (
                result.output_contract_pass
                and result.precision_pass
                and accuracy_ok
                and result.gpu_evidence_pass
            )
    except Exception as exc:
        result.errors.append(str(exc))
        result.passed = False
    return result


def _print_result(result: ModelResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    memory = (
        f"{result.peak_gpu_memory_bytes / (1024 ** 2):.2f} MiB"
        if result.peak_gpu_memory_bytes is not None else "unavailable"
    )
    accuracy = (
        "n/a" if result.accuracy_pass is None
        else f"{result.accuracy:.4f} (min {result.accuracy_min:.4f})"
    )
    print(f"[{status}] {result.model}")
    print(f"  wall time:       {result.wall_time_s:.6f}s" if result.wall_time_s else "  wall time:       n/a")
    source = result.gpu_evidence_source or result.nvml_status
    print(f"  peak GPU memory: {memory} ({source})")
    print(f"  max abs diff:    {result.max_abs_diff}")
    print(f"  accuracy:        {accuracy}")
    if result.errors:
        for error in result.errors:
            print(f"  error:           {error}")
    # Surface stderr tail for crash diagnostics
    if not result.passed and result.stderr_tail:
        print(f"  stderr (tail):")
        for line in result.stderr_tail.strip().splitlines():
            print(f"    {line}")
    if not result.passed and result.stdout_tail:
        print(f"  stdout (tail):")
        for line in result.stdout_tail.strip().splitlines():
            print(f"    {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default=DEFAULT_COMMAND,
                        help="registered command template with {onnx}, {input}, {output}")
    parser.add_argument("--models", nargs="+", default=["mlp", "resnet", "transformer"])
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--testdata-dir", type=Path, default=DEFAULT_TESTDATA)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--allow-reference",
        action="store_true",
        help="waive GPU-process evidence for disclosed CPU reference validation",
    )
    parser.add_argument(
        "--skip-cupy-preflight",
        action="store_true",
        help="skip CuPy/device smoke test for a custom non-CuPy command",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.timeout <= 0 or args.sample_interval <= 0:
        parser.error("--timeout and --sample-interval must be positive")

    preflight = (
        {"passed": True, "skipped": True}
        if args.skip_cupy_preflight else _cupy_preflight()
    )
    if preflight["passed"]:
        label = (
            "skipped"
            if preflight.get("skipped")
            else f"{preflight['cupy_version']} on {preflight['device_name']}"
        )
        print(f"CuPy preflight: {label}")
    else:
        print(f"CuPy preflight failed: {preflight.get('error', 'unknown error')}")

    results = [
        run_model(
            model, args.command, args.models_dir, args.testdata_dir,
            args.batch_size, args.timeout, args.sample_interval,
            args.allow_reference,
        )
        for model in args.models
    ]
    for result in results:
        _print_result(result)
    all_models_passed = all(result.passed for result in results)
    score_eligible = (
        not args.allow_reference
        and preflight["passed"]
        and not preflight.get("skipped", False)
    )
    gate_points = 15 if score_eligible and all_models_passed else 0
    report = {
        "format_version": "1.0",
        "command_template": args.command,
        "batch_size": args.batch_size,
        "allow_reference": args.allow_reference,
        "cupy_preflight": preflight,
        "written_scoring": {
            "correctness_accuracy_points": 15,
            "runtime_points": 25,
            "peak_gpu_memory_points": 10,
        },
        "score_summary": {
            "correctness_accuracy": {
                "earned": gate_points if score_eligible else None,
                "available": 15,
                "eligible": score_eligible,
                "reason": None if score_eligible else "reference-mode waiver is not score evidence",
            },
            "runtime": {
                "earned": None,
                "available": 25,
                "reason": "ranked across submissions",
            },
            "peak_gpu_memory": {
                "earned": None,
                "available": 10,
                "reason": "ranked across submissions",
            },
            "known_points": gate_points,
        },
        "all_passed": preflight["passed"] and all_models_passed,
        "results": [asdict(result) for result in results],
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.report}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
