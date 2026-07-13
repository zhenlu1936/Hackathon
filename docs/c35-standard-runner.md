# C3.5 standard black-box runner

`c35/test_c35.py` is a development regression suite, not a replica of the official evaluator. It mixes internal operator unit tests with CLI tests, hardcodes public model shapes and thresholds, checks golden results only for selected batch cases, invokes only this repository's Python module, and does not sample NVML process-tree memory.

Use `c35.standard_runner` for a closer implementation of the written C3.5 contract:

```bash
python3 -m c35.standard_runner \
  --command 'python3 -m c35.deploy --onnx {onnx} --input {input} --output {output} --batch-size {batch_size}' \
  --batch-size 256 \
  --report c35-standard-report.json
```

The release entrypoint is:

```bash
./run_c35.sh
```

Set `PYTHON` only when selecting a different server-native Python executable.

Set `COMMAND_TEMPLATE` to the exact command registered for evaluation. The template must contain `{onnx}`, `{input}`, and `{output}`; it may contain `{batch_size}`.

The runner performs one cold subprocess execution per model and:

- reads input, golden, and threshold metadata from the released package;
- validates return status and output manifest schema;
- validates output names, files, float32 dtype, shapes, and sample count through golden shape equality;
- applies each model's `rtol` and `atol` from `thresholds.json`;
- computes the requested top-1 accuracy and model-specific minimum;
- measures process wall time from spawn through exit;
- samples GPU memory for the root process and Linux `/proc` descendants using the server-native `nvidia-smi` command, with a process-local CuPy memory-pool fallback for MIG environments;
- performs a CuPy import, CUDA-device, and matrix-multiply preflight;
- emits a machine-readable JSON report;
- returns nonzero if any correctness, accuracy, output-contract, command, timeout, or required GPU-evidence check fails.

The normal mode requires AEC H200 execution evidence from `nvidia-smi` or a
structured record emitted by the child after real CuPy
execution. The child record includes the device, CuPy version, and nonzero
default-pool reservation; its pool total is a process-local high-water proxy
when MIG suppresses per-process accounting, not an NVML-equivalent whole-process
measurement. There is no CPU/reference waiver in the CuPy-only runner.

The report awards the known 15-point correctness/accuracy gate only when every model and the CuPy preflight pass. Runtime and peak-memory points remain `null` because the written rubric ranks submissions against one another; raw measurements are recorded for that ranking.

The official evaluator implementation has not been released, so this runner follows the written specification but must not be described as byte-for-byte identical to the organizer's script.
