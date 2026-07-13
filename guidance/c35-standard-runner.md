# C3.5 standard black-box runner

`c35/test_c35.py` is a development regression suite, not a replica of the official evaluator. It mixes internal operator unit tests with CLI tests, hardcodes public model shapes and thresholds, checks golden results only for selected batch cases, invokes only this repository's Python module, and does not sample NVML process-tree memory.

Use `c35.standard_runner` for a closer implementation of the written C3.5 contract:

```bash
python3 -m c35.standard_runner \
  --command 'python3 -m c35.deploy --onnx {onnx} --input {input} --output {output} --batch-size {batch_size} --backend cupy' \
  --batch-size 256 \
  --report c35-standard-report.json
```

Or run:

```bash
./run_c35_standard.sh
```

Set `PYTHON=.venv/bin/python` when the required packages are installed in the local virtual environment.

Set `COMMAND_TEMPLATE` to the exact command registered for evaluation. The template must contain `{onnx}`, `{input}`, and `{output}`; it may contain `{batch_size}`.

The runner performs one cold subprocess execution per model and:

- reads input, golden, and threshold metadata from the released package;
- validates return status and output manifest schema;
- validates output names, files, float32 dtype, shapes, and sample count through golden shape equality;
- applies each model's `rtol` and `atol` from `thresholds.json`;
- computes the requested top-1 accuracy and model-specific minimum;
- measures process wall time from spawn through exit;
- samples GPU memory for the root process and Linux `/proc` descendants using `pynvml`, with an automatic `nvidia-smi` fallback;
- performs a CuPy import, CUDA-device, and matrix-multiply preflight;
- emits a machine-readable JSON report;
- returns nonzero if any correctness, accuracy, output-contract, command, timeout, or required GPU-evidence check fails.

The normal mode requires GPU process memory to be observed through `pynvml` or `nvidia-smi`. On a disclosed CPU development machine, add `--allow-reference`; the report records that waiver. GPU observation does not by itself prove that the implementation uses the required AEC compiler/runtime, so AEC call-path compliance still requires code review and target-environment evidence.

The report awards the known 15-point correctness/accuracy gate only when every model and the CuPy preflight pass. Runtime and peak-memory points remain `null` because the written rubric ranks submissions against one another; raw measurements are recorded for that ranking.

For a disclosed CPU-only development check, make all exceptions explicit:

```bash
./run_c35_standard.sh \
  --allow-reference \
  --skip-cupy-preflight \
  --command 'python3 -m c35.deploy --onnx {onnx} --input {input} --output {output} --batch-size {batch_size} --backend numpy'
```

The official evaluator implementation has not been released, so this runner follows the written specification but must not be described as byte-for-byte identical to the organizer's script.
