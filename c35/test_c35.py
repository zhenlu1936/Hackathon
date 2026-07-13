"""Tests for C3.5 end-to-end model deployment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
TESTDATA_DIR = ROOT / ".specification" / "testcases" / "release_to_competitors" / "testdata" / "c35"


class C35SpecificationTests(unittest.TestCase):
    """Specification conformance tests for C3.5."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="c35-test-")
        self.temp = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_deploy(
        self,
        model: str,
        batch_size: int | None = None,
    ) -> Path:
        """Run C3.5 deploy and return output directory."""
        model_path = MODELS_DIR / f"{model}_v1.onnx"
        input_dir = TESTDATA_DIR / f"{model}_v1" / "input"
        output_dir = self.temp / f"{model}_out"

        cmd = [
            sys.executable, "-m", "c35.deploy",
            "--onnx", str(model_path),
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--backend", "numpy",
        ]
        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0,
            f"Deploy failed for {model}:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        )
        return output_dir

    def _check_output_manifest(self, output_dir: Path, expected_shape: list[int]) -> None:
        """Verify output manifest.json and logits.npy."""
        manifest_path = output_dir / "manifest.json"
        logits_path = output_dir / "logits.npy"

        self.assertTrue(manifest_path.is_file(), f"Missing manifest at {manifest_path}")
        self.assertTrue(logits_path.is_file(), f"Missing logits at {logits_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        tensors = manifest.get("tensors", [])
        self.assertEqual(len(tensors), 1, "Expected exactly one output tensor")
        entry = tensors[0]

        self.assertEqual(entry["name"], "logits")
        self.assertEqual(entry["dtype"], "float32")

        logits = np.load(logits_path)
        self.assertEqual(logits.dtype, np.float32)
        self.assertEqual(list(logits.shape), expected_shape)
        self.assertTrue(np.all(np.isfinite(logits)), "Output contains NaN/Inf")
        self.assertTrue(logits.flags["C_CONTIGUOUS"], "Output not C-contiguous")

    # ── MLP tests ───────────────────────────────

    def test_mlp_batch_256(self) -> None:
        output_dir = self._run_deploy("mlp", batch_size=256)
        self._check_output_manifest(output_dir, [10000, 10])

        golden = np.load(TESTDATA_DIR / "mlp_v1" / "golden" / "logits.npy")
        logits = np.load(output_dir / "logits.npy")
        self.assertTrue(
            np.allclose(logits, golden, rtol=1e-3, atol=1e-3),
            f"MLP allclose failed: max_diff={np.max(np.abs(logits - golden)):.6e}"
        )

        labels = np.load(TESTDATA_DIR / "mlp_v1" / "labels.npy")
        acc = np.mean(np.argmax(logits, axis=-1) == labels.reshape(-1))
        self.assertGreaterEqual(acc, 0.98, f"MLP accuracy {acc:.4f} < 0.98")

    def test_mlp_batch_size_one(self) -> None:
        output_dir = self._run_deploy("mlp", batch_size=1)
        self._check_output_manifest(output_dir, [10000, 10])

    def test_mlp_batch_size_prime(self) -> None:
        """Test batch size that does not divide sample count."""
        output_dir = self._run_deploy("mlp", batch_size=7)
        self._check_output_manifest(output_dir, [10000, 10])

    def test_mlp_batch_larger_than_samples(self) -> None:
        """Test batch size > total samples."""
        output_dir = self._run_deploy("mlp", batch_size=20000)
        self._check_output_manifest(output_dir, [10000, 10])

    def test_mlp_batch_size_independence(self) -> None:
        """Output must be identical regardless of batch size."""
        out1 = self._run_deploy("mlp", batch_size=1)
        out2 = self._run_deploy("mlp", batch_size=256)
        logits1 = np.load(out1 / "logits.npy")
        logits2 = np.load(out2 / "logits.npy")
        max_diff = np.max(np.abs(logits1 - logits2))
        self.assertLess(
            max_diff, 1e-3,
            f"Batch-size-independent outputs differ by {max_diff:.6e}"
        )

    # ── ResNet tests ────────────────────────────

    def test_resnet_batch_64(self) -> None:
        output_dir = self._run_deploy("resnet", batch_size=64)
        self._check_output_manifest(output_dir, [10000, 10])

        golden = np.load(TESTDATA_DIR / "resnet_v1" / "golden" / "logits.npy")
        logits = np.load(output_dir / "logits.npy")
        self.assertTrue(
            np.allclose(logits, golden, rtol=1e-3, atol=1e-3),
            f"ResNet allclose failed: max_diff={np.max(np.abs(logits - golden)):.6e}"
        )

        labels = np.load(TESTDATA_DIR / "resnet_v1" / "labels.npy")
        acc = np.mean(np.argmax(logits, axis=-1) == labels.reshape(-1))
        self.assertGreaterEqual(acc, 0.85, f"ResNet accuracy {acc:.4f} < 0.85")

    def test_resnet_batch_indivisible(self) -> None:
        """Test batch size that does not divide 10000."""
        output_dir = self._run_deploy("resnet", batch_size=97)
        self._check_output_manifest(output_dir, [10000, 10])

    # ── Transformer tests ───────────────────────

    def test_transformer_batch_128(self) -> None:
        output_dir = self._run_deploy("transformer", batch_size=128)
        self._check_output_manifest(output_dir, [10000, 18, 14])

        golden = np.load(TESTDATA_DIR / "transformer_v1" / "golden" / "logits.npy")
        logits = np.load(output_dir / "logits.npy")
        self.assertTrue(
            np.allclose(logits, golden, rtol=1e-3, atol=1e-3),
            f"Transformer allclose failed: max_diff={np.max(np.abs(logits - golden)):.6e}"
        )

    def test_transformer_batch_one(self) -> None:
        output_dir = self._run_deploy("transformer", batch_size=1)
        self._check_output_manifest(output_dir, [10000, 18, 14])

    # ── Determinism tests ───────────────────────

    def test_repeated_runs_identical(self) -> None:
        """Two runs with the same inputs must produce identical outputs."""
        out1 = self._run_deploy("mlp", batch_size=256)
        out2 = self._run_deploy("mlp", batch_size=256)
        logits1 = np.load(out1 / "logits.npy")
        logits2 = np.load(out2 / "logits.npy")
        np.testing.assert_array_equal(logits1, logits2)

    def test_manifest_self_consistent(self) -> None:
        """Manifest metadata must match actual file content."""
        for model in ["mlp", "resnet", "transformer"]:
            with self.subTest(model=model):
                output_dir = self._run_deploy(model, batch_size=32)
                with open(output_dir / "manifest.json", "r") as f:
                    manifest = json.load(f)
                logits = np.load(output_dir / "logits.npy")
                entry = manifest["tensors"][0]
                self.assertEqual(entry["dtype"], "float32")
                self.assertEqual(logits.dtype, np.float32)
                self.assertEqual(list(logits.shape), entry["shape"])

    # ── CLI validation tests ────────────────────

    def test_missing_model_exits_nonzero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "c35.deploy",
             "--onnx", "/nonexistent/model.onnx",
             "--input", str(TESTDATA_DIR / "mlp_v1" / "input"),
             "--output", str(self.temp / "out")],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_negative_batch_size_rejected(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "c35.deploy",
             "--onnx", str(MODELS_DIR / "mlp_v1.onnx"),
             "--input", str(TESTDATA_DIR / "mlp_v1" / "input"),
             "--output", str(self.temp / "out"),
             "--batch-size", "-1"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_zero_batch_size_rejected(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "c35.deploy",
             "--onnx", str(MODELS_DIR / "mlp_v1.onnx"),
             "--input", str(TESTDATA_DIR / "mlp_v1" / "input"),
             "--output", str(self.temp / "out"),
             "--batch-size", "0"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


class C35OperatorTests(unittest.TestCase):
    """Unit tests for individual ONNX operator implementations."""

    def test_relu(self) -> None:
        from c35.engine import op_relu
        x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        out = op_relu([x], {})
        np.testing.assert_array_equal(out, np.array([[0.0, 0.0, 1.0, 2.0]], dtype=np.float32))

    def test_gemm_no_transpose(self) -> None:
        from c35.engine import op_gemm
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0], [4.0]], dtype=np.float32)
        c = np.array([[0.5]], dtype=np.float32)
        out = op_gemm([a, b, c], {"alpha": 1.0, "beta": 1.0})
        expected = np.dot(a, b) + c
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_gemm_transpose_b(self) -> None:
        from c35.engine import op_gemm
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        # Test transposeB without bias (pass None as third input)
        out = op_gemm([a, b, np.float32(0.0)], {"transB": 1, "alpha": 1.0, "beta": 0.0})
        expected = np.dot(a, b.T)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_softmax_stable(self) -> None:
        from c35.engine import op_softmax
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = op_softmax([x], {"axis": -1})
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=5)
        self.assertTrue(np.all(out >= 0))

    def test_layer_norm(self) -> None:
        from c35.engine import op_layer_normalization
        x = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        bias = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = op_layer_normalization([x, scale, bias], {"axis": -1, "epsilon": 1e-5})
        # Mean should be close to 0, std close to 1
        self.assertAlmostEqual(float(np.mean(out)), 0.0, places=5)
        self.assertAlmostEqual(float(np.std(out)), 1.0, places=2)

    def test_conv_3x3_s1(self) -> None:
        from c35.engine import op_conv
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        w = np.random.randn(16, 3, 3, 3).astype(np.float32)
        out = op_conv([x, w], {
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
        })
        self.assertEqual(out.shape, (1, 16, 32, 32))

    def test_conv_3x3_s2(self) -> None:
        from c35.engine import op_conv
        x = np.random.randn(1, 3, 32, 32).astype(np.float32)
        w = np.random.randn(16, 3, 3, 3).astype(np.float32)
        out = op_conv([x, w], {
            "kernel_shape": [3, 3],
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
        })
        self.assertEqual(out.shape, (1, 16, 15, 15))

    def test_gather(self) -> None:
        from c35.engine import op_gather
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        indices = np.array([0, 2], dtype=np.int64)
        out = op_gather([x, indices], {"axis": 0})
        np.testing.assert_array_equal(out, np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32))

    def test_gather_with_batch(self) -> None:
        from c35.engine import op_gather
        x = np.random.randn(1, 14, 128).astype(np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int64)
        out = op_gather([x, indices], {"axis": 1})
        self.assertEqual(out.shape, (1, 1, 3, 128))

    def test_reshape_with_neg_one(self) -> None:
        from c35.engine import op_reshape
        x = np.random.randn(1, 18, 128).astype(np.float32)
        shape = np.array([-1, 18, 4, 32], dtype=np.int64)
        out = op_reshape([x, shape], {"allowzero": 0})
        self.assertEqual(out.shape, (1, 18, 4, 32))

    def test_split_three_way(self) -> None:
        from c35.engine import op_split
        x = np.random.randn(1, 18, 384).astype(np.float32)
        out = op_split([x, np.array([])], {"axis": -1, "_num_outputs": 3})
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].shape, (1, 18, 128))
        self.assertEqual(out[1].shape, (1, 18, 128))
        self.assertEqual(out[2].shape, (1, 18, 128))

    def test_split_uses_python_integer_boundaries(self) -> None:
        import c35.engine as engine

        original_xp = engine.xp

        class StrictArrayModule:
            float32 = np.float32

            @staticmethod
            def split(value, indices, axis=0):
                if not isinstance(indices, list) or not all(
                    type(index) is int for index in indices
                ):
                    raise TypeError("indices must be Python integers")
                return np.split(value, indices, axis=axis)

        try:
            engine.xp = StrictArrayModule()
            x = np.arange(12, dtype=np.float32).reshape(1, 12)
            out = engine.op_split([x], {"axis": 1, "split": [3, 4, 5]})
        finally:
            engine.xp = original_xp
        self.assertEqual([part.shape for part in out], [(1, 3), (1, 4), (1, 5)])

    def test_transpose(self) -> None:
        from c35.engine import op_transpose
        x = np.random.randn(1, 4, 18, 32).astype(np.float32)
        out = op_transpose([x], {"perm": [0, 2, 1, 3]})
        self.assertEqual(out.shape, (1, 18, 4, 32))

    def test_erf(self) -> None:
        from c35.engine import op_erf
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out = op_erf([x], {})
        # erf(0) = 0
        self.assertAlmostEqual(float(out[2]), 0.0, places=4)
        # erf(inf) -> 1, erf(-inf) -> -1
        self.assertGreater(float(out[4]), 0.99)
        self.assertLess(float(out[0]), -0.99)

    def test_global_average_pool(self) -> None:
        from c35.engine import op_global_average_pool
        x = np.ones((1, 64, 8, 8), dtype=np.float32)
        out = op_global_average_pool([x], {})
        self.assertEqual(out.shape, (1, 64, 1, 1))
        np.testing.assert_allclose(out, 1.0, rtol=1e-5)

    def test_flatten(self) -> None:
        from c35.engine import op_flatten
        x = np.random.randn(1, 64, 1, 1).astype(np.float32)
        out = op_flatten([x], {"axis": 1})
        self.assertEqual(out.shape, (1, 64))


class C35RunnerEvidenceTests(unittest.TestCase):
    def test_valid_cupy_pool_evidence(self) -> None:
        from c35.standard_runner import (
            GPU_EVIDENCE_PREFIX,
            _parse_backend_evidence,
            _valid_cupy_evidence,
        )

        payload = {
            "backend": "cupy",
            "cupy_version": "14.1.1",
            "device_id": 0,
            "device_name": "test-mig",
            "pool_reserved_bytes": 4096,
        }
        stderr = "log line\n" + GPU_EVIDENCE_PREFIX + json.dumps(payload) + "\n"
        parsed = _parse_backend_evidence(stderr)
        self.assertEqual(parsed, payload)
        self.assertTrue(_valid_cupy_evidence(parsed))

    def test_zero_pool_is_not_gpu_evidence(self) -> None:
        from c35.standard_runner import _valid_cupy_evidence

        self.assertFalse(_valid_cupy_evidence({
            "backend": "cupy",
            "cupy_version": "14.1.1",
            "device_id": 0,
            "pool_reserved_bytes": 0,
        }))


if __name__ == "__main__":
    unittest.main()
