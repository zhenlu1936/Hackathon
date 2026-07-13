"""Tests for C3.5 end-to-end model deployment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cupy as cp
import onnx


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / ".specification" / "testcases" / "release_to_competitors"
MODELS_DIR = RELEASE_DIR / "models"
TESTDATA_DIR = RELEASE_DIR / "testdata" / "c35"


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

        logits = cp.load(logits_path)
        self.assertEqual(logits.dtype, cp.float32)
        self.assertEqual(list(logits.shape), expected_shape)
        self.assertTrue(cp.all(cp.isfinite(logits)), "Output contains NaN/Inf")
        self.assertTrue(logits.flags["C_CONTIGUOUS"], "Output not C-contiguous")

    # ── MLP tests ───────────────────────────────

    def test_mlp_batch_256(self) -> None:
        output_dir = self._run_deploy("mlp", batch_size=256)
        self._check_output_manifest(output_dir, [10000, 10])

        golden = cp.load(TESTDATA_DIR / "mlp_v1" / "golden" / "logits.npy")
        logits = cp.load(output_dir / "logits.npy")
        self.assertTrue(
            cp.allclose(logits, golden, rtol=1e-3, atol=1e-3),
            f"MLP allclose failed: max_diff={cp.max(cp.abs(logits - golden)):.6e}"
        )

        labels = cp.load(TESTDATA_DIR / "mlp_v1" / "labels.npy")
        acc = cp.mean(cp.argmax(logits, axis=-1) == labels.reshape(-1))
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
        logits1 = cp.load(out1 / "logits.npy")
        logits2 = cp.load(out2 / "logits.npy")
        max_diff = cp.max(cp.abs(logits1 - logits2))
        self.assertLess(
            max_diff, 1e-3,
            f"Batch-size-independent outputs differ by {max_diff:.6e}"
        )

    # ── ResNet tests ────────────────────────────

    def test_resnet_batch_64(self) -> None:
        output_dir = self._run_deploy("resnet", batch_size=64)
        self._check_output_manifest(output_dir, [10000, 10])

        golden = cp.load(TESTDATA_DIR / "resnet_v1" / "golden" / "logits.npy")
        logits = cp.load(output_dir / "logits.npy")
        self.assertTrue(
            cp.allclose(logits, golden, rtol=1e-3, atol=1e-3),
            f"ResNet allclose failed: max_diff={cp.max(cp.abs(logits - golden)):.6e}"
        )

        labels = cp.load(TESTDATA_DIR / "resnet_v1" / "labels.npy")
        acc = cp.mean(cp.argmax(logits, axis=-1) == labels.reshape(-1))
        self.assertGreaterEqual(acc, 0.85, f"ResNet accuracy {acc:.4f} < 0.85")

    def test_resnet_batch_indivisible(self) -> None:
        """Test batch size that does not divide 10000."""
        output_dir = self._run_deploy("resnet", batch_size=97)
        self._check_output_manifest(output_dir, [10000, 10])

    # ── Transformer tests ───────────────────────

    def test_transformer_batch_128(self) -> None:
        output_dir = self._run_deploy("transformer", batch_size=128)
        self._check_output_manifest(output_dir, [10000, 18, 14])

        golden = cp.load(TESTDATA_DIR / "transformer_v1" / "golden" / "logits.npy")
        logits = cp.load(output_dir / "logits.npy")
        self.assertTrue(
            cp.allclose(logits, golden, rtol=1e-3, atol=1e-3),
            f"Transformer allclose failed: max_diff={cp.max(cp.abs(logits - golden)):.6e}"
        )

    def test_transformer_batch_one(self) -> None:
        output_dir = self._run_deploy("transformer", batch_size=1)
        self._check_output_manifest(output_dir, [10000, 18, 14])

    # ── Determinism tests ───────────────────────

    def test_repeated_runs_identical(self) -> None:
        """Two runs with the same inputs must produce identical outputs."""
        out1 = self._run_deploy("mlp", batch_size=256)
        out2 = self._run_deploy("mlp", batch_size=256)
        logits1 = cp.load(out1 / "logits.npy")
        logits2 = cp.load(out2 / "logits.npy")
        cp.testing.assert_array_equal(logits1, logits2)

    def test_manifest_self_consistent(self) -> None:
        """Manifest metadata must match actual file content."""
        for model in ["mlp", "resnet", "transformer"]:
            with self.subTest(model=model):
                output_dir = self._run_deploy(model, batch_size=32)
                with open(output_dir / "manifest.json", "r") as f:
                    manifest = json.load(f)
                logits = cp.load(output_dir / "logits.npy")
                entry = manifest["tensors"][0]
                self.assertEqual(entry["dtype"], "float32")
                self.assertEqual(logits.dtype, cp.float32)
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
        x = cp.array([[-1.0, 0.0, 1.0, 2.0]], dtype=cp.float32)
        out = op_relu([x], {})
        cp.testing.assert_array_equal(out, cp.array([[0.0, 0.0, 1.0, 2.0]], dtype=cp.float32))

    def test_gemm_no_transpose(self) -> None:
        from c35.engine import op_gemm
        a = cp.array([[1.0, 2.0]], dtype=cp.float32)
        b = cp.array([[3.0], [4.0]], dtype=cp.float32)
        c = cp.array([[0.5]], dtype=cp.float32)
        out = op_gemm([a, b, c], {"alpha": 1.0, "beta": 1.0})
        expected = cp.dot(a, b) + c
        cp.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_gemm_transpose_b(self) -> None:
        from c35.engine import op_gemm
        a = cp.array([[1.0, 2.0]], dtype=cp.float32)
        b = cp.array([[3.0, 4.0]], dtype=cp.float32)
        # Test transposeB without bias (pass None as third input)
        out = op_gemm([a, b, cp.float32(0.0)], {"transB": 1, "alpha": 1.0, "beta": 0.0})
        expected = cp.dot(a, b.T)
        cp.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_softmax_stable(self) -> None:
        from c35.engine import op_softmax
        x = cp.array([[1.0, 2.0, 3.0]], dtype=cp.float32)
        out = op_softmax([x], {"axis": -1})
        self.assertAlmostEqual(float(cp.sum(out)), 1.0, places=5)
        self.assertTrue(bool(cp.all(out >= 0).item()))

    def test_layer_norm(self) -> None:
        from c35.engine import op_layer_normalization
        x = cp.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=cp.float32)
        scale = cp.array([1.0, 1.0, 1.0, 1.0], dtype=cp.float32)
        bias = cp.array([0.0, 0.0, 0.0, 0.0], dtype=cp.float32)
        out = op_layer_normalization([x, scale, bias], {"axis": -1, "epsilon": 1e-5})
        # Mean should be close to 0, std close to 1
        self.assertAlmostEqual(float(cp.mean(out)), 0.0, places=5)
        self.assertAlmostEqual(float(cp.std(out)), 1.0, places=2)

    def test_conv_3x3_s1(self) -> None:
        from c35.engine import op_conv
        x = cp.random.randn(1, 3, 32, 32).astype(cp.float32)
        w = cp.random.randn(16, 3, 3, 3).astype(cp.float32)
        out = op_conv([x, w], {
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
        })
        self.assertEqual(out.shape, (1, 16, 32, 32))

    def test_conv_3x3_s2(self) -> None:
        from c35.engine import op_conv
        x = cp.random.randn(1, 3, 32, 32).astype(cp.float32)
        w = cp.random.randn(16, 3, 3, 3).astype(cp.float32)
        out = op_conv([x, w], {
            "kernel_shape": [3, 3],
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
        })
        self.assertEqual(out.shape, (1, 16, 15, 15))

    def test_gather(self) -> None:
        from c35.engine import op_gather
        x = cp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=cp.float32)
        indices = cp.array([0, 2], dtype=cp.int64)
        out = op_gather([x, indices], {"axis": 0})
        cp.testing.assert_array_equal(out, cp.array([[1.0, 2.0], [5.0, 6.0]], dtype=cp.float32))

    def test_gather_with_batch(self) -> None:
        from c35.engine import op_gather
        x = cp.random.randn(1, 14, 128).astype(cp.float32)
        indices = cp.array([[0, 1, 2]], dtype=cp.int64)
        out = op_gather([x, indices], {"axis": 1})
        self.assertEqual(out.shape, (1, 1, 3, 128))

    def test_reshape_with_neg_one(self) -> None:
        from c35.engine import op_reshape
        x = cp.random.randn(1, 18, 128).astype(cp.float32)
        shape = cp.array([-1, 18, 4, 32], dtype=cp.int64)
        out = op_reshape([x, shape], {"allowzero": 0})
        self.assertEqual(out.shape, (1, 18, 4, 32))

    def test_split_three_way(self) -> None:
        from c35.engine import op_split
        x = cp.random.randn(1, 18, 384).astype(cp.float32)
        out = op_split([x, cp.array([])], {"axis": -1, "_num_outputs": 3})
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].shape, (1, 18, 128))
        self.assertEqual(out[1].shape, (1, 18, 128))
        self.assertEqual(out[2].shape, (1, 18, 128))

    def test_split_uses_python_integer_boundaries(self) -> None:
        import c35.engine as engine

        original_xp = engine.xp

        class StrictArrayModule:
            float32 = cp.float32

            @staticmethod
            def split(value, indices, axis=0):
                if not isinstance(indices, list) or not all(
                    type(index) is int for index in indices
                ):
                    raise TypeError("indices must be Python integers")
                return cp.split(value, indices, axis=axis)

        try:
            engine.xp = StrictArrayModule()
            x = cp.arange(12, dtype=cp.float32).reshape(1, 12)
            out = engine.op_split([x], {"axis": 1, "split": [3, 4, 5]})
        finally:
            engine.xp = original_xp
        self.assertEqual([part.shape for part in out], [(1, 3), (1, 4), (1, 5)])

    def test_transpose(self) -> None:
        from c35.engine import op_transpose
        x = cp.random.randn(1, 4, 18, 32).astype(cp.float32)
        out = op_transpose([x], {"perm": [0, 2, 1, 3]})
        self.assertEqual(out.shape, (1, 18, 4, 32))

    def test_erf(self) -> None:
        from c35.engine import op_erf
        x = cp.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=cp.float32)
        out = op_erf([x], {})
        # erf(0) = 0
        self.assertAlmostEqual(float(out[2]), 0.0, places=4)
        # erf(inf) -> 1, erf(-inf) -> -1
        self.assertGreater(float(out[4]), 0.99)
        self.assertLess(float(out[0]), -0.99)

    def test_global_average_pool(self) -> None:
        from c35.engine import op_global_average_pool
        x = cp.ones((1, 64, 8, 8), dtype=cp.float32)
        out = op_global_average_pool([x], {})
        self.assertEqual(out.shape, (1, 64, 1, 1))
        cp.testing.assert_allclose(out, 1.0, rtol=1e-5)

    def test_flatten(self) -> None:
        from c35.engine import op_flatten
        x = cp.random.randn(1, 64, 1, 1).astype(cp.float32)
        out = op_flatten([x], {"axis": 1})
        self.assertEqual(out.shape, (1, 64))


class C35Opset17AttributeTests(unittest.TestCase):
    """Non-default attributes and valid hidden-style shapes for all 17 ops."""

    def test_flatten_nondefault_axis_is_always_rank_two(self) -> None:
        from c35.engine import op_flatten

        x = cp.arange(120, dtype=cp.float32).reshape(2, 3, 4, 5)
        self.assertEqual(op_flatten([x], {"axis": 2}).shape, (6, 20))
        self.assertEqual(op_flatten([x], {"axis": 0}).shape, (1, 120))
        self.assertEqual(op_flatten([x], {"axis": -1}).shape, (24, 5))

    def test_gemm_transpose_a_alpha_beta_and_broadcast_bias(self) -> None:
        from c35.engine import op_gemm

        a = cp.arange(6, dtype=cp.float32).reshape(3, 2)
        b = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        c = cp.linspace(-1.0, 1.0, 4, dtype=cp.float32)
        actual = op_gemm(
            [a, b, c],
            {"transA": 1, "alpha": 0.5, "beta": 2.0},
        )
        expected = cp.float32(0.5) * (a.T @ b) + cp.float32(2.0) * c
        cp.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_relu_accepts_non_matrix_shape(self) -> None:
        from c35.engine import op_relu

        x = cp.linspace(-2.0, 2.0, 24, dtype=cp.float32).reshape(2, 3, 4)
        cp.testing.assert_array_equal(op_relu([x], {}), cp.maximum(x, 0))

    def test_conv_dilation_same_lower_and_group(self) -> None:
        from c35.engine import op_conv

        x = cp.arange(30, dtype=cp.float32).reshape(1, 1, 5, 6)
        w = cp.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=cp.float32)
        actual = op_conv([x, w], {
            "auto_pad": "SAME_LOWER",
            "strides": [2, 2],
            "dilations": [2, 2],
            "group": 1,
        })
        padded = cp.pad(x, ((0, 0), (0, 0), (1, 1), (1, 0)))
        expected = (
            padded[:, :, 0:6:2, 0:6:2] * w[0, 0, 0, 0]
            + padded[:, :, 0:6:2, 2:8:2] * w[0, 0, 0, 1]
            + padded[:, :, 2:8:2, 0:6:2] * w[0, 0, 1, 0]
            + padded[:, :, 2:8:2, 2:8:2] * w[0, 0, 1, 1]
        )
        cp.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

        grouped_x = cp.arange(18, dtype=cp.float32).reshape(1, 2, 3, 3)
        grouped_w = cp.array([[[[2.0]]], [[[3.0]]]], dtype=cp.float32)
        grouped = op_conv([grouped_x, grouped_w], {"group": 2})
        cp.testing.assert_array_equal(grouped[:, 0], grouped_x[:, 0] * 2.0)
        cp.testing.assert_array_equal(grouped[:, 1], grouped_x[:, 1] * 3.0)

    def test_add_broadcasts_higher_rank(self) -> None:
        from c35.engine import op_add

        x = cp.arange(24, dtype=cp.float32).reshape(2, 3, 4)
        bias = cp.array([1.0, -1.0, 2.0, -2.0], dtype=cp.float32)
        cp.testing.assert_array_equal(op_add([x, bias], {}), x + bias)

    def test_global_average_pool_supports_extra_spatial_rank(self) -> None:
        from c35.engine import op_global_average_pool

        x = cp.arange(2 * 3 * 4 * 5 * 6, dtype=cp.float32).reshape(2, 3, 4, 5, 6)
        actual = op_global_average_pool([x], {})
        self.assertEqual(actual.shape, (2, 3, 1, 1, 1))
        cp.testing.assert_allclose(actual, x.mean(axis=(2, 3, 4), keepdims=True))

    def test_gather_negative_axis_preserves_data_dtype(self) -> None:
        from c35.engine import op_gather

        data = cp.arange(24, dtype=cp.int64).reshape(2, 3, 4)
        indices = cp.array([[3, 1], [0, 2]], dtype=cp.int64)
        actual = op_gather([data, indices], {"axis": -1})
        expected = cp.take(data, indices, axis=2)
        self.assertEqual(actual.dtype, cp.int64)
        cp.testing.assert_array_equal(actual, expected)

    def test_layer_norm_axis_one_and_optional_outputs(self) -> None:
        from c35.engine import op_layer_normalization

        x = cp.arange(24, dtype=cp.float32).reshape(2, 3, 4)
        scale = cp.linspace(0.5, 1.5, 12, dtype=cp.float32).reshape(3, 4)
        bias = cp.linspace(-0.2, 0.2, 12, dtype=cp.float32).reshape(3, 4)
        y, mean, inv_std = op_layer_normalization(
            [x, scale, bias],
            {"axis": 1, "epsilon": 1e-3, "_num_outputs": 3},
        )
        expected_mean = x.mean(axis=(1, 2), keepdims=True)
        centered = x - expected_mean
        expected_inv_std = 1.0 / cp.sqrt(
            cp.mean(centered * centered, axis=(1, 2), keepdims=True) + 1e-3
        )
        expected_y = centered * expected_inv_std * scale + bias
        cp.testing.assert_allclose(y, expected_y, rtol=1e-5, atol=1e-5)
        cp.testing.assert_allclose(mean, expected_mean, rtol=1e-6, atol=1e-6)
        cp.testing.assert_allclose(inv_std, expected_inv_std, rtol=1e-6, atol=1e-6)

    def test_matmul_broadcasts_batch_dimensions(self) -> None:
        from c35.engine import op_matmul

        a = cp.arange(2 * 1 * 3 * 4, dtype=cp.float32).reshape(2, 1, 3, 4)
        b = cp.arange(5 * 4 * 2, dtype=cp.float32).reshape(1, 5, 4, 2)
        cp.testing.assert_allclose(op_matmul([a, b], {}), cp.matmul(a, b))

    def test_constant_tensor_and_scalar_attributes_are_loaded(self) -> None:
        from c35.executor import _extract_constant_values

        tensor = onnx.numpy_helper.from_array(
            cp.asnumpy(cp.arange(6, dtype=cp.float32).reshape(2, 3)),
            name="constant_value",
        )
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Constant", [], ["tensor_out"], value=tensor),
                onnx.helper.make_node("Constant", [], ["scalar_out"], value_int=7),
                onnx.helper.make_node(
                    "Constant", [], ["float_vector_out"],
                    value_floats=[0.25, -0.5],
                ),
            ],
            "constants",
            [],
            [],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        with tempfile.TemporaryDirectory(prefix="c35-constant-") as temp:
            path = Path(temp) / "constants.onnx"
            onnx.save(model, path)
            values = _extract_constant_values(str(path))
        cp.testing.assert_array_equal(
            values["tensor_out"], cp.arange(6, dtype=cp.float32).reshape(2, 3)
        )
        self.assertEqual(values["scalar_out"].dtype, cp.int64)
        self.assertEqual(int(values["scalar_out"].item()), 7)
        self.assertEqual(values["float_vector_out"].dtype, cp.float32)
        cp.testing.assert_array_equal(
            values["float_vector_out"], cp.array([0.25, -0.5], dtype=cp.float32)
        )

    def test_split_uses_opset17_sizes_input(self) -> None:
        from c35.engine import op_split

        x = cp.arange(24, dtype=cp.float32).reshape(2, 12)
        sizes = cp.array([2, 4, 6], dtype=cp.int64)
        parts = op_split([x, sizes], {"axis": -1, "_num_outputs": 3})
        self.assertEqual([part.shape for part in parts], [(2, 2), (2, 4), (2, 6)])
        cp.testing.assert_array_equal(cp.concatenate(parts, axis=-1), x)

    def test_reshape_zero_copies_input_dimension(self) -> None:
        from c35.engine import op_reshape

        x = cp.arange(24, dtype=cp.float32).reshape(2, 3, 4)
        shape = cp.array([0, -1], dtype=cp.int64)
        actual = op_reshape([x, shape], {"allowzero": 0})
        self.assertEqual(actual.shape, (2, 12))
        cp.testing.assert_array_equal(actual.reshape(x.shape), x)

    def test_transpose_default_reverses_axes(self) -> None:
        from c35.engine import op_transpose

        x = cp.arange(24, dtype=cp.float32).reshape(2, 3, 4)
        cp.testing.assert_array_equal(op_transpose([x], {}), cp.transpose(x, (2, 1, 0)))

    def test_div_broadcasts_non_scalar_denominator(self) -> None:
        from c35.engine import op_div

        x = cp.arange(1, 13, dtype=cp.float32).reshape(3, 4)
        denominator = cp.array([1.0, 2.0, 4.0, 8.0], dtype=cp.float32)
        cp.testing.assert_allclose(op_div([x, denominator], {}), x / denominator)

    def test_softmax_nonlast_axis(self) -> None:
        from c35.engine import op_softmax

        x = cp.arange(24, dtype=cp.float32).reshape(2, 3, 4)
        actual = op_softmax([x], {"axis": 1})
        cp.testing.assert_allclose(actual.sum(axis=1), cp.ones((2, 4)), atol=1e-6)

    def test_erf_and_mul_hidden_style_broadcast(self) -> None:
        from c35.engine import op_erf, op_mul

        x = cp.linspace(-3.0, 3.0, 24, dtype=cp.float32).reshape(2, 3, 4)
        erf_out = op_erf([x], {})
        self.assertEqual(erf_out.shape, x.shape)
        self.assertTrue(bool(cp.all(erf_out <= 1.0).item()))
        self.assertTrue(bool(cp.all(erf_out >= -1.0).item()))
        scale = cp.array([0.5, 1.0, 1.5, 2.0], dtype=cp.float32)
        cp.testing.assert_allclose(op_mul([erf_out, scale], {}), erf_out * scale)


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
