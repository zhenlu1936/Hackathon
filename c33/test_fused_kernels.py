"""Device-side numerical checks for generated C3.3 kernels and fallbacks.

Run on the organizer H200 with::

    python3 -m unittest c33.test_fused_kernels
"""

from __future__ import annotations

import unittest

import cupy as cp

from c35.engine import (
    op_conv,
    op_fused_attention_scores,
    op_fused_conv_activation,
    op_fused_conv_residual_activation,
    op_fused_gemm_epilogue,
    op_fused_layer_normalization,
    op_fused_matmul_bias,
    op_fused_residual_norm,
    op_fused_softmax_dropout,
    op_fused_transpose_reshape,
    op_gemm,
    op_layer_normalization,
    op_softmax,
)


class FusedKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        if cp.cuda.runtime.getDeviceCount() < 1:
            self.skipTest("requires a CUDA device")

    def test_softmax_dropout_matches_reference_for_nonlast_axis(self) -> None:
        value = cp.linspace(-3.0, 3.0, 2 * 5 * 7, dtype=cp.float32).reshape(2, 5, 7)
        attrs = {"axis": 1}
        actual = op_fused_softmax_dropout([value], attrs)
        expected = op_softmax([value], attrs)
        cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_softmax_dropout_writes_planned_output(self) -> None:
        value = cp.linspace(-2.0, 2.0, 3 * 11, dtype=cp.float32).reshape(3, 11)
        planned = cp.empty_like(value)
        actual = op_fused_softmax_dropout(
            [value], {"axis": -1, "_planned_output": planned}
        )
        self.assertEqual(actual.data.ptr, planned.data.ptr)
        cp.testing.assert_allclose(
            actual, op_softmax([value], {"axis": -1}), rtol=1e-5, atol=1e-6
        )

    def test_residual_norm_matches_reference(self) -> None:
        x = cp.linspace(-1.0, 1.0, 3 * 4 * 16, dtype=cp.float32).reshape(3, 4, 16)
        residual = cp.linspace(0.5, -0.5, 3 * 4 * 16, dtype=cp.float32).reshape(3, 4, 16)
        scale = cp.linspace(0.75, 1.25, 16, dtype=cp.float32)
        bias = cp.linspace(-0.1, 0.1, 16, dtype=cp.float32)
        attrs = {"axis": -1, "epsilon": 1e-5}
        actual = op_fused_residual_norm([x, residual, scale, bias], attrs)
        expected = op_layer_normalization([x + residual, scale, bias], attrs)
        cp.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_residual_norm_writes_planned_output(self) -> None:
        x = cp.linspace(-1.0, 1.0, 2 * 3 * 8, dtype=cp.float32).reshape(2, 3, 8)
        residual = cp.linspace(0.25, -0.25, 2 * 3 * 8, dtype=cp.float32).reshape(2, 3, 8)
        scale = cp.ones(8, dtype=cp.float32)
        bias = cp.zeros(8, dtype=cp.float32)
        planned = cp.empty_like(x)
        attrs = {"axis": -1, "epsilon": 1e-5, "_planned_output": planned}
        actual = op_fused_residual_norm([x, residual, scale, bias], attrs)
        self.assertEqual(actual.data.ptr, planned.data.ptr)
        cp.testing.assert_allclose(
            actual,
            op_layer_normalization(
                [x + residual, scale, bias], {"axis": -1, "epsilon": 1e-5}
            ),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_gemm_flatten_bias_relu_matches_reference(self) -> None:
        value = cp.linspace(-1.0, 1.0, 2 * 3 * 4, dtype=cp.float32).reshape(2, 3, 4)
        weight = cp.linspace(-0.5, 0.5, 12 * 5, dtype=cp.float32).reshape(12, 5)
        bias = cp.linspace(-0.2, 0.2, 5, dtype=cp.float32)
        planned = cp.empty((2, 5), dtype=cp.float32)
        attrs = {
            "_flatten_axis": 1,
            "_activation": "Relu",
            "_planned_output": planned,
        }
        actual = op_fused_gemm_epilogue([value, weight, bias], attrs)
        expected = cp.maximum(
            op_gemm([value.reshape(2, 12), weight, bias], {}), 0
        )
        self.assertEqual(actual.data.ptr, planned.data.ptr)
        cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_batched_matmul_bias_matches_reference(self) -> None:
        value = cp.linspace(-1.0, 1.0, 2 * 3 * 7, dtype=cp.float32).reshape(2, 3, 7)
        weight = cp.linspace(-0.4, 0.4, 7 * 5, dtype=cp.float32).reshape(7, 5)
        bias = cp.linspace(-0.1, 0.1, 5, dtype=cp.float32)
        actual = op_fused_matmul_bias([value, weight, bias], {})
        expected = cp.matmul(value, weight) + bias
        cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_conv_relu_and_residual_match_reference(self) -> None:
        value = cp.linspace(-1.0, 1.0, 2 * 4 * 7 * 7, dtype=cp.float32).reshape(2, 4, 7, 7)
        weight = cp.linspace(-0.3, 0.3, 6 * 4 * 3 * 3, dtype=cp.float32).reshape(6, 4, 3, 3)
        bias = cp.linspace(-0.2, 0.2, 6, dtype=cp.float32)
        attrs = {
            "kernel_shape": [3, 3], "pads": [1, 1, 1, 1],
            "strides": [1, 1], "dilations": [1, 1], "group": 1,
            "_activation": "Relu",
        }
        conv = op_conv([value, weight, bias], attrs)
        actual = op_fused_conv_activation([value, weight, bias], attrs)
        cp.testing.assert_allclose(
            actual, cp.maximum(conv, 0), rtol=1e-4, atol=1e-4
        )

        residual = cp.linspace(0.2, -0.2, conv.size, dtype=cp.float32).reshape(conv.shape)
        residual_attrs = dict(attrs)
        residual_attrs["_residual_input_index"] = 3
        # ``op_conv`` may return a non-contiguous NCHW transpose view.  The
        # real C3.4 arena contract deliberately supplies contiguous outputs.
        planned = cp.empty(tuple(conv.shape), dtype=cp.float32)
        residual_attrs["_planned_output"] = planned
        actual_residual = op_fused_conv_residual_activation(
            [value, weight, bias, residual], residual_attrs
        )
        self.assertEqual(actual_residual.data.ptr, planned.data.ptr)
        cp.testing.assert_allclose(
            actual_residual, cp.maximum(conv + residual, 0),
            rtol=1e-4, atol=1e-4,
        )

    def test_layer_norm_kernel_matches_reference(self) -> None:
        value = cp.linspace(-2.0, 2.0, 3 * 5 * 16, dtype=cp.float32).reshape(3, 5, 16)
        scale = cp.linspace(0.8, 1.2, 16, dtype=cp.float32)
        bias = cp.linspace(-0.1, 0.1, 16, dtype=cp.float32)
        attrs = {"axis": -1, "epsilon": 1e-5}
        actual = op_fused_layer_normalization([value, scale, bias], attrs)
        expected = op_layer_normalization([value, scale, bias], attrs)
        cp.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_attention_score_kernel_matches_reference(self) -> None:
        query = cp.linspace(-0.5, 0.5, 2 * 3 * 5 * 7, dtype=cp.float32).reshape(2, 3, 5, 7)
        key = cp.linspace(0.4, -0.4, 2 * 3 * 7 * 5, dtype=cp.float32).reshape(2, 3, 7, 5)
        divisor = cp.asarray(7.0 ** 0.5, dtype=cp.float32)
        mask = cp.linspace(-0.2, 0.2, 5 * 5, dtype=cp.float32).reshape(1, 1, 5, 5)
        planned = cp.empty((2, 3, 5, 5), dtype=cp.float32)
        actual = op_fused_attention_scores(
            [query, key, divisor, mask], {"axis": -1, "_planned_output": planned}
        )
        expected = op_softmax(
            [cp.matmul(query, key) / divisor + mask], {"axis": -1}
        )
        self.assertEqual(actual.data.ptr, planned.data.ptr)
        cp.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_transpose_reshape_kernel_matches_reference(self) -> None:
        value = cp.arange(2 * 3 * 4 * 5, dtype=cp.float32).reshape(2, 3, 4, 5)
        shape = cp.asarray([2, 4, 15], dtype=cp.int64)
        planned = cp.empty((2, 4, 15), dtype=cp.float32)
        actual = op_fused_transpose_reshape(
            [value, shape], {
                "perm": [0, 2, 1, 3], "allowzero": 0,
                "_planned_output": planned,
            }
        )
        expected = value.transpose(0, 2, 1, 3).reshape(2, 4, 15)
        self.assertEqual(actual.data.ptr, planned.data.ptr)
        cp.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
