"""Regression tests for executable C3.2 kernel layout handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


def _load_registry_with_numpy():
    """Load the registry with NumPy standing in for CuPy for shape-only tests."""
    module_path = Path(__file__).with_name("kernel_registry.py")
    spec = importlib.util.spec_from_file_location(
        "_c32_kernel_registry_shape_test", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load kernel_registry.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"cupy": np}):
        spec.loader.exec_module(module)
    return module


class KernelRegistryTests(unittest.TestCase):
    def test_unsupported_kernel_fails_closed(self) -> None:
        registry = _load_registry_with_numpy()
        with self.assertRaisesRegex(RuntimeError, "Unknown kernel"):
            registry.lookup("sigmoid_f32")

    def test_nhwo_contract_is_transposed_to_nchw(self) -> None:
        registry = _load_registry_with_numpy()
        n, h, w, o = 2, 2, 3, 4
        logical_nhwo = np.arange(n * h * w * o, dtype=np.float32).reshape(
            n, h, w, o
        )
        contract = logical_nhwo.reshape(n * h * w, o)

        actual = registry._reshape_conv_output(
            contract,
            [],
            {"_batch_size": n, "_output_spatial": [h, w]},
        )

        np.testing.assert_array_equal(
            actual, logical_nhwo.transpose(0, 3, 1, 2)
        )

    def test_im2col_contract_reshape_and_bias_match_nchw_convolution(self) -> None:
        registry = _load_registry_with_numpy()
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2, 2, 4, 5), dtype=np.float32)
        weight = rng.standard_normal((3, 2, 3, 2), dtype=np.float32)
        bias = rng.standard_normal((3,), dtype=np.float32)
        params = {
            "kernel_shape": [3, 2],
            "strides": [1, 2],
            "dilations": [1, 1],
            "pads": [1, 0, 1, 1],
            "lowering_kind": "conv_contract",
            "_batch_size": 2,
            "_output_spatial": [4, 3],
        }

        col_outputs = [np.empty((2 * 4 * 3, 2 * 3 * 2), dtype=np.float32)]
        registry.kernel_im2col([x], col_outputs, params, None)
        contract_outputs = [np.empty((2 * 4 * 3, 3), dtype=np.float32)]
        registry.kernel_matmul(
            [col_outputs[0], weight], contract_outputs, params, None
        )
        reshape_outputs = [np.empty((2, 3, 4, 3), dtype=np.float32)]
        registry.kernel_conv_reshape(
            [contract_outputs[0]], reshape_outputs, params, None
        )
        actual_outputs = [np.empty((2, 3, 4, 3), dtype=np.float32)]
        registry.kernel_add_bias(
            [reshape_outputs[0], bias],
            actual_outputs,
            {"bias_axis": 1},
            None,
        )

        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 1)))
        expected = np.empty((2, 3, 4, 3), dtype=np.float32)
        for n in range(2):
            for out_channel in range(3):
                for out_h in range(4):
                    for out_w in range(3):
                        patch = padded[
                            n,
                            :,
                            out_h:out_h + 3,
                            out_w * 2:out_w * 2 + 2,
                        ]
                        expected[n, out_channel, out_h, out_w] = (
                            np.sum(patch * weight[out_channel])
                            + bias[out_channel]
                        )

        np.testing.assert_allclose(
            actual_outputs[0], expected, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
