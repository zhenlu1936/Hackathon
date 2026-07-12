"""Specification-focused tests for C3.1 graph parsing and DAG export."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from c31.import_onnx import import_onnx
from c3common.ir.graph import Graph, Node


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "specification/testcases/release_to_competitors/models"


class C31SpecificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="c31-test-")
        self.temp = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _save(
        self,
        name: str,
        graph: onnx.GraphProto,
        *,
        opsets: list[onnx.OperatorSetIdProto] | None = None,
        ir_version: int | None = None,
    ) -> Path:
        model = helper.make_model(
            graph,
            opset_imports=opsets or [helper.make_opsetid("", 17)],
        )
        if ir_version is not None:
            model.ir_version = ir_version
        path = self.temp / f"{name}.onnx"
        onnx.save(model, path)
        return path

    def test_public_models_through_required_cli(self) -> None:
        for model_path in sorted(MODELS.glob("*.onnx")):
            with self.subTest(model=model_path.name):
                first = self.temp / f"{model_path.stem}.json"
                second = self.temp / f"{model_path.stem}.repeat.json"
                for output in (first, second):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "export_dag.py"),
                            "--onnx",
                            str(model_path),
                            "--output",
                            str(output),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "")
                    json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(first.read_bytes(), second.read_bytes())

                graph = import_onnx(str(model_path))
                dag = graph.to_dag_json()
                self.assertEqual(len(graph.initializers), len([
                    tensor for tensor in graph.tensors.values()
                    if tensor.is_initializer
                ]))
                self.assertFalse({t.name for t in graph.inputs} & graph.initializers.keys())
                self.assertEqual(len(graph.node_order), len(graph.nodes))
                node_by_name = {node["name"]: node for node in dag["nodes"]}
                for edge in dag["edges"]:
                    self.assertIn(edge["tensor"], node_by_name[edge["src_node"]]["outputs"])
                    self.assertIn(edge["tensor"], node_by_name[edge["dst_node"]]["inputs"])

    def test_fanout_duplicate_names_and_unknown_shape(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", None, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", None, 4])
        nodes = [
            helper.make_node("Relu", ["x"], ["mid"]),
            helper.make_node("Identity", ["mid"], ["a"], name="dup"),
            helper.make_node("Identity", ["mid"], ["b"], name="dup"),
            helper.make_node("Add", ["a", "b"], ["y"], name="join"),
        ]
        graph = import_onnx(str(self._save(
            "fanout", helper.make_graph(nodes, "fanout", [x], [y])
        )))
        dag = graph.to_dag_json()
        self.assertEqual(dag["graph_inputs"][0]["shape"], ["batch", None, 4])
        self.assertEqual(len({node["name"] for node in dag["nodes"]}), 4)
        self.assertEqual(sum(edge["tensor"] == "mid" for edge in dag["edges"]), 2)

    def test_optional_inputs_retain_position_without_edges(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
        clip = helper.make_node("Clip", ["x", "", ""], ["y"], name="clip")
        path = self._save(
            "optional",
            helper.make_graph([clip], "optional", [x], [y]),
            opsets=[helper.make_opsetid("", 11)],
        )
        graph = import_onnx(str(path))
        self.assertEqual(graph.to_dag_json()["nodes"][0]["inputs"], ["x", "", ""])
        self.assertNotIn("", graph.tensors)
        self.assertEqual(graph.build_edges(), [])

    def test_initializer_is_not_a_public_input(self) -> None:
        weight_info = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1])
        weight = numpy_helper.from_array(np.array([1], dtype=np.float32), name="w")
        path = self._save(
            "initializer_input",
            helper.make_graph([], "initializer", [weight_info], [weight_info], [weight]),
            ir_version=3,
        )
        graph = import_onnx(str(path))
        self.assertEqual(graph.inputs, [])
        self.assertIn("w", graph.initializers)

    def test_direct_input_can_be_a_graph_output(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        graph = import_onnx(str(self._save(
            "direct_output", helper.make_graph([], "direct", [x], [x])
        )))
        self.assertEqual(graph.tensor_producer["x"], "INPUT")
        self.assertEqual(graph.nodes, {})

    def test_complex_attributes_are_json_serializable(self) -> None:
        sub_output = helper.make_tensor_value_info("sub_output", TensorProto.FLOAT, [1])
        sub_constant = helper.make_node(
            "Constant",
            [],
            ["sub_output"],
            value=numpy_helper.from_array(np.array([1], dtype=np.float32)),
        )
        body = helper.make_graph([sub_constant], "body", [], [sub_output])
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        custom = helper.make_node(
            "Custom",
            ["x"],
            ["y"],
            domain="test",
            floats=[1.0, 2.0],
            ints=[1, 2],
            strings=[b"text", b"\xff"],
            body=body,
        )
        path = self._save(
            "attributes",
            helper.make_graph([custom], "attributes", [x], [y]),
            opsets=[helper.make_opsetid("", 17), helper.make_opsetid("test", 1)],
        )
        dag = import_onnx(str(path)).to_dag_json()
        json.dumps(dag)
        attrs = dag["nodes"][0]["attributes"]
        self.assertEqual(attrs["floats"], [1.0, 2.0])
        self.assertEqual(attrs["ints"], [1, 2])
        self.assertEqual(attrs["strings"][1], {"base64": "/w=="})
        self.assertIsInstance(attrs["body"], dict)

    def test_malformed_onnx_is_rejected(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        bad = helper.make_node("Add", ["x", "missing"], ["y"])
        path = self.temp / "bad.onnx"
        onnx.save(
            helper.make_model(
                helper.make_graph([bad], "bad", [x], [y]),
                opset_imports=[helper.make_opsetid("", 17)],
            ),
            path,
        )
        with self.assertRaises(Exception):
            import_onnx(str(path))

        output = self.temp / "bad.json"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "export_dag.py"),
                "--onnx",
                str(path),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
