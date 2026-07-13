# C3.1 — Computation graph parsing and representation

## Objective and scoring

Load an arbitrary supplied ONNX file and export a valid DAG JSON. The 10 points split into model loading (4) and correct graph parsing (6). The command must accept `--onnx` and `--output`, write only to the requested output path, and exit with code 0 on success.

The parser must be model-general: never special-case public filenames, graph names, tensor hashes, fixed weights, or the three released model instances. Generate DAG JSON at evaluation time from the supplied ONNX file. Declare the ONNX/protobuf dependencies and package them for offline installation.

## Required work

1. Load and validate the ONNX protobuf, including opset information.
2. Separate true graph inputs from initializers; weights must not appear in `graph_inputs`.
3. Record graph outputs, node names/types, ordered tensor inputs/outputs, attributes, dtypes, and symbolic shapes.
4. Build producer and consumer maps from tensor names.
5. Create an edge for each producer-to-consumer dependency. If one tensor has multiple consumers, emit one edge per consumer.
6. Topologically sort and reject cycles, duplicate producers, and dangling references.
7. Export stable JSON containing at least `format_version`, `graph_inputs`, `graph_outputs`, `nodes`, and `edges`.

## Important edge cases

- ONNX node names may be empty or duplicated. Generate deterministic internal IDs, while retaining the original name separately.
- Optional inputs can be empty strings; do not turn them into tensors or edges.
- `Constant` produces a graph tensor but is not an initializer.
- Initializers may also be listed in `graph.input` in older exports; exclude them from public runtime inputs.
- Shapes can include symbolic dimensions such as `batch`; do not coerce them to zero.
- `Reshape`, `Split`, `Transpose`, and broadcasting operators require attributes and/or constant inputs later, so do not discard these during a C3.1-only export.
- A graph output can be produced directly by a node and need not have a consumer edge.

## Design recommendation

Build the rich internal IR first, then serialize a projection of it. Maintain these indexes:

```text
tensor_producer[tensor_name] -> node_id or INPUT/INITIALIZER
tensor_consumers[tensor_name] -> [node_id, ...]
node_by_id[node_id] -> Node
```

This makes C3.3 validation and C3.4 lifetime analysis direct extensions of C3.1 rather than separate re-parsing implementations.

## Acceptance tests

- Run the CLI on all three public ONNX files.
- Parse the produced JSON with a strict JSON parser.
- Assert graph inputs exclude initializers.
- Assert all 17 published operator types can be represented.
- Assert every edge tensor is in the source outputs and destination inputs.
- Assert every graph output resolves to an input, initializer, constant, or node output.
- Assert topological sorting visits every node exactly once.
- Repeat an export and compare bytes or normalized JSON for determinism.

## Failure modes that lose points

- Treating weights as user inputs.
- Building edges by adjacency instead of tensor producer/consumer relationships.
- Dropping fan-out edges or nodes with multiple outputs.
- Assuming concrete batch sizes.
- Printing JSON to stdout but not writing `--output`.
- Using ONNX display names as unique IDs without handling empty/duplicate names.

---

## 中文赛题要求与实现材料

### 评分构成

| 评分项 | 分值 |
| --- | ---: |
| 模型加载 | 4 分 |
| 正确的计算图解析 | 6 分 |
| **合计** | **10 分** |

### 命令行接口

```text
<选手程序> --onnx <model.onnx> --output <dag.json>
```

- `--onnx`：输入 ONNX 模型文件路径；
- `--output`：输出 DAG JSON 文件路径；
- 成功时必须以退出码 `0` 结束；
- 非零退出码视为模型处理失败；
- `stdout` 内容不参与评测，评测器只读取 `--output` 指定的文件。

报名命令模板示例：

```text
python export_dag.py --onnx {onnx} --output {output}
```

### 标准输出示例

```json
{
  "format_version": "1.0",
  "graph_inputs": [
    {
      "name": "input",
      "dtype": "FLOAT",
      "shape": ["batch", 1, 28, 28]
    }
  ],
  "graph_outputs": [
    {
      "name": "logits",
      "dtype": "FLOAT",
      "shape": ["batch", 10]
    }
  ],
  "nodes": [
    {
      "name": "/fc1/Gemm",
      "op_type": "Gemm",
      "inputs": [
        "/flatten/Flatten_output_0",
        "fc1.weight",
        "fc1.bias"
      ],
      "outputs": ["/fc1/Gemm_output_0"]
    }
  ],
  "edges": [
    {
      "src_node": "/flatten/Flatten",
      "dst_node": "/fc1/Gemm",
      "tensor": "/flatten/Flatten_output_0"
    }
  ]
}
```

### 字段定义

| 字段 | 说明 |
| --- | --- |
| `format_version` | 导出格式版本，建议设为字符串 `"1.0"` |
| `graph_inputs` | 模型输入张量，不包含权重等 initializer |
| `graph_outputs` | 模型输出张量 |
| `nodes` | 节点列表，记录节点名、算子类型、输入张量和输出张量 |
| `edges` | 节点间的数据依赖边，记录源节点、目标节点和流动的张量 |

张量的形状建议按以下规则导出：

- 静态维度：JSON 整数；
- 符号维度：字符串，例如 `"batch"`；
- 未知维度：`null`；
- 不要把未知或符号维度擅自改成 `0` 或 `1`。

### 最小 Python 实现骨架

下面的代码展示核心解析方法，可在此基础上补充严格校验、属性导出和更完整的错误信息。

```python
import argparse
import json
import sys
from pathlib import Path

import onnx
from onnx import TensorProto


def dtype_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(elem_type)
    except ValueError:
        return "UNDEFINED"


def parse_shape(tensor_type) -> list:
    result = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param") and dim.dim_param:
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def value_info_to_json(value_info) -> dict:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "dtype": dtype_name(tensor_type.elem_type),
        "shape": parse_shape(tensor_type),
    }


def export_dag(onnx_path: str, output_path: str) -> None:
    model = onnx.load(onnx_path, load_external_data=True)
    onnx.checker.check_model(model)
    graph = model.graph

    initializer_names = {item.name for item in graph.initializer}
    graph_inputs = [
        value_info_to_json(item)
        for item in graph.input
        if item.name not in initializer_names
    ]
    graph_outputs = [value_info_to_json(item) for item in graph.output]

    nodes = []
    producer_by_tensor = {}
    node_names = []

    for index, node in enumerate(graph.node):
        # ONNX 允许节点名为空；生成名称必须稳定且唯一。
        node_name = node.name or f"{node.op_type}_{index}"
        if node_name in node_names:
            node_name = f"{node_name}_{index}"
        node_names.append(node_name)

        inputs = [name for name in node.input if name]
        outputs = [name for name in node.output if name]
        nodes.append({
            "name": node_name,
            "op_type": node.op_type,
            "inputs": inputs,
            "outputs": outputs,
        })

        for tensor_name in outputs:
            if tensor_name in producer_by_tensor:
                raise ValueError(f"duplicate producer: {tensor_name}")
            producer_by_tensor[tensor_name] = node_name

    edges = []
    for node in nodes:
        for tensor_name in node["inputs"]:
            src_node = producer_by_tensor.get(tensor_name)
            if src_node is not None:
                edges.append({
                    "src_node": src_node,
                    "dst_node": node["name"],
                    "tensor": tensor_name,
                })

    result = {
        "format_version": "1.0",
        "graph_inputs": graph_inputs,
        "graph_outputs": graph_outputs,
        "nodes": nodes,
        "edges": edges,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        export_dag(args.onnx, args.output)
        return 0
    except Exception as exc:
        print(f"failed to export DAG: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### 边的构造原则

先建立生产者映射：

```text
producer_by_tensor[输出张量名] = 生产节点名
```

再遍历每个节点的输入张量：如果某输入存在于生产者映射中，就创建一条生产节点到消费节点的边。

以下输入通常不生成节点间边：

- 模型输入；
- initializer/权重；
- ONNX 可选参数对应的空字符串。

如果同一张量被多个节点消费，必须为每个消费者分别生成一条边。

### 重点边界情况

- initializer 可能同时出现在 `graph.input`，必须从 `graph_inputs` 排除；
- 节点名可能为空或重复，需要生成稳定、唯一的内部名称；
- 多输入、多输出节点必须完整保留所有非空张量名；
- `Constant` 是产生张量的节点，不应当被当成 initializer；
- 图输出可能没有消费者，但仍须保留；
- `If`、`Loop`、`Scan` 的属性可能包含子图；若只导出顶层图，应明确记录实现范围；
- 大模型可能使用外部权重数据，加载时需保留正确的相对路径环境；
- 输出顺序必须稳定，确保同一模型重复运行得到一致 JSON。

### 推荐自测集合

1. 单节点模型；
2. 线性链式模型；
3. 一个张量被多个节点消费的分支模型；
4. 多输入、多输出模型；
5. 含 initializer 的 Gemm 或 Conv 模型；
6. 含符号维度和未知维度的模型；
7. 节点名为空或重复的模型；
8. 含可选空输入的算子模型；
9. 包含 `If` 或 `Loop` 的模型；
10. 非法、截断或不存在的 ONNX 文件。

### 最终验收清单

- [ ] 命令行参数和报名命令模板完全匹配。
- [ ] 成功返回 `0`，失败返回非零退出码。
- [ ] 结果写入 `--output`，不依赖 stdout。
- [ ] 输出可被严格 JSON 解析器读取。
- [ ] `graph_inputs` 已排除 initializer。
- [ ] 图输入、图输出的类型和形状正确。
- [ ] 节点名称稳定且唯一。
- [ ] 节点输入和输出保持 ONNX 原始顺序。
- [ ] 所有节点间张量依赖均生成正确边。
- [ ] fan-out、多输出和空可选输入处理正确。
- [ ] 同一输入重复执行产生确定性输出。
