# C3.1 — Computation graph parsing and representation

## Release contract

Load an arbitrary supplied ONNX file and export a valid DAG JSON. The 10 points split into model loading (4) and correct graph parsing (6). The command must accept `--onnx` and `--output`, write only to the requested output path, and exit with code 0 on success.

The parser is model-general: it does not special-case public filenames, graph
names, tensor hashes, fixed weights, or released model instances. DAG JSON is
generated at evaluation time from the supplied ONNX file. ONNX and protobuf
must be provided natively by the evaluation server.

## Released implementation

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

## Internal representation

Build the rich internal IR first, then serialize a projection of it. Maintain these indexes:

```text
tensor_producer[tensor_name] -> node_id or INPUT/INITIALIZER
tensor_consumers[tensor_name] -> [node_id, ...]
node_by_id[node_id] -> Node
```

This makes C3.3 validation and C3.4 lifetime analysis direct extensions of C3.1 rather than separate re-parsing implementations.

## Validation

- Run the CLI on all three public ONNX files.
- Parse the produced JSON with a strict JSON parser.
- Assert graph inputs exclude initializers.
- Assert all 17 published operator types can be represented.
- Assert every edge tensor is in the source outputs and destination inputs.
- Assert every graph output resolves to an input, initializer, constant, or node output.
- Assert topological sorting visits every node exactly once.
- Repeat an export and compare bytes or normalized JSON for determinism.

## Contract violations

- Treating weights as user inputs.
- Building edges by adjacency instead of tensor producer/consumer relationships.
- Dropping fan-out edges or nodes with multiple outputs.
- Assuming concrete batch sizes.
- Printing JSON to stdout but not writing `--output`.
- Using ONNX display names as unique IDs without handling empty/duplicate names.
