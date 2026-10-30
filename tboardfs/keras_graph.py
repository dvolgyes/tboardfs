"""Convert a Keras model config to the conceptual graph that TensorBoard draws."""

from collections.abc import Iterator
from types import MappingProxyType
from typing import Any, cast

from google.protobuf.message import Message

from tboardfs.proto import graph_pb2

_BASE_DTYPES = MappingProxyType(
    {
        "float32": 1,
        "float64": 2,
        "int32": 3,
        "uint8": 4,
        "int16": 5,
        "int8": 6,
        "string": 7,
        "complex64": 8,
        "int64": 9,
        "bool": 10,
        "qint8": 11,
        "quint8": 12,
        "qint32": 13,
        "bfloat16": 14,
        "qint16": 15,
        "quint16": 16,
        "uint16": 17,
        "complex128": 18,
        "float16": 19,
        "resource": 20,
        "variant": 21,
        "uint32": 22,
        "uint64": 23,
        "half": 19,
        "float": 1,
        "double": 2,
    }
)
# TensorBoard's dtype names: each name and its _ref form, which is 100 higher.
_DTYPE_NAMES = MappingProxyType(
    dict(_BASE_DTYPES)
    | {f"{name}_ref": enum + 100 for name, enum in _BASE_DTYPES.items()}
)
_DTYPE_ENUMS = frozenset(_BASE_DTYPES.values()) | {
    enum + 100 for enum in _BASE_DTYPES.values()
}
HISTORY_FIELDS = 3


def keras_model_to_graph_def(model: object) -> Message:
    """Return the GraphDef that TensorBoard makes from a Keras model.to_json().

    This ports keras_model_to_graph_def of TensorBoard 2.21. Each layer is a
    node named <model scope>/<layer name>, with its Keras class and dtype as
    attributes and its inbound layers as inputs. JSON values are read with
    Python semantics, as TensorBoard reads them: a string or an object where a
    list should be is iterated, and a name that is not text is formatted with
    str() inside a scope.

    Raises:
        ValueError: TensorBoard fails on the config.
    """
    graph = cast(Any, getattr(graph_pb2, "GraphDef")())
    state = _Links()
    for scope, layer in _KerasConfig.walk(model):
        config = _KerasConfig.config(layer)
        if config.get("layers") is not None:
            state.add_model(scope, layer)
            continue
        node = graph.node.add(
            name=_KerasConfig.text(_scoped(scope, config.get("name")))
        )
        class_name = layer.get("class_name")
        if class_name is not None:
            node.attr["keras_class"].s = _KerasConfig.text(class_name).encode("ascii")
        dtype = _KerasConfig.dtype_enum(config.get("dtype"))
        if dtype is not None:
            node.attr["dtype"].type = dtype
        if layer.get("inbound_nodes") is not None:
            node.input.extend(state.inbound_inputs(scope, layer))
        elif state.previous is not None:
            node.input.append(_KerasConfig.text(state.previous))
        if node.name in state.input_to_layer:
            node.input.append(_KerasConfig.text(state.input_to_layer[node.name]))
        state.previous = node.name
    return cast(Message, graph)


def _scoped(scope: object, name: object) -> object:
    """Return <scope>/<name>, or the name alone for an empty scope."""
    return f"{scope}/{name}" if scope else name


class _Links:
    """Track the layer links that TensorBoard follows while it walks a model.

    Attributes:
        input_to_layer: scoped input layer of a nested model, to the layer
            that feeds it in the parent model
        model_outputs: scoped model name, to its scoped output layers
        previous: the previous layer in walk order, for Sequential models
    """

    input_to_layer: dict[object, object]
    model_outputs: dict[object, list[object]]
    previous: object

    def __init__(self) -> None:
        """Start with no links."""
        self.input_to_layer = {}
        self.model_outputs = {}
        self.previous = None

    def add_model(self, scope: object, model: dict[str, Any]) -> None:
        """Record the input and output links of a (nested) model."""
        config = _KerasConfig.config(model)
        layers = config.get("layers")
        if not layers:
            raise ValueError("Keras model config has no layers")
        name = _scoped(scope, config.get("name"))
        inputs = config.get("input_layers")
        outputs = config.get("output_layers")
        inbound = _KerasConfig.inbound_nodes(model)
        functional = bool(inputs and outputs)
        if inbound and functional:
            for input_layer, node in zip(
                _KerasConfig.items(inputs), inbound, strict=False
            ):
                key = _KerasConfig.key(_scoped(name, input_layer))
                self.input_to_layer[key] = _scoped(scope, node[0])
        elif inbound:
            self.previous = _scoped(scope, inbound[0][0])
        elif self.previous and functional:
            # Keras 3 writes one input as a flat [name, node, index] list. Unlike
            # TensorBoard 2.21, which fails on it, read it as one input.
            if _KerasConfig.size(inputs) != 1 and not _KerasConfig.is_history(inputs):
                raise ValueError("Keras model in a Sequential model has several inputs")
            key = _KerasConfig.key(_scoped(name, _KerasConfig.element(inputs, 0)))
            self.input_to_layer[key] = self.previous
        name = _KerasConfig.key(name)
        if functional:
            first = _KerasConfig.element(outputs, 0)
            output_layers = (
                _KerasConfig.items(outputs) if isinstance(first, list) else [outputs]
            )
            self.model_outputs[name] = [
                _scoped(name, _KerasConfig.element(output, 0))
                for output in output_layers
            ]
        else:
            last = _KerasConfig.dict_value(_KerasConfig.element(layers, -1))
            self.model_outputs[name] = [
                _scoped(name, _KerasConfig.config(last).get("name"))
            ]

    def inbound_inputs(self, scope: object, layer: dict[str, Any]) -> Iterator[str]:
        """Yield the input names of a layer, mapping a model to its outputs.

        An index that is not below the number of outputs, NaN included, takes
        the last output, as in TensorBoard.
        """
        for name, _, index in _KerasConfig.inbound_nodes(layer):
            inbound = _scoped(scope, name)
            names = self.model_outputs.get(_KerasConfig.key(inbound), [inbound])
            if not isinstance(index, (int, float)):
                raise ValueError(f"Keras inbound node index {index!r} is not a number")
            last = not index < len(names)
            yield _KerasConfig.text(
                names[-1] if last else _KerasConfig.element(names, index)
            )


class _KerasConfig:
    """Read the parts of a Keras config that TensorBoard reads."""

    @staticmethod
    def walk(layer: object) -> Iterator[tuple[object, dict[str, Any]]]:
        """Yield (scope, layer) for a layer and its nested layers in preorder."""
        layer = _KerasConfig.dict_value(layer)
        yield "", layer
        config = _KerasConfig.config(layer)
        if not config.get("layers"):
            return
        name = config.get("name")
        for sublayer in _KerasConfig.items(config["layers"]):
            for sub_scope, nested in _KerasConfig.walk(sublayer):
                yield (f"{name}/{sub_scope}" if sub_scope else name), nested

    @staticmethod
    def inbound_nodes(layer: dict[str, Any]) -> list[Any]:
        """Return [name, size, index] per inbound tensor of a Keras 3 layer.

        Inbound nodes that are not dicts, as in Keras 2 configs, are skipped.
        """
        nodes = layer.get("inbound_nodes")
        if nodes is None:
            return []
        histories = []
        for node in _KerasConfig.items(nodes):
            if not isinstance(node, dict):
                continue
            for args in _KerasConfig.items(node.get("args", [])):
                for arg in args if isinstance(args, list) else [args]:
                    config = _KerasConfig.dict_value(
                        _KerasConfig.dict_value(arg).get("config", {})
                    )
                    history = config.get("keras_history", [])
                    if _KerasConfig.size(history) >= HISTORY_FIELDS:
                        if not isinstance(history, (list, str)):
                            raise ValueError("Keras keras_history is an object")
                        histories.append(history[:HISTORY_FIELDS])
        return histories

    @staticmethod
    def dtype_enum(dtype_or_policy: object) -> int | None:
        """Return the DataType enum of a layer dtype or dtype policy.

        A value that TensorBoard cannot convert gives no dtype.
        """
        dtype = dtype_or_policy
        if isinstance(dtype_or_policy, dict) and "config" in dtype_or_policy:
            dtype = _KerasConfig.dict_value(dtype_or_policy["config"]).get("name")
        # JSON numbers match the enum table as Python dict keys do: 1.0 and true
        # are 1.
        if isinstance(dtype, (int, float)) and dtype in _DTYPE_ENUMS:
            return int(dtype)
        if isinstance(dtype, str):
            return _DTYPE_NAMES.get(dtype)
        return None

    @staticmethod
    def config(layer: dict[str, Any]) -> dict[str, Any]:
        """Return the config dict of a layer."""
        return _KerasConfig.dict_value(layer.get("config"))

    @staticmethod
    def dict_value(value: object) -> dict[str, Any]:
        """Return a JSON object, or raise ValueError for another value."""
        if not isinstance(value, dict):
            raise ValueError(f"Keras config has {type(value).__name__} for an object")
        return value

    @staticmethod
    def items(value: object) -> list[Any]:
        """Return the items of a JSON value as Python iterates it.

        A string gives its characters and an object its keys.
        """
        if not isinstance(value, (list, str, dict)):
            raise ValueError(f"Keras config has {type(value).__name__} for a list")
        return list(value)

    @staticmethod
    def size(value: object) -> int:
        """Return len() of a JSON array, string or object."""
        if not isinstance(value, (list, str, dict)):
            raise ValueError(f"Keras config has {type(value).__name__} for a list")
        return len(value)

    @staticmethod
    def is_history(value: object) -> bool:
        """Return true for one flat [name, node, index] Keras 3 tensor reference."""
        return (
            isinstance(value, list)
            and len(value) == HISTORY_FIELDS
            and isinstance(value[0], str)
        )

    @staticmethod
    def element(value: object, index: object) -> object:
        """Return value[index] of a JSON array or string, as Python indexes it."""
        if not isinstance(value, (list, str)) or not isinstance(index, int):
            raise ValueError("Keras config cannot be indexed")
        if not -len(value) <= index < len(value):
            raise ValueError(f"Keras config index {index} is out of range")
        return value[index]

    @staticmethod
    def text(value: object) -> str:
        """Return a node name or class name, or raise ValueError for no text."""
        if not isinstance(value, str):
            raise ValueError(f"Keras config has {type(value).__name__} for a name")
        return value

    @staticmethod
    def key(value: object) -> object:
        """Return a hashable dict key, or raise ValueError."""
        if isinstance(value, (list, dict)):
            raise ValueError("Keras config has a list or object for a name")
        return value
