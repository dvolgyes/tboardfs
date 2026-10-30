"""Keras configs that Keras does not write in the golden fixtures convert as in TensorBoard.

The expected node inputs are the output of TensorBoard 2.21
keras_model_to_graph_def for the same configs.
"""

from typing import Any

import pytest

from tboardfs.keras_graph import keras_model_to_graph_def


def _history(*histories: list[object]) -> list[dict[str, Any]]:
    """Return the inbound_nodes of one Keras 3 call with these keras_history values."""
    tensors = [{"config": {"keras_history": history}} for history in histories]
    return [{"args": [tensors] if len(tensors) > 1 else tensors, "kwargs": {}}]


def _layer(
    name: str, inbound: list[Any] | None = None, **fields: Any
) -> dict[str, Any]:
    layer = {"class_name": "Dense", "config": {"name": name}, **fields}
    if inbound is not None:
        layer["inbound_nodes"] = inbound
    return layer


def _functional(
    name: str, layers: list[Any], outputs: list[Any], **fields: Any
) -> dict[str, Any]:
    config = {
        "name": name,
        "layers": layers,
        "input_layers": [["x", 0, 0]],
        "output_layers": outputs,
    }
    return {"class_name": "Functional", "config": config, **fields}


def _inputs(model: dict[str, Any]) -> list[tuple[str, list[str]]]:
    return [
        (node.name, list(node.input)) for node in keras_model_to_graph_def(model).node
    ]


def test_nested_sequential_takes_its_inbound_layer() -> None:
    """A called Sequential model starts from its inbound layer, not the previous layer."""
    inner = {
        "class_name": "Sequential",
        "config": {"name": "inner", "layers": [_layer("d1")]},
    }
    model = _functional(
        "m",
        [
            _layer("x", []),
            _layer("a", _history(["x", 0, 0])),
            {**inner, "inbound_nodes": _history(["x", 0, 0])},
            _layer("out", _history(["a", 0, 0], ["inner", 0, 0])),
        ],
        [["out", 0, 0]],
    )

    assert _inputs(model) == [
        ("m/x", []),
        ("m/a", ["m/x"]),
        ("m/inner/d1", ["m/x"]),
        ("m/out", ["m/a", "m/inner/d1"]),
    ]


def test_an_index_past_the_outputs_takes_the_last_output() -> None:
    """An inbound index not below the number of outputs is the last output."""
    two = _functional(
        "two",
        [
            _layer("i", []),
            _layer("p", _history(["i", 0, 0])),
            _layer("q", _history(["i", 0, 0])),
        ],
        [["p", 0, 0], ["q", 0, 1]],
        inbound_nodes=_history(["x", 0, 0]),
    )
    model = _functional(
        "m",
        [
            _layer("x", []),
            two,
            _layer("at_len", _history(["two", 0, 2])),
            _layer("first", _history(["two", 0, 0])),
            _layer("single", _history(["x", 0, 1])),
        ],
        [["single", 0, 0]],
    )

    assert _inputs(model)[4:] == [
        ("m/at_len", ["m/two/q"]),
        ("m/first", ["m/two/p"]),
        ("m/single", ["m/x"]),
    ]


def test_short_histories_and_keras_2_nodes_give_no_inputs() -> None:
    """A keras_history with fewer than 3 fields and a Keras 2 list node are skipped."""
    model = _functional(
        "m",
        [
            _layer("x", []),
            _layer("short", _history(["x", 0])),
            _layer("keras2", [[["x", 0, 0, {}]]]),
        ],
        [["keras2", 0, 0]],
    )

    assert _inputs(model) == [("m/x", []), ("m/short", []), ("m/keras2", [])]


def test_a_nested_model_without_layers_fails() -> None:
    """TensorBoard fails on a nested model whose layers list is empty."""
    empty = {"class_name": "Sequential", "config": {"name": "empty", "layers": []}}
    model = _functional(
        "m",
        [_layer("x", []), {**empty, "inbound_nodes": _history(["x", 0, 0])}],
        [["x", 0, 0]],
    )

    with pytest.raises(ValueError, match="no layers"):
        keras_model_to_graph_def(model)


def test_a_functional_model_in_a_sequential_model_takes_the_previous_layer() -> None:
    """The flat Keras 3 input of a nested model is one input, unlike in TensorBoard."""
    inner = _functional(
        "f", [_layer("fi", []), _layer("fd", _history(["fi", 0, 0]))], [["fd", 0, 0]]
    )
    inner["config"]["input_layers"] = ["fi", 0, 0]
    model = {
        "class_name": "Sequential",
        "config": {"name": "s", "layers": [_layer("a"), inner]},
    }

    assert _inputs(model) == [("s/a", []), ("s/f/fi", ["s/a"]), ("s/f/fd", ["s/f/fi"])]


def test_a_nested_model_with_several_inputs_in_a_sequential_model_fails() -> None:
    """Three Keras 3 inputs are not one flat input."""
    inner = _functional("f", [_layer("fi", [])], [["fi", 0, 0]])
    inner["config"]["input_layers"] = [["a", 0, 0], ["b", 0, 0], ["c", 0, 0]]
    model = {
        "class_name": "Sequential",
        "config": {"name": "s", "layers": [_layer("a"), inner]},
    }

    with pytest.raises(ValueError, match="several inputs"):
        keras_model_to_graph_def(model)
