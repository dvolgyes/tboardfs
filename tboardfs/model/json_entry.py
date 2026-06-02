from dataclasses import dataclass
from typing import Any


@dataclass
class JsonEntry:
    """Index entry for plugin metadata rendered as JSON.

    :ivar tag: TensorBoard summary tag
    :ivar tab: virtual tab name
    :ivar step: TensorBoard step
    :ivar wall_time: event wall time
    :ivar payload: JSON-safe payload
    """

    tag: str
    tab: str
    step: int | float
    wall_time: float
    payload: dict[str, Any]
