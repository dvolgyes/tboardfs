"""Load puremagic signatures only when an encoded payload needs classification."""

from typing import Any, cast

import puremagic


def magic_matches(blob: bytes) -> list[Any] | None:
    """Return matching signatures or no result when puremagic finds none."""
    try:
        return cast(list[Any], puremagic.magic_string(blob))
    except puremagic.PureError:
        return None
