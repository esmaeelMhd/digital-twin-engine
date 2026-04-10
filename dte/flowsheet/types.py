"""Core type definitions for flowsheet graph modeling."""

from __future__ import annotations

EXTERNAL_SOURCE = "__source__"
EXTERNAL_SINK = "__sink__"

STREAM_KINDS = (
    "process",
    "utility",
    "recycle",
    "source",
    "sink",
)


def is_external_node(node_name: str) -> bool:
    """Return whether a node name refers to an external source/sink."""

    return node_name in {EXTERNAL_SOURCE, EXTERNAL_SINK}
