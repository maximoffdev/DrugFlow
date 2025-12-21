"""Compatibility shim.

This module was renamed to `src.model.graph_builders` because it now contains
multiple graph construction utilities.

Prefer importing from `src.model.graph_builders` directly.
"""

from src.model.graph_builders import (  # noqa: F401
    build_batched_biknn_edges,
    build_batched_fully_connected_edges,
    build_biknn_radius_graph_no_pbc,
    safe_norm,
)
