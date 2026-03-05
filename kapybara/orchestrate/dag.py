"""Field-dependency DAG for TPS scheduling.

Represents the topological ordering of (T, field_value) simulation jobs.
The DAG knows only graph structure; it does not interact with SLURM or
StateDB. The Scheduler queries DAG topology and combines it with StateDB
state to make scheduling decisions.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from kapybara.config.schema import SimulationConfig
from kapybara.utils.convert import str2npy


@dataclass
class DAGNode:
    """A node in the field-dependency graph representing one (T, field_value) job.

    Attributes:
        T: Temperature string.
        field_value: Field value string (g or s depending on run type).
        parent: Parent node whose job must run (or reach n_branch) before
            this node's job can start. ``None`` for root nodes (field = 0).
        children: List of child nodes that depend on this node.
    """
    T: str
    field_value: str
    parent: Optional['DAGNode'] = None
    children: list['DAGNode'] = field(default_factory=list)


class DependencyDAG:
    """Pure field-dependency topology for TPS scheduling.

    Knows only graph structure (which field values depend on which parents).
    Does NOT interact with StateDB, SLURM, or branching logic — those
    concerns belong to the Scheduler.

    Attributes:
        _nodes: Nested dict ``{T: {field_value: DAGNode}}`` containing all
            nodes in the graph.
    """

    def __init__(self, config: SimulationConfig):
        """Build the DAG from a SimulationConfig.

        Args:
            config: Frozen :class:`~kapybara.config.schema.SimulationConfig`
                providing ``runtype``, temperature and field value lists, and
                decimal places.
        """
        self._nodes: dict[str, dict[str, DAGNode]] = {}
        self._build(config)

    def _build(self, config: SimulationConfig) -> None:
        if config.runtype == "g":
            self._build_field_chain(config, config.g, config.n_decimals[2])
        elif config.runtype == "s":
            self._build_field_chain(config, config.s, config.n_decimals[1])
        elif config.runtype == "sg":
            self._build_sg(config)

    def _build_field_chain(self, config: SimulationConfig,
                           field_values: list[str], n_decimals: int) -> None:
        """Build dependency chain for a single scan axis (g or s).

        Algorithm (from __KAPyBARA _init_tps / original create_dependency):
        1. Convert string field values to float, sort by |value|
        2. value=0 is root (depends only on prerun)
        3. For each value: parent is the previous value with the same sign
        4. If no same-sign predecessor exists: parent is 0
        """
        field_npy = str2npy(field_values)
        sorted_vals = field_npy[np.argsort(np.abs(field_npy))]

        zero = f"{0.0:.{n_decimals}f}"

        # Build dependency dict: {field_str: parent_field_str or None}
        dependency: dict[str, str | None] = {zero: None}

        for i in range(1, len(sorted_vals)):
            current = np.round(sorted_vals[i], n_decimals)
            current_str = f"{current:.{n_decimals}f}"
            parent_str = zero  # default fallback

            for j in range(i - 1, -1, -1):
                previous = sorted_vals[j]
                if np.sign(previous) == np.sign(current):
                    parent_str = f"{np.round(previous, n_decimals):.{n_decimals}f}"
                    break

            dependency[current_str] = parent_str

        # Build nodes for each (T, field_value) pair
        for T in config.T:
            self._nodes[T] = {}
            for fv_str in dependency:
                self._nodes[T][fv_str] = DAGNode(T=T, field_value=fv_str)

            # Link parent/children
            for fv_str, parent_str in dependency.items():
                node = self._nodes[T][fv_str]
                if parent_str is not None:
                    parent_node = self._nodes[T][parent_str]
                    node.parent = parent_node
                    parent_node.children.append(node)

    def _build_sg(self, config: SimulationConfig) -> None:
        raise NotImplementedError(
            "Doubly-biased sampling (sg) dependency DAG is not implemented yet."
        )

    # ── Topology queries (pure reads) ──

    def get_node(self, T: str, field_value: str) -> DAGNode | None:
        """Look up a node by (T, field_value).

        Args:
            T: Temperature string.
            field_value: Field value string.

        Returns:
            The :class:`DAGNode`, or ``None`` if not found.
        """
        return self._nodes.get(T, {}).get(field_value)

    def get_roots(self) -> list[DAGNode]:
        """Return all root nodes (nodes with no parent, i.e. field = 0).

        Returns:
            List of :class:`DAGNode` objects with ``parent=None``.
        """
        return [
            node
            for t_nodes in self._nodes.values()
            for node in t_nodes.values()
            if node.parent is None
        ]

    def get_parent(self, T: str, field_value: str) -> DAGNode | None:
        """Return the parent node of (T, field_value).

        Args:
            T: Temperature string.
            field_value: Field value string.

        Returns:
            Parent :class:`DAGNode`, or ``None`` for root nodes.
        """
        node = self.get_node(T, field_value)
        return node.parent if node else None

    def get_children(self, T: str, field_value: str) -> list[DAGNode]:
        """Return the child nodes of (T, field_value).

        Args:
            T: Temperature string.
            field_value: Field value string.

        Returns:
            List of child :class:`DAGNode` objects (empty if none).
        """
        node = self.get_node(T, field_value)
        return node.children if node else []

    def all_nodes(self) -> list[DAGNode]:
        """Return all nodes in the DAG as a flat list.

        Returns:
            List of all :class:`DAGNode` objects across all temperatures and
            field values.
        """
        return [
            node
            for t_nodes in self._nodes.values()
            for node in t_nodes.values()
        ]

    def get_dependency_map(self) -> dict[str, str | None]:
        """Return the field-value dependency map for a single temperature.

        Extracts the ``{field_value: parent_field_value}`` mapping from the
        first temperature in the DAG. Because the dependency structure is
        identical for all temperatures, one temperature is sufficient.

        Returns:
            Dict mapping each field value string to its parent field value
            string, or ``None`` for the root node.
        """
        if not self._nodes:
            return {}
        first_T = next(iter(self._nodes))
        return {
            fv: (node.parent.field_value if node.parent else None)
            for fv, node in self._nodes[first_T].items()
        }
