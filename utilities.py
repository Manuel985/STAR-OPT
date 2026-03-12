from __future__ import annotations 
import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

NodeId = str
EdgeId = str


@dataclass
class AttackGraph:
    nodes: Set[NodeId]
    edges: Dict[EdgeId, Tuple[NodeId, NodeId]]
    sources: Set[NodeId]
    targets: Set[NodeId]
    p_edge: Dict[EdgeId, float]
    I_base: Dict[NodeId, float]
    log_p_edge_base: Dict[EdgeId, float] | None = None
    log_I_base: Dict[NodeId, float] | None = None
    _in_degree: Dict[NodeId, int] | None = None
    _out_degree: Dict[NodeId, int] | None = None

    def __post_init__(self) -> None:

        # Basic consistency checks on the graph structure
        if not self.sources.issubset(self.nodes):
            raise ValueError("All sources must be contained in nodes")

        if not self.targets.issubset(self.nodes):
            raise ValueError("All targets must be contained in nodes")

        if self.sources & self.targets:
            raise ValueError("Sources and targets must be disjoint")

        in_deg: Dict[NodeId, int] = {n: 0 for n in self.nodes}
        out_deg: Dict[NodeId, int] = {n: 0 for n in self.nodes}

        for e_id, (u, v) in self.edges.items():

            if u not in self.nodes or v not in self.nodes:
                raise ValueError(
                    f"Edge {e_id} references unknown node(s) {u}, {v}"
                )

            out_deg[u] += 1
            in_deg[v] += 1

        # Sources must start attack paths (no incoming edges)
        for s in self.sources:

            if in_deg[s] != 0:
                raise ValueError(
                    f"Source node {s} must have in-degree 0"
                )

            if out_deg[s] == 0:
                raise ValueError(
                    f"Source node {s} must have out-degree > 0"
                )

        # Targets must terminate attack paths (no outgoing edges)
        for t in self.targets:

            if out_deg[t] != 0:
                raise ValueError(
                    f"Target node {t} must have out-degree 0"
                )

            if in_deg[t] == 0:
                raise ValueError(
                    f"Target node {t} must have in-degree > 0"
                )

        # Each edge must have a base compromise probability
        if set(self.p_edge.keys()) != set(self.edges.keys()):
            raise ValueError(
                "p_edge must be defined exactly for the edges of the graph"
            )

        for e_id in self.edges.keys():

            p = self.p_edge[e_id]

            if not (0.0 < p <= 1.0):
                raise ValueError(
                    f"p_edge[{e_id}] must be in (0,1], got {p}"
                )

        # Impact must be specified for each target
        if set(self.I_base.keys()) != self.targets:
            raise ValueError(
                "I_base must be defined exactly for the target nodes"
            )

        for f in self.targets:

            I = self.I_base[f]

            if I <= 0.0:
                raise ValueError(
                    f"I_base[{f}] must be > 0, got {I} for {f}"
                )

        # Build adjacency list to verify reachability
        adjacency: Dict[NodeId, List[NodeId]] = {n: [] for n in self.nodes}

        for _, (u, v) in self.edges.items():
            adjacency[u].append(v)

        # Check that at least one attack path exists from a source to a target
        reachable: Set[NodeId] = set(self.sources)
        stack: List[NodeId] = list(self.sources)

        while stack:
            u = stack.pop()

            for v in adjacency[u]:
                if v not in reachable:
                    reachable.add(v)
                    stack.append(v)

        if not any(t in reachable for t in self.targets):
            raise ValueError(
                "Attack graph must contain at least one path from a source to a target"
            )

        self._in_degree = in_deg
        self._out_degree = out_deg

        # Precompute logs of probabilities and impacts.
        # This avoids repeated log computations during optimization.
        self.log_p_edge_base = {
            e: math.log(self.p_edge[e]) for e in self.edges
        }

        self.log_I_base = {
            t: math.log(self.I_base[t]) for t in self.targets
        }


@dataclass
class Countermeasure:
    id: str
    cost: float
    scope: Set[str]
    effectiveness: float
    log_effectiveness: float | None = None
    implemented: bool = False

    def __post_init__(self) -> None:

        # Basic identifier validation
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(
                "Countermeasure id must be a non-empty string"
            )

        if self.cost < 0.0:
            raise ValueError(
                f"Cost for countermeasure {self.id} must be >= 0"
            )

        # Ensure scope is a set
        if not isinstance(self.scope, set):
            self.scope = set(self.scope)

        if not self.scope:
            raise ValueError(
                f"Countermeasure {self.id} must act on at least one element"
            )

        # Effectiveness is modeled as a multiplicative reduction factor
        if not (0.0 < self.effectiveness < 1.0):
            raise ValueError(
                f"Effectiveness for countermeasure {self.id} must satisfy 0 < effectiveness < 1"
            )
        
        # Already implemented controls must not consume additional budget
        if self.implemented and self.cost != 0.0:
            raise ValueError(
                f"Countermeasure {self.id} is already implemented so cost must be 0"
            )

        # Precompute log(effectiveness) for faster risk updates
        self.log_effectiveness = math.log(self.effectiveness)


@dataclass
class CountermeasureCatalog:
    items: List[Countermeasure]
    edge_to_cm: Dict[str, List[Countermeasure]] | None = None
    target_to_cm: Dict[str, List[Countermeasure]] | None = None

    def __post_init__(self) -> None:

        # Ensure all countermeasure IDs are unique
        seen: Set[str] = set()

        for cm in self.items:

            if cm.id in seen:
                raise ValueError(
                    f"Duplicate countermeasure id: {cm.id}"
                )

            seen.add(cm.id)

        self.edge_to_cm = {}
        self.target_to_cm = {}

        # Initialize structures used later for fast lookup
        # (actual mapping is completed in ModelInput validation)
        for cm in self.items:

            for elem in cm.scope:

                if elem not in self.edge_to_cm:
                    self.edge_to_cm[elem] = []

                if elem not in self.target_to_cm:
                    self.target_to_cm[elem] = []

    def by_id(self) -> Dict[str, Countermeasure]:
        return {cm.id: cm for cm in self.items}


@dataclass
class OptimizationConfig:
    risk_threshold: Optional[float] = None
    budget: Optional[float] = None

    def __post_init__(self) -> None:

        # Risk threshold must be strictly positive if specified
        if self.risk_threshold is not None and self.risk_threshold <= 0.0:
            raise ValueError(
                "risk_threshold must be > 0"
            )

        if self.budget is not None and self.budget < 0.0:
            raise ValueError(
                "budget must be >= 0"
            )


@dataclass
class ModelInput:
    graph: AttackGraph
    catalog: CountermeasureCatalog
    config: OptimizationConfig

    def __post_init__(self) -> None:

        edge_ids = set(self.graph.edges.keys())
        target_ids = set(self.graph.targets)

        # Ensure each countermeasure acts either on edges or targets, but not both
        for cm in self.catalog.items:

            if cm.scope.issubset(edge_ids):
                continue

            if cm.scope.issubset(target_ids):
                continue

            raise ValueError(
                f"Countermeasure {cm.id} must reference only edges OR only targets"
            )

        # Build lookup tables used by the optimizer to quickly find
        # which countermeasures affect a given edge or target
        self.catalog.edge_to_cm = {}
        self.catalog.target_to_cm = {}

        for cm in self.catalog.items:

            if cm.scope.issubset(edge_ids):

                for e in cm.scope:
                    self.catalog.edge_to_cm.setdefault(e, []).append(cm)

            else:

                for t in cm.scope:
                    self.catalog.target_to_cm.setdefault(t, []).append(cm)