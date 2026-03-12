from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import pulp

from utilities import (
    AttackGraph,
    NodeId,
    EdgeId,
    CountermeasureCatalog,
    OptimizationConfig,
    ModelInput,
)


def attacker_best_response(
    graph: AttackGraph,
    log_p_edge: Dict[EdgeId, float],
    log_I: Dict[NodeId, float],
) -> Tuple[List[NodeId], List[EdgeId], float]:

    nodes = list(graph.nodes)
    sources = graph.sources
    targets = graph.targets

    # Longest-path dynamic programming in log-space.
    # Since probabilities multiply along paths, we maximize the sum of logs.
    dist: Dict[NodeId, float] = {n: float("-inf") for n in nodes}
    parent: Dict[NodeId, Optional[NodeId]] = {n: None for n in nodes}
    parent_edge: Dict[NodeId, Optional[EdgeId]] = {n: None for n in nodes}

    for s in sources:
        dist[s] = 0.0

    # Bellman-Ford style relaxation over edges
    n_nodes = len(nodes)
    for _ in range(n_nodes - 1):
        updated = False
        for e_id, (u, v) in graph.edges.items():
            w = log_p_edge[e_id]
            if dist[u] == float("-inf"):
                continue
            cand = dist[u] + w
            if cand > dist[v]:
                dist[v] = cand
                parent[v] = u
                parent_edge[v] = e_id
                updated = True
        if not updated:
            break

    # Attacker chooses the target maximizing probability * impact
    best_target = max(targets, key=lambda t: dist[t] + log_I[t])
    best_score = dist[best_target] + log_I[best_target]

    # Reconstruct attack path
    path_nodes: List[NodeId] = []
    path_edges: List[EdgeId] = []
    cur = best_target

    while cur is not None:
        path_nodes.append(cur)
        prev = parent[cur]
        if prev is not None:
            e_id = parent_edge[cur]
            assert e_id is not None
            path_edges.append(e_id)
        cur = prev

    path_nodes.reverse()
    path_edges.reverse()

    return path_nodes, path_edges, best_score


def compute_log_p_and_log_I(
    graph: AttackGraph,
    catalog: CountermeasureCatalog,
    x_vals: Dict[str, float],
) -> Tuple[Dict[EdgeId, float], Dict[NodeId, float]]:

    # Start from base log probabilities and impacts
    log_p_edge = graph.log_p_edge_base.copy()
    log_I = graph.log_I_base.copy()

    # Apply countermeasure effects
    for cm in catalog.items:

        x = x_vals.get(cm.id, 0.0)
        if x <= 0.0:
            continue

        log_factor = cm.log_effectiveness

        # Edge countermeasures reduce compromise probability
        if cm.scope.issubset(graph.edges):
            for e in cm.scope:
                log_p_edge[e] += x * log_factor

        # Target countermeasures reduce impact
        else:
            for t in cm.scope:
                log_I[t] += x * log_factor

    return log_p_edge, log_I


def build_path_constraint_coeffs(
    path_nodes: Sequence[NodeId],
    path_edges: Sequence[EdgeId],
    target: NodeId,
    graph: AttackGraph,
    catalog: CountermeasureCatalog,
) -> Tuple[float, Dict[str, float]]:

    # Constant term = baseline log-risk of the path
    const_term = sum(graph.log_p_edge_base[e] for e in path_edges)
    const_term += graph.log_I_base[target]

    coeffs: Dict[str, float] = {}

    # Contributions from countermeasures acting on edges of the path
    for e in path_edges:

        for cm in catalog.edge_to_cm.get(e, []):

            log_factor = cm.log_effectiveness
            coeffs[cm.id] = coeffs.get(cm.id, 0.0) + log_factor

    # Contributions from countermeasures acting on the target
    for cm in catalog.target_to_cm.get(target, []):

        log_factor = math.log(cm.effectiveness)
        coeffs[cm.id] = coeffs.get(cm.id, 0.0) + log_factor

    return const_term, coeffs


def _solve_min_cost_given_risk_cap_and_budget(
    graph: AttackGraph,
    catalog: CountermeasureCatalog,
    budget: float,
    risk_cap: float,
    iter_max: int = 50,
    tol: float = 1e-6,
) -> Dict[str, object]:

    # Risk cap is enforced in log-space
    log_R_cap = math.log(risk_cap)

    prob = pulp.LpProblem("MinCostGivenRiskCapAndBudget", pulp.LpMinimize)

    # Binary decision variables: whether to deploy a countermeasure
    x_vars = {
        cm.id: pulp.LpVariable(
            f"x_{cm.id}",
            lowBound=1 if cm.implemented else 0,
            upBound=1,
            cat="Binary"
        )
        for cm in catalog.items
    }

    # Objective: minimize total cost
    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items)

    # Budget constraint
    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items) <= budget

    active_paths: List[List[NodeId]] = []
    best_solution = None

    # Constraint generation loop
    for it in range(iter_max):

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        x_vals = {cid: var.value() or 0.0 for cid, var in x_vars.items()}

        # Compute attacker best response under current defense
        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x_vals)
        path, path_edges, log_risk = attacker_best_response(graph, log_p, log_I)

        worst_risk = math.exp(log_risk)

        # If worst-case risk satisfies the constraint, we are done
        if worst_risk <= risk_cap + tol:

            best_solution = {
                "status": "Optimal",
                "selected_cms": [cm.id for cm in catalog.items if x_vals[cm.id] > 0.5],
                "x_values": x_vals,
                "total_cost": sum(cm.cost * x_vals[cm.id] for cm in catalog.items),
                "worst_path": path,
                "worst_path_risk": worst_risk,
                "active_paths": active_paths,
            }
            break

        # Otherwise add a new path constraint
        target = path[-1]

        const_term, coeffs = build_path_constraint_coeffs(
            path, path_edges, target, graph, catalog
        )

        prob += (
            const_term
            + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs)
            <= log_R_cap
        )

        active_paths.append(list(path))

    if best_solution is None:

        best_solution = {
            "status": "IterLimit",
            "selected_cms": [],
            "x_values": {},
            "total_cost": None,
            "worst_path": None,
            "worst_path_risk": None,
            "active_paths": active_paths,
        }

    return best_solution


def solve_min_cost_given_risk_threshold(
    model_input: ModelInput,
    iter_max: int = 50,
    tol: float = 1e-6,
) -> Dict[str, object]:

    graph = model_input.graph
    catalog = model_input.catalog
    config = model_input.config

    if config.risk_threshold is None:
        raise ValueError("risk_threshold not set")

    R_max = config.risk_threshold
    log_R_max = math.log(R_max)

    prob = pulp.LpProblem("MinCostUnderRiskThreshold", pulp.LpMinimize)

    x_vars = {
        cm.id: pulp.LpVariable(
            f"x_{cm.id}",
            lowBound=1 if cm.implemented else 0,
            upBound=1,
            cat="Binary"
        )
        for cm in catalog.items
    }

    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items)

    active_paths = []
    best_solution = None

    # Same constraint-generation approach but enforcing risk threshold
    for it in range(iter_max):

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        x_vals = {cid: var.value() or 0.0 for cid, var in x_vars.items()}

        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x_vals)
        path, path_edges, log_risk = attacker_best_response(graph, log_p, log_I)

        worst_risk = math.exp(log_risk)

        if worst_risk <= R_max + tol:

            best_solution = {
                "status": "Optimal",
                "selected_cms": [cm.id for cm in catalog.items if x_vals[cm.id] > 0.5],
                "x_values": x_vals,
                "total_cost": sum(cm.cost * x_vals[cm.id] for cm in catalog.items),
                "worst_path": path,
                "worst_path_risk": worst_risk,
                "active_paths": active_paths,
            }
            break

        target = path[-1]

        const_term, coeffs = build_path_constraint_coeffs(
            path, path_edges, target, graph, catalog
        )

        prob += (
            const_term
            + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs)
            <= log_R_max
        )

        active_paths.append(list(path))

    if best_solution is None:
        return {
            "status": "IterLimit",
            "selected_cms": [],
            "x_values": {},
            "total_cost": None,
            "worst_path": None,
            "worst_path_risk": None,
            "active_paths": active_paths,
        }

    # Lexicographic refinement: minimize risk given the optimal cost
    cost_star = float(best_solution["total_cost"])

    lex_config = OptimizationConfig(risk_threshold=None, budget=cost_star)
    lex_input = ModelInput(graph=graph, catalog=catalog, config=lex_config)

    return solve_min_risk_given_budget(lex_input, iter_max, tol)


def solve_min_risk_given_budget(
    model_input: ModelInput,
    iter_max: int = 50,
    tol: float = 1e-6,
) -> Dict[str, object]:

    graph = model_input.graph
    catalog = model_input.catalog
    config = model_input.config

    if config.budget is None:
        raise ValueError("budget not set")

    budget = config.budget

    prob = pulp.LpProblem("MinRiskGivenBudget", pulp.LpMinimize)

    x_vars = {
        cm.id: pulp.LpVariable(
            f"x_{cm.id}",
            lowBound=1 if cm.implemented else 0,
            upBound=1,
            cat="Binary"
        )
        for cm in catalog.items
    }

    # z represents the maximum log-risk over all attack paths
    z = pulp.LpVariable("z", lowBound=math.log(1e-12))

    prob += z

    prob += pulp.lpSum(cm.cost * x_vars[cm.id] for cm in catalog.items) <= budget

    active_paths = []
    best_solution = None
    risk_cap = None

    # Iteratively add worst-path constraints
    for it in range(iter_max):

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        x_vals = {cid: var.value() or 0.0 for cid, var in x_vars.items()}

        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x_vals)
        path, path_edges, log_risk = attacker_best_response(graph, log_p, log_I)

        worst_risk = math.exp(log_risk)

        z_val = z.value()
        if z_val is not None and z_val + tol >= log_risk:

            best_solution = {
                "status": "Optimal",
                "selected_cms": [cm.id for cm in catalog.items if x_vals[cm.id] > 0.5],
                "x_values": x_vals,
                "total_cost": sum(cm.cost * x_vals[cm.id] for cm in catalog.items),
                "worst_path": path,
                "worst_path_risk": worst_risk,
                "active_paths": active_paths,
            }

            risk_cap = worst_risk
            break

        target = path[-1]

        const_term, coeffs = build_path_constraint_coeffs(
            path, path_edges, target, graph, catalog
        )

        prob += z >= const_term + pulp.lpSum(coeffs[c] * x_vars[c] for c in coeffs)

        active_paths.append(list(path))

    if best_solution is None:
        return {
            "status": "IterLimit",
            "selected_cms": [],
            "x_values": {},
            "total_cost": None,
            "worst_path": None,
            "worst_path_risk": None,
            "active_paths": active_paths,
        }

    # Second phase: minimize cost while keeping the same optimal risk
    lex = _solve_min_cost_given_risk_cap_and_budget(
        graph,
        catalog,
        budget,
        risk_cap,
        iter_max,
        tol,
    )

    return lex