import math

# Data structures used to define the attack graph model
from utilities import (
    AttackGraph,
    Countermeasure,
    CountermeasureCatalog,
    ModelInput,
    OptimizationConfig,
)

# Optimization and evaluation routines
from staropt import (
    compute_log_p_and_log_I,
    attacker_best_response,
    solve_min_cost_given_risk_threshold,
    solve_min_risk_given_budget,
)


def build_case_study():
    """
    Define the attack graph, probabilities, impacts, and available countermeasures.

    To create your own scenario, modify:
        - nodes
        - edges
        - sources and targets
        - p_edge (attack probabilities)
        - I_base (impact of each target)
        - countermeasures
    """

    # Nodes of the attack graph (system components)
    nodes = {
        "Remote",
        "Insider/Local",
        "External Services",
        "Dev Machine",
        "3D LiDAR",
        "Edge Computing Unit",
        "Robotic Arm",
    }

    # Directed attack steps: edge_id -> (source_node, destination_node)
    edges = {
        "T1190_AccessFromUntrustedNetworks": ("Remote", "Dev Machine"),
        "T1566_CredentialPhishing": ("Remote", "Dev Machine"),
        "T1021_MisconfiguredNetPolicies": ("Remote", "External Services"),
        "T1190_UnauthAccessExposedServices": ("Remote", "External Services"),
        "T1032_InsecureTraffic": ("External Services", "Robotic Arm"),
        "T1078_InappropriateUserAccess": ("Dev Machine", "Robotic Arm"),
        "T1078_PrivilegeElevation": ("Dev Machine", "Robotic Arm"),
        "T1078_AdminAccess": ("Dev Machine", "Robotic Arm"),
        "T0836_ModifyParameter": ("Insider/Local", "3D LiDAR"),
        "T1548_002_WeakRBAC": ("3D LiDAR", "Edge Computing Unit"),
        "T1562_MisconfiguredPolicies": ("3D LiDAR", "Edge Computing Unit"),
        "T1082_WeakDetectionRules": ("3D LiDAR", "Edge Computing Unit"),
        "T1569_002_ProcessingManipulation": ("Edge Computing Unit", "Robotic Arm"),
        "AML_T0029_AppSettingsManip": ("Edge Computing Unit", "Robotic Arm"),
        "T1068_CodeExecution": ("Edge Computing Unit", "Robotic Arm"),
    }

    # Attacker entry points
    sources = {
        "Remote",
        "Insider/Local",
    }

    # Critical assets
    targets = {
        "Robotic Arm",
    }

    # Probability of successfully exploiting each attack step
    p_edge = {
        "T1190_AccessFromUntrustedNetworks": 0.3,
        "T1566_CredentialPhishing": 0.4,
        "T1021_MisconfiguredNetPolicies": 0.25,
        "T1190_UnauthAccessExposedServices": 0.35,
        "T1032_InsecureTraffic": 0.3,
        "T1078_InappropriateUserAccess": 0.45,
        "T1078_PrivilegeElevation": 0.4,
        "T1078_AdminAccess": 0.5,
        "T0836_ModifyParameter": 0.1,
        "T1548_002_WeakRBAC": 0.35,
        "T1562_MisconfiguredPolicies": 0.3,
        "T1082_WeakDetectionRules": 0.25,
        "T1569_002_ProcessingManipulation": 0.4,
        "AML_T0029_AppSettingsManip": 0.35,
        "T1068_CodeExecution": 0.3,
    }

    # Impact value associated with compromising each target
    I_base = {
        "Robotic Arm": 50,
    }

    graph = AttackGraph(
        nodes=nodes,
        edges=edges,
        sources=sources,
        targets=targets,
        p_edge=p_edge,
        I_base=I_base,
    )

    # Available countermeasures.
    # Each countermeasure reduces the probability of the attack steps in its scope.
    catalog = CountermeasureCatalog(
        items=[
            Countermeasure(
                id="M1018_UserAccountManagement",
                cost=4.0,
                scope={
                    "T1078_InappropriateUserAccess",
                    "T1078_PrivilegeElevation",
                    "T1078_AdminAccess",
                },
                effectiveness=0.7,
            ),
            Countermeasure(
                id="M1018_ProjectAdminRole",
                cost=7.0,
                scope={
                    "T1078_InappropriateUserAccess",
                    "T1078_PrivilegeElevation",
                    "T1078_AdminAccess",
                },
                effectiveness=0.3,
            ),
            Countermeasure(
                id="M1030_TrustedNetworks",
                cost=3.0,
                scope={"T1190_AccessFromUntrustedNetworks"},
                effectiveness=0.6,
            ),
            Countermeasure(
                id="M1026_ElevationPolicy",
                cost=5.0,
                scope={
                    "T1078_PrivilegeElevation",
                    "T1078_AdminAccess",
                },
                effectiveness=0.5,
            ),
            Countermeasure(
                id="M1032_SSO",
                cost=2.0,
                scope={"T1566_CredentialPhishing"},
                effectiveness=0.6,
            ),
            Countermeasure(
                id="M1040_TLS",
                cost=3.0,
                scope={"T1032_InsecureTraffic"},
                effectiveness=0.7,
            ),
            Countermeasure(
                id="M1026_AuthACL",
                cost=5.0,
                scope={
                    "T1021_MisconfiguredNetPolicies",
                    "T1190_UnauthAccessExposedServices",
                },
                effectiveness=0.5,
            ),
            Countermeasure(
                id="M1030_K8sNetPol",
                cost=7.0,
                scope={
                    "T1021_MisconfiguredNetPolicies",
                    "T1190_UnauthAccessExposedServices",
                    "T1032_InsecureTraffic",
                },
                effectiveness=0.4,
            ),
            Countermeasure(
                id="M1018_FalconPolicies",
                cost=7.0,
                scope={
                    "T1548_002_WeakRBAC",
                    "T1562_MisconfiguredPolicies",
                    "T1082_WeakDetectionRules",
                },
                effectiveness=0.5,
            ),
            Countermeasure(
                id="M1049_FalconPrevention",
                cost=5.0,
                scope={
                    "T1562_MisconfiguredPolicies",
                    "T1082_WeakDetectionRules",
                },
                effectiveness=0.6,
            ),
            Countermeasure(
                id="M1018_RBAC",
                cost=3.0,
                scope={"T1548_002_WeakRBAC"},
                effectiveness=0.7,
            ),
            Countermeasure(
                id="M1047_TuneDetection",
                cost=4.0,
                scope={"T1082_WeakDetectionRules"},
                effectiveness=0.65,
            ),
            Countermeasure(
                id="M1026_AppSettingsACL",
                cost=5.0,
                scope={
                    "T1569_002_ProcessingManipulation",
                    "AML_T0029_AppSettingsManip",
                },
                effectiveness=0.55,
            ),
            Countermeasure(
                id="M1053_DriverPatching",
                cost=8.0,
                scope={"T1068_CodeExecution"},
                effectiveness=0.4,
            ),
            Countermeasure(
                id="M1048_InputValidation",
                cost=4.0,
                scope={"T1569_002_ProcessingManipulation"},
                effectiveness=0.6,
            ),
        ]
    )

    # Risk thresholds for the "minimize cost given risk" problem
    risk_thresholds = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5]

    # Budget levels for the "minimize risk given budget" problem
    budgets = [5, 10, 15, 20, 25, 30]

    return graph, catalog, risk_thresholds, budgets


def main() -> None:
    """
    Runs the experiments:
        1) Compute baseline risk
        2) Minimize cost given a risk threshold
        3) Minimize risk given a budget
    """

    graph, catalog, risk_thresholds, budgets = build_case_study()

    x_zero = {cm.id: 0.0 for cm in catalog.items}

    log_p0, log_I0 = compute_log_p_and_log_I(graph, catalog, x_zero)

    base_nodes, base_edges, base_log_risk = attacker_best_response(
        graph, log_p0, log_I0
    )

    base_risk = math.exp(base_log_risk) if base_log_risk is not None else 0.0

    if base_nodes:
        seq = []
        for i, n in enumerate(base_nodes):
            seq.append(n)
            if i < len(base_edges):
                seq.append(base_edges[i])
        base_attack_strategy = " -> ".join(seq)
    else:
        base_attack_strategy = "(no attack path)"

    print("\n=== BASELINE (no countermeasures) ===")
    print(f" Worst attack strategy: {base_attack_strategy}")
    print(f" Baseline risk R_base = {base_risk:.4f}")

    print("\n=== PARETO FRONT (1) — Minimize Cost given Risk Threshold ===")

    print(
        f"{'R_max':>10} | {'Cost':>10} | {'Residual Risk':>15} | "
        f"{'Defensive Strategy':>25} | Attack Strategy"
    )

    print("-" * 150)

    for R_max in risk_thresholds:

        config = OptimizationConfig(risk_threshold=R_max, budget=None)

        mi = ModelInput(graph=graph, catalog=catalog, config=config)

        res = solve_min_cost_given_risk_threshold(mi, iter_max=500)

        if res["status"] != "Optimal":
            print(
                f"{R_max:10.2f} |   ---      |     ---        | "
                f"{'(no solution)':>25} | (no attack path)"
            )
            continue

        sel = sorted(res["selected_cms"])

        defensive_strategy = ", ".join(sel) if sel else "(none)"

        x = {cm.id: (1.0 if cm.id in sel else 0.0) for cm in catalog.items}

        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x)

        nodes, edges, log_risk = attacker_best_response(graph, log_p, log_I)

        if nodes:
            seq = []
            for i, n in enumerate(nodes):
                seq.append(n)
                if i < len(edges):
                    seq.append(edges[i])
            attack_strategy = " -> ".join(seq)
        else:
            attack_strategy = "(no attack path)"

        print(
            f"{R_max:10.2f} | "
            f"{res['total_cost']:10.2f} | "
            f"{res['worst_path_risk']:15.4f} | "
            f"{defensive_strategy:25} | "
            f"{attack_strategy}"
        )

    print("\n=== PARETO FRONT (2) — Minimize Risk given Budget ===")

    print(
        f"{'Budget':>10} | {'Cost':>10} | {'Residual Risk':>15} | "
        f"{'Defensive Strategy':>25} | Attack Strategy"
    )

    print("-" * 150)

    for B in budgets:

        config = OptimizationConfig(risk_threshold=None, budget=B)

        mi = ModelInput(graph=graph, catalog=catalog, config=config)

        res = solve_min_risk_given_budget(mi, iter_max=500)

        if res["status"] != "Optimal":
            print(
                f"{B:10.2f} |   ---      |     ---        | "
                f"{'(no solution)':>25} | (no attack path)"
            )
            continue

        sel = sorted(res["selected_cms"])

        defensive_strategy = ", ".join(sel) if sel else "(none)"

        x = {cm.id: (1.0 if cm.id in sel else 0.0) for cm in catalog.items}

        log_p, log_I = compute_log_p_and_log_I(graph, catalog, x)

        nodes, edges, log_risk = attacker_best_response(graph, log_p, log_I)

        if nodes:
            seq = []
            for i, n in enumerate(nodes):
                seq.append(n)
                if i < len(edges):
                    seq.append(edges[i])
            attack_strategy = " -> ".join(seq)
        else:
            attack_strategy = "(no attack path)"

        print(
            f"{B:10.2f} | "
            f"{res['total_cost']:10.2f} | "
            f"{res['worst_path_risk']:15.4f} | "
            f"{defensive_strategy:25} | "
            f"{attack_strategy}"
        )


if __name__ == "__main__":
    main()