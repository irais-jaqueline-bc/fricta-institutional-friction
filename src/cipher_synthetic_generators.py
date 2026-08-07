from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import norm

FEATURE_NAMES = [
    "willingness_constraint_score",
    "digital_usage_constraint_score",
    "training_deficit_score",
    "device_constraint",
    "digital_tool_variety_constraint",
    "internet_stability_constraint",
    "staffing_constraint_score",
    "time_constraint_score",
    "administrative_disorganization_constraint",
    "recording_system_constraint",
    "system_change_resistance_constraint",
    "admin_time_load_constraint",
    "resource_constraint_score",
]

CONFIG_VECTOR = np.array(
    [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 0],
    dtype=float,
)

DISCRIMINATIVE_INDICES = np.arange(12, dtype=int)
NEUTRAL_INDEX = 12

IMPLEMENTATION_SPEC: dict[str, Any] = {
    "version": "STAGE8_GENERATOR_IMPLEMENTATION_V1",
    "status": "FROZEN_BEFORE_SMOKE_RESULTS",
    "common": {
        "n": 80,
        "p": 13,
        "center": 0.50,
        "profile_deviation": 0.24,
        "core_noise_sd": 0.06,
        "boundary_noise_sd": 0.04,
        "continuum_noise_sd": 0.08,
        "clip": [0.0, 1.0],
        "master_seed": 20260807,
        "numeric_label_swap_probability": 0.50,
    },
    "S1_CONFIG_TWO_PROFILE": {
        "latent_counts": {"A": 40, "B": 40},
        "noise_sd": 0.06,
    },
    "S2_CORE_BOUNDARY": {
        "core_counts": {"A": 28, "B": 28},
        "boundary_counts": {"A_side": 12, "B_side": 12},
        "A_side_target_weight_interval": [0.35, 0.50],
        "B_side_target_weight_interval": [0.50, 0.65],
        "boundary_noise_sd": 0.04,
    },
    "S3_DIRECTIONAL_REACHABILITY": {
        "accessible_source_is_randomized_each_replicate": True,
        "accessible_source_core_count": 24,
        "accessible_source_bridge_count": 16,
        "target_profile_count": 40,
        "gate_feature_count": 3,
        "reverse_lock_feature_count": 6,
        "bridge_target_weight": 0.45,
        "bridge_non_gate_noise_sd": 0.06,
        "bridge_gate_noise_sd": 0.04,
        "target_non_gate_noise_sd": 0.30,
        "target_gate_noise_sd": 0.06,
        "target_neutral_noise_sd": 0.06,
        "oracle_positive_required_changes": 3,
        "oracle_negative_minimum_required_changes": 6,
        "oracle_candidate_gate_values": (
            "copied from a paired observed target-profile institution"
        ),
        "smoke_forward_plausibility_rate_min": 0.75,
        "smoke_mirrored_reverse_plausibility_rate_max": 0.25,
        "note": (
            "The accessible source has a narrow core plus 16 targetward bridges. "
            "Only three gate features remain source-like on each bridge. The target "
            "profile is deliberately broader on non-gate dimensions. This creates "
            "a planted sparse source-to-target candidate while making the mirrored "
            "three-gate reverse move fail source-manifold plausibility in the smoke "
            "geometry. Official CIPHER recovery remains an empirical test."
        ),
    },
    "S4_SEVERITY_CONTINUUM": {
        "latent_z": "Uniform(0,1)",
        "feature_loading_distribution": "Uniform(0.65,1.00)",
        "formula": "x_j = 0.5 + loading_j * (z - 0.5) + Normal(0,0.08)",
        "smoke_median_spearman_with_z_min": 0.75,
    },
    "S5_GOVERNANCE_CONFOUNDED": {
        "exact_counts": {
            "NGO": 26,
            "PUBLIC": 22,
            "PRIVATE": 17,
            "MIXED": 15,
        },
        "feature_blocks": {
            "NGO": [0, 1, 2],
            "PUBLIC": [3, 4, 5],
            "PRIVATE": [6, 7, 8, 9],
            "MIXED": [10, 11, 12],
        },
        "positive_block_offset": 0.18,
        "compensation_rule": (
            "all non-block features receive -0.18*m/(13-m), "
            "so each governance offset vector sums to zero"
        ),
        "noise_sd": 0.08,
        "smoke_min_pairwise_governance_centroid_distance_min": 0.20,
    },
    "S6_NO_CLUSTER_NULL": {
        "gaussian_copula_pairwise_rho": 0.20,
        "beta_marginal_alpha": 2.5,
        "beta_marginal_beta": 2.5,
    },
}


@dataclass
class SyntheticBundle:
    data: pd.DataFrame
    truth: pd.DataFrame
    metadata: dict[str, Any]
    oracle_candidates: pd.DataFrame | None = None


def _seed_for(master_seed: int, scenario_id: str, replicate: int) -> int:
    payload = f"{master_seed}|{scenario_id}|{replicate}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _rng(master_seed: int, scenario_id: str, replicate: int) -> np.random.Generator:
    return np.random.default_rng(_seed_for(master_seed, scenario_id, replicate))


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _prototypes() -> tuple[np.ndarray, np.ndarray]:
    center = IMPLEMENTATION_SPEC["common"]["center"]
    deviation = IMPLEMENTATION_SPEC["common"]["profile_deviation"]
    a = center + deviation * CONFIG_VECTOR
    b = center - deviation * CONFIG_VECTOR
    return a.astype(float), b.astype(float)


def _numeric_labels(
    latent: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool, dict[str, int]]:
    swapped = bool(rng.random() < 0.50)
    mapping = {"A": 2, "B": 1} if swapped else {"A": 1, "B": 2}
    numeric = np.array([mapping[str(value)] for value in latent], dtype=int)
    return numeric, swapped, mapping


def _bundle_from_arrays(
    scenario_id: str,
    replicate: int,
    rng: np.random.Generator,
    X: np.ndarray,
    latent_profile: np.ndarray | None,
    extra_truth: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    oracle_candidates: pd.DataFrame | None = None,
) -> SyntheticBundle:
    n = X.shape[0]
    ids = np.array(
        [f"SYN_{scenario_id}_R{replicate:03d}_{i+1:03d}" for i in range(n)],
        dtype=object,
    )

    perm = rng.permutation(n)
    X = X[perm]
    ids = ids[perm]

    data = pd.DataFrame(X, columns=FEATURE_NAMES)
    data.insert(0, "institution_id", ids)

    truth = pd.DataFrame({"institution_id": ids})

    meta = dict(metadata or {})
    meta.update(
        {
            "scenario_id": scenario_id,
            "replicate": int(replicate),
            "n": int(n),
        }
    )

    if latent_profile is not None:
        latent_profile = np.asarray(latent_profile, dtype=object)[perm]
        numeric, swapped, mapping = _numeric_labels(latent_profile, rng)
        truth["latent_profile"] = latent_profile
        truth["true_profile"] = numeric
        meta["numeric_label_swap"] = swapped
        meta["latent_to_numeric_mapping"] = mapping
    else:
        truth["latent_profile"] = ""
        truth["true_profile"] = pd.Series([pd.NA] * n, dtype="Int64")
        meta["numeric_label_swap"] = None
        meta["latent_to_numeric_mapping"] = None

    if extra_truth:
        for key, value in extra_truth.items():
            arr = np.asarray(value, dtype=object)
            if len(arr) != n:
                raise ValueError(
                    f"{scenario_id}: truth field {key} has len {len(arr)}, expected {n}"
                )
            truth[key] = arr[perm]

    return SyntheticBundle(
        data=data,
        truth=truth,
        metadata=meta,
        oracle_candidates=oracle_candidates,
    )


def generate_s1(
    replicate: int,
    master_seed: int,
) -> SyntheticBundle:
    scenario_id = "S1_CONFIG_TWO_PROFILE"
    rng = _rng(master_seed, scenario_id, replicate)
    a, b = _prototypes()
    sd = 0.06

    Xa = _clip(a + rng.normal(0.0, sd, size=(40, 13)))
    Xb = _clip(b + rng.normal(0.0, sd, size=(40, 13)))
    X = np.vstack([Xa, Xb])
    latent = np.array(["A"] * 40 + ["B"] * 40, dtype=object)

    return _bundle_from_arrays(
        scenario_id,
        replicate,
        rng,
        X,
        latent,
        extra_truth={
            "true_boundary": np.array([False] * 80),
            "oracle_reachable": np.array([False] * 80),
        },
        metadata={
            "prototype_A": a.tolist(),
            "prototype_B": b.tolist(),
        },
    )


def generate_s2(
    replicate: int,
    master_seed: int,
) -> SyntheticBundle:
    scenario_id = "S2_CORE_BOUNDARY"
    rng = _rng(master_seed, scenario_id, replicate)
    a, b = _prototypes()

    A_core = _clip(a + rng.normal(0.0, 0.06, size=(28, 13)))
    B_core = _clip(b + rng.normal(0.0, 0.06, size=(28, 13)))

    # t is the weight on prototype B.
    tA = rng.uniform(0.35, 0.50, size=12)
    tB = rng.uniform(0.50, 0.65, size=12)

    A_boundary = np.vstack(
        [_clip((1.0 - t) * a + t * b + rng.normal(0.0, 0.04, size=13)) for t in tA]
    )
    B_boundary = np.vstack(
        [_clip((1.0 - t) * a + t * b + rng.normal(0.0, 0.04, size=13)) for t in tB]
    )

    X = np.vstack([A_core, A_boundary, B_core, B_boundary])
    latent = np.array(
        ["A"] * 40 + ["B"] * 40,
        dtype=object,
    )
    boundary = np.array([False] * 28 + [True] * 12 + [False] * 28 + [True] * 12)
    mixture_weight_on_B = np.concatenate(
        [
            np.zeros(28),
            tA,
            np.ones(28),
            tB,
        ]
    )

    return _bundle_from_arrays(
        scenario_id,
        replicate,
        rng,
        X,
        latent,
        extra_truth={
            "true_boundary": boundary,
            "mixture_weight_on_B": mixture_weight_on_B,
            "oracle_reachable": np.array([False] * 80),
        },
        metadata={
            "prototype_A": a.tolist(),
            "prototype_B": b.tolist(),
        },
    )


def _within_profile_5nn_threshold(X: np.ndarray) -> float:
    if len(X) < 6:
        raise ValueError("Need at least six target observations for 5-NN threshold.")
    distances = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    ordered = np.sort(distances, axis=1)
    fifth_neighbor = ordered[:, 5]  # col 0 is self
    return float(np.quantile(fifth_neighbor, 0.95))


def _candidate_5nn_distance(
    candidates: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    distances = np.sqrt(
        ((candidates[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
    )
    return np.sort(distances, axis=1)[:, 4]


def generate_s3(
    replicate: int,
    master_seed: int,
) -> SyntheticBundle:
    scenario_id = "S3_DIRECTIONAL_REACHABILITY"
    rng = _rng(master_seed, scenario_id, replicate)
    a, b = _prototypes()

    source_latent = "A" if rng.random() < 0.50 else "B"
    target_latent = "B" if source_latent == "A" else "A"
    source_proto = a if source_latent == "A" else b
    target_proto = b if source_latent == "A" else a

    gates = np.sort(rng.choice(DISCRIMINATIVE_INDICES, size=3, replace=False))
    remaining = np.array(
        [idx for idx in DISCRIMINATIVE_INDICES if idx not in set(gates)],
        dtype=int,
    )
    reverse_locks = np.sort(rng.choice(remaining, size=6, replace=False))

    source_core = _clip(source_proto + rng.normal(0.0, 0.06, size=(24, 13)))

    target_sd = np.full(13, 0.30, dtype=float)
    target_sd[gates] = 0.06
    target_sd[NEUTRAL_INDEX] = 0.06
    target_obs = _clip(target_proto + rng.normal(0.0, target_sd, size=(40, 13)))

    bridge_center = 0.55 * source_proto + 0.45 * target_proto
    bridges = _clip(bridge_center + rng.normal(0.0, 0.06, size=(16, 13)))
    bridges[:, gates] = _clip(
        source_proto[gates] + rng.normal(0.0, 0.04, size=(16, len(gates)))
    )

    # Paired observed target anchors provide legal observed-level values
    # for the planted 3-feature oracle candidates.
    anchor_indices = rng.choice(
        np.arange(40, dtype=int),
        size=16,
        replace=False,
    )
    oracle_cf = bridges.copy()
    oracle_cf[:, gates] = target_obs[
        anchor_indices[:, None],
        gates[None, :],
    ]

    source_X = np.vstack([source_core, bridges])

    if source_latent == "A":
        X = np.vstack([source_X, target_obs])
        latent = np.array(["A"] * 40 + ["B"] * 40, dtype=object)
        bridge_rows = np.arange(24, 40, dtype=int)
        target_rows = np.arange(40, 80, dtype=int)
    else:
        X = np.vstack([target_obs, source_X])
        latent = np.array(["A"] * 40 + ["B"] * 40, dtype=object)
        bridge_rows = np.arange(64, 80, dtype=int)
        target_rows = np.arange(0, 40, dtype=int)

    oracle_reachable = np.zeros(80, dtype=bool)
    oracle_reachable[bridge_rows] = True

    oracle_required = np.full(80, 6, dtype=int)
    oracle_required[bridge_rows] = 3

    bridge_flag = np.zeros(80, dtype=bool)
    bridge_flag[bridge_rows] = True

    source_direction = np.array(
        [source_latent] * 80,
        dtype=object,
    )

    # Build stable IDs before the bundle's row permutation so the oracle
    # candidates can reference their originating latent rows unambiguously.
    pre_ids = np.array(
        [f"SYN_{scenario_id}_R{replicate:03d}_{i+1:03d}" for i in range(80)],
        dtype=object,
    )

    oracle_rows = []
    for local_idx, raw_row in enumerate(bridge_rows):
        candidate = oracle_cf[local_idx]
        row = {
            "source_institution_id_pre_permutation": pre_ids[raw_row],
            "source_latent_profile": source_latent,
            "target_latent_profile": target_latent,
            "oracle_required_features": 3,
            "oracle_gate_features_json": json.dumps(
                [FEATURE_NAMES[int(i)] for i in gates]
            ),
            "paired_target_anchor_index": int(anchor_indices[local_idx]),
        }
        row.update(
            {feature: float(candidate[j]) for j, feature in enumerate(FEATURE_NAMES)}
        )
        oracle_rows.append(row)

    oracle_candidates = pd.DataFrame(oracle_rows)

    bundle = _bundle_from_arrays(
        scenario_id,
        replicate,
        rng,
        X,
        latent,
        extra_truth={
            "true_boundary": np.array([False] * 80),
            "oracle_reachable": oracle_reachable,
            "oracle_required_features": oracle_required,
            "is_accessible_bridge": bridge_flag,
            "accessible_source_latent": source_direction,
        },
        metadata={
            "prototype_A": a.tolist(),
            "prototype_B": b.tolist(),
            "accessible_source_latent": source_latent,
            "target_latent": target_latent,
            "gate_feature_indices": [int(i) for i in gates],
            "gate_feature_names": [FEATURE_NAMES[int(i)] for i in gates],
            "reverse_lock_feature_indices": [int(i) for i in reverse_locks],
            "reverse_lock_feature_names": [
                FEATURE_NAMES[int(i)] for i in reverse_locks
            ],
            "bridge_target_weight": 0.45,
        },
        oracle_candidates=oracle_candidates,
    )

    return bundle


def generate_s4(
    replicate: int,
    master_seed: int,
) -> SyntheticBundle:
    scenario_id = "S4_SEVERITY_CONTINUUM"
    rng = _rng(master_seed, scenario_id, replicate)

    z = rng.uniform(0.0, 1.0, size=80)
    loadings = rng.uniform(0.65, 1.00, size=13)
    X = (
        0.50
        + (z[:, None] - 0.50) * loadings[None, :]
        + rng.normal(0.0, 0.08, size=(80, 13))
    )
    X = _clip(X)

    return _bundle_from_arrays(
        scenario_id,
        replicate,
        rng,
        X,
        None,
        extra_truth={
            "true_boundary": np.array([False] * 80),
            "oracle_reachable": np.array([False] * 80),
            "latent_severity": z,
        },
        metadata={
            "feature_loadings": loadings.tolist(),
        },
    )


def _governance_offset(block: list[int]) -> np.ndarray:
    p = 13
    magnitude = 0.18
    m = len(block)
    compensation = -(magnitude * m) / (p - m)
    offset = np.full(p, compensation, dtype=float)
    offset[np.array(block, dtype=int)] = magnitude

    if abs(float(offset.sum())) > 1e-12:
        raise RuntimeError("Governance offset must sum to zero.")

    return offset


def generate_s5(
    replicate: int,
    master_seed: int,
) -> SyntheticBundle:
    scenario_id = "S5_GOVERNANCE_CONFOUNDED"
    rng = _rng(master_seed, scenario_id, replicate)

    exact_counts = {
        "NGO": 26,
        "PUBLIC": 22,
        "PRIVATE": 17,
        "MIXED": 15,
    }
    blocks = {
        "NGO": [0, 1, 2],
        "PUBLIC": [3, 4, 5],
        "PRIVATE": [6, 7, 8, 9],
        "MIXED": [10, 11, 12],
    }

    governance = np.concatenate(
        [np.array([name] * count, dtype=object) for name, count in exact_counts.items()]
    )
    rng.shuffle(governance)

    X = np.empty((80, 13), dtype=float)

    offset_vectors = {name: _governance_offset(block) for name, block in blocks.items()}

    for i, name in enumerate(governance):
        X[i] = _clip(0.50 + offset_vectors[str(name)] + rng.normal(0.0, 0.08, size=13))

    return _bundle_from_arrays(
        scenario_id,
        replicate,
        rng,
        X,
        None,
        extra_truth={
            "true_boundary": np.array([False] * 80),
            "oracle_reachable": np.array([False] * 80),
            "governance_type": governance,
        },
        metadata={
            "governance_counts": exact_counts,
            "governance_offset_vectors": {
                key: value.tolist() for key, value in offset_vectors.items()
            },
        },
    )


def generate_s6(
    replicate: int,
    master_seed: int,
) -> SyntheticBundle:
    scenario_id = "S6_NO_CLUSTER_NULL"
    rng = _rng(master_seed, scenario_id, replicate)

    rho = 0.20
    corr = np.full((13, 13), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)

    gaussian = rng.multivariate_normal(
        mean=np.zeros(13),
        cov=corr,
        size=80,
    )
    u = norm.cdf(gaussian)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)

    X = beta_dist.ppf(u, a=2.5, b=2.5)
    X = _clip(X)

    return _bundle_from_arrays(
        scenario_id,
        replicate,
        rng,
        X,
        None,
        extra_truth={
            "true_boundary": np.array([False] * 80),
            "oracle_reachable": np.array([False] * 80),
        },
        metadata={
            "gaussian_copula_rho": rho,
            "beta_alpha": 2.5,
            "beta_beta": 2.5,
        },
    )


GENERATORS = {
    "S1_CONFIG_TWO_PROFILE": generate_s1,
    "S2_CORE_BOUNDARY": generate_s2,
    "S3_DIRECTIONAL_REACHABILITY": generate_s3,
    "S4_SEVERITY_CONTINUUM": generate_s4,
    "S5_GOVERNANCE_CONFOUNDED": generate_s5,
    "S6_NO_CLUSTER_NULL": generate_s6,
}


def generate_scenario(
    scenario_id: str,
    replicate: int,
    master_seed: int = 20260807,
) -> SyntheticBundle:
    if scenario_id not in GENERATORS:
        raise KeyError(f"Unknown synthetic scenario: {scenario_id}")

    bundle = GENERATORS[scenario_id](
        replicate=replicate,
        master_seed=master_seed,
    )

    feature_matrix = bundle.data[FEATURE_NAMES].to_numpy(dtype=float)

    if bundle.data.shape != (80, 14):
        raise ValueError(f"{scenario_id}: expected 80 rows and 13 features + ID.")
    if not np.isfinite(feature_matrix).all():
        raise ValueError(f"{scenario_id}: non-finite feature values.")
    if not ((feature_matrix >= 0.0) & (feature_matrix <= 1.0)).all():
        raise ValueError(f"{scenario_id}: feature values outside [0,1].")
    if bundle.data["institution_id"].nunique() != 80:
        raise ValueError(f"{scenario_id}: duplicate institution IDs.")

    return bundle


def plausibility_audit_s3(
    bundle: SyntheticBundle,
) -> dict[str, float]:
    if bundle.oracle_candidates is None:
        raise ValueError("S3 bundle has no oracle candidates.")

    source_latent = str(bundle.metadata["accessible_source_latent"])
    target_latent = str(bundle.metadata["target_latent"])
    gates = [int(i) for i in bundle.metadata["gate_feature_indices"]]

    merged = bundle.data.merge(
        bundle.truth[["institution_id", "latent_profile", "is_accessible_bridge"]],
        on="institution_id",
        validate="one_to_one",
    )

    target = merged[merged["latent_profile"].astype(str) == target_latent][
        FEATURE_NAMES
    ].to_numpy(dtype=float)

    source = merged[merged["latent_profile"].astype(str) == source_latent][
        FEATURE_NAMES
    ].to_numpy(dtype=float)

    oracle = bundle.oracle_candidates[FEATURE_NAMES].to_numpy(dtype=float)

    target_threshold = _within_profile_5nn_threshold(target)
    forward_distance = _candidate_5nn_distance(oracle, target)
    forward_rate = float(np.mean(forward_distance <= target_threshold))

    # Mirror audit: take 16 deterministic target observations and change only the
    # three gate coordinates toward observed source-bridge gate values. This is
    # not the official reverse search; it checks that the generator does not make
    # the obvious three-gate reverse move equally plausible.
    target_subset = target[:16].copy()

    bridge_source = merged[merged["is_accessible_bridge"].astype(bool)][
        FEATURE_NAMES
    ].to_numpy(dtype=float)

    reverse = target_subset.copy()
    reverse[:, gates] = bridge_source[:16, gates]

    source_threshold = _within_profile_5nn_threshold(source)
    reverse_distance = _candidate_5nn_distance(reverse, source)
    reverse_rate = float(np.mean(reverse_distance <= source_threshold))

    return {
        "forward_oracle_plausibility_rate": forward_rate,
        "mirrored_reverse_three_gate_plausibility_rate": reverse_rate,
        "target_5nn_threshold": float(target_threshold),
        "source_5nn_threshold": float(source_threshold),
    }
