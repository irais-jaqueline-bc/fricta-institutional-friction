from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PLAN_V1_PATH = ROOT / "cipher" / "design" / "stage8_official_run_plan_freeze_v1.json"

REPLICATE_002_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "synthetic"
    / "official"
    / "convergence_audit"
    / "replicate_002"
)

REPLICATE_003_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "synthetic"
    / "official"
    / "convergence_audit"
    / "replicate_003"
)

FREEZE_PATH = ROOT / "cipher" / "design" / "stage8_official_run_plan_freeze_v2.json"

AUDIT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "audit"
    / "stage8e1_convergence_edge_semantics_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Stage 8E1 freeze already exists: {FREEZE_PATH}")

    plan_v1 = load_json(PLAN_V1_PATH)

    replicate_002_audit = load_json(REPLICATE_002_PATH / "replicate_audit.json")

    checks = {
        "stage8e_v1_passed": (
            plan_v1.get("gate_status") == "PASS_STAGE_8E_OFFICIAL_RUN_PLAN_FREEZE"
        ),
        "official_replicate_002_finalized": (
            REPLICATE_002_PATH.exists()
            and replicate_002_audit.get("status") == "PASS_STAGE_8F1_OFFICIAL_REPLICATE"
        ),
        "replicate_003_not_started": (not REPLICATE_003_PATH.exists()),
        "replicate_002_is_frozen_official_index": (
            replicate_002_audit.get("official_replicate") == 2
        ),
    }

    if not all(checks.values()):
        print("\n=== CIPHER STAGE 8E1 — CONVERGENCE EDGE-SEMANTICS FREEZE ===\n")

        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")

        print("\nGATE STATUS: FAIL_STAGE_8E1_CONVERGENCE_EDGE_SEMANTICS")

        raise SystemExit(1)

    plan_v2 = dict(plan_v1)

    plan_v2["version"] = "STAGE8_OFFICIAL_RUN_PLAN_FREEZE_V2"

    plan_v2["status"] = (
        "FROZEN_AFTER_OFFICIAL_REPLICATE_002_TECHNICAL_REVIEW_BEFORE_REPLICATE_003"
    )

    plan_v2["parent_plan"] = {
        "path": str(PLAN_V1_PATH.relative_to(ROOT)),
        "sha256": sha256_file(PLAN_V1_PATH),
    }

    plan_v2["official_replicate_002_status"] = {
        "finalized": True,
        "immutable": True,
        "scientific_results_retained": True,
        "rerun_required": False,
        "reason": (
            "Replicate 002 passed every frozen technical gate. "
            "The post-run review identified only an undefined-correlation "
            "edge case and a provenance-label clarification; neither changes "
            "the fitted models, synthetic data, thresholds, or replicate-002 outputs."
        ),
    }

    plan_v2["convergence_audit"]["uncertainty_metric"]["undefined_spearman_policy"] = {
        "trigger": (
            "scipy Spearman correlation is undefined because at least one "
            "normalized-entropy vector is constant"
        ),
        "status": ("UNDEFINED_CONSTANT_INPUT"),
        "reported_rho": None,
        "passes_0_95": False,
        "escalation_trigger": True,
        "interpretation": (
            "Undefined rank correlation is treated conservatively as failure "
            "of the frozen >=.95 convergence criterion. It is not replaced "
            "with 1.0, even when both vectors appear constant."
        ),
        "reason_for_freeze": (
            "Official replicate 002 produced this mathematical edge case in "
            "S1 and S3. The existing runner already treated NaN >= .95 as False. "
            "This amendment only makes that existing conservative behavior explicit."
        ),
    }

    plan_v2["convergence_audit"]["uncertainty_metric"][
        "constant_input_warning_handling"
    ] = {
        "future_runner_behavior": (
            "detect constant entropy vectors before calling spearmanr, "
            "record UNDEFINED_CONSTANT_INPUT, and avoid emitting warning spam"
        ),
        "scientific_decision_changed": False,
    }

    plan_v2["data_provenance_boundary"] = {
        "latent_truth_posthoc_only": True,
        "observed_auxiliary_pretruth_allowlist": [
            "institution_id",
            "governance_type",
        ],
        "governance_type_role": (
            "Observed auxiliary institutional covariate used only by the "
            "predeclared governance falsification test; it is not a planted "
            "profile/reachability label."
        ),
        "future_runner_requirement": (
            "Before governance falsification, construct an observed-auxiliary "
            "frame containing only institution_id and governance_type. "
            "Do not pass the full synthetic truth table into the governance audit."
        ),
        "forbidden_before_pretruth_outputs_are_frozen": [
            "true_profile",
            "latent_profile",
            "true_boundary",
            "oracle_reachable",
            "accessible_source_latent",
            "planted gate/reverse-lock truth",
            "any other planted outcome/structure label",
        ],
        "replicate_002_effect": (
            "No scientific rerun is required: the governance audit implementation "
            "used only institution_id and governance_type, so the numerical result "
            "is unchanged. This amendment tightens the provenance interface for "
            "replicates 003 onward and corrects the wording of the audit contract."
        ),
    }

    plan_v2["final_aggregation_semantics"]["general"][
        "truth_used_only_post_pipeline"
    ] = "LATENT_TRUTH_ONLY_POSTHOC; GOVERNANCE_TYPE IS ALLOWLISTED OBSERVED AUXILIARY METADATA"

    plan_v2["final_aggregation_semantics"]["general"][
        "undefined_uncertainty_convergence_count_reported"
    ] = True

    plan_v2["no_post_result_changes_addendum"] = [
        "do not reinterpret undefined constant-input Spearman as 1.0",
        "do not relax the .95 uncertainty convergence threshold",
        "do not rerun or replace finalized replicate 002",
        "do not expose latent synthetic truth to pretruth discovery/CF code",
        "do not treat governance_type as latent profile truth",
    ]

    plan_v2["next_stage"] = {
        "id": ("STAGE_8F2"),
        "name": ("Remaining convergence-audit official replicates"),
        "replicates": list(
            range(
                3,
                12,
            )
        ),
        "scope": (
            "Run official replicates 003..011 using the frozen v2 semantics. "
            "Keep undefined constant-input Spearman as convergence failure and "
            "use the observed-auxiliary governance interface."
        ),
    }

    plan_v2["gate_status"] = "PASS_STAGE_8E1_CONVERGENCE_EDGE_SEMANTICS"

    FREEZE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FREEZE_PATH.write_text(
        json.dumps(
            plan_v2,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_PATH.write_text(
        json.dumps(
            {
                "checks": checks,
                "parent_plan_sha256": sha256_file(PLAN_V1_PATH),
                "official_replicate_002_audit_sha256": sha256_file(
                    REPLICATE_002_PATH / "replicate_audit.json"
                ),
                "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "gate_status": ("PASS_STAGE_8E1_CONVERGENCE_EDGE_SEMANTICS"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8E1 — CONVERGENCE EDGE-SEMANTICS FREEZE ===\n")

    print("Official replicate 002:")
    print("  remains finalized and immutable")
    print("  no rerun required")

    print("\nUndefined uncertainty Spearman:")
    print("  constant input -> status UNDEFINED_CONSTANT_INPUT")
    print("  rho = NA")
    print("  convergence pass = False")
    print("  triggers frozen escalation to 1000")
    print("  threshold .95 remains unchanged")

    print("\nSynthetic data provenance:")
    print("  latent truth remains post-hoc only")
    print("  governance_type is allowlisted observed auxiliary metadata")
    print("  future governance audit receives only institution_id + governance_type")
    print("  full truth table is forbidden pretruth")

    print("\n=== FREEZE CHECKS ===\n")

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_8E1_CONVERGENCE_EDGE_SEMANTICS")

    print(
        "Frozen plan v2:",
        FREEZE_PATH.relative_to(ROOT),
    )

    print("Next: Stage 8F2 official replicates 003..011. " "Do not run 012..101 yet.")


if __name__ == "__main__":
    main()
