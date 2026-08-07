from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PLAN_V2_PATH = ROOT / "cipher" / "design" / "stage8_official_run_plan_freeze_v2.json"

OFFICIAL_AUDIT_ROOT = (
    ROOT / "cipher" / "outputs" / "synthetic" / "official" / "convergence_audit"
)

FREEZE_PATH = ROOT / "cipher" / "design" / "stage8_official_run_plan_freeze_v3.json"

AUDIT_PATH = (
    ROOT
    / "cipher"
    / "outputs"
    / "audit"
    / "stage8e2_optional_governance_interface_audit.json"
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
        raise FileExistsError(f"Stage 8E2 freeze already exists: {FREEZE_PATH}")

    plan_v2 = load_json(PLAN_V2_PATH)

    replicate_002 = OFFICIAL_AUDIT_ROOT / "replicate_002"

    attempted_indices = list(
        range(
            3,
            12,
        )
    )

    final_paths = {
        replicate: (OFFICIAL_AUDIT_ROOT / f"replicate_{replicate:03d}")
        for replicate in attempted_indices
    }

    working_paths = {
        replicate: (OFFICIAL_AUDIT_ROOT / f"replicate_{replicate:03d}__WORKING")
        for replicate in attempted_indices
    }

    final_existing = [
        replicate for replicate, path in final_paths.items() if path.exists()
    ]

    working_existing = [
        replicate for replicate, path in working_paths.items() if path.exists()
    ]

    checks = {
        "stage8e1_plan_v2_passed": (
            plan_v2.get("gate_status") == "PASS_STAGE_8E1_CONVERGENCE_EDGE_SEMANTICS"
        ),
        "official_replicate_002_still_finalized": (replicate_002.exists()),
        "no_003_011_finalized_after_failed_runner": (len(final_existing) == 0),
        "failed_working_attempts_detected": (len(working_existing) >= 1),
    }

    if not all(checks.values()):
        print("\n=== CIPHER STAGE 8E2 — OPTIONAL GOVERNANCE INTERFACE FREEZE ===\n")

        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")

        print(
            "Detected finalized 003..011:",
            final_existing,
        )

        print(
            "Detected failed working 003..011:",
            working_existing,
        )

        print("\nGATE STATUS: FAIL_STAGE_8E2_OPTIONAL_GOVERNANCE_INTERFACE")

        raise SystemExit(1)

    plan_v3 = dict(plan_v2)

    plan_v3["version"] = "STAGE8_OFFICIAL_RUN_PLAN_FREEZE_V3"

    plan_v3["status"] = (
        "FROZEN_AFTER_8F2_V1_GOVERNANCE_INTERFACE_FAILURE_BEFORE_ANY_003_011_FINALIZATION"
    )

    plan_v3["parent_plan"] = {
        "path": str(PLAN_V2_PATH.relative_to(ROOT)),
        "sha256": sha256_file(PLAN_V2_PATH),
    }

    plan_v3["stage8f2_v1_failure_record"] = {
        "failure_type": ("TECHNICAL_OPTIONAL_FIELD_INTERFACE_BUG"),
        "error": ("KeyError: ['governance_type'] not in index"),
        "root_cause": (
            "The Stage 8E1 provenance hardening attempted to slice "
            "['institution_id', 'governance_type'] from every scenario's "
            "synthetic truth table. The generator only defines governance_type "
            "for S5_GOVERNANCE_CONFOUNDED. The inherited governance_audit "
            "already treats a missing governance variable as NOT_APPLICABLE."
        ),
        "scientific_method_changed": False,
        "thresholds_changed": False,
        "generator_changed": False,
        "official_indices_changed": False,
        "deterministic_seeds_changed": False,
        "finalized_replicates_003_011": final_existing,
        "failed_working_attempts_detected": working_existing,
        "replicate_002_affected": False,
    }

    plan_v3["data_provenance_boundary"]["future_runner_requirement"] = (
        "Construct a fresh observed-auxiliary DataFrame that always contains "
        "institution_id and contains governance_type only when that field exists "
        "in the generator output. Pass only that frame into governance_audit. "
        "Never pass the full synthetic truth table."
    )

    plan_v3["data_provenance_boundary"]["observed_auxiliary_interface"] = {
        "all_scenarios": [
            "institution_id",
        ],
        "S5_GOVERNANCE_CONFOUNDED_only": [
            "institution_id",
            "governance_type",
        ],
        "other_scenarios_governance_audit_status": (
            "NOT_APPLICABLE_NO_GOVERNANCE_VARIABLE"
        ),
        "latent_truth_exposed_pretruth": False,
    }

    plan_v3["failed_attempt_recovery_policy"] = {
        "affected_indices": working_existing,
        "finalized_official_result_exists": False,
        "action_before_rerun": (
            "Archive each existing replicate_XXX__WORKING directory by renaming "
            "it to a unique replicate_XXX__FAILED_8F2_V1_GOVERNANCE_INTERFACE "
            "path. Never delete or overwrite the failed attempt."
        ),
        "rerun_rule": (
            "Rerun the same official replicate index with the same deterministic "
            "seed after the semantics-preserving interface fix."
        ),
        "replacement_index_forbidden": True,
        "failed_attempts_excluded_from_official_inference": True,
    }

    plan_v3["shell_execution_policy"] = {
        "problem_observed": (
            "A shell pipeline using Python | tee without pipefail returns tee's "
            "exit status, so `|| break` may fail to stop after Python crashes."
        ),
        "required_for_future_multi_replicate_loop": (
            "Run `set -o pipefail` before the loop, or inspect PIPESTATUS, so a "
            "nonzero Python exit stops the loop."
        ),
        "scientific_method_changed": False,
    }

    plan_v3["no_post_result_changes_addendum_v2"] = [
        "do not add governance_type to scenarios that do not generate it",
        "do not change S5 governance geometry or counts",
        "do not treat missing governance_type as a scientific failure",
        "do not delete failed 8F2 v1 working directories",
        "do not replace official indices 003..011",
        "do not rerun finalized replicate 002",
    ]

    plan_v3["next_stage"] = {
        "id": ("STAGE_8F2_V2"),
        "name": (
            "Archive failed working attempts and rerun official convergence replicates 003..011"
        ),
        "scope": (
            "Use conditional observed-auxiliary governance frames and a pipefail-safe "
            "execution loop. Preserve official indices and deterministic seeds."
        ),
    }

    plan_v3["gate_status"] = "PASS_STAGE_8E2_OPTIONAL_GOVERNANCE_INTERFACE"

    FREEZE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FREEZE_PATH.write_text(
        json.dumps(
            plan_v3,
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
                "failed_working_attempts_detected": working_existing,
                "finalized_003_011_detected": final_existing,
                "parent_plan_sha256": sha256_file(PLAN_V2_PATH),
                "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "gate_status": ("PASS_STAGE_8E2_OPTIONAL_GOVERNANCE_INTERFACE"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CIPHER STAGE 8E2 — OPTIONAL GOVERNANCE INTERFACE FREEZE ===\n")

    print("8F2 v1 failure:")
    print("  technical interface bug only")
    print("  governance_type exists only in S5")
    print("  no official replicate 003..011 was finalized")

    print("\nFrozen governance interface:")
    print("  all scenarios: institution_id")
    print("  S5 only: institution_id + governance_type")
    print(
        "  other scenarios: governance audit -> NOT_APPLICABLE_NO_GOVERNANCE_VARIABLE"
    )
    print("  full latent truth table is never passed pretruth")

    print("\nFailed-attempt recovery:")
    print("  archive existing __WORKING directories; never delete/overwrite")
    print("  rerun SAME official indices with SAME deterministic seeds")
    print("  failed attempts do not enter official inference")

    print("\nShell execution:")
    print("  future loop MUST use: set -o pipefail")
    print("  this makes Python | tee propagate Python failures")

    print(
        "\nDetected failed working attempts:",
        working_existing,
    )

    print("\n=== FREEZE CHECKS ===\n")

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nGATE STATUS: PASS_STAGE_8E2_OPTIONAL_GOVERNANCE_INTERFACE")

    print(
        "Frozen plan v3:",
        FREEZE_PATH.relative_to(ROOT),
    )

    print("Next: Stage 8F2 v2 recovery + rerun. " "Do not run official 012..101.")


if __name__ == "__main__":
    main()
