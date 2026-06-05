import os
import pandas as pd

OUTPUT_FILE = os.path.join("data", "processed", "instrument_validation_table.csv")


def main():
    rows = [
        {
            "validation_dimension": "Content validity",
            "evidence_used": "Survey-to-variable mapping and FRICTA branch assignment",
            "key_result": (
                "Each survey item was mapped to a canonical FRICTA variable "
                "and assigned to one analytical branch."
            ),
            "interpretation_strength": "Supported",
            "paper_interpretation": (
                "The instrument shows content validity because each item has "
                "a defined analytical role within the framework."
            ),
        },
        {
            "validation_dimension": "Structural validity",
            "evidence_used": "Branch correlation matrix",
            "key_result": (
                "Some branches are weakly associated, but ICI-OLI and OCI-HCARI "
                "show high overlap."
            ),
            "interpretation_strength": "Partially supported",
            "paper_interpretation": (
                "The branches should be interpreted as related dimensions of "
                "digital adoption friction, not as fully independent constructs."
            ),
        },
        {
            "validation_dimension": "Weight robustness",
            "evidence_used": "Sensitivity analysis under alternative weighting schemes",
            "key_result": (
                "Top-10 institutional overlap remained between 0.8 and 0.9 "
                "across alternative weighting schemes."
            ),
            "interpretation_strength": "Supported",
            "paper_interpretation": (
                "The institutional ranking is robust to moderate changes in "
                "branch weights."
            ),
        },
        {
            "validation_dimension": "Discriminant validity",
            "evidence_used": "Low-friction vs high-friction profile analysis",
            "key_result": (
                "High-friction institutions were strongly separated from "
                "low-friction institutions by multiple indicators."
            ),
            "interpretation_strength": "Supported",
            "paper_interpretation": (
                "The instrument distinguishes between institutional profiles "
                "with different levels of adoption friction."
            ),
        },
        {
            "validation_dimension": "Hypothesis robustness",
            "evidence_used": "Digital maturity vs human-capacity stress test",
            "key_result": (
                "Digital maturity deficit showed a standardized difference of "
                "1.53, compared with 0.90 for human-capacity deficit."
            ),
            "interpretation_strength": "Supported",
            "paper_interpretation": (
                "Digital maturity indicators distinguished high-friction "
                "institutions more strongly than isolated human-capacity barriers."
            ),
        },
        {
            "validation_dimension": "Threshold robustness",
            "evidence_used": "20/80, 25/75, and 30/70 friction group thresholds",
            "key_result": (
                "Digital maturity remained stronger than human capacity across "
                "all tested thresholds."
            ),
            "interpretation_strength": "Supported",
            "paper_interpretation": (
                "The central finding does not depend on a single arbitrary "
                "definition of high and low friction."
            ),
        },
        {
            "validation_dimension": "Branch-removal robustness",
            "evidence_used": "Leave-one-branch-out hypothesis test",
            "key_result": (
                "Digital maturity remained stronger in three of four branch-removal "
                "conditions; when ICI was removed, human capacity was slightly stronger."
            ),
            "interpretation_strength": "Partially supported",
            "paper_interpretation": (
                "The finding is broadly robust, but ICI appears to act as a critical "
                "bridge between digital maturity and adoption friction."
            ),
        },
        {
            "validation_dimension": "Component-level interpretability",
            "evidence_used": "ICI decomposition analysis",
            "key_result": (
                "Device constraint showed the strongest standardized difference "
                "among ICI-related components."
            ),
            "interpretation_strength": "Supported",
            "paper_interpretation": (
                "Within the infrastructure branch, device availability appears to be "
                "the most informative component for distinguishing friction profiles."
            ),
        },
        {
            "validation_dimension": "Applied validity",
            "evidence_used": "Institutional pilots or external feedback",
            "key_result": "Pending or limited depending on institutional response.",
            "interpretation_strength": "Limited / pending",
            "paper_interpretation": (
                "External validation through institutional feedback remains a "
                "limitation and future work if pilots are not completed before submission."
            ),
        },
    ]

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("[SUCCESS] Instrument validation table created.")
    print(f"[OUTPUT] {OUTPUT_FILE}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
