import pandas as pd

INPUT_FILE = "data/processed/fricta_scored.csv"

OUTPUT_ARCHETYPES = "data/processed/friction_archetypes.csv"
OUTPUT_SUMMARY = "data/processed/friction_archetype_summary.csv"


def classify_archetype(row):

    scores = {
        "ICI": row["ICI"],
        "OCI": row["OCI"],
        "OLI": row["OLI"],
        "HCARI": row["HCARI"],
    }

    high_branches = sum(score >= 0.60 for score in scores.values())

    if high_branches >= 3:
        return "Multi-Constraint"

    dominant_branch = max(scores, key=scores.get)

    mapping = {
        "ICI": "Infrastructure-Limited",
        "OCI": "Organizationally-Limited",
        "OLI": "Operationally-Limited",
        "HCARI": "Human-Capacity-Limited",
    }

    return mapping[dominant_branch]


def main():

    print("[PIPELINE] Running friction_archetypes.py")

    df = pd.read_csv(INPUT_FILE)

    print(f"[INFO] Institutions loaded: {len(df)}")

    df["friction_archetype"] = df.apply(classify_archetype, axis=1)

    summary = df["friction_archetype"].value_counts().reset_index()

    summary.columns = ["archetype", "institution_count"]

    summary["percentage"] = (summary["institution_count"] / len(df)).round(4)

    df.to_csv(OUTPUT_ARCHETYPES, index=False)

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\nArchetype Summary:\n")
    print(summary.to_string(index=False))

    print("\n[SUCCESS] Friction archetypes completed.")


if __name__ == "__main__":
    main()
