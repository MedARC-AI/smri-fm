"""Check SynthSeg QC scores and flag any below the 0.65 threshold."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Check SynthSeg QC scores")
    parser.add_argument("--seg-dir", required=True, nargs="+", help="Directories with QC CSVs")
    parser.add_argument("--threshold", type=float, default=0.65, help="QC threshold")
    args = parser.parse_args()

    for seg_dir in args.seg_dir:
        seg_dir = Path(seg_dir)
        print(f"\n=== {seg_dir.name} ===")
        qc_files = sorted(seg_dir.glob("*_qc.csv"))
        if not qc_files:
            print("  No QC files found.")
            continue

        all_qc = []
        for qc_path in qc_files:
            df = pd.read_csv(qc_path)
            all_qc.append(df)

        combined = pd.concat(all_qc, ignore_index=True)
        score_cols = [c for c in combined.columns if c != "subject"]

        print(f"  Subjects: {len(combined)}")
        print(f"  Score columns: {score_cols}")
        print(f"\n  Mean scores:")
        for col in score_cols:
            mean = combined[col].mean()
            print(f"    {col}: {mean:.4f}")

        # Check for failures
        print(f"\n  Scores below {args.threshold}:")
        any_below = False
        for _, row in combined.iterrows():
            for col in score_cols:
                if row[col] < args.threshold:
                    print(f"    {row['subject']} - {col}: {row[col]:.4f}")
                    any_below = True
        if not any_below:
            print("    None! All scores above threshold.")


if __name__ == "__main__":
    main()
