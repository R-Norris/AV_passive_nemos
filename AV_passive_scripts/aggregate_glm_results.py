"""Aggregate and update GLM population summary tables across all animals, runs, and sessions.

Scans the clean GLM results directory (or custom paths), collects individual unit
results and factors, de-duplicates units by unit_id, and writes:
  - population_units_summary.csv
  - population_factors_long.csv

Usage:
    # Run across default directory (_scripts/extracted_data/clean_glm)
    python _scripts/aggregate_glm_results.py

    # Specify custom directory or multiple source directories
    python _scripts/aggregate_glm_results.py --dirs _scripts/extracted_data/clean_glm another_results_folder/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR if (SCRIPT_DIR / "data_tables").exists() or (SCRIPT_DIR / "_data_tables").exists()
    else SCRIPT_DIR.parent
)

DEFAULT_TABLES_DIR = (
    (REPO_ROOT / "data_tables") if (REPO_ROOT / "data_tables").exists()
    else ((REPO_ROOT / "_data_tables") if (REPO_ROOT / "_data_tables").exists()
    else (SCRIPT_DIR / "data_tables" if (SCRIPT_DIR / "data_tables").exists() else SCRIPT_DIR / "_data_tables"))
)

def _find_dir(preferred: list[pathlib.Path], fallback: pathlib.Path) -> pathlib.Path:
    for p in preferred:
        if p.exists():
            return p
    return fallback

DEFAULT_GLM_DIR = _find_dir(
    [
        REPO_ROOT / "output" / "clean_glm",
        REPO_ROOT / "extracted_data" / "clean_glm",
        SCRIPT_DIR / "extracted_data" / "clean_glm",
    ],
    REPO_ROOT / "output" / "clean_glm",
)


def lookup_cortical_field(tables_dir: pathlib.Path, animal_id: str, rec_name: str, channel_mcs: int) -> str:
    """Lookup anatomical cortical field (MEG, PEG, VC, SOM, AEG) from Ferret_Side_Chan_Field.csv."""
    csv_path = tables_dir / "Ferret_Side_Chan_Field.csv"
    if not csv_path.exists():
        return ""
    try:
        df = pd.read_csv(csv_path)
        base = animal_id.split("_Depth")[0].split("_D")[0].strip()
        sub = df[df["Ferret"].str.lower() == base.lower()]
        if len(sub) == 0:
            sub = df[df["Ferret"].str.lower().str.contains(base.lower())]
        if len(sub) == 0:
            return ""
        side = "Left" if ("A" in rec_name.upper() or "LEFT" in rec_name.upper()) else "Right"
        sub_side = sub[sub["Side"].str.lower() == side.lower()]
        row = sub_side[sub_side["Channel_mcs"] == int(channel_mcs)]
        if len(row) > 0:
            return str(row["Field"].iloc[0]).strip()
    except Exception:
        pass
    return ""


def aggregate_results(
    input_dirs: list[pathlib.Path],
    out_dir: pathlib.Path,
    tables_dir: pathlib.Path = DEFAULT_TABLES_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover all unit results across input directories and build consolidated tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    wide_records: dict[str, dict] = {}
    long_dfs: list[pd.DataFrame] = []

    # 1. First, load existing population_units_summary.csv if present as baseline
    master_wide_csv = out_dir / "population_units_summary.csv"
    if master_wide_csv.exists():
        try:
            df_old = pd.read_csv(master_wide_csv)
            for _, row in df_old.iterrows():
                uid = str(row["unit_id"])
                wide_records[uid] = row.to_dict()
        except Exception:
            pass

    master_long_csv = out_dir / "population_factors_long.csv"
    if master_long_csv.exists():
        try:
            df_old_long = pd.read_csv(master_long_csv)
            long_dfs.append(df_old_long)
        except Exception:
            pass

    # 2. Discover unit summaries (*_summary.json) across all input directories
    for d in input_dirs:
        d = pathlib.Path(d)
        if not d.exists():
            continue

        # Check JSON summaries
        for jf in d.glob("*_summary.json"):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                uid = rec.get("unit_id")
                if uid:
                    wide_records[uid] = rec
            except Exception:
                pass

        # Check individual factor files (*_factors.csv)
        factor_files = list(d.glob("*_passive_glm_factors.csv")) + list(d.glob("*_clean_glm_factors.csv"))
        if not factor_files:
            factor_files = list(d.glob("*_factors.csv"))
        for ff in factor_files:
            try:
                sub_f = pd.read_csv(ff)
                if len(sub_f) > 0:
                    long_dfs.append(sub_f)
            except Exception:
                pass

    if not wide_records:
        print(f"No unit results found across {[str(d) for d in input_dirs]}.")
        return pd.DataFrame(), pd.DataFrame()

    wide_df = pd.DataFrame(list(wide_records.values()))

    # Fill in missing region labels if needed
    if "region_label" in wide_df.columns:
        for idx, row in wide_df.iterrows():
            if pd.isna(row["region_label"]) or str(row["region_label"]).strip() in ("", "nan"):
                reg = lookup_cortical_field(tables_dir, str(row["animal_id"]), str(row["rec_name"]), int(row["channel_num"]))
                wide_df.at[idx, "region_label"] = reg

    # Sort wide_df cleanly: animal_id -> depth -> rec_name -> channel_num
    sort_cols = [c for c in ["animal_id", "depth", "rec_name", "channel_num"] if c in wide_df.columns]
    if sort_cols:
        wide_df = wide_df.sort_values(sort_cols).reset_index(drop=True)

    # Consolidate factors long dataframe
    if long_dfs:
        combined_long = pd.concat(long_dfs, ignore_index=True)
        # Deduplicate factors on unit_id + factor_key
        if "unit_id" in combined_long.columns and "factor_key" in combined_long.columns:
            combined_long = combined_long.drop_duplicates(subset=["unit_id", "factor_key"], keep="last")
            combined_long = combined_long.sort_values(["unit_id", "factor_key"]).reset_index(drop=True)
    else:
        combined_long = pd.DataFrame()

    # Save consolidated CSV tables
    wide_df.to_csv(master_wide_csv, index=False)
    if not combined_long.empty:
        combined_long.to_csv(master_long_csv, index=False)

    print("\n=====================================================================")
    print("           POPULATION GLM RESULTS AGGREGATION COMPLETE")
    print("=====================================================================")
    print(f"Total Unique Units: {len(wide_df)}")
    print(f"Output Directory:   {out_dir}")
    print(f"Master Summary CSV: {master_wide_csv.name} ({len(wide_df)} units)")
    if not combined_long.empty:
        print(f"Master Factors CSV: {master_long_csv.name} ({len(combined_long)} factor rows)")

    # Breakdown by Animal & Depth
    print("\nBreakdown by Animal & Depth:")
    grp_cols = [c for c in ["animal_id", "depth", "rec_name"] if c in wide_df.columns]
    if grp_cols:
        counts = wide_df.groupby(grp_cols).size().reset_index(name="n_units")
        for _, r in counts.iterrows():
            d_str = f"Depth {r['depth']}" if "depth" in r else "Single"
            print(f"  - {r['animal_id']} ({d_str}, {r['rec_name']}): {r['n_units']} units")

    # Coding strategy breakdown
    if "coding_strategy" in wide_df.columns:
        print("\nCoding Strategy Breakdown:")
        strat_counts = wide_df["coding_strategy"].value_counts().to_dict()
        for strat, cnt in strat_counts.items():
            pct = 100.0 * cnt / len(wide_df)
            print(f"  - {strat:26s}: {cnt:3d} ({pct:5.1f}%)")

    # Cortical field breakdown
    if "region_label" in wide_df.columns:
        print("\nCortical Field Breakdown:")
        reg_counts = wide_df["region_label"].value_counts().to_dict()
        for reg, cnt in reg_counts.items():
            r_name = str(reg) if reg and str(reg) != "nan" else "Unannotated"
            print(f"  - {r_name:26s}: {cnt:3d}")

    print("=====================================================================\n")
    return wide_df, combined_long


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GLM population summary tables across runs and animals.")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[DEFAULT_GLM_DIR],
        help="One or more directories containing GLM results to aggregate.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_GLM_DIR,
        help="Output directory to save consolidated population tables.",
    )
    parser.add_argument(
        "--tables-dir",
        type=pathlib.Path,
        default=DEFAULT_TABLES_DIR,
        help="Path to directory containing Ferret_Side_Chan_Field.csv.",
    )
    args = parser.parse_args()

    input_paths = [pathlib.Path(d) for d in args.dirs]
    aggregate_results(input_dirs=input_paths, out_dir=args.out_dir, tables_dir=args.tables_dir)


if __name__ == "__main__":
    main()
