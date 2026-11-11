"""Automated Pipeline Script for Passive AV Data (Stages 1 & 2).

Loads configuration from pipeline_config.json (or CLI overrides) to run:
  1. [Stage 0]: Data extraction from raw spike files if not yet extracted.
                Supports single-depth animals (Nala, Rajah, Crumble, Gnocchi) and
                multi-depth animals (Eclair) segmented by recording depth from Elair_Reps_GLM.csv.
  2. [Stage 1]: Automated channel QC & sensory modality screening (plot_channel_diagnostics_auto.py),
                annotated with anatomical cortical field from Ferret_Side_Chan_Field.csv.
  3. [Stage 2]: Publication-standard unified causal NeMoS Poisson GLM (fit_glm_passive.py) across all viable channels.

Usage Examples:
    # Run full pipeline with default config (F1812_Nala Rec_A)
    python run_pipeline.py

    # Run for multi-depth animal F1902_Eclair at Depth 1
    python run_pipeline.py --animal F1902_Eclair --depth 1 --rec Rec_A

    # Run for all cortical depths of Eclair (Depths 1, 2, 3, 4)
    python run_pipeline.py --animal F1902_Eclair --depth all

    # Override animal or recording stream
    python run_pipeline.py --animal F1813_Rajah --rec Rec_A

    # Run only Stage 1 (Screening) or only Stage 2 (GLM)
    python run_pipeline.py --stage 1
    python run_pipeline.py --stage 2

    # Enable diagnostic dashboard PNG generation (by default fast screening skips PNGs)
    python run_pipeline.py --plots

    # Use a custom config JSON file
    python run_pipeline.py --config my_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)


def find_repo_root(start_dir: pathlib.Path) -> pathlib.Path:
    # 1. Look for root data tables or repository markers
    for d in [start_dir, start_dir.parent]:
        if (d / "data_tables").exists() or (d / "_data_tables").exists():
            return d
    for d in [start_dir.parent, start_dir]:
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            return d
    for d in [start_dir.parent, start_dir]:
        if (d / "output").exists() or (d / "stimuli").exists():
            return d
    for d in [start_dir.parent, start_dir]:
        if (d / "pipeline_config.json").exists():
            return d
    return start_dir


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_config.json"
if not DEFAULT_CONFIG_PATH.exists():
    for c in [
        REPO_ROOT / "pipeline_config.json",
        REPO_ROOT / "AV_passive_scripts" / "pipeline_config.json",
        REPO_ROOT / "scripts" / "pipeline_config.json",
        REPO_ROOT / "_scripts" / "pipeline_config.json",
    ]:
        if c.exists():
            DEFAULT_CONFIG_PATH = c
            break

# Ensure scripts / _scripts / AV_passive_scripts is in sys.path
for s_dir in [
    SCRIPT_DIR,
    REPO_ROOT,
    REPO_ROOT / "AV_passive_scripts",
    REPO_ROOT / "scripts",
    REPO_ROOT / "_scripts",
]:
    if s_dir.exists() and str(s_dir) not in sys.path:
        sys.path.insert(0, str(s_dir))


def resolve_path(
    p: str | pathlib.Path, base_dir: pathlib.Path = REPO_ROOT
) -> pathlib.Path:
    """Resolve absolute or repository-relative paths cleanly."""
    path_obj = pathlib.Path(p)
    if path_obj.is_absolute():
        return path_obj
    return (base_dir / path_obj).resolve()


def load_config(config_path: pathlib.Path | None = None) -> dict:
    """Load JSON pipeline configuration with fallback searching."""
    candidates = []
    if config_path:
        candidates.append(pathlib.Path(config_path))
    candidates.extend(
        [
            DEFAULT_CONFIG_PATH,
            REPO_ROOT / "pipeline_config.json",
            REPO_ROOT / "AV_passive_scripts" / "pipeline_config.json",
            REPO_ROOT / "scripts" / "pipeline_config.json",
            REPO_ROOT / "_scripts" / "pipeline_config.json",
        ]
    )

    cfg_file = None
    for c in candidates:
        if c.exists():
            cfg_file = c
            break

    if not cfg_file:
        raise FileNotFoundError(
            f"Pipeline config file not found. Checked: {[str(c) for c in candidates]}"
        )

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Resolve all directories with automatic non-underscore / underscore fallback
    def find_dir(preferred_rel: str, fallback_rel: str) -> pathlib.Path:
        pref = resolve_path(preferred_rel)
        if pref.exists():
            return pref
        fall = resolve_path(fallback_rel)
        if fall.exists():
            return fall
        return pref

    cfg["data_root"] = resolve_path(
        cfg.get("data_root", r"E:\AV_passive_extracted_spikes")
    )

    raw_stim = cfg.get("stimuli_dir")
    cfg["stimuli_dir"] = (
        resolve_path(raw_stim) if raw_stim else find_dir("stimuli", "_stimuli")
    )

    raw_tables = cfg.get("data_tables_dir")
    cfg["data_tables_dir"] = (
        resolve_path(raw_tables)
        if raw_tables
        else find_dir("data_tables", "_data_tables")
    )

    raw_ext = cfg.get("extracted_dir")
    if raw_ext:
        cfg["extracted_dir"] = resolve_path(raw_ext)
    else:
        for cand in ["output", "extracted_data", "_scripts/extracted_data"]:
            p = resolve_path(cand)
            if p.exists():
                cfg["extracted_dir"] = p
                break
        else:
            cfg["extracted_dir"] = resolve_path("output")

    raw_diag = cfg.get("diagnostics_dir")
    cfg["diagnostics_dir"] = (
        resolve_path(raw_diag) if raw_diag else (cfg["extracted_dir"] / "diagnostics")
    )

    raw_glm = cfg.get("clean_glm_dir")
    cfg["clean_glm_dir"] = (
        resolve_path(raw_glm) if raw_glm else (cfg["extracted_dir"] / "clean_glm")
    )
    cfg["_config_source"] = str(cfg_file)
    return cfg


def ensure_extracted_dataset(
    animal_id: str,
    data_root: pathlib.Path,
    stimuli_dir: pathlib.Path,
    data_tables_dir: pathlib.Path,
    extracted_dir: pathlib.Path,
    rec_name: str = "Rec_A",
    depth: int | None = None,
    depth_table_name: str = "Elair_Reps_GLM.csv",
    bin_size_s: float = 0.02,
    force_extract: bool = False,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Check if 20ms extracted arrays exist for an animal/depth/stream; extract automatically if missing."""
    extracted_dir.mkdir(parents=True, exist_ok=True)
    bin_ms = int(round(bin_size_s * 1000))
    dataset_stem = f"{animal_id}_Depth{depth}" if depth is not None else animal_id

    # Check candidates for existing pre-extracted data containing rec_name
    candidates = [
        (
            extracted_dir / f"{dataset_stem}_{rec_name}_all_blocks_{bin_ms}ms.npz",
            extracted_dir / f"{dataset_stem}_{rec_name}_all_blocks_trials.csv",
        ),
        (
            extracted_dir / f"{dataset_stem}_all_blocks_{bin_ms}ms.npz",
            extracted_dir / f"{dataset_stem}_all_blocks_trials.csv",
        ),
    ]

    if not force_extract:
        for c_npz, c_csv in candidates:
            if c_npz.exists() and c_csv.exists():
                try:
                    with np.load(c_npz) as d:
                        lbls = [str(x) for x in d["channel_labels"]]
                    if any(
                        lbl.startswith(f"{rec_name}_")
                        or ("B" in rec_name and "Rec_B" in lbl)
                        or ("A" in rec_name and "Rec_A" in lbl)
                        for lbl in lbls
                    ):
                        return c_npz, c_csv
                except Exception:
                    pass

    # Extract base animal folder (e.g. F1902_Eclair)
    base_animal = animal_id.split("_Depth")[0].split("_D")[0].strip()
    animal_sessions_dir = data_root / base_animal
    if not animal_sessions_dir.exists():
        raise FileNotFoundError(
            f"Extracted data missing and raw animal directory not found:\n"
            f"  {animal_sessions_dir}\n"
            f"Please verify data_root in pipeline_config.json."
        )

    depth_info = f" (Depth {depth})" if depth is not None else ""
    print(f"\n=======================================================")
    print(
        f"[Stage 0: Extraction] Extracting 20ms arrays for {animal_id}{depth_info} ({rec_name})"
    )
    print(f"  Raw blocks: {animal_sessions_dir}")
    print(f"  Stimuli:    {stimuli_dir}")
    print(f"  Output:     {extracted_dir}")
    print(f"=======================================================")

    led_csv = data_tables_dir / "Chamber_LED.csv"
    if not led_csv.exists():
        raise FileNotFoundError(f"Chamber_LED.csv not found in {data_tables_dir}")

    depth_table_path = data_tables_dir / depth_table_name
    reps_path = (
        depth_table_path
        if (depth_table_path.exists() and "eclair" in animal_id.lower())
        else None
    )

    # Import and run extractor
    from extract_sessions import extract_sessions

    t0 = time.perf_counter()
    save_as_stream_specific = False
    try:
        # Try joint extraction (Rec_A + Rec_B_2) if possible
        res = extract_sessions(
            sessions_dir=animal_sessions_dir,
            stim_dir=stimuli_dir,
            led_csv_path=led_csv,
            depth=depth,
            depth_table_path=reps_path,
            bin_size=bin_size_s,
        )
    except Exception as e:
        # Fallback to single recording stream extraction
        print(
            f"Joint extraction skipped ({e}); extracting specifically for stream {rec_name} ..."
        )
        rec_tag = "Rec_B_2" if ("B" in rec_name) else "Rec_A"
        res = extract_sessions(
            sessions_dir=animal_sessions_dir,
            stim_dir=stimuli_dir,
            led_csv_path=led_csv,
            recordings=(rec_tag,),
            depth=depth,
            depth_table_path=reps_path,
            bin_size=bin_size_s,
        )
        save_as_stream_specific = True

    elapsed = time.perf_counter() - t0

    if save_as_stream_specific:
        out_npz = extracted_dir / f"{dataset_stem}_{rec_name}_all_blocks_{bin_ms}ms.npz"
        out_csv = extracted_dir / f"{dataset_stem}_{rec_name}_all_blocks_trials.csv"
    else:
        out_npz = extracted_dir / f"{dataset_stem}_all_blocks_{bin_ms}ms.npz"
        out_csv = extracted_dir / f"{dataset_stem}_all_blocks_trials.csv"

    np.savez_compressed(
        out_npz,
        envelopes=res["envelopes"],
        visual_voltage=res["visual_voltage"],
        binned_spikes=res["binned_spikes"],
        bin_edges_s=res["bin_edges_s"],
        bin_centers_s=res["bin_centers_s"],
        channel_labels=res["channel_labels"],
        spike_trial_idx=res["spike_trial_idx"],
        spike_channel_idx=res["spike_channel_idx"],
        spike_rel_times_s=res["spike_rel_times_s"],
    )
    res["trials_df"].to_csv(out_csv, index=False)
    print(
        f"Extracted {len(res['trials_df'])} trials across {res['binned_spikes'].shape[2]} channels in {elapsed:.1f}s."
    )
    return out_npz, out_csv


def run_depth_pipeline(
    animal: str,
    rec: str,
    current_depth: int | None,
    cfg: dict,
    channels_override: list[int] | None,
    stages_to_run: list[int],
    skip_plots: bool,
    force_extract: bool,
    fig_formats: tuple[str, ...] = ("png",),
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Execute Stage 0, Stage 1 and Stage 2 for a single depth / recording site."""
    data_root = cfg["data_root"]
    stimuli_dir = cfg["stimuli_dir"]
    data_tables_dir = cfg["data_tables_dir"]
    extracted_dir = cfg["extracted_dir"]
    diagnostics_dir = cfg["diagnostics_dir"]
    clean_glm_dir = cfg["clean_glm_dir"]

    st1_cfg = cfg.get("stage1_diagnostics", {})
    min_rate = float(st1_cfg.get("min_rate_hz", 5.0))
    max_gap = float(st1_cfg.get("max_silent_gap_s", 0.50))
    min_pass_sessions = int(st1_cfg.get("min_pass_sessions", 2))
    min_clean_trials = int(st1_cfg.get("min_clean_trials", 100))

    st2_cfg = cfg.get("stage2_glm", {})
    cv_folds = int(st2_cfg.get("cv_folds", 5))
    reg_strength = float(st2_cfg.get("reg_strength", 1e-3))
    filter_sessions = bool(st2_cfg.get("filter_sessions", True))

    depth_str = f" Depth {current_depth}" if current_depth is not None else ""
    depth_tag = f"_Depth{current_depth}" if current_depth is not None else ""
    print(f"\n---------------------------------------------------------------------")
    print(f"  DATASET EXECUTION: {animal}{depth_str} | Stream: {rec}")
    print(f"---------------------------------------------------------------------")

    # Stage 0: Ensure extraction
    ensure_extracted_dataset(
        animal_id=animal,
        data_root=data_root,
        stimuli_dir=stimuli_dir,
        data_tables_dir=data_tables_dir,
        extracted_dir=extracted_dir,
        rec_name=rec,
        depth=current_depth,
        depth_table_name=cfg.get("depth_table", "Elair_Reps_GLM.csv"),
        bin_size_s=0.02,
        force_extract=force_extract,
    )

    stage1_summary_df = None
    stage2_summary_df = None

    # Stage 1: Automated Diagnostics & Sensory Modality Screening
    if 1 in stages_to_run:
        print(
            f"\n>>> Running Stage 1: Diagnostics Screening ({animal}{depth_str} {rec}) ..."
        )
        from plot_channel_diagnostics_auto import run_batch_diagnostics

        stage1_summary_df = run_batch_diagnostics(
            data_dir=extracted_dir,
            out_dir=diagnostics_dir,
            animal_id=animal,
            rec_name=rec,
            depth=current_depth,
            channel_list=channels_override,
            min_rate_hz=min_rate,
            max_silent_gap_s=max_gap,
            min_pass_sessions=min_pass_sessions,
            min_clean_trials=min_clean_trials,
            skip_plots=skip_plots,
            fig_formats=fig_formats,
        )
    else:
        stage1_csv = (
            diagnostics_dir
            / f"{animal}{depth_tag}_{rec}_population_diagnostics_summary.csv"
        )
        if stage1_csv.exists():
            stage1_summary_df = pd.read_csv(stage1_csv)

    # Stage 2: NeMoS Poisson GLM
    if 2 in stages_to_run:
        print(
            f"\n>>> Running Stage 2: NeMoS Poisson GLM ({animal}{depth_str} {rec}) ..."
        )
        from fit_glm_passive import UnitSpec, run_population_pipeline

        if channels_override is not None:
            selected_channels = channels_override
            print(
                f"Fitting {len(selected_channels)} user-specified channels: {selected_channels}"
            )
        elif stage1_summary_df is not None and "run_glm" in stage1_summary_df.columns:
            viable_df = stage1_summary_df.loc[stage1_summary_df["run_glm"] == 1]
            selected_channels = []
            for _, r in viable_df.iterrows():
                lbl = str(r["channel_name"])
                if "Channel " in lbl:
                    ch_num = int(lbl.split("Channel ")[1].split()[0])
                elif "_C" in lbl:
                    ch_num = int(lbl.split("_C")[-1].split()[0].replace(")", ""))
                else:
                    ch_num = int(r["channel_idx"]) + 1
                selected_channels.append(ch_num)
            n_excluded = len(stage1_summary_df) - len(selected_channels)
            print(
                f"[Diagnostics Filter] Retained {len(selected_channels)} channels passing QC "
                f"({n_excluded} channels excluded from GLM):\n  Channels: {selected_channels}\n"
            )
        else:
            selected_channels = [26, 5]
            print(
                f"No diagnostics summary found; falling back to default test channels {selected_channels}"
            )

        if len(selected_channels) == 0:
            print(
                f"Warning: 0 channels passed Stage 1 QC for {animal}{depth_str} {rec}. Skipping Stage 2 GLM."
            )
        else:
            units = [
                UnitSpec(
                    animal_id=animal,
                    rec_name=rec,
                    channel_num=ch,
                    depth=current_depth,
                    min_rate_hz=min_rate,
                    max_silent_gap_s=max_gap,
                )
                for ch in selected_channels
            ]
            _, stage2_summary_df = run_population_pipeline(
                units=units,
                data_dir=extracted_dir,
                out_dir=clean_glm_dir,
                n_splits=cv_folds,
                reg_strength=reg_strength,
                filter_sessions=filter_sessions,
                fig_formats=fig_formats,
            )

    return stage1_summary_df, stage2_summary_df


def run_pipeline(
    config_path: pathlib.Path | None = None,
    animal_override: str | None = None,
    rec_override: str | None = None,
    depth_override: str | int | None = None,
    channels_override: list[int] | None = None,
    stimset_override: str | int | None = None,
    stages: list[int] | None = None,
    force_plots: bool | None = None,
    force_extract: bool = False,
    fig_format_override: str | None = None,
) -> dict:
    """Execute Stage 1 and Stage 2 pipeline across single or multiple recording depths."""
    cfg = load_config(config_path)

    raw_fmt = (
        fig_format_override
        if fig_format_override is not None
        else ("both" if cfg.get("save_pdf", False) else cfg.get("fig_format", "png"))
    )
    if str(raw_fmt).lower() == "both":
        fig_formats: tuple[str, ...] = ("png", "pdf")
    elif str(raw_fmt).lower() == "pdf":
        fig_formats = ("pdf",)
    else:
        fig_formats = ("png",)

    # Resolve animals to run
    raw_animals = (
        animal_override
        if animal_override is not None
        else cfg.get("animals", cfg.get("animal", "F1812_Nala"))
    )
    if isinstance(raw_animals, str):
        if raw_animals.lower() == "all":
            discovered = [
                d.name
                for d in sorted(data_root.iterdir())
                if d.is_dir() and d.name.startswith("F")
            ]
            animals_to_run = (
                discovered
                if discovered
                else [
                    "F1812_Nala",
                    "F1813_Rajah",
                    "F1901_Crumble",
                    "F1902_Eclair",
                    "F2001_Gnocchi",
                ]
            )
        elif "," in raw_animals:
            animals_to_run = [a.strip() for a in raw_animals.split(",")]
        else:
            animals_to_run = [raw_animals.strip()]
    elif isinstance(raw_animals, (list, tuple)):
        flat = []
        for a in raw_animals:
            if str(a).lower() == "all":
                discovered = [
                    d.name
                    for d in sorted(data_root.iterdir())
                    if d.is_dir() and d.name.startswith("F")
                ]
                flat.extend(
                    discovered
                    if discovered
                    else [
                        "F1812_Nala",
                        "F1813_Rajah",
                        "F1901_Crumble",
                        "F1902_Eclair",
                        "F2001_Gnocchi",
                    ]
                )
            elif "," in str(a):
                flat.extend([x.strip() for x in str(a).split(",")])
            else:
                flat.append(str(a).strip())
        animals_to_run = list(dict.fromkeys(flat))
    else:
        animals_to_run = ["F1812_Nala"]

    # Resolve recording streams to run
    raw_recs = (
        rec_override
        if rec_override is not None
        else cfg.get("recording_streams", cfg.get("recording_stream", "Rec_A"))
    )
    if isinstance(raw_recs, str):
        if raw_recs.lower() == "all":
            recs_to_run = ["Rec_A", "Rec_B"]
        elif "," in raw_recs:
            recs_to_run = [r.strip() for r in raw_recs.split(",")]
        else:
            recs_to_run = [raw_recs.strip()]
    elif isinstance(raw_recs, (list, tuple)):
        flat_r = []
        for r in raw_recs:
            if str(r).lower() == "all":
                flat_r.extend(["Rec_A", "Rec_B"])
            elif "," in str(r):
                flat_r.extend([x.strip() for x in str(r).split(",")])
            else:
                flat_r.append(str(r).strip())
        recs_to_run = list(dict.fromkeys(flat_r))
    else:
        recs_to_run = ["Rec_A"]

    channels_to_use = (
        channels_override if channels_override is not None else cfg.get("channels")
    )
    stimset = (
        stimset_override if stimset_override is not None else cfg.get("stimulus_set")
    )
    stages_to_run = stages if stages is not None else [1, 2]

    # Stage 1 plot settings
    st1_cfg = cfg.get("stage1_diagnostics", {})
    skip_plots = st1_cfg.get("skip_plots", True)
    if force_plots is True:
        skip_plots = False
    elif force_plots is False:
        skip_plots = True

    data_root = cfg["data_root"]
    stimuli_dir = cfg["stimuli_dir"]
    data_tables_dir = cfg["data_tables_dir"]
    extracted_dir = cfg["extracted_dir"]
    diagnostics_dir = cfg["diagnostics_dir"]
    clean_glm_dir = cfg["clean_glm_dir"]

    print("=====================================================================")
    print("           PASSIVE SENSORY NEURAL CODING PIPELINE")
    print("=====================================================================")
    print(f"Config Source:    {cfg['_config_source']}")
    print(f"Animals to Run:   {animals_to_run}")
    print(f"Recording Streams:{recs_to_run}")
    print(
        f"Channels:         {channels_to_use if channels_to_use is not None else 'All available channels in stream'}"
    )
    print(
        f"Stimulus Set:     {stimset if stimset is not None else 'All stimulus sets'}"
    )
    print(f"Data Root:        {data_root}")
    print(f"Stimuli Dir:      {stimuli_dir}")
    print(f"Data Tables Dir:  {data_tables_dir}")
    print(f"Stages Selected:  {stages_to_run}")
    print("=====================================================================\n")

    t_start = time.perf_counter()
    all_s1 = []
    all_s2 = []
    dataset_records = []

    depth_table_path = data_tables_dir / cfg.get("depth_table", "Elair_Reps_GLM.csv")

    for animal in animals_to_run:
        is_multi_depth = depth_table_path.exists() and "eclair" in animal.lower()
        available_depths = [1, 2, 3, 4]
        if is_multi_depth:
            try:
                reps_df = pd.read_csv(depth_table_path)
                available_depths = sorted(
                    reps_df["Depth"].dropna().unique().astype(int).tolist()
                )
            except Exception:
                available_depths = [1, 2, 3, 4]

        raw_depth = depth_override if depth_override is not None else cfg.get("depth")
        if raw_depth is not None and str(raw_depth).lower() not in ("auto", "none"):
            if isinstance(raw_depth, str) and raw_depth.lower() == "all":
                depths_to_run = available_depths if is_multi_depth else [None]
            elif isinstance(raw_depth, (list, tuple)):
                depths_to_run = [int(d) for d in raw_depth]
            else:
                depths_to_run = [int(raw_depth)]
        elif is_multi_depth:
            depths_to_run = available_depths
        else:
            depths_to_run = [None]

        for rec in recs_to_run:
            for d in depths_to_run:
                d_str = f"Depth {d}" if d is not None else "Single"
                dataset_records.append(f"{animal} ({d_str}, {rec})")
                s1_df, s2_df = run_depth_pipeline(
                    animal=animal,
                    rec=rec,
                    current_depth=d,
                    cfg=cfg,
                    channels_override=channels_to_use,
                    stages_to_run=stages_to_run,
                    skip_plots=skip_plots,
                    force_extract=force_extract,
                    fig_formats=fig_formats,
                )
                if s1_df is not None:
                    all_s1.append(s1_df)
                if s2_df is not None:
                    all_s2.append(s2_df)

    elapsed_total = time.perf_counter() - t_start

    final_stage1_df = pd.concat(all_s1, ignore_index=True) if all_s1 else None
    if all_s2:
        final_stage2_df = pd.concat(all_s2, ignore_index=True)
        if "unit_id" in final_stage2_df.columns:
            final_stage2_df = final_stage2_df.drop_duplicates(
                subset=["unit_id"], keep="last"
            ).reset_index(drop=True)
        if "animal_id" in final_stage2_df.columns and "rec_name" in final_stage2_df.columns:
            mask_run = final_stage2_df["animal_id"].isin(animals_to_run) & final_stage2_df[
                "rec_name"
            ].isin(recs_to_run)
            if mask_run.any():
                final_stage2_df = final_stage2_df.loc[mask_run].reset_index(drop=True)
    else:
        final_stage2_df = None

    # =========================================================================
    # Final Pipeline Summary
    # =========================================================================
    print("\n=====================================================================")
    print("                     PIPELINE EXECUTION COMPLETE")
    print("=====================================================================")
    print(f"Total Wall Time:    {elapsed_total:.1f}s ({elapsed_total/60.0:.2f} min)")
    print(
        f"Datasets Processed: {len(dataset_records)} across {len(animals_to_run)} animals"
    )

    if final_stage1_df is not None:
        n_tot = len(final_stage1_df)
        n_pass = (
            int((final_stage1_df["run_glm"] == 1).sum())
            if "run_glm" in final_stage1_df.columns
            else n_tot
        )
        print(
            f"Stage 1 Screened:   {n_tot} sites/channels -> {n_pass} passed QC for GLM ({n_tot - n_pass} excluded)"
        )
        print(f"  Diagnostics Directory: {diagnostics_dir}")

    if final_stage2_df is not None and len(final_stage2_df) > 0:
        print(f"\nStage 2 GLM Units Fitted: {len(final_stage2_df)} recording sites")
        if "coding_strategy" in final_stage2_df.columns:
            strat_counts = final_stage2_df["coding_strategy"].value_counts().to_dict()
            print("  Coding Strategy Breakdown:")
            for strat, cnt in strat_counts.items():
                print(
                    f"    - {strat:25s}: {cnt} ({100.0*cnt/len(final_stage2_df):.1f}%)"
                )
        if "region_label" in final_stage2_df.columns:
            region_counts = final_stage2_df["region_label"].value_counts().to_dict()
            print("  Cortical Field Breakdown:")
            for reg, cnt in region_counts.items():
                reg_name = reg if reg else "Unannotated"
                print(f"    - {reg_name:25s}: {cnt}")
        glm_summary_csv = clean_glm_dir / "population_units_summary.csv"
        print(f"  GLM Summary CSV: {glm_summary_csv}")

    print("=====================================================================\n")

    return {
        "animals": animals_to_run,
        "recs": recs_to_run,
        "stage1_summary": final_stage1_df,
        "stage2_summary": final_stage2_df,
        "elapsed_seconds": elapsed_total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated passive sensory neural coding pipeline across animals."
    )
    parser.add_argument(
        "--config", type=pathlib.Path, default=None, help="Path to pipeline_config.json"
    )
    parser.add_argument(
        "--animal",
        nargs="+",
        default=None,
        help="One or more animal IDs (e.g. F1812_Nala or 'all')",
    )
    parser.add_argument(
        "--rec",
        nargs="+",
        default=None,
        help="One or more recording streams (e.g. Rec_A, Rec_B, or 'all')",
    )
    parser.add_argument(
        "--depth",
        default=None,
        help="Cortical depth index (e.g. 1, 2, 3, 4, or 'all'). Used for multi-depth animals like Eclair.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="Specific 1-based channels to evaluate (default: all channels in stream)",
    )
    parser.add_argument(
        "--stimset",
        default=None,
        help="Optional stimulus set identifier (e.g. 1, 2, or AVPilot1)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        nargs="+",
        default=None,
        choices=[1, 2],
        help="Stages to run (1: Diagnostics, 2: GLM). Default: both 1 and 2.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=None,
        help="Generate 8-panel diagnostic dashboard figures in Stage 1 (default: False for fast screening)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Force re-extraction of raw data from data_root into 20ms arrays",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate and update population summary tables from all existing GLM results without re-running models",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save editable vector PDF figures alongside PNG dashboards (ideal for paper figure preparation)",
    )
    parser.add_argument(
        "--fig-format",
        type=str,
        default=None,
        choices=["png", "pdf", "both"],
        help="Output figure format override: 'png', 'pdf', or 'both'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate:
        from aggregate_glm_results import aggregate_results

        cfg = load_config(args.config)
        aggregate_results(
            input_dirs=[cfg["clean_glm_dir"]],
            out_dir=cfg["clean_glm_dir"],
            tables_dir=cfg["data_tables_dir"],
        )
        return

    fig_fmt_ovr = "both" if args.pdf else args.fig_format

    run_pipeline(
        config_path=args.config,
        animal_override=args.animal,
        rec_override=args.rec,
        depth_override=args.depth,
        channels_override=args.channels,
        stimset_override=args.stimset,
        stages=args.stage,
        force_plots=args.plots,
        force_extract=args.extract,
        fig_format_override=fig_fmt_ovr,
    )


if __name__ == "__main__":
    main()
