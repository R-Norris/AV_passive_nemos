"""Sense-check plots for recording channels against unique auditory and calibrated
visual luminance envelopes across animal sessions.

Generates wide, vertically-stacked PSTH plots for specified channels and modalities,
with solid traces for spiking PSTHs (with SEM ribbons) and dotted traces for stimulus envelopes.
Also provides an optional plot of the Jumbo_LED voltage-to-luminance PCHIP calibration curve.

Usage Examples:
    # Default: Run test channels 26 and 5 for F1812_Nala Rec_A (both modalities)
    python _scripts/plot_sense_check.py

    # Specific channels for an animal
    python _scripts/plot_sense_check.py --animal F1812_Nala --rec Rec_A --channels 26 5 1 20

    # Specific modality only
    python _scripts/plot_sense_check.py --channels 26 --modality auditory
    python _scripts/plot_sense_check.py --channels 5 --modality visual

    # Multi-depth animals (e.g., F1902_Eclair)
    python _scripts/plot_sense_check.py --animal F1902_Eclair --depth 1 --rec Rec_A --channels 2

    # All channels in a recording stream
    python _scripts/plot_sense_check.py --animal F1812_Nala --rec Rec_A --all-channels
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
import numpy as np
import pandas as pd

try:
    from extract_sessions import DEFAULT_LED_CSV, build_led_luminance_model
except Exception:
    DEFAULT_LED_CSV = pathlib.Path(
        r"C:\Users\rhcno\OneDrive - University College London\Project - AV binding\Passive_Task\Jumbo_LED.csv"
    )
    build_led_luminance_model = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR if (SCRIPT_DIR / "data_tables").exists() or (SCRIPT_DIR / "_data_tables").exists()
    else SCRIPT_DIR.parent
)

DATA_TABLES_DIR = (
    (REPO_ROOT / "data_tables") if (REPO_ROOT / "data_tables").exists()
    else ((REPO_ROOT / "_data_tables") if (REPO_ROOT / "_data_tables").exists()
    else (SCRIPT_DIR / "data_tables" if (SCRIPT_DIR / "data_tables").exists() else SCRIPT_DIR / "_data_tables"))
)

def _find_dir(preferred: list[pathlib.Path], fallback: pathlib.Path) -> pathlib.Path:
    for p in preferred:
        if p.exists():
            return p
    return fallback

DATA_DIR = _find_dir(
    [REPO_ROOT / "output", REPO_ROOT / "extracted_data", SCRIPT_DIR / "extracted_data"],
    REPO_ROOT / "output",
)
SENSE_OUT_DIR = _find_dir(
    [DATA_DIR / "sense_check_plots", SCRIPT_DIR / "extracted_data" / "sense_check_plots"],
    DATA_DIR / "sense_check_plots",
)


def lookup_cortical_field(
    data_tables_dir: pathlib.Path,
    animal_id: str,
    rec_name: str,
    channel_mcs: int,
) -> str:
    """Lookup anatomical cortical field (MEG, PEG, VC, SOM, AEG) from Ferret_Side_Chan_Field.csv."""
    tables_dir = pathlib.Path(data_tables_dir)
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
        side = (
            "Left"
            if ("A" in rec_name.upper() or "LEFT" in rec_name.upper())
            else "Right"
        )
        sub_side = sub[sub["Side"].str.lower() == side.lower()]
        row = sub_side[sub_side["Channel_mcs"] == int(channel_mcs)]
        if len(row) > 0:
            return str(row["Field"].iloc[0]).strip()
    except Exception:
        pass
    return ""


def find_channel_index(channel_labels: np.ndarray, rec_tag: str, ch_num: int) -> int:
    """Find index in channel_labels matching recording tag and channel number."""
    target = f"{rec_tag}_C{ch_num:02d}"
    matches = np.where(channel_labels == target)[0]
    if len(matches) > 0:
        return int(matches[0])
    alt_target = f"{rec_tag}_Ch{ch_num:02d}"
    matches = np.where(channel_labels == alt_target)[0]
    if len(matches) > 0:
        return int(matches[0])
    for idx, lbl in enumerate(channel_labels):
        if str(lbl).endswith(f"_{ch_num}") or str(lbl).endswith(f"_C{ch_num:02d}"):
            return idx
    if 0 <= (ch_num - 1) < len(channel_labels):
        return ch_num - 1
    raise ValueError(
        f"Channel {ch_num} (tag: {rec_tag}) not found in channel labels: {channel_labels.tolist()[:10]}..."
    )


def plot_led_calibration_curve(
    led_csv_path: pathlib.Path,
    out_path: pathlib.Path,
    fig_formats: tuple[str, ...] = ("png",),
) -> None:
    """Plot measured Chamber_LED.csv points alongside the fitted monotonic PCHIP curve."""
    if not led_csv_path.exists():
        print(f"Skipping LED calibration curve (file not found: {led_csv_path})")
        return
    if build_led_luminance_model is None:
        print("Skipping LED calibration curve (build_led_luminance_model unavailable)")
        return

    df_led = pd.read_csv(led_csv_path).sort_values("Voltage")
    v_to_l = build_led_luminance_model(led_csv_path)
    v_grid = np.linspace(0.0, 4.1, 500)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(
        v_grid, v_to_l(v_grid), color="#1f77b4", lw=2.2, label="Monotonic PCHIP fit"
    )
    ax.scatter(
        df_led["Voltage"],
        df_led["Luminance"],
        color="#222222",
        s=36,
        zorder=5,
        label="Chamber_LED.csv measurements",
    )
    ax.axvline(2.5, color="gray", ls=":", lw=1.0, label="Turn-on knee (~2.5 V)")
    ax.set_xlabel("LED Drive Voltage (V)")
    ax.set_ylabel("Chamber Luminance")
    ax.set_title("Chamber LED Voltage-to-Luminance Calibration Fit")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in fig_formats:
        target = out_path.with_suffix(f".{ext}")
        fig.savefig(target, dpi=150)
        print(f"Saved LED calibration curve: {target}")
    plt.close(fig)


def assign_unique_envelope_shapes(
    trials_df: pd.DataFrame,
    envelopes: np.ndarray,
    visual_voltage: np.ndarray | None,
    stim_slice: slice,
) -> tuple[list[dict], list[dict]]:
    """Discover all unique auditory and visual temporal envelope shapes per StimFile."""
    aud_groups: list[dict] = []
    vis_groups: list[dict] = []

    for stim_file in sorted(trials_df["StimFile"].unique()):
        sub_mask = (trials_df["StimFile"] == stim_file).values
        sub_indices = np.where(sub_mask)[0]

        # 1. Unique auditory shapes within this StimFile
        file_aud_shapes: list[tuple[np.ndarray, list[int]]] = []
        for idx in sub_indices:
            trace = envelopes[idx, stim_slice, 0]
            norm_trace = trace / (trace.max() + 1e-12)
            placed = False
            for ref_norm, member_list in file_aud_shapes:
                if np.allclose(norm_trace, ref_norm, atol=1e-4):
                    member_list.append(int(idx))
                    placed = True
                    break
            if not placed:
                file_aud_shapes.append((norm_trace, [int(idx)]))

        aud_entries = []
        for _, member_indices in file_aud_shapes:
            m_idx = np.array(member_indices, dtype=int)
            sub_df = trials_df.iloc[m_idx]
            aud_entries.append((int(sub_df["StimIdx"].min()), m_idx, sub_df))
        aud_entries.sort(key=lambda x: x[0])

        for shape_num, (_, m_idx, sub_df) in enumerate(aud_entries, start=1):
            aud_groups.append(
                {
                    "stim_file": stim_file,
                    "shape_num": shape_num,
                    "trial_indices": m_idx,
                    "incoh_indices": m_idx[(sub_df["Coherence"] == 0).values],
                    "stim_idxs": sorted(sub_df["StimIdx"].unique().tolist()),
                    "vowels": (
                        sorted(sub_df["Vowel"].dropna().unique().tolist())
                        if "Vowel" in sub_df.columns
                        else []
                    ),
                    "n_env": (
                        sorted(sub_df["Envelope"].unique().tolist())
                        if "Envelope" in sub_df.columns
                        else []
                    ),
                    "mean_env": envelopes[m_idx, :, 0].mean(axis=0),
                }
            )

        # 2. Unique underlying visual shapes within this StimFile
        file_vis_shapes: list[tuple[np.ndarray, list[int]]] = []
        for idx in sub_indices:
            v_base = (
                float(trials_df.iloc[idx]["vBase"])
                if "vBase" in trials_df.columns
                else 0.0
            )
            v_scale = (
                float(trials_df.iloc[idx]["vScale"])
                if "vScale" in trials_df.columns
                else 1.0
            )
            v_scale = v_scale if abs(v_scale) > 1e-6 else 1.0

            if visual_voltage is not None:
                raw_vis = (visual_voltage[idx, stim_slice] - v_base) / v_scale
            else:
                raw_vis = envelopes[idx, stim_slice, 1]

            norm_vis = raw_vis / (raw_vis.max() + 1e-12)
            placed = False
            for ref_norm, member_list in file_vis_shapes:
                if np.allclose(norm_vis, ref_norm, atol=1e-4):
                    member_list.append(int(idx))
                    placed = True
                    break
            if not placed:
                file_vis_shapes.append((norm_vis, [int(idx)]))

        vis_entries = []
        for _, member_indices in file_vis_shapes:
            m_idx = np.array(member_indices, dtype=int)
            sub_df = trials_df.iloc[m_idx]
            vis_entries.append((int(sub_df["StimIdx"].min()), m_idx, sub_df))
        vis_entries.sort(key=lambda x: x[0])

        for shape_num, (_, m_idx, sub_df) in enumerate(vis_entries, start=1):
            vis_groups.append(
                {
                    "stim_file": stim_file,
                    "shape_num": shape_num,
                    "trial_indices": m_idx,
                    "incoh_indices": m_idx[(sub_df["Coherence"] == 0).values],
                    "stim_idxs": sorted(sub_df["StimIdx"].unique().tolist()),
                    "n_env": (
                        sorted(sub_df["Envelope"].unique().tolist())
                        if "Envelope" in sub_df.columns
                        else []
                    ),
                    "mean_env": envelopes[m_idx, :, 1].mean(axis=0),
                }
            )

    return aud_groups, vis_groups


def plot_single_channel_stacked_envelopes(
    groups: list[dict],
    t: np.ndarray,
    spk_hz: np.ndarray,
    ch_idx: int,
    ch_title: str,
    spike_color: str,
    modality_label: str,
    env_color: str,
    env_unit: str,
    stim_slice: slice,
    out_path: pathlib.Path,
    fig_formats: tuple[str, ...] = ("png",),
) -> None:
    """Plot one channel's PSTH (solid) stacked vertically across all unique envelopes (dotted, unshaded)."""
    n_rows = len(groups)
    if n_rows == 0:
        print(f"Warning: No {modality_label} envelope groups found to plot.")
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.15 * n_rows), sharex=True)
    axes_list = np.atleast_1d(axes).ravel()
    xticks = np.arange(-0.25, 3.51, 0.25)

    for r_i, grp in enumerate(groups):
        ax = axes_list[r_i]
        ax_env = ax.twinx()

        idx_all = grp["trial_indices"]
        idx_inc = grp["incoh_indices"]
        env_trace = grp["mean_env"]

        # Spiking PSTH (solid line + subtle SEM ribbon) on left axis
        rates = spk_hz[idx_all, :, ch_idx]
        mean_r = rates.mean(axis=0)
        sem_r = rates.std(axis=0, ddof=1) / max(np.sqrt(len(idx_all)), 1.0)
        l_spk = ax.plot(
            t,
            mean_r,
            color=spike_color,
            lw=1.7,
            ls="-",
            label=f"{ch_title} firing rate (Hz)",
        )
        ax.fill_between(
            t, mean_r - sem_r, mean_r + sem_r, color=spike_color, alpha=0.15
        )

        # Stimulus envelope (unshaded, crisp dotted line) on right axis
        max_e = float(env_trace.max())
        l_env = ax_env.plot(
            t,
            env_trace,
            color=env_color,
            lw=2.0,
            ls=(0, (1.5, 1.8)),
            label=f"{modality_label} envelope (dotted)",
        )
        ax_env.set_ylim(-0.04 * max_e, max_e * 1.15 if max_e > 0 else 1.0)
        ax_env.set_ylabel(env_unit, color=env_color, fontsize=8.5)
        ax_env.tick_params(axis="y", colors=env_color, labelsize=8)

        ax.axvline(0.0, color="gray", ls="--", lw=0.9, alpha=0.6)
        ax.axvline(3.0, color="gray", ls="--", lw=0.9, alpha=0.6)
        ax.set_xticks(xticks)
        ax.grid(axis="x", color="0.88", lw=0.6, ls="-")
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel("Rate (Hz)", color=spike_color, fontsize=8.5)
        ax.tick_params(axis="y", colors=spike_color, labelsize=8)

        # Incoherent PSTH correlation at +40ms lag
        s0, s1 = stim_slice.start + 2, stim_slice.stop
        if len(idx_inc) > 0 and (s1 > s0) and (s0 >= 2):
            psth_inc = spk_hz[idx_inc, s0:s1, ch_idx].mean(axis=0)
            e_sub = env_trace[s0 - 2 : s1 - 2]
            if np.std(e_sub) > 1e-6 and np.std(psth_inc) > 1e-6:
                r_inc = float(np.corrcoef(e_sub, psth_inc)[0, 1])
            else:
                r_inc = 0.0
        else:
            r_inc = 0.0

        short_sf = grp["stim_file"].replace("_SS.mat", "")
        extra_parts = []
        if grp.get("vowels"):
            extra_parts.append(f"vowel={','.join(grp['vowels'])}")
        if grp.get("n_env"):
            extra_parts.append(f"Env={grp['n_env']}")
        extra = f", {', '.join(extra_parts)}" if extra_parts else ""

        ax.set_title(
            f"{short_sf} — {modality_label} Envelope #{grp['shape_num']}  "
            f"(StimIdx {grp['stim_idxs']}{extra}, n={len(idx_all)} trials | incoh PSTH r@40ms = {r_inc:+.2f})",
            fontsize=9.5,
            loc="left",
            pad=4,
        )

        if r_i == 0:
            lines = l_spk + l_env
            ax.legend(
                lines,
                [ln.get_label() for ln in lines],
                loc="upper right",
                fontsize=8.5,
                framealpha=0.9,
            )

    axes_list[-1].set_xlabel("Time from stimulus onset (s)", fontsize=10)
    fig.suptitle(
        f"{ch_title} vs. All {len(groups)} {modality_label} Envelopes (20 ms bins)",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in fig_formats:
        target = out_path.with_suffix(f".{ext}")
        fig.savefig(target, dpi=150, bbox_inches="tight")
        print(f"  Saved: {target}")
    plt.close(fig)


def run_sense_checks(
    data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    animal_id: str = "F1812_Nala",
    rec_name: str = "Rec_A",
    depth: int | None = None,
    channel_list: list[int] | None = None,
    modality: str = "both",
    plot_led: bool = False,
    led_csv_path: pathlib.Path | None = None,
    tables_dir: pathlib.Path | None = None,
    fig_formats: tuple[str, ...] = ("png",),
) -> list[pathlib.Path]:
    """Execute sense-check stacked envelope PSTH generation across requested channels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[pathlib.Path] = []

    candidates = []
    if depth is not None:
        candidates.append(
            (
                data_dir / f"{animal_id}_Depth{depth}_all_blocks_20ms.npz",
                data_dir / f"{animal_id}_Depth{depth}_all_blocks_trials.csv",
            )
        )
        candidates.append(
            (
                data_dir / f"{animal_id}_D{depth}_all_blocks_20ms.npz",
                data_dir / f"{animal_id}_D{depth}_all_blocks_trials.csv",
            )
        )
    candidates.append(
        (
            data_dir / f"{animal_id}_all_blocks_20ms.npz",
            data_dir / f"{animal_id}_all_blocks_trials.csv",
        )
    )

    npz_path, csv_path = None, None
    for cand_npz, cand_csv in candidates:
        if cand_npz.exists() and cand_csv.exists():
            npz_path, csv_path = cand_npz, cand_csv
            break

    if not npz_path or not csv_path:
        raise FileNotFoundError(
            f"Missing extracted data files for {animal_id} (depth={depth}) in {data_dir}.\n"
            f"Looked for: {[(c[0].name, c[1].name) for c in candidates]}"
        )

    # Optional LED calibration curve
    if plot_led or not (
        (out_dir / "chamber_led_calibration_fit.png").exists()
        or (data_dir / "chamber_led_calibration_fit.png").exists()
    ):
        actual_led_csv = led_csv_path or DEFAULT_LED_CSV
        plot_led_calibration_curve(
            actual_led_csv,
            out_dir / "chamber_led_calibration_fit.png",
            fig_formats=fig_formats,
        )

    depth_info = f" (Depth {depth})" if depth is not None else ""
    print(f"\n=======================================================")
    print(f"Loading data for Sense Check: {animal_id}{depth_info} ({rec_name})")
    print(f"=======================================================")
    data = np.load(npz_path)
    trials_df = pd.read_csv(csv_path)

    envelopes = data["envelopes"]
    visual_voltage = data["visual_voltage"] if "visual_voltage" in data else None
    binned_spikes = data["binned_spikes"]
    t = data["bin_centers_s"]
    bin_edges = data["bin_edges_s"]
    channel_labels = data["channel_labels"]
    bin_size = float(bin_edges[1] - bin_edges[0])

    # Filter by depth if trials_df has 'depth' column and depth is specified
    if depth is not None and "depth" in trials_df.columns:
        d_mask = (trials_df["depth"] == int(depth)).to_numpy()
        if not d_mask.all():
            envelopes = envelopes[d_mask]
            if visual_voltage is not None:
                visual_voltage = visual_voltage[d_mask]
            binned_spikes = binned_spikes[d_mask]
            trials_df = trials_df.iloc[d_mask].reset_index(drop=True)

    spk_hz = binned_spikes.astype(np.float64) / bin_size
    stim_start = int(np.searchsorted(bin_edges, -1e-9))
    stim_end = int(np.searchsorted(bin_edges, 3.0 - 1e-9))
    stim_slice = slice(stim_start, stim_end)

    aud_groups, vis_groups = assign_unique_envelope_shapes(
        trials_df=trials_df,
        envelopes=envelopes,
        visual_voltage=visual_voltage,
        stim_slice=stim_slice,
    )
    print(
        f"Discovered {len(aud_groups)} unique auditory & {len(vis_groups)} visual luminance envelope shapes.\n"
        f"Total trials: {len(trials_df)} across {trials_df['block'].nunique()} sessions.\n"
    )

    # Resolve target channels
    rec_matches = [
        (i, lbl)
        for i, lbl in enumerate(channel_labels)
        if str(lbl).startswith(f"{rec_name}_")
    ]
    if not rec_matches:
        rec_matches = list(enumerate(channel_labels))

    if channel_list is not None:
        target_indices = []
        for ch in channel_list:
            target_indices.append(find_channel_index(channel_labels, rec_name, ch))
    else:
        target_indices = [idx for idx, _ in rec_matches]

    print(
        f"Processing sense-checks for {len(target_indices)} channels (Modality: {modality.upper()}) ...\n"
    )

    act_tables_dir = tables_dir or DATA_TABLES_DIR
    depth_str = f" Depth {depth}" if depth is not None else ""
    depth_tag = f"_Depth{depth}" if depth is not None else ""

    for k, ch_idx in enumerate(target_indices):
        ch_label = str(channel_labels[ch_idx])
        ch_num = int(ch_label.split("_C")[-1]) if "_C" in ch_label else ch_idx + 1
        reg_field = lookup_cortical_field(act_tables_dir, animal_id, rec_name, ch_num)
        reg_str = f" ({reg_field})" if reg_field else ""
        ch_title = f"{animal_id}{depth_str} — {rec_name} Ch {ch_num:02d}{reg_str}"
        t0 = time.perf_counter()

        print(
            f"[{k+1}/{len(target_indices)}] Generating sense-check for {ch_title} ..."
        )

        # 1. Auditory envelopes
        if modality in ("both", "auditory"):
            out_aud = (
                out_dir
                / f"sense_check_{animal_id}{depth_tag}_{ch_label}_auditory_envelopes.png"
            )
            plot_single_channel_stacked_envelopes(
                groups=aud_groups,
                t=t,
                spk_hz=spk_hz,
                ch_idx=ch_idx,
                ch_title=ch_title,
                spike_color="#d95f02",
                modality_label="Auditory",
                env_color="#8c510a",
                env_unit="Aud env",
                stim_slice=stim_slice,
                out_path=out_aud,
                fig_formats=fig_formats,
            )
            for ext in fig_formats:
                generated_paths.append(out_aud.with_suffix(f".{ext}"))

            # Backward-compatible alias for standard F1812_Nala test channels
            if animal_id == "F1812_Nala" and depth is None and ch_num in (26, 5) and "png" in fig_formats:
                legacy_aud = (
                    out_dir / f"sense_check_ch{ch_num:02d}_auditory_envelopes.png"
                )
                if legacy_aud.resolve() != out_aud.resolve() and out_aud.exists():
                    shutil.copy2(out_aud, legacy_aud)

        # 2. Visual luminance envelopes
        if modality in ("both", "visual"):
            out_vis = (
                out_dir
                / f"sense_check_{animal_id}{depth_tag}_{ch_label}_visual_envelopes.png"
            )
            plot_single_channel_stacked_envelopes(
                groups=vis_groups,
                t=t,
                spk_hz=spk_hz,
                ch_idx=ch_idx,
                ch_title=ch_title,
                spike_color="#7b68ee",
                modality_label="Visual Luminance",
                env_color="#1f77b4",
                env_unit="Luminance",
                stim_slice=stim_slice,
                out_path=out_vis,
                fig_formats=fig_formats,
            )
            for ext in fig_formats:
                generated_paths.append(out_vis.with_suffix(f".{ext}"))

            # Backward-compatible alias for standard F1812_Nala test channels
            if animal_id == "F1812_Nala" and depth is None and ch_num in (26, 5) and "png" in fig_formats:
                legacy_vis = (
                    out_dir / f"sense_check_ch{ch_num:02d}_visual_envelopes.png"
                )
                if legacy_vis.resolve() != out_vis.resolve() and out_vis.exists():
                    shutil.copy2(out_vis, legacy_vis)

        elapsed = time.perf_counter() - t0
        print(f"    -> Done in {elapsed:.1f}s")

    print(
        f"\nAll requested sense-checks complete! Generated {len(generated_paths)} figure(s)."
    )
    return generated_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot channel PSTH sense-checks stacked vertically across unique auditory and calibrated visual envelopes."
    )
    parser.add_argument(
        "--animal", type=str, default="F1812_Nala", help="Animal ID (e.g., F1812_Nala)"
    )
    parser.add_argument(
        "--rec", type=str, default="Rec_A", help="Recording stream (Rec_A or Rec_B)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Cortical depth index (e.g. 1, 2, 3, 4 for multi-depth animals like Eclair)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="One or more 1-based channel numbers (e.g., --channels 26 5). If omitted, defaults to 26 and 5.",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Process all available channels in the recording stream",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="both",
        choices=["both", "auditory", "visual"],
        help="Stimulus envelope modality to plot: 'both', 'auditory', or 'visual' (default: 'both')",
    )
    parser.add_argument(
        "--plot-led",
        action="store_true",
        help="Plot the Chamber LED voltage-to-luminance calibration curve",
    )
    parser.add_argument(
        "--led-csv",
        type=pathlib.Path,
        default=DEFAULT_LED_CSV,
        help="Path to Chamber_LED.csv calibration file",
    )
    parser.add_argument(
        "--tables-dir",
        type=pathlib.Path,
        default=DATA_TABLES_DIR,
        help="Path to data tables directory (containing Ferret_Side_Chan_Field.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=DATA_DIR,
        help="Path to extracted data directory (containing {animal}_all_blocks_20ms.npz)",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=SENSE_OUT_DIR,
        help="Output directory for generated sense-check figures (default: output/sense_check_plots directory)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save editable vector PDF figures alongside PNG plots",
    )
    parser.add_argument(
        "--fig-format",
        type=str,
        default="png",
        choices=["png", "pdf", "both"],
        help="Output figure format: 'png' (default), 'pdf' (vector), or 'both'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all_channels:
        channel_list = None
    elif args.channels is not None:
        channel_list = args.channels
    else:
        # Default test channels
        channel_list = [26, 5]

    if args.pdf or args.fig_format == "both":
        fig_formats = ("png", "pdf")
    elif args.fig_format == "pdf":
        fig_formats = ("pdf",)
    else:
        fig_formats = ("png",)

    run_sense_checks(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        animal_id=args.animal,
        rec_name=args.rec,
        depth=args.depth,
        channel_list=channel_list,
        modality=args.modality,
        plot_led=args.plot_led,
        led_csv_path=args.led_csv,
        tables_dir=args.tables_dir,
        fig_formats=fig_formats,
    )


if __name__ == "__main__":
    main()
