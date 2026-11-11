"""Extract trial-aligned stimulus envelopes and spike times across ferret recording blocks.

Supports all animals (F1812_Nala, F1813_Rajah, F1901_Crumble, F1902_Eclair, F2001_Gnocchi):
  - Automatically matches probe recording files (*Rec_A*, *Rec_B*), accommodating variant
    suffixes such as Rec_A_2, Rec_A_3, Rec_A_6, Rec_B_2, Rec_B_3, Rec_B_4.
  - Handles single-depth animals as well as multi-depth segmentation (F1902_Eclair via Elair_Reps_GLM.csv).
  - Automatically excludes quiet blocks (for Eclair) and empty/corrupted files.
  2. Loads the block's stimulus set (AVPilot*_SS.mat), applies the 10 ms cosine ramp
     (`envelopeJumbo`), scales audio by `10^(-Atten/20)` (using the exact generative
     envelope from the matching coherent condition, or Hilbert envelope) and converts
     48.8 kHz visual LED voltage (`vis * vScale + vBase`) to chamber luminance using a
     monotonic PCHIP spline fitted to `Jumbo_LED.csv`. Pre/post stimulus intervals
     (`[-0.4s, 0.0s)` and `[3.0s, 3.5s)`) are 0.
  3. Bins both envelopes into `(n_trials, n_bins, 2)` (`[..., 0]` = auditory amplitude,
     `[..., 1]` = visual luminance) and aligns/bins spike times on `[-0.4s, +3.5s]`
     at 20 ms resolution (`n_bins = 195`).
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
from scipy.interpolate import PchipInterpolator
from tqdm.auto import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR if (SCRIPT_DIR / "data_tables").exists() or (SCRIPT_DIR / "_data_tables").exists()
    else SCRIPT_DIR.parent
)
DEFAULT_DATA_ROOT = pathlib.Path(r"E:\AV_passive_extracted_spikes")
DEFAULT_SESSIONS_DIR = DEFAULT_DATA_ROOT / "F1812_Nala"

DEFAULT_STIM_DIR = (
    (REPO_ROOT / "stimuli") if (REPO_ROOT / "stimuli").exists()
    else ((REPO_ROOT / "_stimuli") if (REPO_ROOT / "_stimuli").exists()
    else (SCRIPT_DIR / "stimuli" if (SCRIPT_DIR / "stimuli").exists() else SCRIPT_DIR / "_stimuli"))
)

DATA_TABLES_DIR = (
    (REPO_ROOT / "data_tables") if (REPO_ROOT / "data_tables").exists()
    else ((REPO_ROOT / "_data_tables") if (REPO_ROOT / "_data_tables").exists()
    else (SCRIPT_DIR / "data_tables" if (SCRIPT_DIR / "data_tables").exists() else SCRIPT_DIR / "_data_tables"))
)
DEFAULT_LED_CSV = DATA_TABLES_DIR / "Chamber_LED.csv"

def _find_dir(preferred: list[pathlib.Path], fallback: pathlib.Path) -> pathlib.Path:
    for p in preferred:
        if p.exists():
            return p
    return fallback

DEFAULT_OUT_DIR = _find_dir(
    [REPO_ROOT / "output", REPO_ROOT / "extracted_data", SCRIPT_DIR / "extracted_data"],
    REPO_ROOT / "output",
)


def natural_block_key(path: pathlib.Path) -> List[object]:
    """Sort Block_J1-2 before Block_J1-10 chronologically."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path.name)]


def build_led_luminance_model(led_csv_path: pathlib.Path) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a shape-preserving monotonic C1 spline (PCHIP) mapping LED Voltage -> Luminance.

    `Jumbo_LED.csv` has 0 luminance up to 2.0 V, a turn-on knee around 2.5-2.7 V, and a
    steep quasi-linear rise up to 4.0 V (765 luminance units). PCHIP passes through every
    measured point without overshoot, stays >= 0, and has a continuous first derivative.
    """
    df_led = pd.read_csv(led_csv_path).sort_values("Voltage")
    v_pts = df_led["Voltage"].to_numpy(dtype=np.float64)
    l_pts = df_led["Luminance"].to_numpy(dtype=np.float64)
    if v_pts[0] > 0.0:
        v_pts = np.concatenate([[0.0], v_pts])
        l_pts = np.concatenate([[0.0], l_pts])
    spline = PchipInterpolator(v_pts, l_pts, extrapolate=True)

    def voltage_to_luminance(voltage: np.ndarray) -> np.ndarray:
        v = np.asarray(voltage, dtype=np.float64)
        lum = np.where(v <= 2.0, 0.0, spline(v))
        return np.maximum(lum, 0.0)

    return voltage_to_luminance


def envelope_jumbo(signal: np.ndarray, samples: int) -> np.ndarray:
    """Port of MATLAB `envelopeJumbo(signal, samples)`: cosine fade-in/fade-out ramp."""
    out = np.array(signal, dtype=np.float64, copy=True)
    x = -np.pi + np.arange(samples, dtype=np.float64) * (np.pi / samples)
    env_func = np.cos(x) / 2.0 + 0.5
    out[:samples] *= env_func
    out[-samples:] *= env_func[::-1]
    return out


def bin_waveform(waveform: np.ndarray, fs: float, bin_size: float, duration: float) -> np.ndarray:
    """Average a high-rate waveform into exact `bin_size` bins over `[0, duration)`."""
    n_stim_bins = int(round(duration / bin_size))
    binned = np.empty(n_stim_bins, dtype=np.float64)
    for b in range(n_stim_bins):
        s0 = int(round(b * bin_size * fs))
        s1 = int(round((b + 1) * bin_size * fs))
        binned[b] = waveform[s0:s1].mean()
    return binned


def preload_stimulus_sets(
    stim_dir: pathlib.Path,
    stim_files: Iterable[str],
    voltage_to_luminance: Callable[[np.ndarray], np.ndarray],
    vis_scale_combos: Iterable[Tuple[float, float]],
    bin_size: float = 0.02,
    stim_duration: float = 3.0,
    audio_mode: str = "generative",
) -> Dict[Tuple[str, int], dict]:
    """Precompute 20 ms base audio envelopes & calibrated visual luminance traces per (StimFile, StimIdx).

    Crucially, the nonlinear `voltage -> luminance` calibration is applied at the native
    48,828 Hz stimulus rate (`vis_ramped * vScale + vBase -> luminance`) BEFORE averaging
    into 20 ms bins.
    """
    unique_combos = sorted(set((float(vb), float(vs)) for vb, vs in vis_scale_combos))
    cache: Dict[Tuple[str, int], dict] = {}

    for stim_file in sorted(set(stim_files)):
        mat_path = stim_dir / stim_file
        if not mat_path.exists():
            raise FileNotFoundError(f"Stimulus file not found: {mat_path}")
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        stim_array = mat["stimulus"]

        for idx_zero, st in enumerate(stim_array):
            stim_idx = idx_zero + 1  # MATLAB 1-based StimIdx
            fs = float(st.fStim)
            ramp_samples = int(np.ceil(10e-3 * fs))
            st_aud = np.asarray(st.audio, dtype=np.float64)

            if audio_mode == "generative":
                coh_matches = [
                    s2
                    for s2 in stim_array
                    if int(getattr(s2, "coh", -1)) == 1 and np.array_equal(st_aud, np.asarray(s2.audio))
                ]
                if len(coh_matches) != 1:
                    raise ValueError(
                        f"Expected 1 coherent audio match for {stim_file} idx={stim_idx}, found {len(coh_matches)}"
                    )
                aud_env_raw = np.asarray(coh_matches[0].visual, dtype=np.float64)
                aud_env_ramped = envelope_jumbo(aud_env_raw, ramp_samples)
                aud_binned_base = bin_waveform(aud_env_ramped, fs=fs, bin_size=bin_size, duration=stim_duration)
            elif audio_mode == "hilbert":
                aud_ramped = envelope_jumbo(st_aud, ramp_samples)
                aud_hilbert_env = np.abs(scipy.signal.hilbert(aud_ramped))
                aud_binned_base = bin_waveform(aud_hilbert_env, fs=fs, bin_size=bin_size, duration=stim_duration)
            else:
                raise ValueError(f"Unknown audio_mode={audio_mode!r}; expected 'generative' or 'hilbert'.")

            # Visual: 48.8 kHz cosine ramp -> scale to Voltage -> convert to Luminance -> 20 ms bin average
            vis_ramped = envelope_jumbo(np.asarray(st.visual, dtype=np.float64), ramp_samples)
            vis_binned_base = bin_waveform(vis_ramped, fs=fs, bin_size=bin_size, duration=stim_duration)

            vis_lum_by_combo: Dict[Tuple[float, float], np.ndarray] = {}
            for vb, vs in unique_combos:
                vis_volt_48k = vis_ramped * vs + vb
                vis_lum_48k = voltage_to_luminance(vis_volt_48k)
                vis_lum_by_combo[(vb, vs)] = bin_waveform(
                    vis_lum_48k, fs=fs, bin_size=bin_size, duration=stim_duration
                )

            cache[(stim_file, stim_idx)] = {
                "aud_base": aud_binned_base,
                "vis_base": vis_binned_base,
                "vis_lum_by_combo": vis_lum_by_combo,
                "vowel": str(getattr(st, "vowel", "")),
                "fStim": fs,
                "coh_mat": int(getattr(st, "coh", -1)),
                "nEnv_mat": int(getattr(st, "nEnv", -1)),
            }
    return cache


def find_recording_mat(block_dir: pathlib.Path, rec_tag: str) -> pathlib.Path:
    """Locate `*<rec_tag>_spiketimes.mat` inside `block_dir`, with flexible probe tag matching."""
    # 1. Exact match
    matches = sorted(block_dir.glob(f"*{rec_tag}_spiketimes.mat"))
    if len(matches) == 1:
        return matches[0]

    # 2. Rec_B / Rec_B_2 standard alias
    if not matches and rec_tag in ("Rec_B", "Rec_B_2"):
        alt = "Rec_B_2" if rec_tag == "Rec_B" else "Rec_B"
        matches = sorted(block_dir.glob(f"*{alt}_spiketimes.mat"))
        if len(matches) == 1:
            return matches[0]

    # 3. Flexible probe suffix match (e.g. Rec_A_2, Rec_A_3, Rec_A_6, Rec_B_3, Rec_B_4)
    if not matches:
        if "A" in rec_tag.upper():
            matches = sorted(block_dir.glob("*Rec_A*_spiketimes.mat"))
        elif "B" in rec_tag.upper():
            matches = sorted(block_dir.glob("*Rec_B*_spiketimes.mat"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 '*{rec_tag}_spiketimes.mat' in {block_dir}, found {len(matches)}: {matches}"
        )
    return matches[0]


def extract_sessions(
    sessions_dir: pathlib.Path = DEFAULT_SESSIONS_DIR,
    stim_dir: pathlib.Path = DEFAULT_STIM_DIR,
    led_csv_path: pathlib.Path = DEFAULT_LED_CSV,
    blocks: Sequence[str] | None = None,
    recordings: Sequence[str] = ("Rec_A", "Rec_B_2"),
    depth: int | None = None,
    depth_table_path: pathlib.Path | None = None,
    t_pre: float = -0.4,
    t_post: float = 3.5,
    stim_duration: float = 3.0,
    bin_size: float = 0.02,
    audio_mode: str = "generative",
) -> dict:
    """Loop over animal blocks and assemble envelopes, binned spikes, relative spike times, and trial info.

    Optionally filters by cortical recording depth if depth_table_path (e.g. Elair_Reps_GLM.csv) is provided.
    """
    if t_pre > 0.0 or t_post < stim_duration:
        raise ValueError(f"Expected t_pre <= 0 and t_post >= {stim_duration}, got [{t_pre}, {t_post}].")

    voltage_to_luminance = build_led_luminance_model(led_csv_path)

    # Optional depth / rep filtering (only applied to multi-depth animal F1902_Eclair)
    reps_df = None
    if depth_table_path and pathlib.Path(depth_table_path).exists() and "eclair" in sessions_dir.name.lower():
        reps_df = pd.read_csv(depth_table_path)
        reps_df["_Start"] = pd.to_datetime(reps_df["Start_date"], format="%d/%m/%Y")
        reps_df["_End"] = pd.to_datetime(reps_df["End_date"], format="%d/%m/%Y")

    n_bins = int(round((t_post - t_pre) / bin_size))
    bin_edges = np.linspace(t_pre, t_post, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    stim_start_bin = int(round((0.0 - t_pre) / bin_size))
    n_stim_bins = int(round(stim_duration / bin_size))
    stim_end_bin = stim_start_bin + n_stim_bins

    all_block_dirs = sorted(
        [d for d in sessions_dir.glob("Block_*") if d.is_dir()],
        key=natural_block_key,
    )
    if blocks:
        wanted = set(blocks)
        all_block_dirs = [d for d in all_block_dirs if d.name in wanted]
    if not all_block_dirs:
        raise ValueError(f"No matching Block_* directories found in {sessions_dir} (filter={blocks}).")

    # First pass: validate complete blocks (.txt + requested *_spiketimes.mat), filter by depth/dates, and discover StimFiles
    valid_block_dirs: List[pathlib.Path] = []
    block_tables: Dict[str, pd.DataFrame] = {}
    block_metadata: Dict[str, dict] = {}
    needed_stim_files = set()
    vis_scale_combos = set()
    for bdir in all_block_dirs:
        txts = list(bdir.glob("*.txt"))
        if len(txts) != 1:
            print(f"[SKIP] {bdir.name}: expected 1 .txt trial table, found {len(txts)}.")
            continue

        # Exclude quiet sound level blocks (constraint for Eclair: e.g. 'Passive_Quiet_Eclair')
        if "quiet" in txts[0].name.lower():
            print(f"[SKIP] {bdir.name}: quiet sound level block ({txts[0].name}).")
            continue

        # Extract date from trial table filename (e.g., '2022_04_11 level81...' or '2022-04-11...')
        m_date = re.search(r"(\d{4})[_-](\d{2})[_-](\d{2})", txts[0].name)
        block_date_str = f"{m_date.group(1)}-{m_date.group(2)}-{m_date.group(3)}" if m_date else ""

        # Depth table verification if provided
        block_depth = 1
        rec_a_valid = 1
        rec_b_valid = 1
        if reps_df is not None:
            if not block_date_str:
                print(f"[SKIP] {bdir.name}: cannot extract date for depth lookup from {txts[0].name}.")
                continue
            b_dt = pd.to_datetime(block_date_str)
            sub_rep = reps_df[(reps_df["_Start"] <= b_dt) & (b_dt <= reps_df["_End"])]
            if len(sub_rep) == 0:
                # Excluded: date outside configured depth recording ranges
                continue
            rep_row = sub_rep.iloc[0]
            block_depth = int(rep_row["Depth"])
            rec_a_valid = int(rep_row.get("Rec A", 1))
            rec_b_valid = int(rep_row.get("Rec B", 1))
            if depth is not None and block_depth != int(depth):
                # Excluded: block is at a different cortical depth
                continue

        missing_recs = []
        for r in recordings:
            try:
                mat_f = find_recording_mat(bdir, r)
                if mat_f.stat().st_size == 0:
                    missing_recs.append(f"{r} (0 bytes)")
            except FileNotFoundError:
                missing_recs.append(r)
        if missing_recs:
            print(f"[SKIP] {bdir.name}: missing or empty spiketimes.mat for {missing_recs}.")
            continue

        df = pd.read_csv(txts[0], sep="\t").dropna(axis=1, how="all")
        valid_block_dirs.append(bdir)
        block_tables[bdir.name] = df
        block_metadata[bdir.name] = {
            "session_date": block_date_str,
            "depth": block_depth,
            "valid_rec_a": rec_a_valid,
            "valid_rec_b": rec_b_valid,
        }
        needed_stim_files.update(df["StimFile"].unique().tolist())
        for vb, vs in df[["vBase", "vScale"]].drop_duplicates().itertuples(index=False):
            vis_scale_combos.add((float(vb), float(vs)))

    if not valid_block_dirs:
        raise ValueError(f"No complete blocks found to extract in {sessions_dir} (depth={depth}).")

    print(
        f"Preloading stimulus sets {sorted(needed_stim_files)} from '{stim_dir}' "
        f"(audio_mode='{audio_mode}', LED calibration='{led_csv_path.name}')..."
    )
    stim_cache = preload_stimulus_sets(
        stim_dir=stim_dir,
        stim_files=needed_stim_files,
        voltage_to_luminance=voltage_to_luminance,
        vis_scale_combos=vis_scale_combos,
        bin_size=bin_size,
        stim_duration=stim_duration,
        audio_mode=audio_mode,
    )

    all_envelopes: List[np.ndarray] = []
    all_vis_voltage: List[np.ndarray] = []
    all_binned_spikes: List[np.ndarray] = []
    all_trial_rows: List[pd.DataFrame] = []

    flat_spike_trial: List[np.ndarray] = []
    flat_spike_chan: List[np.ndarray] = []
    flat_spike_time: List[np.ndarray] = []

    channel_labels: List[str] | None = None
    global_trial_offset = 0

    for bdir in tqdm(valid_block_dirs, desc="Extracting blocks"):
        df_txt = block_tables[bdir.name].copy()

        rec_data = []
        common_trial_ids = set(df_txt["Trial"].astype(int).tolist())
        for rec_tag in recordings:
            mat_path = find_recording_mat(bdir, rec_tag)
            smat = scipy.io.loadmat(
                str(mat_path),
                squeeze_me=True,
                struct_as_record=False,
                variable_names=["matched_times", "trial_idx", "spike_times"],
            )
            t_idx = np.atleast_1d(smat["trial_idx"]).astype(int)
            m_times = np.atleast_1d(smat["matched_times"]).astype(np.float64)
            spk_cells = [np.sort(np.atleast_1d(ch).astype(np.float64)) for ch in smat["spike_times"]]

            rec_data.append(
                {
                    "rec_tag": rec_tag,
                    "trial_map": dict(zip(t_idx.tolist(), m_times.tolist())),
                    "spike_times": spk_cells,
                }
            )
            common_trial_ids &= set(t_idx.tolist())

        df_matched = df_txt[df_txt["Trial"].astype(int).isin(common_trial_ids)].sort_values("Trial").reset_index(drop=True)
        n_block_trials = len(df_matched)

        block_labels = [
            f"{r['rec_tag']}_C{ch_i + 1:02d}"
            for r in rec_data
            for ch_i in range(len(r["spike_times"]))
        ]
        if channel_labels is None:
            channel_labels = block_labels
        elif channel_labels != block_labels:
            raise ValueError(f"Channel mismatch in {bdir.name}: expected {len(channel_labels)}, got {len(block_labels)}")

        n_channels = len(channel_labels)
        flat_channels = [ch_arr for r in rec_data for ch_arr in r["spike_times"]]

        matched_onsets = np.array(
            [rec_data[0]["trial_map"][int(tr)] for tr in df_matched["Trial"]],
            dtype=np.float64,
        )

        # 1. Construct (n_block_trials, n_bins, 2) stimulus envelopes: [audio_env, visual_luminance]
        block_env = np.zeros((n_block_trials, n_bins, 2), dtype=np.float64)
        block_volt = np.zeros((n_block_trials, n_bins), dtype=np.float64)
        vowels = []
        for row_i, row in enumerate(df_matched.itertuples(index=False)):
            key = (str(row.StimFile), int(row.StimIdx))
            base = stim_cache[key]
            atten_gain = 10.0 ** (-float(row.Atten) / 20.0)
            vb, vs = float(row.vBase), float(row.vScale)

            # Channel 0: Auditory amplitude envelope (0 pre/post stimulus)
            block_env[row_i, stim_start_bin:stim_end_bin, 0] = base["aud_base"] * atten_gain
            # Channel 1: Calibrated Visual LED Luminance (0 pre/post stimulus)
            block_env[row_i, stim_start_bin:stim_end_bin, 1] = base["vis_lum_by_combo"][(vb, vs)]
            # Also retain 20ms visual voltage for reference
            block_volt[row_i, stim_start_bin:stim_end_bin] = base["vis_base"] * vs + vb
            vowels.append(base["vowel"])

        # 2. Extract trial-relative spike times in [t_pre, t_post] & bin at 20 ms
        block_counts = np.zeros((n_block_trials, n_bins, n_channels), dtype=np.int32)
        for row_i, onset in enumerate(matched_onsets):
            win_start = onset + t_pre
            win_end = onset + t_post
            g_trial = global_trial_offset + row_i

            for ch_i, spk_arr in enumerate(flat_channels):
                i0 = int(np.searchsorted(spk_arr, win_start, side="left"))
                i1 = int(np.searchsorted(spk_arr, win_end, side="right"))
                if i1 > i0:
                    rel_spk = spk_arr[i0:i1] - onset
                    counts, _ = np.histogram(rel_spk, bins=bin_edges)
                    block_counts[row_i, :, ch_i] = counts

                    flat_spike_trial.append(np.full(rel_spk.shape, g_trial, dtype=np.int32))
                    flat_spike_chan.append(np.full(rel_spk.shape, ch_i, dtype=np.int16))
                    flat_spike_time.append(rel_spk)

        df_matched.insert(0, "global_trial_idx", np.arange(global_trial_offset, global_trial_offset + n_block_trials))
        df_matched.insert(1, "block", bdir.name)
        df_matched["Vowel"] = vowels
        df_matched["matched_onset_s"] = matched_onsets

        meta = block_metadata.get(bdir.name, {})
        df_matched["session_date"] = meta.get("session_date", "")
        df_matched["depth"] = meta.get("depth", 1)
        df_matched["valid_rec_a"] = meta.get("valid_rec_a", 1)
        df_matched["valid_rec_b"] = meta.get("valid_rec_b", 1)

        all_envelopes.append(block_env)
        all_vis_voltage.append(block_volt)
        all_binned_spikes.append(block_counts)
        all_trial_rows.append(df_matched)
        global_trial_offset += n_block_trials

        print(
            f"  {bdir.name}: {n_block_trials} trials | StimFile={df_matched['StimFile'].unique().tolist()} "
            f"| channels={n_channels} | total binned spikes={int(block_counts.sum()):,}"
        )

    envelopes = np.concatenate(all_envelopes, axis=0)
    visual_voltage = np.concatenate(all_vis_voltage, axis=0)
    binned_spikes = np.concatenate(all_binned_spikes, axis=0)
    trials_df = pd.concat(all_trial_rows, ignore_index=True)

    spike_trial_idx = np.concatenate(flat_spike_trial) if flat_spike_trial else np.empty(0, dtype=np.int32)
    spike_channel_idx = np.concatenate(flat_spike_chan) if flat_spike_chan else np.empty(0, dtype=np.int16)
    spike_rel_times_s = np.concatenate(flat_spike_time) if flat_spike_time else np.empty(0, dtype=np.float64)

    return {
        "envelopes": envelopes,  # (n_trials, n_bins, 2): [..., 0]=audio_env, [..., 1]=visual_luminance
        "visual_voltage": visual_voltage,  # (n_trials, n_bins): raw LED voltage (V)
        "binned_spikes": binned_spikes,  # (n_trials, n_bins, n_channels)
        "bin_edges_s": bin_edges,
        "bin_centers_s": bin_centers,
        "channel_labels": np.array(channel_labels),
        "trials_df": trials_df,
        "spike_trial_idx": spike_trial_idx,
        "spike_channel_idx": spike_channel_idx,
        "spike_rel_times_s": spike_rel_times_s,
    }


def save_sanity_check_plot(result: dict, out_png: pathlib.Path) -> None:
    """Save a quick visual check of coherent vs incoherent envelopes and population PSTH."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = result["bin_centers_s"]
    env = result["envelopes"]
    spk = result["binned_spikes"]
    df = result["trials_df"]
    bin_size = float(result["bin_edges_s"][1] - result["bin_edges_s"][0])

    coh_idx = int(df.index[df["Coherence"] == 1][0])
    inc_idx = int(df.index[df["Coherence"] == 0][0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

    r_coh = df.iloc[coh_idx]
    ax0 = axes[0]
    ax0_r = ax0.twinx()
    l1 = ax0.plot(t, env[coh_idx, :, 0], color="tab:orange", lw=1.8, label="Audio amplitude env")
    l2 = ax0_r.plot(t, env[coh_idx, :, 1], color="tab:blue", lw=1.8, linestyle="--", label="Visual luminance")
    ax0.set_ylabel("Audio env", color="tab:orange")
    ax0_r.set_ylabel("Luminance", color="tab:blue")
    ax0.set_title(
        f"Coherent Trial #{coh_idx} ({r_coh['block']}, {r_coh['StimFile']} idx={r_coh['StimIdx']}, "
        f"Atten={r_coh['Atten']}dB, vBase={r_coh['vBase']}, vScale={r_coh['vScale']})"
    )
    lines = l1 + l2
    ax0.legend(lines, [l.get_label() for l in lines], loc="upper right")

    r_inc = df.iloc[inc_idx]
    ax1 = axes[1]
    ax1_r = ax1.twinx()
    ax1.plot(t, env[inc_idx, :, 0], color="tab:orange", lw=1.8, label="Audio amplitude env")
    ax1_r.plot(t, env[inc_idx, :, 1], color="tab:blue", lw=1.8, linestyle="--", label="Visual luminance")
    ax1.set_ylabel("Audio env", color="tab:orange")
    ax1_r.set_ylabel("Luminance", color="tab:blue")
    ax1.set_title(
        f"Incoherent Trial #{inc_idx} ({r_inc['block']}, {r_inc['StimFile']} idx={r_inc['StimIdx']}, "
        f"Atten={r_inc['Atten']}dB, vBase={r_inc['vBase']}, vScale={r_inc['vScale']})"
    )

    mean_psth_hz = spk.mean(axis=(0, 2)) / bin_size
    axes[2].plot(t, mean_psth_hz, color="black", lw=1.8)
    axes[2].axvline(0.0, color="0.35", ls=":", lw=1.2, label="Stim onset (0.0s)")
    axes[2].axvline(3.0, color="0.35", ls="--", lw=1.2, label="Stim offset (3.0s)")
    axes[2].set_ylabel("Mean rate (Hz)")
    axes[2].set_xlabel("Time from stimulus onset (s)")
    axes[2].set_title(f"Grand-Average PSTH ({len(df)} trials x {spk.shape[2]} channels, 20 ms bins)")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract animal stimulus envelopes and spike times across recording blocks.")
    parser.add_argument("--animal", type=str, default="F1812_Nala", help="Animal ID folder (default: F1812_Nala)")
    parser.add_argument("--sessions-dir", type=pathlib.Path, default=None, help="Path to animal block folders")
    parser.add_argument("--stim-dir", type=pathlib.Path, default=DEFAULT_STIM_DIR)
    parser.add_argument("--led-csv", type=pathlib.Path, default=DEFAULT_LED_CSV)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--blocks",
        nargs="*",
        default=None,
        help="Optional subset of block folder names (e.g. --blocks Block_J1-2). Omit to process all blocks.",
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=["Rec_A", "Rec_B_2"],
        help="Recording tags to include per block (default: Rec_A Rec_B_2).",
    )
    parser.add_argument("--bin-size", type=float, default=0.02, help="Bin width in seconds (default: 0.02).")
    parser.add_argument(
        "--t-pre",
        type=float,
        default=-0.4,
        help="Window start rel. to stim onset in s (default: -0.4, matching peristimulus spike cleaning cutoff).",
    )
    parser.add_argument("--t-post", type=float, default=3.5, help="Window end rel. to stim onset in s (default: 3.5).")
    parser.add_argument(
        "--audio-mode",
        choices=["generative", "hilbert"],
        default="generative",
        help="Use exact generative [0,1] envelope from matching coherent condition ('generative', default) or Hilbert envelope ('hilbert').",
    )
    parser.add_argument("--plot-check", action="store_true", help="Save a sanity-check plot of envelopes & PSTH.")
    args = parser.parse_args()

    actual_sessions_dir = args.sessions_dir or (DEFAULT_DATA_ROOT / args.animal)

    result = extract_sessions(
        sessions_dir=actual_sessions_dir,
        stim_dir=args.stim_dir,
        led_csv_path=args.led_csv,
        blocks=args.blocks,
        recordings=args.recordings,
        t_pre=args.t_pre,
        t_post=args.t_post,
        bin_size=args.bin_size,
        audio_mode=args.audio_mode,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{'_'.join(args.blocks)}" if args.blocks else "_all_blocks"
    bin_ms = int(round(args.bin_size * 1000))
    npz_path = args.out_dir / f"{args.animal}{suffix}_{bin_ms}ms.npz"
    csv_path = args.out_dir / f"{args.animal}{suffix}_trials.csv"

    np.savez_compressed(
        npz_path,
        envelopes=result["envelopes"],
        visual_voltage=result["visual_voltage"],
        binned_spikes=result["binned_spikes"],
        bin_edges_s=result["bin_edges_s"],
        bin_centers_s=result["bin_centers_s"],
        channel_labels=result["channel_labels"],
        spike_trial_idx=result["spike_trial_idx"],
        spike_channel_idx=result["spike_channel_idx"],
        spike_rel_times_s=result["spike_rel_times_s"],
    )
    result["trials_df"].to_csv(csv_path, index=False)

    print("\n=== Extraction Summary ===")
    print(f"Envelopes shape:     {result['envelopes'].shape}  (n_trials, n_bins, 2 -> [audio_env, visual_luminance])")
    print(f"Binned spikes shape: {result['binned_spikes'].shape}  (n_trials, n_bins, n_channels)")
    print(f"Relative spikes:     {len(result['spike_rel_times_s']):,} total spikes in [{args.t_pre}s, {args.t_post}s]")
    print(f"Saved arrays to:     {npz_path}")
    print(f"Saved trial info to: {csv_path}")

    if args.plot_check:
        png_path = args.out_dir / f"sanity_check{suffix}.png"
        save_sanity_check_plot(result, png_path)
        print(f"Saved check plot to: {png_path}")


if __name__ == "__main__":
    main()
