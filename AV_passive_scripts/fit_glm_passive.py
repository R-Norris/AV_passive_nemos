"""Publication-Standard Passive Audiovisual NeMoS Poisson GLM Pipeline (`fit_glm_passive.py`).

Designed for publication-ready Methods & population scaling across multiple neurons and animals.

Key Methodological Design:
--------------------------
1. **Single Unified Model (`RaisedCosineLog` 0-200 ms Causal Temporal-Filter Poisson GLM)**:
   Every panel, parameter estimate, temporal kernel, and held-out ablation comes from ONE unified
   model, avoiding contradictions between separate 1-lag and multi-lag fits and accommodating
   arbitrary neural latencies (0-180 ms) across brain regions.

2. **Orthogonal, Unconfounded Predictor Decomposition**:
   - **Acoustic Dynamics (3 groups, 4 basis funcs each, 0-200 ms)**:
       `audio_env` (level), `d_audio_pos` (+dA/dt rising onsets), `d_audio_neg` (-dA/dt offsets).
   - **Visual Calibrated Luminance Dynamics (3 groups, 4 basis funcs each, 0-200 ms)**:
       Peak-normalized Jumbo-LED luminance `visual_lum` (L / L_max in [0, 1]),
       `d_lum_pos` (+dL/dt flash onsets), `d_lum_neg` (-dL/dt trough dips).
   - **Multisensory Synchrony & Coherence (3 groups: 2 phasic 0-200 ms + 1 tonic [0, 3s) contrast)**:
       * `av_coincidence`  : (`+dA/dt * +dL/dt`, 4 basis funcs, 0-200 ms), nonzero ONLY at synchronous
                             auditory + visual rising onsets.
       * `av_offset_coinc` : (`-dA/dt * -dL/dt`, 4 basis funcs, 0-200 ms), nonzero ONLY at synchronous
                             auditory + visual falling offsets.
       * `av_coherence`    : Centered [-0.5, +0.5] tonic boxcar on [0, 3s) for Coherent vs Incoherent
                             streams, capturing sustained cross-modal state / LFP entrainment effects.
     - beta > 0 : Super-additive multisensory enhancement (phasic burst or sustained tonic gain)
     - beta < 0 : Sub-additive saturation / divisive normalization / suppression
     - beta = 0 : Purely independent log-additive A and V responses
   - **Orthogonal Stimulus-Epoch Contrasts on [0, 3s) (decoupled from silence adaptation)**:
       * `stim_sustained`: [0, 3s) boxcar vs pre/post silence (absorbs tonic adaptation / mean step)
       * `vowel_identity`: Centered [-0.5, +0.5] contrast for Vowel /u/ vs /a/
       * `vowel_f0`      : Centered [-0.5, +0.5] contrast for F0 195 Hz vs 175 Hz
       * `led_regime`    : Orthogonal 2x2 LED pedestal `vBase` (2.0V vs 1.0V) & ceiling `vMax`
                           (4.0V vs 3.5V), including both tonic shifts and flash-gain interactions
                           (`+dL/dt * c_vbase`, `+dL/dt * c_vmax`) to test invariance vs scaling.
   - **Dynamic Per-Neuron Nuisance Controls (auto-adapting to observed sessions & stimsets)**:
       * `envelope_pair`   : Sum-to-zero deviation contrasts across whichever (StimSet x Envelope)
                             pairs (up to 12) were presented on QC-valid trials of that neuron.
       * `session_baseline`: Recording block indicators absorbing cross-session firing-rate drift.

3. **Two-Stage Robust Criterion for "Which Factors Impact Firing?"**:
   - **Primary Gate (Predictive Necessity — 5-Fold CV Held-Out Leave-One-Factor-Out Delta-LL)**:
     Stratified 5-fold cross-validation across trials predicts every QC-valid trial strictly
     out-of-sample. Dropping factor group `g` yields per-trial held-out log-likelihood loss
     `Delta-LL_g` (millibits/spike), tested via one-sided paired Wilcoxon signed-rank test across
     all valid trials with Benjamini-Hochberg FDR (`q_lofo < 0.05`).
   - **Secondary Gate & Directionality (Trial-Clustered Huber-White Sandwich Inference)**:
     Full-sample fit provides directional effect sizes (`net_beta`, rate multiplier `exp(beta)`,
     peak latency `ms`), 95% trial-clustered CIs, and omnibus Group Wald `q_wald < 0.05`.
   - A factor is classified as `drives_firing = True` if it passes BOTH `q_lofo < 0.05`
     (genuinely improves unseen-trial spike prediction) AND `q_wald < 0.05`.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
import numpy as np
import pandas as pd
from plot_channel_diagnostics_auto import (
    compute_session_qc,
    compute_trial_qc,
    find_channel_index,
)
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold

import nemos as nmo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", write_through=True)
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = (
    SCRIPT_DIR
    if (SCRIPT_DIR / "data_tables").exists() or (SCRIPT_DIR / "_data_tables").exists()
    else SCRIPT_DIR.parent
)

DATA_TABLES_DIR = (
    (REPO_ROOT / "data_tables")
    if (REPO_ROOT / "data_tables").exists()
    else (
        (REPO_ROOT / "_data_tables")
        if (REPO_ROOT / "_data_tables").exists()
        else (
            SCRIPT_DIR / "data_tables"
            if (SCRIPT_DIR / "data_tables").exists()
            else SCRIPT_DIR / "_data_tables"
        )
    )
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
DIAG_DIR = _find_dir(
    [DATA_DIR / "diagnostics", SCRIPT_DIR / "extracted_data" / "diagnostics"],
    DATA_DIR / "diagnostics",
)
CLEAN_OUT_DIR = _find_dir(
    [DATA_DIR / "clean_glm", SCRIPT_DIR / "extracted_data" / "clean_glm"],
    DATA_DIR / "clean_glm",
)

# Time-varying continuous groups convolved with the 0-200 ms RaisedCosineLog basis
CONT_KEYS = [
    "audio_env",
    "d_audio_pos",
    "d_audio_neg",
    "visual_lum",
    "d_lum_pos",
    "d_lum_neg",
    "av_coincidence",
    "av_offset_coinc",
]

FACTOR_LABELS = {
    "audio_env": "Audio Env A",
    "d_audio_pos": "+dAudio/dt (rising)",
    "d_audio_neg": "-dAudio/dt (falling)",
    "visual_lum": "LED Lum (L/Lmax)",
    "d_lum_pos": "+dLum/dt (brighten)",
    "d_lum_neg": "-dLum/dt (dimming)",
    "av_coincidence": "AV Onset Coinc (+dA×+dL)",
    "av_offset_coinc": "AV Offset Coinc (-dA×-dL)",
    "stim_sustained": "Sustained Stim (0-3s vs sil.)",
    "av_coherence": "AV Coherence (Coh vs Inc)",
    "vowel_identity": "Vowel (/u/ vs /a/)",
    "vowel_f0": "Vowel F0 (195 vs 175 Hz)",
    "led_regime": "LED Regime (vBase, vMax)",
    "envelope_pair": "StimSet Pair (control)",
    "session_baseline": "Session Drift (control)",
}

FACTOR_COLORS = {
    # Auditory factors: tones of yellow / orange / brown
    "audio_env": "#d95f02",  # Warm rich dark orange
    "d_audio_pos": "#e69f00",  # Golden yellow / bright amber
    "d_audio_neg": "#8c510a",  # Sienna / saddlebrown
    "vowel_identity": "#b45309",  # Warm golden brown
    "vowel_f0": "#d97706",  # Rich amber / dark gold
    # Visual factors: tones of blue and purple
    "visual_lum": "#1f77b4",  # Royal blue
    "d_lum_pos": "#7b68ee",  # Medium purple / slate blue
    "d_lum_neg": "#4a148c",  # Deep dark purple / indigo
    "led_regime": "#6366f1",  # Indigo / violet
    # Audiovisual / interaction factors: greens / teal colours
    "av_coincidence": "#009e73",  # Sea green / teal
    "av_offset_coinc": "#00695c",  # Deep dark teal / emerald
    "av_coherence": "#26a69a",  # Medium sea-green / teal
    # Controls & baselines: neutral grays
    "stim_sustained": "#555555",
    "envelope_pair": "#607d8b",
    "session_baseline": "#333333",
}

# Biological / stimulus factors subjected to BH-FDR correction for factor identification
BIOLOGICAL_FACTOR_ORDER = [
    "audio_env",
    "d_audio_pos",
    "d_audio_neg",
    "visual_lum",
    "d_lum_pos",
    "d_lum_neg",
    "av_coincidence",
    "av_offset_coinc",
    "stim_sustained",
    "av_coherence",
    "vowel_identity",
    "vowel_f0",
    "led_regime",
]


# =============================================================================
# Cortical Field Annotation Helper
# =============================================================================
def lookup_cortical_field(
    data_tables_dir: pathlib.Path | str,
    animal_id: str,
    rec_name: str,
    channel_mcs: int,
) -> str:
    """Lookup anatomical cortical field (MEG, PEG, VC, SOM, AEG) from Ferret_Side_Chan_Field.csv.

    Matches:
      - Ferret: animal ID (e.g. 'F1812_Nala', 'F1902_Eclair', stripped of '_Depth*')
      - Side: 'Left' for Rec_A, 'Right' for Rec_B / Rec_B_2
      - Channel_mcs: 1-based MCS channel number
    """
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


@dataclass
class UnitSpec:
    """Specification for a single recorded channel/unit in a given animal."""

    animal_id: str
    rec_name: str
    channel_num: int
    depth: int | None = None
    region_label: str = ""
    min_rate_hz: float = 5.0
    max_silent_gap_s: float = 0.50

    def __post_init__(self):
        if not self.region_label:
            self.region_label = lookup_cortical_field(
                DATA_TABLES_DIR, self.animal_id, self.rec_name, self.channel_num
            )

    @property
    def unit_id(self) -> str:
        d_str = f"_D{self.depth}" if self.depth is not None else ""
        return f"{self.animal_id}{d_str}_{self.rec_name}_C{self.channel_num:02d}"

    @property
    def display_title(self) -> str:
        depth_str = f" Depth {self.depth}" if self.depth is not None else ""
        extra = f" ({self.region_label})" if self.region_label else ""
        return f"{self.animal_id}{depth_str} — {self.rec_name} Ch {self.channel_num}{extra}"


# =============================================================================
# Statistical Helpers
# =============================================================================
def benjamini_hochberg_fdr(p_vals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values (handles NaNs safely)."""
    p_arr = np.asarray(p_vals, dtype=np.float64)
    q_out = np.full_like(p_arr, np.nan, dtype=np.float64)
    mask = np.isfinite(p_arr)
    if np.any(mask):
        q_out[mask] = stats.false_discovery_control(
            np.clip(p_arr[mask], 1e-300, 1.0), method="bh"
        )
    return q_out


def q_to_stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def per_trial_poisson_ll(y_2d: np.ndarray, mu_2d: np.ndarray) -> np.ndarray:
    """Per-trial Poisson log-likelihood up to additive log(y!) constant."""
    mu_safe = np.maximum(mu_2d, 1e-15)
    return np.sum(y_2d * np.log(mu_safe) - mu_safe, axis=1)


def build_trial_cv_splits(
    trials_sub_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct stratified K-fold trial splits with automatic fallback for rare strata."""
    n_valid = len(trials_sub_df)
    if n_valid < n_splits * 2:
        raise ValueError(f"Too few QC-valid trials ({n_valid}) for {n_splits}-fold CV.")

    fine_strata = (
        trials_sub_df["StimFile"].astype(str)
        + "_"
        + trials_sub_df["StimIdx"].astype(str)
        + "_vb"
        + trials_sub_df["vBase"].astype(str)
        + "_vs"
        + trials_sub_df["vScale"].astype(str)
    )
    coarse_strata = (
        trials_sub_df["StimFile"].astype(str)
        + "_"
        + trials_sub_df["Vowel"].astype(str)
        + "_coh"
        + trials_sub_df["Coherence"].astype(str)
    )

    dummy_x = np.zeros(n_valid)
    for strata in (fine_strata, coarse_strata):
        if strata.value_counts().min() >= n_splits:
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            return list(skf.split(dummy_x, strata.to_numpy()))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(dummy_x))


# =============================================================================
# Animal-Level & Unit-Level Feature Construction
# =============================================================================
def prepare_animal_base_features(
    data_dir: pathlib.Path,
    animal_id: str,
    depth: int | None = None,
    rec_name: str | None = None,
    window_bins: int = 10,
    n_basis_funcs: int = 4,
) -> dict:
    """Load an animal's extracted `.npz`/`.csv` once and precompute shared 0-200 ms features."""
    candidates = []
    if depth is not None:
        candidates.append(
            (
                data_dir / f"{animal_id}_{rec_name}_Depth{depth}_all_blocks_20ms.npz",
                data_dir / f"{animal_id}_{rec_name}_Depth{depth}_all_blocks_trials.csv",
            )
        )
        candidates.append(
            (
                data_dir / f"{animal_id}_Depth{depth}_all_blocks_20ms.npz",
                data_dir / f"{animal_id}_Depth{depth}_all_blocks_trials.csv",
            )
        )
    candidates.append(
        (
            data_dir / f"{animal_id}_{rec_name}_all_blocks_20ms.npz",
            data_dir / f"{animal_id}_{rec_name}_all_blocks_trials.csv",
        )
    )
    candidates.append(
        (
            data_dir / f"{animal_id}_all_blocks_20ms.npz",
            data_dir / f"{animal_id}_all_blocks_trials.csv",
        )
    )

    npz_path, csv_path = None, None
    for npz_c, csv_c in candidates:
        if npz_c.exists() and csv_c.exists():
            try:
                with np.load(npz_c) as tmp_d:
                    lbls = [str(x) for x in tmp_d["channel_labels"]]
                if any(
                    lbl.startswith(f"{rec_name}_")
                    or ("B" in rec_name and "Rec_B" in lbl)
                    or ("A" in rec_name and "Rec_A" in lbl)
                    for lbl in lbls
                ):
                    npz_path, csv_path = npz_c, csv_c
                    break
            except Exception:
                pass

    if not npz_path:
        for npz_c, csv_c in candidates:
            if npz_c.exists() and csv_c.exists():
                npz_path, csv_path = npz_c, csv_c
                break

    if not npz_path or not csv_path:
        raise FileNotFoundError(
            f"Missing extracted files for {animal_id} (depth={depth}): checked {[c[0].name for c in candidates]}"
        )

    data = np.load(npz_path)
    trials_df = pd.read_csv(csv_path)

    envelopes = data[
        "envelopes"
    ]  # (n_trials, n_bins, 2): [audio, calibrated_luminance_cd_m2]
    binned_spikes = data["binned_spikes"]
    t_centers = data["bin_centers_s"]
    bin_edges = data["bin_edges_s"]
    channel_labels = data["channel_labels"]
    bin_size = round(float(bin_edges[1] - bin_edges[0]), 6)

    # Filter by depth if trials_df has 'depth' column and depth is specified
    if depth is not None and "depth" in trials_df.columns:
        d_mask = (trials_df["depth"] == int(depth)).to_numpy()
        if not d_mask.all():
            envelopes = envelopes[d_mask]
            binned_spikes = binned_spikes[d_mask]
            trials_df = trials_df.loc[d_mask].reset_index(drop=True)

    # Filter out invalid recording stream trials (e.g. Rec B = 0 periods)
    if rec_name is not None:
        if "A" in rec_name.upper() and "valid_rec_a" in trials_df.columns:
            v_mask = (trials_df["valid_rec_a"] == 1).to_numpy()
            if not v_mask.all():
                envelopes = envelopes[v_mask]
                binned_spikes = binned_spikes[v_mask]
                trials_df = trials_df.loc[v_mask].reset_index(drop=True)
        elif ("B" in rec_name.upper()) and "valid_rec_b" in trials_df.columns:
            v_mask = (trials_df["valid_rec_b"] == 1).to_numpy()
            if not v_mask.all():
                envelopes = envelopes[v_mask]
                binned_spikes = binned_spikes[v_mask]
                trials_df = trials_df.loc[v_mask].reset_index(drop=True)

    trials_df["global_trial_idx"] = np.arange(len(trials_df))

    n_trials, n_bins, _ = envelopes.shape
    keep = slice(window_bins, n_bins)
    n_keep = n_bins - window_bins
    stim_start = int(np.searchsorted(bin_edges, -1e-9))
    stim_end = int(np.searchsorted(bin_edges, 3.0 - 1e-9))

    aud = envelopes[:, :, 0].astype(np.float64)
    lum_phys = envelopes[:, :, 1].astype(np.float64)

    # Orthogonal (vBase, vMax) LED parameters & trial-level contrasts
    vb_vec = trials_df["vBase"].to_numpy(dtype=np.float64)
    vs_vec = trials_df["vScale"].to_numpy(dtype=np.float64)
    vmax_vec = np.round(vb_vec + vs_vec, 4)

    # Peak-normalize calibrated luminance per trial ceiling so L_unit(t) lies in [0, 1]
    # while preserving the physical Jumbo-LED deadzone (0 cd/m^2 below turn-on voltage).
    # Generalizes automatically to any (vBase, vScale) ceiling via trial peak or regime max!
    regime_peaks = (
        pd.Series(lum_phys[:, stim_start:stim_end].max(axis=1))
        .groupby(vmax_vec)
        .transform("max")
        .to_numpy()
    )
    lum_unit = lum_phys / np.maximum(regime_peaks[:, np.newaxis], 1e-6)

    d_aud = np.diff(aud, axis=1, prepend=0.0)
    d_lum = np.diff(lum_unit, axis=1, prepend=0.0)

    def norm_sd(arr: np.ndarray) -> np.ndarray:
        sd = float(arr[:, stim_start:stim_end].std())
        return arr / (sd if sd > 0 else 1.0)

    aud_n = norm_sd(aud)
    da_pos_n = norm_sd(np.maximum(d_aud, 0.0))
    da_neg_n = norm_sd(np.maximum(-d_aud, 0.0))
    lum_n = norm_sd(lum_unit)
    dl_pos_n = norm_sd(np.maximum(d_lum, 0.0))
    dl_neg_n = norm_sd(np.maximum(-d_lum, 0.0))
    av_coinc_n = norm_sd(da_pos_n * dl_pos_n)
    av_off_coinc_n = norm_sd(da_neg_n * dl_neg_n)

    # Causal 0-200 ms RaisedCosineLog temporal basis (columns normalized to unit sum so
    # net integrated filter weight sum_b w_b equals log-rate gain per +1 SD sustained input)
    _, bk = nmo.basis.RaisedCosineLogEval(n_basis_funcs=n_basis_funcs).evaluate_on_grid(
        window_bins
    )
    basis_kernels = np.array(bk, dtype=np.float64, copy=True)
    basis_kernels /= np.maximum(basis_kernels.sum(axis=0, keepdims=True), 1e-12)

    def conv_basis(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(
            nmo.convolve.create_convolutional_predictor(
                basis_kernels, arr, predictor_causality="causal", shift=False, axis=1
            ),
            dtype=np.float64,
        )
        return out[:, keep, :]

    conv_cont_3d = {
        "audio_env": conv_basis(aud_n),
        "d_audio_pos": conv_basis(da_pos_n),
        "d_audio_neg": conv_basis(da_neg_n),
        "visual_lum": conv_basis(lum_n),
        "d_lum_pos": conv_basis(dl_pos_n),
        "d_lum_neg": conv_basis(dl_neg_n),
        "av_coincidence": conv_basis(av_coinc_n),
        "av_offset_coinc": conv_basis(av_off_coinc_n),
    }

    # Centered [-0.5, +0.5] contrasts during [0, 3s), decoupled from `stim_sustained`
    c_coh = np.where(trials_df["Coherence"].to_numpy(dtype=int) == 1, +0.5, -0.5)
    c_vowel = np.where(trials_df["Vowel"].astype(str).to_numpy() == "u", +0.5, -0.5)
    c_f0 = np.where(trials_df["Pitch"].to_numpy(dtype=int) == 195, +0.5, -0.5)
    c_vbase = (
        np.where(np.isclose(vb_vec, vb_vec.max()), +0.5, -0.5)
        if np.ptp(vb_vec) > 0
        else np.zeros(n_trials)
    )
    c_vmax = (
        np.where(np.isclose(vmax_vec, vmax_vec.max()), +0.5, -0.5)
        if np.ptp(vmax_vec) > 0
        else np.zeros(n_trials)
    )

    def make_epoch_boxcar(trial_weights: np.ndarray) -> np.ndarray:
        arr = np.zeros((n_trials, n_bins, 1), dtype=np.float64)
        arr[:, stim_start:stim_end, 0] = trial_weights[:, np.newaxis]
        return arr[:, keep, :]

    stim_sustained_3d = make_epoch_boxcar(np.ones(n_trials, dtype=np.float64))
    coh_3d = make_epoch_boxcar(c_coh)
    vowel_3d = make_epoch_boxcar(c_vowel)
    f0_3d = make_epoch_boxcar(c_f0)
    vbase_tonic_3d = make_epoch_boxcar(c_vbase)
    vmax_tonic_3d = make_epoch_boxcar(c_vmax)

    gain_vbase_3d = conv_basis(dl_pos_n * c_vbase[:, np.newaxis])
    gain_vmax_3d = conv_basis(dl_pos_n * c_vmax[:, np.newaxis])

    short_sf = (
        trials_df["StimFile"]
        .astype(str)
        .str.replace("_SS.mat", "", regex=False)
        .str.replace("AVPilot", "AVP", regex=False)
    )
    pair_tags = (
        short_sf + "_Env" + trials_df["Envelope"].astype(int).astype(str)
    ).to_numpy(dtype=object)

    return {
        "animal_id": animal_id,
        "trials_df": trials_df,
        "binned_spikes": binned_spikes,
        "channel_labels": channel_labels,
        "t_centers": t_centers,
        "t_keep": t_centers[keep],
        "bin_edges": bin_edges,
        "bin_size": bin_size,
        "keep_slice": keep,
        "n_bins": n_bins,
        "n_keep": n_keep,
        "stim_start": stim_start,
        "stim_end": stim_end,
        "basis_kernels": basis_kernels,
        "conv_cont_3d": conv_cont_3d,
        "stim_sustained_3d": stim_sustained_3d,
        "coh_3d": coh_3d,
        "vowel_3d": vowel_3d,
        "f0_3d": f0_3d,
        "vbase_tonic_3d": vbase_tonic_3d,
        "vmax_tonic_3d": vmax_tonic_3d,
        "gain_vbase_3d": gain_vbase_3d,
        "gain_vmax_3d": gain_vmax_3d,
        "c_coh": c_coh,
        "c_vowel": c_vowel,
        "c_f0": c_f0,
        "c_vbase": c_vbase,
        "c_vmax": c_vmax,
        "pair_tags": pair_tags,
    }


def assemble_unit_design_matrix(animal_bundle: dict, valid_idx: np.ndarray) -> dict:
    """Build full-rank 3D design tensor `(M_valid, T_keep, P)` and group slices for one unit.

    Automatically adapts to whichever recording blocks, stimulus-set pairs, and experimental
    contrasts actually vary across `valid_idx` (safe across any subset of sessions/animals).
    """
    trials_df = animal_bundle["trials_df"]
    keep = animal_bundle["keep_slice"]
    n_bins = animal_bundle["n_bins"]
    n_keep = animal_bundle["n_keep"]
    stim_start = animal_bundle["stim_start"]
    stim_end = animal_bundle["stim_end"]

    # Filter out any rare recording block with <2 valid trials so CV folds always see every block
    sub_blocks = trials_df.iloc[valid_idx]["block"].astype(str)
    block_counts = sub_blocks.value_counts()
    keep_blocks = set(block_counts[block_counts >= 2].index.tolist())
    if len(keep_blocks) < len(block_counts):
        valid_idx = valid_idx[sub_blocks.isin(keep_blocks).to_numpy()]

    M = len(valid_idx)
    blocks_list: list[np.ndarray] = []
    group_slices: dict[str, slice] = {}
    sub_slices: dict[str, slice] = {}
    cursor = 0

    def add_block(
        group_key: str, arr_3d: np.ndarray, sub_map: dict[str, slice] | None = None
    ) -> None:
        nonlocal cursor
        w = arr_3d.shape[2]
        blocks_list.append(arr_3d)
        group_slices[group_key] = slice(cursor, cursor + w)
        if sub_map:
            for sk, ssl in sub_map.items():
                sub_slices[sk] = slice(cursor + ssl.start, cursor + ssl.stop)
        cursor += w

    # 1. Continuous 0-200 ms temporal-filter groups (8 groups x 4 basis = 32 cols)
    for k in CONT_KEYS:
        add_block(k, animal_bundle["conv_cont_3d"][k][valid_idx])

    # 2. Sustained stimulus-epoch indicator [0, 3s) vs pre/post silence (1 col)
    add_block("stim_sustained", animal_bundle["stim_sustained_3d"][valid_idx])

    # 2b. Sustained AV Coherence state (Coherent vs Incoherent on [0, 3s)) if both present
    if np.ptp(animal_bundle["c_coh"][valid_idx]) > 0:
        add_block("av_coherence", animal_bundle["coh_3d"][valid_idx])

    # 3. Vowel identity (/u/ vs /a/) if both vowels present
    if np.ptp(animal_bundle["c_vowel"][valid_idx]) > 0:
        add_block("vowel_identity", animal_bundle["vowel_3d"][valid_idx])

    # 4. Vowel F0 (195 vs 175 Hz) if both pitches present
    if np.ptp(animal_bundle["c_f0"][valid_idx]) > 0:
        add_block("vowel_f0", animal_bundle["f0_3d"][valid_idx])

    # 5. LED Regime (vBase, vMax) tonic + flash-gain interactions
    led_parts = []
    led_sub: dict[str, slice] = {}
    led_cur = 0
    if np.ptp(animal_bundle["c_vbase"][valid_idx]) > 0:
        vb_t = animal_bundle["vbase_tonic_3d"][valid_idx]
        vb_g = animal_bundle["gain_vbase_3d"][valid_idx]
        led_parts.extend([vb_t, vb_g])
        led_sub["vbase_tonic"] = slice(led_cur, led_cur + 1)
        led_cur += 1
        led_sub["gain_vbase"] = slice(led_cur, led_cur + vb_g.shape[2])
        led_cur += vb_g.shape[2]
    if np.ptp(animal_bundle["c_vmax"][valid_idx]) > 0:
        vm_t = animal_bundle["vmax_tonic_3d"][valid_idx]
        vm_g = animal_bundle["gain_vmax_3d"][valid_idx]
        led_parts.extend([vm_t, vm_g])
        led_sub["vmax_tonic"] = slice(led_cur, led_cur + 1)
        led_cur += 1
        led_sub["gain_vmax"] = slice(led_cur, led_cur + vm_g.shape[2])
        led_cur += vm_g.shape[2]
    if led_parts:
        add_block("led_regime", np.concatenate(led_parts, axis=2), sub_map=led_sub)

    # 6. Dynamic `envelope_pair` (`StimSet x Envelope`, sum-to-zero deviation coding on [0, 3s))
    pair_tags_valid = animal_bundle["pair_tags"][valid_idx]
    obs_pairs = sorted(pd.unique(pair_tags_valid).tolist())
    if len(obs_pairs) > 1:
        pair_grid = np.full((M, n_bins), "silence", dtype=object)
        pair_grid[:, stim_start:stim_end] = pair_tags_valid[:, np.newaxis]
        pair_basis = nmo.basis.Category(obs_pairs, out_of_category=True)
        assert list(pair_basis.categories) == obs_pairs
        M_all = np.asarray(
            pair_basis.compute_features(pair_grid[:, keep].ravel()), dtype=np.float64
        ).reshape(M, n_keep, len(obs_pairs))
        X_pair_3d = M_all[:, :, 1:] - M_all[:, :, [0]]
        add_block("envelope_pair", X_pair_3d)

    # 7. Dynamic `session_baseline` (`block`, active across entire trial including pre/post silence)
    blocks_valid = trials_df.iloc[valid_idx]["block"].astype(str).to_numpy(dtype=object)
    obs_blocks = sorted(pd.unique(blocks_valid).tolist())
    if len(obs_blocks) > 1:
        block_grid = np.repeat(blocks_valid[:, np.newaxis], n_keep, axis=1)
        sess_basis = nmo.basis.Category(obs_blocks)
        assert list(sess_basis.categories) == obs_blocks
        X_sess_3d = np.asarray(
            sess_basis.compute_features(block_grid.ravel()), dtype=np.float64
        )[:, 1:].reshape(M, n_keep, len(obs_blocks) - 1)
        add_block("session_baseline", X_sess_3d)

    X_3d = np.concatenate(blocks_list, axis=2)  # (M, n_keep, P)

    # Full-rank check on categorical + scalar columns (guarantees zero collinearity)
    scalar_cols = [np.ones(M * n_keep)]
    for gk in [
        "stim_sustained",
        "av_coherence",
        "vowel_identity",
        "vowel_f0",
        "envelope_pair",
        "session_baseline",
    ]:
        if gk in group_slices:
            scalar_cols.append(X_3d[:, :, group_slices[gk]].reshape(M * n_keep, -1))
    for sk in ["vbase_tonic", "vmax_tonic"]:
        if sk in sub_slices:
            scalar_cols.append(X_3d[:, :, sub_slices[sk]].reshape(M * n_keep, -1))
    cat_mat = np.column_stack(scalar_cols)
    rk = int(np.linalg.matrix_rank(cat_mat))
    assert (
        rk == cat_mat.shape[1]
    ), f"Categorical design rank-deficient: {rk} != {cat_mat.shape[1]}"

    return {
        "valid_idx": valid_idx,
        "X_3d": X_3d,
        "group_slices": group_slices,
        "sub_slices": sub_slices,
        "obs_blocks": obs_blocks,
        "obs_pairs": obs_pairs,
    }


# =============================================================================
# Trial-Clustered Huber-White Sandwich Covariance on 3D Matrix
# =============================================================================
def compute_matrix_clustered_sandwich(
    glm: nmo.glm.GLM,
    X_3d: np.ndarray,
    y_2d: np.ndarray,
    reg_strength: float,
) -> dict:
    """Compute trial-clustered Huber-White sandwich covariance for a matrix-fitted NeMoS GLM.

    Index 0 of `theta` / `cov` is the unpenalized intercept; indices `1..P` match `X_3d` columns
    shifted by +1 (`col_offset = 1`).
    """
    M, T, P = X_3d.shape
    N = M * T
    X_with_int = np.concatenate([np.ones((M, T, 1), dtype=np.float64), X_3d], axis=2)
    X_flat = X_with_int.reshape(N, P + 1)

    intercept_val = float(np.asarray(glm.intercept_).ravel()[0])
    coef_vec = np.asarray(glm.coef_, dtype=np.float64).ravel()
    theta = np.concatenate([[intercept_val], coef_vec])

    mu_2d = np.asarray(glm.predict(X_3d.reshape(N, P)), dtype=np.float64).reshape(M, T)
    mu_flat = mu_2d.ravel()

    # Summed score vector per trial cluster
    U = np.einsum("mt,mtp->mp", y_2d - mu_2d, X_with_int)
    meat = (M / max(M - 1.0, 1.0)) * (U.T @ U)

    pen = np.full(P + 1, N * reg_strength, dtype=np.float64)
    pen[0] = 0.0
    bread = (X_flat.T * mu_flat) @ X_flat + np.diag(pen)
    bread_inv = np.linalg.inv(bread)
    cov = bread_inv @ meat @ bread_inv

    return {
        "intercept": intercept_val,
        "coef": coef_vec,
        "cov_coef": cov[1:, 1:],
        "se_coef": np.sqrt(np.maximum(np.diag(cov[1:, 1:]), 0.0)),
        "mu_2d": mu_2d,
    }


# =============================================================================
# Core Per-Unit Fitting, 5-Fold CV LOFO, Inference & Plotting
# =============================================================================
def fit_and_evaluate_unit(
    unit: UnitSpec,
    animal_bundle: dict,
    out_dir: pathlib.Path,
    n_splits: int = 5,
    reg_strength: float = 1e-3,
    filter_sessions: bool = True,
    fig_formats: tuple[str, ...] = ("png",),
) -> dict:
    """Fit unified Poisson GLM, run 5-fold CV LOFO across all factors, and save figure + CSVs."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_idx = find_channel_index(
        animal_bundle["channel_labels"], unit.rec_name, unit.channel_num
    )
    counts_all = animal_bundle["binned_spikes"][:, :, ch_idx]
    bin_size = animal_bundle["bin_size"]
    keep = animal_bundle["keep_slice"]
    t_keep = animal_bundle["t_keep"]
    trials_df = animal_bundle["trials_df"]

    qc = compute_trial_qc(
        counts_ch=counts_all,
        all_counts=animal_bundle["binned_spikes"],
        bin_size=bin_size,
        ch_idx=ch_idx,
        min_rate_hz=unit.min_rate_hz,
        max_silent_gap_s=unit.max_silent_gap_s,
        exclude_single_ch_gap=False,
    )

    # Look for cached session QC from diagnostics or out_dir
    diag_session_csv = (
        DIAG_DIR
        / f"{unit.animal_id}_{unit.rec_name}_C{unit.channel_num:02d}_diagnostics_session_qc.csv"
    )
    alt_diag_csv = DIAG_DIR / f"{unit.unit_id}_session_qc.csv"
    local_session_csv = out_dir / f"{unit.unit_id}_session_qc.csv"

    if diag_session_csv.exists():
        session_qc_df = pd.read_csv(diag_session_csv)
        print(
            f"  Session QC: loaded cached diagnostics table from {diag_session_csv.name}"
        )
    elif alt_diag_csv.exists():
        session_qc_df = pd.read_csv(alt_diag_csv)
        print(f"  Session QC: loaded cached diagnostics table from {alt_diag_csv.name}")
    elif local_session_csv.exists():
        session_qc_df = pd.read_csv(local_session_csv)
        print(
            f"  Session QC: loaded cached session table from {local_session_csv.name}"
        )
    else:
        session_qc_df = compute_session_qc(
            counts_ch=counts_all,
            all_counts=animal_bundle["binned_spikes"],
            trials_df=trials_df,
            bin_edges=animal_bundle["bin_edges"],
            ch_idx=ch_idx,
            min_rate_hz=unit.min_rate_hz,
            max_silent_gap_s=unit.max_silent_gap_s,
        )
    session_qc_df.to_csv(local_session_csv, index=False)

    if filter_sessions:
        kept_blocks = set(session_qc_df.loc[session_qc_df["status"] == "KEEP", "block"])
        sess_mask = trials_df["block"].isin(kept_blocks).values
        raw_valid_idx = np.where(qc["is_valid"] & sess_mask)[0]
        n_pass = len(kept_blocks)
        print(
            f"  Session QC: kept {n_pass} / {len(session_qc_df)} blocks ({len(raw_valid_idx)} trials)"
        )
    else:
        raw_valid_idx = np.where(qc["is_valid"])[0]

    design = assemble_unit_design_matrix(animal_bundle, raw_valid_idx)
    valid_idx = design["valid_idx"]
    X_3d = design["X_3d"]
    group_slices = design["group_slices"]
    sub_slices = design["sub_slices"]
    obs_blocks = design["obs_blocks"]
    obs_pairs = design["obs_pairs"]

    y_2d = counts_all[valid_idx, keep].astype(np.float64)
    M, T, P = X_3d.shape
    valid_df = trials_df.iloc[valid_idx].reset_index(drop=True)

    print(f"\n=== {unit.display_title} [{unit.unit_id}] ===")
    print(
        f"  QC-valid trials: {M} / {len(trials_df)} | Bins/trial: {T} | Design cols: {P} | "
        f"Blocks: {len(obs_blocks)} | StimSet Pairs: {len(obs_pairs)}"
    )

    splits = build_trial_cv_splits(valid_df, n_splits=n_splits, random_state=42)
    solver_kw = {"tol": 1e-6, "maxiter": 250}
    glm_null_runner = nmo.glm.GLM(
        regularizer="Ridge", regularizer_strength=reg_strength, solver_kwargs=solver_kw
    )
    glm_runner = nmo.glm.GLM(
        regularizer="Ridge", regularizer_strength=reg_strength, solver_kwargs=solver_kw
    )

    # Out-of-sample prediction containers across all M valid trials
    mu_cv_null = np.zeros((M, T), dtype=np.float64)
    mu_cv_full = np.zeros((M, T), dtype=np.float64)
    lofo_keys = [
        k for k in group_slices.keys()
    ]  # includes biological factors + envelope_pair + session_baseline
    mu_cv_abl: dict[str, np.ndarray] = {
        k: np.zeros((M, T), dtype=np.float64) for k in lofo_keys
    }

    fold_coefs = []
    fold_intercepts = []
    verified_zero_mask = False

    # Fixed-shape `(N_tr, P)` masked Ridge fitting: compiles at most 2 XLA kernels total!
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr_full = X_3d[tr_idx].reshape(len(tr_idx) * T, P)
        X_te_full = X_3d[te_idx].reshape(len(te_idx) * T, P)
        y_tr = y_2d[tr_idx].ravel()

        # 1. Session-baseline Null model (full-rank session slice, or mean-rate if single session)
        c_warm = np.zeros(P, dtype=np.float64)
        if "session_baseline" in group_slices:
            sl_s = group_slices["session_baseline"]
            glm_null_runner.fit(X_tr_full[:, sl_s], y_tr)
            mu_cv_null[te_idx] = np.asarray(
                glm_null_runner.predict(X_te_full[:, sl_s]), dtype=np.float64
            ).reshape(len(te_idx), T)
            c_warm[sl_s] = np.asarray(glm_null_runner.coef_, dtype=np.float64)
            int_warm = np.asarray(glm_null_runner.intercept_, dtype=np.float64)
        else:
            mean_y = max(float(y_tr.mean()), 1e-12)
            mu_cv_null[te_idx] = mean_y
            int_warm = np.array([np.log(mean_y)], dtype=np.float64)

        # 2. Full Unified GLM (warm-started from session null)
        glm_runner.fit(X_tr_full, y_tr, init_params=(c_warm, int_warm))
        coef_f = np.asarray(glm_runner.coef_, dtype=np.float64).copy()
        int_f = np.asarray(glm_runner.intercept_, dtype=np.float64).copy()
        fold_coefs.append(coef_f)
        fold_intercepts.append(int_f)
        mu_cv_full[te_idx] = np.asarray(
            glm_runner.predict(X_te_full), dtype=np.float64
        ).reshape(len(te_idx), T)

        # 3. Leave-One-Factor-Out refits (warm-started from full model; exact zero on ablated slice)
        # Avoid allocating new (N_tr, P) arrays repeatedly; zero and restore in-place to minimize RAM
        X_tr_m = X_tr_full.copy()
        for gk in lofo_keys:
            sl_g = group_slices[gk]
            orig_cols = X_tr_m[:, sl_g].copy()
            X_tr_m[:, sl_g] = 0.0
            c_init = coef_f.copy()
            c_init[sl_g] = 0.0
            glm_runner.fit(X_tr_m, y_tr, init_params=(c_init, int_f))
            if not verified_zero_mask:
                assert float(np.max(np.abs(np.asarray(glm_runner.coef_)[sl_g]))) < 1e-8
                verified_zero_mask = True
            X_te_m = X_te_full.copy()
            X_te_m[:, sl_g] = 0.0
            mu_cv_abl[gk][te_idx] = np.asarray(
                glm_runner.predict(X_te_m), dtype=np.float64
            ).reshape(len(te_idx), T)
            X_tr_m[:, sl_g] = orig_cols
        del X_tr_m, X_tr_full, X_te_full

    # Overall 5-fold out-of-sample metrics
    ll_null_trials = per_trial_poisson_ll(y_2d, mu_cv_null)
    ll_full_trials = per_trial_poisson_ll(y_2d, mu_cv_full)
    mean_spk_per_trial = float(y_2d.sum(axis=1).mean())
    denom_bits = np.log(2.0) * max(mean_spk_per_trial, 1e-12)
    denom_mbits = denom_bits / 1000.0

    cv_bits_per_spk = float(
        (ll_full_trials.mean() - ll_null_trials.mean()) / denom_bits
    )

    emp_hz_all = y_2d / bin_size
    pred_hz_cv = mu_cv_full / bin_size
    null_hz_cv = mu_cv_null / bin_size
    cond_emp_dyn, cond_pred_dyn = [], []
    for (_, _), grp in valid_df.groupby(["StimFile", "StimIdx"]):
        idx_c = grp.index.to_numpy()
        # Subtract out-of-sample session-null baseline so PSTH r reflects pure stimulus-locked temporal dynamics!
        cond_emp_dyn.append((emp_hz_all[idx_c] - null_hz_cv[idx_c]).mean(axis=0))
        cond_pred_dyn.append((pred_hz_cv[idx_c] - null_hz_cv[idx_c]).mean(axis=0))
    cv_psth_r = float(
        np.corrcoef(np.concatenate(cond_emp_dyn), np.concatenate(cond_pred_dyn))[0, 1]
    )

    # Fit full model on all M QC-valid trials (warm-started from CV mean) for exact sandwich inference
    mean_coef_init = np.mean(fold_coefs, axis=0)
    mean_int_init = np.mean(fold_intercepts, axis=0)

    # Free CV memory buffers before full dataset fit
    import gc

    gc.collect()
    jax.clear_caches()

    glm_full_all = nmo.glm.GLM(
        regularizer="Ridge", regularizer_strength=reg_strength, solver_kwargs=solver_kw
    )
    try:
        glm_full_all.fit(
            X_3d.reshape(M * T, P),
            y_2d.ravel(),
            init_params=(mean_coef_init, mean_int_init),
        )
    except (jax.errors.JaxRuntimeError, Exception) as e:
        print(
            f"  [Memory recovery] Full refit encountered ({e}). Clearing caches and retrying..."
        )
        gc.collect()
        jax.clear_caches()
        try:
            glm_full_all.fit(
                X_3d.reshape(M * T, P),
                y_2d.ravel(),
                init_params=(mean_coef_init, mean_int_init),
            )
        except Exception as e2:
            print(
                f"  [Fallback] Using 5-fold CV mean parameters for sandwich estimation ({e2})."
            )
            glm_full_all.coef_ = mean_coef_init
            glm_full_all.intercept_ = mean_int_init

    sand = compute_matrix_clustered_sandwich(
        glm_full_all, X_3d, y_2d, reg_strength=reg_strength
    )

    basis_kernels = animal_bundle["basis_kernels"]  # (window_bins, n_basis_funcs)
    lags_ms = np.arange(basis_kernels.shape[0]) * (bin_size * 1000.0)

    # Session baseline firing-rate span
    if "session_baseline" in group_slices:
        sess_b = np.concatenate([[0.0], sand["coef"][group_slices["session_baseline"]]])
        session_ratio = float(np.exp(np.ptp(sess_b)))
    else:
        session_ratio = 1.0

    # =========================================================================
    # Build Factor-Level Summary Table (Two-Stage Decision Criterion)
    # =========================================================================
    factor_rows = []
    eval_order = [k for k in BIOLOGICAL_FACTOR_ORDER if k in group_slices] + [
        k for k in ["envelope_pair", "session_baseline"] if k in group_slices
    ]

    for fk in eval_order:
        sl = group_slices[fk]
        w_vec = sand["coef"][sl]
        cov_blk = sand["cov_coef"][sl, sl]

        # Omnibus trial-clustered Group Wald test
        chi2_val = float(w_vec @ np.linalg.pinv(cov_blk) @ w_vec)
        p_wald = float(stats.chi2.sf(chi2_val, df=len(w_vec)))

        # 5-fold CV held-out Delta-LL across all M valid trials
        ll_abl_trials = per_trial_poisson_ll(y_2d, mu_cv_abl[fk])
        diff_mbits = (ll_full_trials - ll_abl_trials) / denom_mbits
        d_mean = float(diff_mbits.mean())
        d_se = float(diff_mbits.std(ddof=1) / np.sqrt(M))
        try:
            _, p_lofo = stats.wilcoxon(diff_mbits, alternative="greater")
        except ValueError:
            p_lofo = 1.0

        # Directional summary (net 0-200 ms weight + peak lag for temporal filters)
        if fk in CONT_KEYS:
            ones_k = np.ones(len(w_vec), dtype=np.float64)
            net_beta = float(ones_k @ w_vec)
            net_se = float(np.sqrt(max(ones_k @ cov_blk @ ones_k, 0.0)))
            kern = basis_kernels @ w_vec
            pk_idx = int(np.argmax(np.abs(kern)))
            peak_val = float(kern[pk_idx])
            peak_lag = float(lags_ms[pk_idx])
        elif len(w_vec) == 1:
            net_beta = float(w_vec[0])
            net_se = float(np.sqrt(max(cov_blk[0, 0], 0.0)))
            peak_val = np.nan
            peak_lag = np.nan
        else:
            # Multi-column categorical / LED regime group: summarize largest standardized sub-effect
            net_beta = np.nan
            net_se = np.nan
            peak_val = np.nan
            peak_lag = np.nan

        is_bio = fk in BIOLOGICAL_FACTOR_ORDER
        factor_rows.append(
            {
                "unit_id": unit.unit_id,
                "animal_id": unit.animal_id,
                "depth": unit.depth if unit.depth is not None else 1,
                "rec_name": unit.rec_name,
                "channel_num": unit.channel_num,
                "region_label": unit.region_label,
                "factor_key": fk,
                "factor_label": FACTOR_LABELS[fk],
                "is_biological_factor": is_bio,
                "df": len(w_vec),
                "cv_delta_mbits": d_mean,
                "cv_delta_se": d_se,
                "p_lofo_raw": float(p_lofo),
                "wald_chi2": chi2_val,
                "p_wald_raw": p_wald,
                "net_beta": net_beta,
                "net_se": net_se,
                "ci95_low": (
                    net_beta - 1.96 * net_se if np.isfinite(net_beta) else np.nan
                ),
                "ci95_high": (
                    net_beta + 1.96 * net_se if np.isfinite(net_beta) else np.nan
                ),
                "rate_mult": (
                    float(np.exp(net_beta)) if np.isfinite(net_beta) else np.nan
                ),
                "peak_kernel": peak_val,
                "peak_lag_ms": peak_lag,
            }
        )

    factor_df = pd.DataFrame(factor_rows)
    bio_mask = factor_df["is_biological_factor"].to_numpy()
    factor_df["q_lofo"] = np.nan
    factor_df["q_wald"] = np.nan
    factor_df.loc[bio_mask, "q_lofo"] = benjamini_hochberg_fdr(
        factor_df.loc[bio_mask, "p_lofo_raw"].to_numpy()
    )
    factor_df.loc[bio_mask, "q_wald"] = benjamini_hochberg_fdr(
        factor_df.loc[bio_mask, "p_wald_raw"].to_numpy()
    )
    # For the 2 nuisance controls, store their direct test p-values in q columns for display convenience
    factor_df.loc[~bio_mask, "q_lofo"] = factor_df.loc[~bio_mask, "p_lofo_raw"]
    factor_df.loc[~bio_mask, "q_wald"] = factor_df.loc[~bio_mask, "p_wald_raw"]

    factor_df["drives_firing"] = (
        (factor_df["cv_delta_mbits"] > 0.0)
        & (factor_df["q_lofo"] < 0.05)
        & (factor_df["q_wald"] < 0.05)
    )
    factor_df["lofo_stars"] = [
        q_to_stars(q) if d > 0 else ""
        for d, q in zip(factor_df["cv_delta_mbits"], factor_df["q_lofo"])
    ]

    # Human-readable interpretation column for instant table scanning
    interpretations = []
    for _, r in factor_df.iterrows():
        if not r["drives_firing"]:
            if r["q_wald"] < 0.05 and r["is_biological_factor"]:
                interpretations.append("No (in-sample ripple only; CV ns)")
            else:
                interpretations.append("No")
        else:
            if not r["is_biological_factor"]:
                interpretations.append(
                    f"Active Control (+{r['cv_delta_mbits']:.2f} mbits/spk)"
                )
            elif np.isfinite(r["net_beta"]):
                pct = (r["rate_mult"] - 1.0) * 100.0
                if np.isfinite(r["peak_lag_ms"]):
                    interpretations.append(
                        f"YES ({pct:+.1f}% net; peak {r['peak_kernel']:+.3f} @ {r['peak_lag_ms']:.0f}ms)"
                    )
                else:
                    interpretations.append(f"YES ({pct:+.1f}% rate change)")
            else:
                interpretations.append(f"YES (+{r['cv_delta_mbits']:.2f} mbits/spk)")
    factor_df["interpretation"] = interpretations

    driver_map = dict(zip(factor_df["factor_key"], factor_df["drives_firing"]))

    # =========================================================================
    # Build Detailed Directional Contrasts Table for Panel A (14 clean rows)
    # =========================================================================
    contrast_rows = []

    def add_contrast_row(parent_key: str, label: str, sl_c: slice, color: str) -> None:
        w = sand["coef"][sl_c]
        c_blk = sand["cov_coef"][sl_c, sl_c]
        ones_v = np.ones(len(w), dtype=np.float64)
        b_net = float(ones_v @ w)
        se_net = float(np.sqrt(max(ones_v @ c_blk @ ones_v, 0.0)))
        p_2s = float(2.0 * stats.norm.sf(abs(b_net / max(se_net, 1e-15))))
        contrast_rows.append(
            {
                "parent_factor": parent_key,
                "label": label,
                "beta": b_net,
                "se_cluster": se_net,
                "ci95_low": b_net - 1.96 * se_net,
                "ci95_high": b_net + 1.96 * se_net,
                "rate_mult": float(np.exp(b_net)),
                "p_raw": p_2s,
                "parent_drives_firing": bool(driver_map.get(parent_key, False)),
                "color": color,
            }
        )

    for ck in CONT_KEYS:
        if ck in group_slices:
            add_contrast_row(ck, FACTOR_LABELS[ck], group_slices[ck], FACTOR_COLORS[ck])

    for sk_key in ["stim_sustained", "av_coherence", "vowel_identity", "vowel_f0"]:
        if sk_key in group_slices:
            add_contrast_row(
                sk_key,
                FACTOR_LABELS[sk_key],
                group_slices[sk_key],
                FACTOR_COLORS[sk_key],
            )

    led_sub_specs = [
        ("vbase_tonic", "LED vBase Tonic (2V vs 1V)", "#5c6bc0"),
        ("vmax_tonic", "LED vMax Tonic (4V vs 3.5V)", "#7b1fa2"),
        ("gain_vbase", "Flash Gain × vBase (2V-1V)", "#2563eb"),
        ("gain_vmax", "Flash Gain × vMax (4V-3.5V)", "#4a148c"),
    ]
    for sub_k, sub_lbl, sub_col in led_sub_specs:
        if sub_k in sub_slices:
            add_contrast_row("led_regime", sub_lbl, sub_slices[sub_k], sub_col)

    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df["q_wald_term"] = benjamini_hochberg_fdr(contrast_df["p_raw"].to_numpy())
    contrast_df["term_stars"] = [q_to_stars(q) for q in contrast_df["q_wald_term"]]

    elapsed = time.perf_counter() - t0
    confirmed_bio = factor_df[
        factor_df["is_biological_factor"] & factor_df["drives_firing"]
    ]["factor_label"].tolist()

    print(
        f"  [5-Fold Out-of-Sample CV] Gain = +{cv_bits_per_spk:.4f} bits/spk "
        f"(+{cv_bits_per_spk*1000:.2f} mbits/spk) over Session-Null | CV PSTH r = {cv_psth_r:.3f} "
        f"({elapsed:.1f}s)"
    )
    print(
        f"  Confirmed Biological Drivers ({len(confirmed_bio)}): {confirmed_bio if confirmed_bio else ['None']}"
    )
    print("  Factor Summary (5-Fold Held-Out LOFO ΔLL + Trial-Clustered Sandwich):")
    for _, r in factor_df.iterrows():
        st = r["lofo_stars"] if r["lofo_stars"] else "ns"
        tag = "[Control]" if not r["is_biological_factor"] else "         "
        print(
            f"    {tag} {r['factor_label']:28s} | CV ΔLL = {r['cv_delta_mbits']:+6.2f} ± {r['cv_delta_se']:4.2f} mbits/spk "
            f"(q={r['q_lofo']:.3g}, {st:3s}) -> {r['interpretation']}"
        )

    # Save CSV outputs
    stem = f"{unit.unit_id}_passive_glm"
    factor_df.to_csv(out_dir / f"{stem}_factors.csv", index=False)
    contrast_df.to_csv(out_dir / f"{stem}_contrasts.csv", index=False)

    # =========================================================================
    # Plot Clean, Publication-Ready 5-Panel Dashboard
    # =========================================================================
    fig = plt.figure(figsize=(19.5, 13.6))
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.30, 0.88, 0.88], width_ratios=[1.12, 1.02, 1.06]
    )
    ax_w = fig.add_subplot(gs[0, 0])
    ax_k = fig.add_subplot(gs[0, 1])
    ax_lofo = fig.add_subplot(gs[0, 2])
    ax_p1 = fig.add_subplot(gs[1, :])
    ax_p2 = fig.add_subplot(gs[2, :], sharex=ax_p1)

    # --- Panel A: Directional Effects (Net 0-200 ms Log-Weight ± 95% Trial CI) ---
    n_c = len(contrast_df)
    y_pos = np.arange(n_c)[::-1]
    c_vals = contrast_df["beta"].to_numpy()
    c_ci = 1.96 * contrast_df["se_cluster"].to_numpy()

    for yp, (_, row_c) in zip(y_pos, contrast_df.iterrows()):
        if row_c["parent_factor"] == "led_regime":
            active_drv = bool(row_c["parent_drives_firing"]) and (
                row_c["q_wald_term"] < 0.05
            )
        else:
            active_drv = bool(row_c["parent_drives_firing"])
        bv = float(row_c["beta"])
        ci_h = 1.96 * float(row_c["se_cluster"])
        ax_w.barh(
            yp,
            bv,
            color=row_c["color"],
            alpha=0.90 if active_drv else 0.20,
            edgecolor="black" if active_drv else "0.65",
            linewidth=0.75 if active_drv else 0.5,
        )
        ax_w.errorbar(
            bv,
            yp,
            xerr=ci_h,
            fmt="none",
            ecolor="0.12" if active_drv else "0.62",
            elinewidth=1.25 if active_drv else 0.85,
            capsize=2.4,
            zorder=4,
        )
        if active_drv and row_c["term_stars"]:
            off = (ci_h + 0.004) if bv >= 0 else -(ci_h + 0.004)
            ha = "left" if bv >= 0 else "right"
            ax_w.text(
                bv + off,
                yp,
                row_c["term_stars"],
                va="center",
                ha=ha,
                fontsize=8.8,
                fontweight="bold",
            )

    # Divider between 8 continuous 0-200 ms factors and 8 epoch/regime contrasts
    n_cont_present = sum(1 for k in CONT_KEYS if k in group_slices)
    if 0 < n_cont_present < n_c:
        ax_w.axhline(
            (y_pos[n_cont_present - 1] + y_pos[n_cont_present]) / 2.0,
            color="0.65",
            ls="--",
            lw=0.8,
        )

    ax_w.axvline(0.0, color="black", lw=0.9)
    ax_w.set_yticks(y_pos)
    ytick_labels = ax_w.set_yticklabels(contrast_df["label"].tolist(), fontsize=8.3)
    for lbl_obj, (_, row_c) in zip(ytick_labels, contrast_df.iterrows()):
        is_bold = (
            bool(row_c["parent_drives_firing"]) and (row_c["q_wald_term"] < 0.05)
            if row_c["parent_factor"] == "led_regime"
            else bool(row_c["parent_drives_firing"])
        )
        if is_bold:
            lbl_obj.set_fontweight("bold")

    x_pad = max(float(np.max(np.abs(c_vals) + c_ci)) * 0.24, 0.025)
    ax_w.set_xlim(
        min(-0.025, float(np.min(c_vals - c_ci)) - x_pad),
        max(0.025, float(np.max(c_vals + c_ci)) + x_pad),
    )
    ax_w.set_xlabel("Net Log-Rate Weight β ± 95% Trial-Clustered CI")
    ax_w.set_title(
        "A. Directional Effect Sizes (Unified GLM)\n[Solid = Confirmed CV Driver; Faded = CV Non-Driver]",
        fontsize=9.8,
        fontweight="bold",
    )
    ax_w.grid(axis="x", alpha=0.25)

    # --- Panel B: Causal 0-200 ms Temporal Filters (Highlighting Confirmed Drivers) ---
    k_styles = [
        ("audio_env", "-", 2.2),
        ("d_audio_pos", "--", 2.3),
        ("d_audio_neg", ":", 1.9),
        ("visual_lum", "-", 2.2),
        ("d_lum_pos", "--", 2.3),
        ("d_lum_neg", ":", 1.9),
        ("av_coincidence", "-.", 2.3),
        ("av_offset_coinc", (0, (3, 1, 1, 1, 1, 1)), 2.1),
    ]
    for k_name, ls, lw in k_styles:
        if k_name not in group_slices:
            continue
        sl_k = group_slices[k_name]
        w_vec = sand["coef"][sl_k]
        cov_k = sand["cov_coef"][sl_k, sl_k]
        kernel = basis_kernels @ w_vec
        kern_se = np.sqrt(
            np.maximum(np.diag(basis_kernels @ cov_k @ basis_kernels.T), 0.0)
        )
        is_drv = bool(driver_map.get(k_name, False))
        col = FACTOR_COLORS[k_name]
        lbl_k = f"{FACTOR_LABELS[k_name]} ({'DRIVER' if is_drv else 'CV ns'})"
        ax_k.plot(
            lags_ms,
            kernel,
            color=col,
            ls=ls,
            lw=lw if is_drv else 1.15,
            alpha=0.95 if is_drv else 0.32,
            zorder=4 if is_drv else 2,
            label=lbl_k,
        )
        if is_drv:
            ax_k.fill_between(
                lags_ms,
                kernel - 1.96 * kern_se,
                kernel + 1.96 * kern_se,
                color=col,
                alpha=0.15,
                zorder=3,
            )

    ax_k.axhline(0.0, color="black", lw=0.8, ls=":")
    ax_k.set_xlabel("Time lag after stimulus feature (ms)")
    ax_k.set_ylabel("Temporal Filter Weight h(τ) ± 95% CI")
    ax_k.set_title(
        f"B. Causal Temporal Filters (0–200 ms)\n"
        f"5-Fold CV: +{cv_bits_per_spk:.4f} bits/spk, Out-of-Sample PSTH r={cv_psth_r:.3f}",
        fontsize=9.8,
        fontweight="bold",
    )
    ax_k.grid(True, alpha=0.25)
    ax_k.legend(fontsize=7.3, ncol=1, loc="best", framealpha=0.92)

    # --- Panel C: Primary Decision Gate — 5-Fold CV Held-Out LOFO Delta-LL ---
    plot_lofo_df = factor_df[factor_df["factor_key"] != "session_baseline"].reset_index(
        drop=True
    )
    y_lofo = np.arange(len(plot_lofo_df))[::-1]
    d_vals = plot_lofo_df["cv_delta_mbits"].to_numpy()
    d_ses = plot_lofo_df["cv_delta_se"].to_numpy()

    for yl, (_, row_f) in zip(y_lofo, plot_lofo_df.iterrows()):
        fk = row_f["factor_key"]
        is_drv = bool(row_f["drives_firing"])
        dm = float(row_f["cv_delta_mbits"])
        ds = float(row_f["cv_delta_se"])
        col = FACTOR_COLORS.get(fk, "#607d8b")
        ax_lofo.barh(
            yl,
            dm,
            color=col,
            alpha=0.90 if is_drv else 0.22,
            edgecolor="black" if is_drv else "0.65",
            linewidth=0.75 if is_drv else 0.5,
        )
        ax_lofo.errorbar(
            dm,
            yl,
            xerr=ds,
            fmt="none",
            ecolor="0.12" if is_drv else "0.60",
            elinewidth=1.2 if is_drv else 0.85,
            capsize=2.4,
            zorder=4,
        )
        if is_drv and row_f["lofo_stars"]:
            ax_lofo.text(
                max(dm + ds, 0.0)
                + max(float(np.max(np.abs(d_vals) + d_ses)) * 0.03, 0.04),
                yl,
                row_f["lofo_stars"],
                va="center",
                ha="left",
                fontsize=9.0,
                fontweight="bold",
            )

    if 0 < n_cont_present < len(plot_lofo_df):
        ax_lofo.axhline(
            (y_lofo[n_cont_present - 1] + y_lofo[n_cont_present]) / 2.0,
            color="0.65",
            ls="--",
            lw=0.8,
        )
    if "envelope_pair" in plot_lofo_df["factor_key"].values and len(plot_lofo_df) > 1:
        ax_lofo.axhline((y_lofo[-2] + y_lofo[-1]) / 2.0, color="0.75", ls=":", lw=0.8)

    ax_lofo.axvline(0.0, color="black", lw=0.9)
    ax_lofo.set_yticks(y_lofo)
    lofo_yticklabels = ax_lofo.set_yticklabels(
        plot_lofo_df["factor_label"].tolist(), fontsize=8.3
    )
    for lbl_obj, (_, row_f) in zip(lofo_yticklabels, plot_lofo_df.iterrows()):
        if row_f["drives_firing"]:
            lbl_obj.set_fontweight("bold")

    l_pad = max(float(np.max(np.abs(d_vals) + d_ses)) * 0.26, 0.25)
    ax_lofo.set_xlim(
        min(-0.15, float(np.min(d_vals - d_ses)) - l_pad * 0.4),
        max(0.40, float(np.max(d_vals + d_ses)) + l_pad),
    )
    ax_lofo.set_ylim(-1.65, len(plot_lofo_df) - 0.35)
    ax_lofo.set_xlabel("5-Fold CV Held-Out Info Loss When Dropped (mbits/spk ± SE)")
    ax_lofo.set_title(
        "C. Which Factors Impact Firing? (5-Fold CV LOFO)\n[Paired Wilcoxon BH-FDR: *<0.05, **<0.01, ***<0.001]",
        fontsize=9.8,
        fontweight="bold",
    )
    ax_lofo.grid(axis="x", alpha=0.25)

    # Session drift control annotation box in dedicated bottom margin of Panel C
    if "session_baseline" in factor_df["factor_key"].values:
        s_row = factor_df[factor_df["factor_key"] == "session_baseline"].iloc[0]
        sess_txt = (
            f"Session Drift Control ({len(obs_blocks)} blocks): "
            f"CV ΔLL = {s_row['cv_delta_mbits']:+.2f}±{s_row['cv_delta_se']:.2f} mbits/spk "
            f"({s_row['lofo_stars'] or 'ns'}), max/min = {session_ratio:.2f}×"
        )
        ax_lofo.text(
            0.98,
            0.015,
            sess_txt,
            transform=ax_lofo.transAxes,
            fontsize=7.5,
            va="bottom",
            ha="right",
            bbox={
                "boxstyle": "round,pad=0.30",
                "facecolor": "0.96",
                "edgecolor": "0.75",
            },
        )

    # --- Panels D & E: Strictly Out-of-Sample 5-Fold CV PSTH Predictions across ALL Valid Trials ---
    # Model-blind representative condition selection: pick the Coherent and Incoherent condition
    # (>=20 valid trials) with highest empirical PSTH temporal modulation (avoids chronic-drift dead blocks)
    def pick_representative_condition(coh_val: int) -> tuple[str, int] | None:
        sub_c = valid_df[valid_df["Coherence"] == coh_val]
        best_key = None
        best_mod = -1.0
        for (sf_k, sidx_k), grp_c in sub_c.groupby(["StimFile", "StimIdx"]):
            if len(grp_c) < 20:
                continue
            emp_psth = emp_hz_all[grp_c.index.to_numpy()].mean(axis=0)
            mod_sd = float(emp_psth.std())
            if mod_sd > best_mod:
                best_mod = mod_sd
                best_key = (str(sf_k), int(sidx_k))
        if best_key is None and len(sub_c) > 0:
            best_key = sub_c.groupby(["StimFile", "StimIdx"]).size().idxmax()
        return best_key

    coh_pick = pick_representative_condition(1)
    inc_pick = pick_representative_condition(0)
    ex_specs = []
    if coh_pick is not None:
        ex_specs.append((ax_p1, coh_pick, "Coherent (synchronous A+V onsets)"))
    if inc_pick is not None:
        ex_specs.append((ax_p2, inc_pick, "Incoherent (independent A & V onsets)"))

    for ax_p, (sf, sidx), coh_desc in ex_specs:
        mask_c = (
            (valid_df["StimFile"] == sf) & (valid_df["StimIdx"] == sidx)
        ).to_numpy()
        n_rep = int(mask_c.sum())
        emp_c = emp_hz_all[mask_c].mean(axis=0)
        pred_c = pred_hz_cv[mask_c].mean(axis=0)
        null_c = (mu_cv_null[mask_c] / bin_size).mean(axis=0)
        r_cond = float(np.corrcoef(emp_c, pred_c)[0, 1])

        row0 = valid_df[mask_c].iloc[0]
        vow = row0["Vowel"]
        f0 = int(row0["Pitch"])

        ax_p.plot(
            t_keep,
            emp_c,
            color="0.50",
            lw=1.25,
            alpha=0.88,
            label=f"Empirical PSTH (n={n_rep} trials)",
        )
        ax_p.plot(
            t_keep,
            null_c,
            color="0.35",
            lw=1.1,
            ls=":",
            alpha=0.75,
            label="Session-Null Baseline (5-fold CV)",
        )
        ax_p.plot(
            t_keep,
            pred_c,
            color="#1e3a8a",
            lw=2.1,
            label="Unified NeMoS GLM (5-fold out-of-sample CV)",
        )
        ax_p.axvline(0.0, color="0.25", ls="--", lw=0.9)
        ax_p.axvline(3.0, color="0.25", ls="--", lw=0.9)
        ax_p.set_ylabel("Firing Rate (Hz)")
        ax_p.set_title(
            f"Out-of-Sample 5-Fold CV — {sf.replace('_SS.mat', '')} StimIdx={sidx} "
            f"(Vowel /{vow}/, F0={f0} Hz, {coh_desc}) | Out-of-Sample PSTH r = {r_cond:.2f}",
            fontsize=9.8,
            fontweight="bold",
            loc="left",
        )
        ax_p.legend(loc="upper right", fontsize=8.1, ncol=3, framealpha=0.92)
        ax_p.grid(True, alpha=0.22)

    ax_p2.set_xlabel("Time from stimulus onset (s)")
    ax_p2.set_xlim(float(t_keep[0]), float(t_keep[-1]))

    import textwrap

    drv_str = ", ".join(confirmed_bio) if confirmed_bio else "None"
    sub_line = f"[Confirmed Drivers (5-Fold CV LOFO q<0.05 & Wald q<0.05): {drv_str}]"
    if len(sub_line) > 125:
        sub_line = textwrap.fill(sub_line, width=125)
        sup_fs = 10.6
        top_rect = 0.948
    else:
        sup_fs = 12.0 if len(sub_line) <= 105 else 11.0
        top_rect = 0.962
    fig.suptitle(
        f"Unified NeMoS Poisson GLM — {unit.display_title}\n{sub_line}",
        fontsize=sup_fs,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, top_rect])
    saved_figs = []
    for ext in fig_formats:
        out_fig = out_dir / f"{stem}_dashboard.{ext}"
        fig.savefig(out_fig, dpi=175)
        saved_figs.append(out_fig)
        print(f"  Saved dashboard: {out_fig}")
    out_png = out_dir / f"{stem}_dashboard.png"
    plt.close(fig)

    # Wide one-row summary for population table across neurons & animals
    wide_row = {
        "unit_id": unit.unit_id,
        "animal_id": unit.animal_id,
        "depth": unit.depth if unit.depth is not None else 1,
        "rec_name": unit.rec_name,
        "channel_num": unit.channel_num,
        "region_label": unit.region_label,
        "n_valid_trials": M,
        "n_total_trials": len(trials_df),
        "n_blocks": len(obs_blocks),
        "n_stimset_pairs": len(obs_pairs),
        "mean_rate_hz": float(emp_hz_all.mean()),
        "session_max_min_ratio": session_ratio,
        "cv_bits_per_spk": cv_bits_per_spk,
        "cv_mbits_per_spk": cv_bits_per_spk * 1000.0,
        "cv_psth_r": cv_psth_r,
        "n_confirmed_drivers": len(confirmed_bio),
        "confirmed_drivers": "; ".join(confirmed_bio) if confirmed_bio else "None",
    }
    for fk in BIOLOGICAL_FACTOR_ORDER + ["envelope_pair"]:
        sub = factor_df[factor_df["factor_key"] == fk]
        if len(sub) == 1:
            wide_row[f"drives_{fk}"] = bool(sub["drives_firing"].iloc[0])
            wide_row[f"mbits_{fk}"] = float(sub["cv_delta_mbits"].iloc[0])
            wide_row[f"q_lofo_{fk}"] = float(sub["q_lofo"].iloc[0])
        else:
            wide_row[f"drives_{fk}"] = False
            wide_row[f"mbits_{fk}"] = np.nan
            wide_row[f"q_lofo_{fk}"] = np.nan

    tracks_audio = bool(
        wide_row.get("drives_audio_env", False)
        or wide_row.get("drives_d_audio_pos", False)
        or wide_row.get("drives_d_audio_neg", False)
    )
    tracks_visual = bool(
        wide_row.get("drives_visual_lum", False)
        or wide_row.get("drives_d_lum_pos", False)
        or wide_row.get("drives_d_lum_neg", False)
    )
    tracks_coincidence = bool(
        wide_row.get("drives_av_coincidence", False)
        or wide_row.get("drives_av_offset_coinc", False)
        or wide_row.get("drives_av_coherence", False)
    )
    is_sustained_only = bool(
        wide_row.get("drives_stim_sustained", False)
        and not (tracks_audio or tracks_visual or tracks_coincidence)
    )

    if (tracks_audio and tracks_visual) or tracks_coincidence:
        coding_strategy = "Multisensory Tracking"
    elif tracks_audio:
        coding_strategy = "Auditory Tracking"
    elif tracks_visual:
        coding_strategy = "Visual Tracking"
    elif is_sustained_only:
        coding_strategy = "Onset/Sustained Only"
    else:
        coding_strategy = "Unresponsive / Non-sensory"
    wide_row["coding_strategy"] = coding_strategy

    # Save atomic per-unit summary JSON for robust cross-session aggregation
    try:
        import json

        clean_row = {}
        for k, v in wide_row.items():
            if pd.isna(v):
                clean_row[k] = None
            elif isinstance(v, (np.integer, int)):
                clean_row[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_row[k] = float(v)
            elif isinstance(v, (np.bool_, bool)):
                clean_row[k] = bool(v)
            else:
                clean_row[k] = v
        with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f_json:
            json.dump(clean_row, f_json, indent=2)
    except Exception:
        pass

    return {
        "unit": unit,
        "factor_df": factor_df,
        "contrast_df": contrast_df,
        "wide_row": wide_row,
        "out_png": out_png,
    }


def run_population_pipeline(
    units: list[UnitSpec],
    data_dir: pathlib.Path = DATA_DIR,
    out_dir: pathlib.Path = CLEAN_OUT_DIR,
    n_splits: int = 5,
    reg_strength: float = 1e-3,
    filter_sessions: bool = True,
    fig_formats: tuple[str, ...] = ("png",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run unified GLM across any list of units from one or more animals, caching per-animal features."""
    animal_cache: dict[tuple, dict] = {}
    all_factor_dfs = []
    wide_rows = []

    for unit in units:
        stems_to_check = [f"{unit.unit_id}_passive_glm", f"{unit.unit_id}_clean_glm"]
        fac_file, dash_exists, sum_file, stem = None, False, None, stems_to_check[0]
        for s in stems_to_check:
            f_cand = out_dir / f"{s}_factors.csv"
            all_figs_present = all(
                (out_dir / f"{s}_dashboard.{ext}").exists() for ext in fig_formats
            )
            if f_cand.exists() and all_figs_present:
                fac_file, dash_exists, stem = f_cand, True, s
                sum_file = out_dir / f"{s}_summary.json"
                break

        # Resume optimization: if unit was already successfully fitted with all current factors & requested image formats, reuse metrics!
        if fac_file is not None and fac_file.exists() and dash_exists:
            try:
                fac_df = pd.read_csv(fac_file)
                existing_keys = set(fac_df["factor_key"].astype(str))
                if "av_offset_coinc" in existing_keys and "av_coherence" in existing_keys:
                    print(f"\n=== {unit.display_title} [{unit.unit_id}] ===")
                    print(f"  [Resume] Loading cached GLM results from {fac_file.name}")
                    all_factor_dfs.append(fac_df)
                    if sum_file.exists():
                        import json

                        with open(sum_file, "r", encoding="utf-8") as f_sum:
                            wide_rows.append(json.load(f_sum))
                    else:
                        master_wide = out_dir / "population_units_summary.csv"
                        if master_wide.exists():
                            m_df = pd.read_csv(master_wide)
                            row_match = m_df[m_df["unit_id"] == unit.unit_id]
                            if len(row_match) > 0:
                                wide_rows.append(row_match.iloc[0].to_dict())
                    continue
            except Exception:
                pass

        cache_key = (unit.animal_id, unit.depth, unit.rec_name)
        if cache_key not in animal_cache:
            d_info = f" (Depth {unit.depth})" if unit.depth is not None else ""
            print(
                f"Loading & precomputing stimulus basis features for animal: {unit.animal_id}{d_info} {unit.rec_name} ..."
            )
            animal_cache[cache_key] = prepare_animal_base_features(
                data_dir=data_dir,
                animal_id=unit.animal_id,
                depth=unit.depth,
                rec_name=unit.rec_name,
            )
        res = fit_and_evaluate_unit(
            unit=unit,
            animal_bundle=animal_cache[cache_key],
            out_dir=out_dir,
            n_splits=n_splits,
            reg_strength=reg_strength,
            filter_sessions=filter_sessions,
            fig_formats=fig_formats,
        )
        all_factor_dfs.append(res["factor_df"])
        wide_rows.append(res["wide_row"])
        del res
        import gc

        gc.collect()
        jax.clear_caches()

    pop_long_df = pd.concat(all_factor_dfs, ignore_index=True)
    pop_wide_df = pd.DataFrame(wide_rows)

    # Merge with existing population tables if present to aggregate across channels, depths, and animals
    long_csv = out_dir / "population_factors_long.csv"
    if long_csv.exists():
        try:
            old_long = pd.read_csv(long_csv)
            cur_uids = set(pop_long_df["unit_id"])
            kept_long = old_long[~old_long["unit_id"].isin(cur_uids)]
            pop_long_df = pd.concat([kept_long, pop_long_df], ignore_index=True)
        except Exception:
            pass
    pop_long_df.to_csv(long_csv, index=False)

    wide_csv = out_dir / "population_units_summary.csv"
    if wide_csv.exists():
        try:
            old_wide = pd.read_csv(wide_csv)
            cur_uids = set(pop_wide_df["unit_id"])
            kept_wide = old_wide[~old_wide["unit_id"].isin(cur_uids)]
            pop_wide_df = pd.concat([kept_wide, pop_wide_df], ignore_index=True)
        except Exception:
            pass
    pop_wide_df.to_csv(wide_csv, index=False)
    print(f"\nSaved population summary tables in: {out_dir}")
    return pop_long_df, pop_wide_df


# Default unit registry (add new channels or new animals here, or pass via CLI!)
DEFAULT_UNITS = [
    UnitSpec(
        animal_id="F1812_Nala",
        rec_name="Rec_A",
        channel_num=26,
        region_label="Left — Auditory",
        min_rate_hz=15.0,
        max_silent_gap_s=0.40,
    ),
    UnitSpec(
        animal_id="F1812_Nala",
        rec_name="Rec_A",
        channel_num=5,
        region_label="Left — Visual",
        min_rate_hz=10.0,
        max_silent_gap_s=0.50,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean unified NeMoS GLM across one or more units/animals."
    )
    parser.add_argument(
        "--animal",
        type=str,
        default="F1812_Nala",
        help="Animal ID (default: F1812_Nala)",
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
        help="One or more 1-based channel numbers to fit (e.g., --channels 26 5). If omitted and --all-channels is set, loads from diagnostics summary table.",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Automatically load and fit all channels flagged with run_glm==1 in the diagnostics summary table",
    )
    parser.add_argument(
        "--from-diagnostics",
        type=pathlib.Path,
        default=None,
        help="Explicit path to population_diagnostics_summary.csv",
    )
    parser.add_argument(
        "--min-rate", type=float, default=5.0, help="Minimum trial firing rate QC (Hz)"
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.50,
        help="Max silent gap QC for wireless dropout (s)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds (default: 5)",
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=1e-3,
        help="Ridge regularization strength (default: 1e-3)",
    )
    parser.add_argument(
        "--filter-sessions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out recording blocks that fail two-gate session QC (default: True)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save editable vector PDF figures alongside PNG dashboards (ideal for paper figure editing)",
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
    animal_id = args.animal or "F1812_Nala"
    rec_name = args.rec or "Rec_A"

    if args.channels is not None:
        units = [
            UnitSpec(
                animal_id=animal_id,
                rec_name=rec_name,
                channel_num=ch,
                depth=args.depth,
                min_rate_hz=args.min_rate,
                max_silent_gap_s=args.max_gap,
            )
            for ch in args.channels
        ]
    elif args.all_channels or (args.from_diagnostics is not None):
        depth_tag = f"_Depth{args.depth}" if args.depth is not None else ""
        diag_candidates = [
            DIAG_DIR
            / f"{animal_id}{depth_tag}_{rec_name}_population_diagnostics_summary.csv",
            DIAG_DIR / f"{animal_id}_{rec_name}_population_diagnostics_summary.csv",
        ]
        summary_path = args.from_diagnostics
        if summary_path is None:
            for cand in diag_candidates:
                if cand.exists():
                    summary_path = cand
                    break
            if summary_path is None:
                summary_path = diag_candidates[0]

        if not summary_path.exists():
            raise FileNotFoundError(
                f"Diagnostics summary table not found: {summary_path}.\n"
                f"Please run diagnostics first, or pass explicit '--channels'."
            )
        diag_df = pd.read_csv(summary_path)
        if "run_glm" not in diag_df.columns:
            raise KeyError(f"'run_glm' column not found in {summary_path}.")

        glm_channels_df = diag_df.loc[diag_df["run_glm"] == 1]
        ch_list = []
        for _, r in glm_channels_df.iterrows():
            lbl = str(r["channel_name"])
            if "Channel " in lbl:
                ch_num = int(lbl.split("Channel ")[1].split()[0])
            elif "_C" in lbl:
                ch_num = int(lbl.split("_C")[-1].split()[0].replace(")", ""))
            else:
                ch_num = int(r["channel_idx"]) + 1
            ch_list.append(ch_num)

        n_excl = len(diag_df) - len(ch_list)
        print(f"\n[Diagnostics Filter] Loaded {summary_path.name}:")
        print(
            f"  Selected {len(ch_list)} channels flagged with run_glm==1 ({n_excl} channels excluded)."
        )
        print(f"  Channels: {ch_list}\n")
        units = [
            UnitSpec(
                animal_id=animal_id,
                rec_name=rec_name,
                channel_num=ch,
                depth=args.depth,
                min_rate_hz=args.min_rate,
                max_silent_gap_s=args.max_gap,
            )
            for ch in ch_list
        ]
    else:
        units = DEFAULT_UNITS

    if args.pdf or args.fig_format == "both":
        fig_formats = ("png", "pdf")
    elif args.fig_format == "pdf":
        fig_formats = ("pdf",)
    else:
        fig_formats = ("png",)

    run_population_pipeline(
        units=units,
        data_dir=DATA_DIR,
        out_dir=CLEAN_OUT_DIR,
        n_splits=args.folds,
        reg_strength=args.reg,
        filter_sessions=args.filter_sessions,
        fig_formats=fig_formats,
    )


if __name__ == "__main__":
    main()
