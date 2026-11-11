"""Generate an educational & publication-ready figure illustrating the 4 Raised-Cosine
Log-Stretched Basis Functions (0-200 ms) used in `fit_glm_passive.py`.
"""

import argparse
import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
import numpy as np
import nemos as nmo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the 0-200 ms raised-cosine log-stretched basis functions."
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save editable vector PDF alongside PNG",
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
    if args.pdf or args.fig_format == "both":
        fig_formats = ("png", "pdf")
    elif args.fig_format == "pdf":
        fig_formats = ("pdf",)
    else:
        fig_formats = ("png",)
    window_bins = 10
    n_basis_funcs = 4
    bin_ms = 20.0
    lags_ms = np.arange(window_bins) * bin_ms  # 0, 20, ..., 180 ms (10 bins covering 0-200 ms)

    basis_obj = nmo.basis.RaisedCosineLogEval(n_basis_funcs=n_basis_funcs)

    # Exact discrete basis evaluated on 10 bins as in fit_glm_passive.py
    u_disc, bk_disc = basis_obj.evaluate_on_grid(window_bins)
    bk_disc = np.array(bk_disc, dtype=np.float64)
    col_sums = bk_disc.sum(axis=0, keepdims=True)
    bk_disc_norm = bk_disc / col_sums

    # Smooth curve evaluated on the exact same [0, 1] domain scaled by the discrete column sums
    u_fine, bk_fine = basis_obj.evaluate_on_grid(300)
    bk_fine_norm = np.array(bk_fine, dtype=np.float64) / col_sums
    lags_fine_ms = np.asarray(u_fine) * ((window_bins - 1) * bin_ms)

    # Graduated shades of cyan (neutral w.r.t. orange/brown Auditory and blue/purple Visual palettes)
    colors = ["#4dd0e1", "#00acc1", "#007c91", "#004d5a"]
    labels = [
        r"Basis 1 ($B_1$: Immediate bin, 0 ms)",
        r"Basis 2 ($B_2$: Short lag, 0–20 ms)",
        r"Basis 3 ($B_3$: Medium lag, peak 20–40 ms)",
        r"Basis 4 ($B_4$: Broad late tail, 40–160 ms)",
    ]

    fig = plt.figure(figsize=(14.5, 9.8), dpi=180)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.05, 1.0], hspace=0.36, wspace=0.24
    )

    # -------------------------------------------------------------------------
    # Panel A: The 4 Raised-Cosine Log Basis Bumps
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    for b in range(n_basis_funcs):
        ax_a.plot(
            lags_fine_ms,
            bk_fine_norm[:, b],
            color=colors[b],
            lw=1.8,
            alpha=0.45,
            ls="--",
        )
        ax_a.plot(
            lags_ms,
            bk_disc_norm[:, b],
            "o-",
            color=colors[b],
            lw=2.5,
            ms=6.5,
            label=labels[b],
        )
        ax_a.fill_between(lags_ms, 0, bk_disc_norm[:, b], color=colors[b], alpha=0.11)

    ax_a.set_title(
        "A. 4 Causal Raised-Cosine Log Basis Functions (0–200 ms)\n"
        r"[Unit-sum normalized across 20 ms bins: $\sum_\tau B_b(\tau) = 1$]",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax_a.set_xlabel(r"Time Lag $\tau$ after Stimulus Event (ms)", fontsize=10)
    ax_a.set_ylabel(r"Basis Weight $B_b(\tau)$", fontsize=10)
    ax_a.set_xticks(np.arange(0, 181, 20))
    ax_a.set_xlim(-6, 186)
    ax_a.set_ylim(0, 1.08)
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(loc="upper right", fontsize=8.7, framealpha=0.94)

    # -------------------------------------------------------------------------
    # Panel B: How Weighted Sums Reconstruct Flexible Neural Filters h(tau)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ex_weights = [
        (
            r"Fast Auditory ON ($w = [+0.03, +0.10, +0.01, 0.00]$)",
            np.array([0.03, 0.10, 0.01, 0.00]),
            "#d95f02",
            "-",
        ),
        (
            r"Biphasic Excitation–Suppression ($w = [+0.07, +0.04, -0.05, -0.02]$)",
            np.array([0.07, 0.04, -0.05, -0.02]),
            "#8c510a",
            "--",
        ),
        (
            r"Medium-Latency Visual Flash ($w = [0.00, +0.02, +0.11, +0.03]$)",
            np.array([0.00, 0.02, +0.11, 0.03]),
            "#7b68ee",
            "-",
        ),
        (
            r"Slow Broad Integration ($w = [0.00, 0.00, +0.03, +0.12]$)",
            np.array([0.00, 0.00, 0.03, 0.12]),
            "#009e73",
            "-.",
        ),
    ]
    ax_b.axhline(0, color="black", lw=0.9, ls=":")
    for lbl, w, col, ls in ex_weights:
        h_disc = bk_disc_norm @ w
        ax_b.plot(
            lags_ms,
            h_disc,
            marker="o",
            ms=5.5,
            color=col,
            lw=2.3,
            ls=ls,
            label=f"{lbl}  ($\\beta_{{\\mathrm{{net}}}} = {w.sum():+.2f}$)",
        )

    ax_b.set_title(
        r"B. Reconstructed Impulse Responses: $h(\tau) = \sum_{b=1}^4 w_b B_b(\tau)$"
        "\n[Weighted combinations capture fast, delayed, or biphasic filters]",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax_b.set_xlabel(r"Time Lag $\tau$ after Stimulus Feature (ms)", fontsize=10)
    ax_b.set_ylabel(r"Filter Log-Rate Weight $h(\tau)$", fontsize=10)
    ax_b.set_xticks(np.arange(0, 181, 20))
    ax_b.set_xlim(-6, 186)
    ax_b.set_ylim(-0.045, 0.165)
    ax_b.grid(True, alpha=0.25)
    ax_b.legend(loc="upper right", fontsize=7.6, framealpha=0.94)

    # -------------------------------------------------------------------------
    # Panel C: Convolving a Stimulus Stream with the 4 Basis Bumps
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, :])
    T_bins = 50  # 1.0 second window (50 x 20ms bins)
    t_ms = np.arange(T_bins) * bin_ms
    stim = np.zeros(T_bins)
    # Example 1: sharp single-bin onset at t = 100 ms
    stim[5] = 2.0
    # Example 2: broader multi-bin modulation pulse at t = 500..560 ms
    stim[25:29] = np.array([0.8, 1.8, 1.5, 0.6])

    # Prepend window_bins padding zeros (just as fit_glm_passive.py uses 10 pre-stim padding bins)
    stim_padded = np.concatenate([np.zeros(window_bins), stim])
    conv_padded = np.asarray(
        nmo.convolve.create_convolutional_predictor(
            bk_disc_norm,
            stim_padded[:, None],
            predictor_causality="causal",
            shift=False,
            axis=0,
        )
    )[:, 0, :]
    conv_out = conv_padded[window_bins:, :]

    ax_c.bar(
        t_ms,
        stim,
        width=14,
        color="0.35",
        alpha=0.26,
        edgecolor="0.2",
        lw=1.0,
        label=r"Input Stimulus Stream $s(t)$ (e.g. $+d\mathrm{Audio}/dt$ or $+d\mathrm{Lum}/dt$)",
    )
    for b in range(n_basis_funcs):
        ax_c.plot(
            t_ms,
            conv_out[:, b],
            "o-",
            ms=4,
            lw=2.1,
            color=colors[b],
            label=f"Convolved Predictor $X_{b+1}(t) = (s * B_{b+1})(t)$",
        )

    ax_c.annotate(
        "1. Single-bin onset (t = 100 ms)\ncreates 4 staggered basis waves\nspanning 100 to 260 ms",
        xy=(140, 1.25),
        xytext=(210, 1.55),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="0.25"),
        fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="0.75"),
    )
    ax_c.annotate(
        "2. Multi-bin stimulus pulse (500–560 ms)\nis smoothly integrated across\nfast & slow basis channels",
        xy=(580, 1.05),
        xytext=(660, 1.55),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="0.25"),
        fontsize=9.2,
        bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="0.75"),
    )

    ax_c.set_title(
        "C. Convolving a Stimulus Stream with the 4 Basis Bumps\n"
        "[Each stimulus event generates 4 staggered predictor columns in the GLM Design Matrix $X$]",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax_c.set_xlabel("Trial Time $t$ (ms)", fontsize=10)
    ax_c.set_ylabel("Predictor Amplitude (SD units)", fontsize=10)
    ax_c.set_xlim(-20, 980)
    ax_c.set_ylim(0, 2.55)
    ax_c.grid(True, alpha=0.25)
    ax_c.legend(loc="upper left", ncol=2, fontsize=8.6, framealpha=0.94)

    fig.suptitle(
        "Raised-Cosine Log-Stretched Temporal Basis Expansion (fit_glm_passive.py)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = (
        script_dir
        if (script_dir / "data_tables").exists() or (script_dir / "_data_tables").exists()
        else script_dir.parent
    )
    out_dir = (
        repo_root / "output"
        if (repo_root / "output").exists()
        else (repo_root / "extracted_data" if (repo_root / "extracted_data").exists() else repo_root / "output")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in fig_formats:
        out_path = out_dir / f"basis_functions_explained.{ext}"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved figure to: {out_path.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()
