"""Quickstart demo: fetch NWB head-direction data and fit NeMoS GLMs."""

import pathlib
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# Recommended JAX precision for GLM optimization
jax.config.update("jax_enable_x64", True)
nap.nap_config.suppress_conversion_warnings = True
warnings.filterwarnings("ignore", message="Ignoring cached namespace.*", category=UserWarning)


def main() -> None:
    print(f"NeMoS version: {nmo.__version__}")
    print(f"JAX version:   {jax.__version__} (devices: {jax.devices()})")
    print(f"Pynapple:      {nap.__version__}\n")

    # 1. Fetch and load example NWB dataset (anterodorsal thalamic head-direction cells)
    print("Fetching dataset 'A2929-200711.nwb'...")
    nwb_path = nmo.fetch.fetch_data("A2929-200711.nwb")
    print(f"Loaded from: {nwb_path}\n")

    data = nap.load_file(nwb_path)
    print("NWB contents:")
    print(data)

    spikes = data["units"]
    head_dir = data["ry"]
    print(f"\nNumber of units: {len(spikes)}")
    print(f"Head-direction time support: {head_dir.time_support}\n")

    # 2. Preprocess with pynapple: 10 ms bins restricted to head-direction tracking epoch
    bin_size = 0.01  # seconds
    binned_hd = head_dir.bin_average(bin_size)
    unit_id = 6
    unit_counts = spikes[unit_id].count(bin_size, ep=head_dir.time_support)
    pop_counts = spikes.count(bin_size, ep=head_dir.time_support)

    # 3. Construct cyclic B-spline basis features for head direction in [0, 2*pi]
    basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=10)
    X = basis.compute_features(binned_hd)
    print(f"Design matrix X shape: {X.shape} (time bins x basis features)")

    # 4. Fit a single-neuron Poisson GLM on unit 6
    print(f"\nFitting single-neuron Poisson GLM on unit {unit_id}...")
    glm = nmo.glm.GLM().fit(X, unit_counts)
    ll = float(glm.score(X, unit_counts, score_type="log-likelihood"))
    pr2 = float(glm.score(X, unit_counts, score_type="pseudo-r2-McFadden"))
    print(f"  Learned coef shape:      {glm.coef_.shape}")
    print(f"  Learned intercept:       {np.asarray(glm.intercept_)}")
    print(f"  Mean log-likelihood:     {ll:.4f}")
    print(f"  McFadden pseudo-R^2:     {pr2:.4f}")

    # 5. Fit a PopulationGLM simultaneously across all units
    print(f"\nFitting PopulationGLM across all {pop_counts.shape[1]} units...")
    pop_glm = nmo.glm.PopulationGLM().fit(X, pop_counts)
    pop_ll = float(pop_glm.score(X, pop_counts, score_type="log-likelihood"))
    pop_pr2 = float(pop_glm.score(X, pop_counts, score_type="pseudo-r2-McFadden"))
    print(f"  Population coef shape:   {pop_glm.coef_.shape}")
    print(f"  Population intercept:    {pop_glm.intercept_.shape}")
    print(f"  Mean log-likelihood:     {pop_ll:.4f}")
    print(f"  McFadden pseudo-R^2:     {pop_pr2:.4f}")

    # 6. Compare empirical vs. GLM-predicted head-direction tuning curve for unit 6
    raw_tuning = nap.compute_tuning_curves(
        spikes, head_dir, bins=100, feature_names=["angles"]
    ).sel(unit=unit_id)

    pred_rate_hz = glm.predict(X)[:, np.newaxis] * X.rate
    model_tuning = nap.compute_tuning_curves(
        pred_rate_hz, head_dir, bins=100, feature_names=["angles"]
    ).sel(unit=0)

    out_plot = pathlib.Path(__file__).parent / "head_direction_glm_tuning.png"
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(raw_tuning.angles, raw_tuning.values, label="Empirical", linewidth=2)
    ax.plot(
        model_tuning.angles,
        model_tuning.values,
        label="NeMoS Poisson GLM",
        linewidth=2,
        linestyle="--",
    )
    ax.set_title(f"Unit {unit_id} Head-Direction Tuning (Hz)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved tuning curve comparison plot to: {out_plot}")


if __name__ == "__main__":
    main()
