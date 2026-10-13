# Robust Beamforming via Low-rank and Sparse Recovery

This MATLAB demo extracts the proposed beamformer and the output SINR versus input SIR experiment from the original implementation. It compares **Proposed**, **MVDR**, and an **Optimal** reference. It is a reduced demonstration, not the complete comparison with all baselines in the original experiment.

## Files

- `beamformer_proposed.m`: alternating low-rank factorization, sparse recovery, and beamforming updates.
- `SINR_SIR.m`: signal generation, Monte Carlo evaluation, plotting, and result export.
- `results/`: generated numerical results and figures.

## Requirements and usage

The demo uses MATLAB built-in functions and does not require CVX, Statistics and Machine Learning Toolbox, or Parallel Computing Toolbox.

Open this folder in MATLAB and run:

```matlab
SINR_SIR
```

The default is `MC = 1500` trials per JNR value. For a quick check, temporarily reduce `MC` near the top of the script. The script uses the current MATLAB random stream without resetting it, as in the original source. Results may vary between runs.

The script writes `results/SINR_SIR.mat`, `results/SINR_SIR.fig`, and `results/SINR_SIR.png`. Re-running the script overwrites these outputs.

Before removal of the explicit random-state initialization, the demo was validated in MATLAB R2024b with 3 trials at each of the 7 JNR values. The smoke test checked finite SINR values, the distortionless constraint, the oracle reference bounds, and MAT/FIG/PNG export. All five outputs of the cleaned proposed function exactly matched the original function for identical input and random initialization. The default 1500-trial experiment has not been run as part of this validation; the smoke test does not establish statistical performance.

## Proposed algorithm

```matlab
[U, V, S, w, L] = beamformer_proposed(Y, a0, r, eta, mu1, mu2, Kmax);
```

For an M-by-N training matrix `Y`, the algorithm initializes rank-r factors `U` and `V`, then alternates the following updates:

1. Update `U` and `V` by least squares using `Y-S`.
2. Update the sparse component `S` by hard thresholding the stacked real and imaginary residual components. Adapt `mu1` using the original median absolute deviation (MAD) rule.
3. Update the auxiliary matrix `T` using `S` and the current beamforming weights.
4. Form a covariance matrix from `Y-T`, apply diagonal loading, and update the distortionless weight vector `w`.

The iteration count is fixed at `Kmax`; no convergence stopping rule is used. The returned `L` is `Y-S`, not `U*V`. The covariance used to update `w` is based on `Y-T`. The original threshold formula and matrix updates are retained.

## Experiment settings

| Parameter | Value |
| --- | --- |
| Array | 16-element uniform linear array |
| Carrier frequency | 100 MHz |
| Element spacing | Half wavelength |
| Desired / interference direction | 0 / 30 degrees |
| Total snapshots | 64 |
| Training / evaluation snapshots | 32 / 32 |
| Nominal input SNR | -10 dB |
| JNR sweep | 0:5:30 dB |
| Nominal input SIR | SNR - JNR, from -10 to -40 dB |
| Training impulse probability | 0.1 |
| Impulsive/background standard deviation ratio | 100 |
| Monte Carlo trials per point | 1500 |
| Proposed parameters | r=2, eta=0.01, mu1=100, mu2=1, Kmax=100 |
| Relative diagonal loading | 0.1 for both MVDR and Proposed |

The first 32 snapshots contain Gaussian mixture noise and are used to estimate beamforming weights. The last 32 snapshots contain only background Gaussian noise and provide the interference and noise powers for evaluation. The noise scaling follows the original implementation and depends on the realized desired-signal power. The JNR and SNR labels therefore represent nominal generation settings, not ratios to background Gaussian noise alone.

For each trial, the evaluation covariance is the sum of the sample interference covariance and a spatially white noise covariance scaled by the realized background noise power. The Optimal curve uses an oracle MVDR weight computed from this same covariance.

Output SINR is computed as:

```matlab
SINR = P_sig * abs(w' * as)^2 / abs(w' * R_in * w);
```

The plotted statistic is `10*log10(mean(SINR))`: trials are averaged in the linear domain before conversion to dB.

## Changes from the source experiment

- Retained only Proposed, MVDR, and Optimal.
- Preserved the proposed algorithm's initialization, update formulas, loading, and default parameters.
- Removed unused variables, obsolete commented formulas, and per-iteration console output.
- Added English comments and automatic result export.
- Used a serial `for` loop to avoid a Parallel Computing Toolbox dependency.
- Replaced `binornd(1, c_1, M, N_half)` with `rand(M, N_half) < c_1`, which generates the same Bernoulli distribution without a toolbox dependency. Exact sample sequences are not claimed to match the original experiment.
