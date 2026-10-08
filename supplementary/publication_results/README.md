# Publication result summaries

These CSV files assemble the archived repeated runs and supplementary experiments used in the revised manuscript. Paths in a CSV's `source_files` column are relative to `supplementary/`. The root `MANUSCRIPT_MAP.csv` connects manuscript locations to code, configuration and results.

Regenerate the summaries from any working directory using:

```sh
python /path/to/repository/supplementary/export_results.py
python /path/to/repository/supplementary/export_results.py --out /path/to/results
```

The exporter uses Python's standard library. It reads JSON and CSV records, computes descriptive statistics and reconstructs the converged FEM displacement field. It performs zero training updates and preserves the archived records.

## Files and statistical conventions

| Output | Manuscript content | SD convention used in manuscript |
| --- | --- | --- |
| `table01_heat_supplementary_repeats.csv` | Table 1, repeated tanh PINNs | Sample SD, ddof=1, five seeds |
| `table02_poisson_repeats.csv` | Table 2, PIHCQNN and tanh PINNs | Sample SD, ddof=1, five seeds |
| `table04_modern_baselines.csv` | Table 4, modern classical baselines | Population SD, ddof=0, three seeds |
| `table05_and_supplementary_recorded_costs.csv` | Recorded parameter counts and elapsed times | Convention follows the source experiment group |
| `table06_frequency_diagnostics.csv` | Table 6 and supplementary frequency controls | Population SD, ddof=0, three seeds |
| `table07_elastostatic_controls.csv` | Table 7 | Sample SD, ddof=1, three seeds |
| `table08_sensitivity_and_ansatz.csv` | Table 8 | Sample SD, ddof=1, two seeds |
| `table09_gradient_diagnostics.csv` | Table 9 | Sample SD, ddof=1 |
| `table10_classical_solvers.csv` | Table 10 | Individual deterministic solutions |
| `dem_energy_method.csv` | Section 3.5.6, energy method | Sample SD, ddof=1, three seeds |
| `table11_variable_inverse.csv` | Table 11 | Neural runs: sample SD, ddof=1, three seeds; FEM: one deterministic solution |
| `table11_fem_reconstructed_displacement.csv` | FEM displacement field on 201 evaluation points | Reconstructed from archived converged coefficients |
| `fixed_weight_study_cpu.csv` | Section 3.5.4, fixed boundary weights | Sample SD, ddof=1, three fresh CPU runs per setting |
| `flow_common_observation_final.csv` | Section 3.4, common-observation comparison | One seed, 1234 |

The named `*_sd_sample` and `*_sd_population` columns expose both conventions. The `sd_ddof` and `*_sd_manuscript` columns select the manuscript convention. SD cells for individual solutions are left blank. Error ratios in the archived JSON are converted to percentages by multiplication by 100. Boundary norms, residual MSEs, gradient norms and amplitude ratios retain their native units.

Table 4's final Fourier-feature variants use `poisson_variants/fourier05` and `inverse_variants/fourier05`. The heat Fourier-feature baseline uses `results/heat/fourier`. The earlier Fourier variants for Poisson and inverse elastostatics have a separate configuration.

For Table 6, the archived float32 reference-amplitude convention is recovered from the recorded `fourier_summary.csv` ratios and the individual final amplitudes. Threshold times use recorded checkpoints. A run below half amplitude at the budget is marked as censored; the manuscript prints `>1500` when all runs in the group remain below that threshold.

Table 9 combines a circuit-only initialization diagnostic with n=3 and a full physics-informed training diagnostic with n=1. These are separate measurements. The ten-qubit endpoint sample SD is computed from the two stored quantum-gradient norms and gives 44.078 after rounding to three decimals.

The FEM reconstruction uses the archived fitted coefficient vector, midpoint element coefficients, endpoint-trapezoidal loads, a 100-element linear mesh, homogeneous boundary values and linear interpolation to 201 points. It solves the forward tridiagonal system once and leaves the inverse optimization unchanged. The resulting displacement error is approximately 0.006522386%.

The flow summary uses held-out velocity errors with every row sharing an observed coordinate excluded. The hybrid physics-training record comprises 600 updates followed by a 2,000-update continuation that resumes the optimizer state. Its total closure count is assembled from the data-initialization result, the interrupted segment's history and the continuation result. Comparative elapsed times are confined to the recorded experiment groups; the flow summary reports accuracy and optimizer counts.

The historical heat PIHCQNN summary, original inverse results in Table 3, original flow results and approximate historical training times remain manuscript-reported values. This export covers the individual records present in the release. `export_coverage.json` lists output coverage and any missing input records. `full_training_gradient_audit.json` preserves the archived checkpoint audit.
