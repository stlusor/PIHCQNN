# Repeated benchmarks and supplementary baselines

This module supplies the final supplementary experiment scripts and archived individual-run results used in manuscript revision 10. The original manuscript scripts and original figures are preserved in the release's original-code section.

## Files and manuscript mapping

| Manuscript item | Archived results |
| --- | --- |
| Table 1, five-run classical heat benchmarks | `reviewer2_experiments/results/heat_pinn/` |
| Table 2, five-run Poisson benchmarks; corresponding Table 5 timings | `reviewer2_experiments/results/poisson_pinn/` and `poisson_qpinn/` |
| Table 4, heat variants | `reviewer1_experiments/results/heat/` |
| Table 4, Poisson variants | `reviewer1_experiments/results/poisson/` plus `poisson_variants/fourier05_seed*.json` |
| Table 4, inverse elastostatic variants | `reviewer1_experiments/results/inverse/` plus `inverse_variants/fourier05_seed*.json` |
| Table 6, reuploading count and supervised-frequency diagnostics | `reviewer1_experiments/results/fourier/` |
| Per-layer parameter counts for the heat, Poisson and original scalar-inverse architectures | `reviewer1_experiments/tables/r2_1_layer_parameters.csv` |

The Fourier-feature variant uses a Gaussian feature scale of 4.0 for the heat case and 0.5 for Poisson and inverse elastostatics. The Poisson and inverse `fourier05` records supply the values reported in Table 4. Classical modern baselines use three seeds and population standard deviations; the repeated benchmarks use five seeds and sample standard deviations. The Fourier summaries use three seeds and population standard deviations. Errors inside JSON and numerical summary CSV files are fractions; manuscript percentage values are obtained by multiplying by 100.

`modern_baselines_summary.csv` is regenerated from the selected final JSON records and includes individual-run training-time statistics. `Reviewer2_Repeatability_Summary.csv` includes the median seed for each repeated benchmark. The original figures retain their original source-run data, and the median identifiers describe these added repetitions.

## Implementation and environment

The supplementary neural runs used Python 3.11.11, PyTorch 2.5.1+cu121 and an NVIDIA GeForce RTX 4070 with 12 GB of memory. The manual quantum implementation uses float32 real parameters and complex64 statevectors. PennyLane 0.45.1 supplies the reference circuit for verification. Installation follows the release's environment instructions.

- `common.py`: neural blocks, automatic differentiation and hybrid model.
- `quantum_layer.py`: differentiable batched PyTorch implementation of the Rot, RY and strongly entangling circuit.
- `classical_baselines.py`: tanh, SIREN, adaptive sinusoidal activation, Fourier-feature and gradient-balancing baselines.
- `poisson_experiment.py`: the repeated PIHCQNN Poisson benchmark.
- `fourier_diagnostic.py`: supervised fits to the two frequency targets, for n = 1, 2, 3 and classical controls.
- `analyze_fourier.py`: frequency amplitudes, error statistics and threshold-step aggregation from the archived records.

The archived JSON files preserve the originally recorded metrics and timings. Device-dependent timings can vary when rerunning on another machine. Each script selects CUDA when available and otherwise selects CPU.

## Run the experiments

`jobs_baselines.json` lists 109 explicit single-run configurations: 10 heat repetitions, 15 Poisson repetitions, 48 modern-baseline runs and 36 frequency fits. Paths are relative to the `supplementary` directory. The central runner creates each output directory and substitutes `{output}` with the selected output path. Reruns write to `rerun_results/`, preserving the archived results.

All heat jobs explicitly use 50,000 Adam updates, a 50 by 50 residual grid and learning rate 0.002. All Poisson jobs explicitly use 5,000 updates, a 40 by 40 residual grid and learning rate 0.002. Residual evaluation uses the full collocation set (`--batch 0`). Modern scalar-inverse jobs use 20,000 updates, 50 observations and learning rate 0.005. Frequency jobs use 1,500 updates, 201 labelled points on [-pi, pi] and learning rate 0.002.

For example, from `supplementary`:

```sh
python reviewer1_experiments/classical_baselines.py --case heat --model fourier --seed 0 --steps 50000 --grid 50 --lr 0.002 --batch 0 --out rerun_results/modern_heat/fourier_seed0.json
python reviewer1_experiments/fourier_diagnostic.py --target f1715 --model pih3 --seed 0 --steps 1500 --lr 0.002 --out rerun_results/frequency/f1715_pih3_seed0.json
python reviewer1_experiments/quantum_layer.py
python reviewer1_experiments/analyze_fourier.py
```

The Poisson quantum script expects its output parent directory to exist; the central runner supplies this directory. Its jobs explicitly set `--batch 0`.

## Publication copy

The training algorithms and archived individual-run JSON records are preserved from their final local sources. The frequency-analysis paths were made relative to `__file__`. The curated job list supplies explicit settings in place of exploratory batch-driver defaults. Summary files are selected for the final manuscript values.
