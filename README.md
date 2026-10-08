# PIHCQNN: original code and revision experiments

Code and archived supplementary results for **Integrating a quantum layer into a physics-informed neural network for solving forward and inverse problems involving differential equations**.

This repository combines the original author code archive with the final supplementary experiments used in the revised manuscript and reviewer responses, version 10. `MANUSCRIPT_MAP.csv` connects the scripts and result files to the manuscript tables and sections.

## Repository layout

```text
original_code/
  archive/PIHCQNN.rar              Original author archive
  src/                            Original Python scripts
  inventory.json                  Original file identities and checksums
supplementary/
  reviewer1_experiments/           Modern baselines and frequency diagnostics
  reviewer2_experiments/           Five-seed heat and Poisson repetitions
  reviewer11_experiments/          Elastostatics, sensitivity, gradients and solvers
  reviewer_supplement_20261007/
    fixed_weights/                Boundary-weight scan and 18 saved runs
    gradient_audit/               Training-gradient statistics
    flow_controls/                Final classical flow controls
  flow_reproduction_20261007/      Hybrid flow code, reference data and final states
  jobs_baselines.json             Explicit benchmark recipes
  jobs_training.json              Explicit training-diagnostic recipes
  run_experiments.py              Central supplementary runner
  run_flow_final.py               Staged shared-protocol flow runner
  export_results.py               Rebuild the supplementary result tables
  publication_results/            Tables exported from the archived records
MANUSCRIPT_MAP.csv
requirements.txt
verify_release.py
release_manifest.json
SHA256SUMS.txt
validation/                       Packaging and short-run verification records
```

## Environment

The supplementary experiments used Python **3.11.11**, PyTorch **2.5.1+cu121**, PennyLane **0.45.1**, NumPy **2.4.3** and SciPy **1.17.1**. GPU runs used an **NVIDIA GeForce RTX 4070, 12 GB**. The fixed-weight scan and the classical supplementary flow controls ran on CPU. Each experiment's configuration identifies its dtype and execution settings; `environment.json` records the environment checked when this repository was assembled.

Create a Python 3.11 environment and install the dependencies:

```console
python -m pip install -r requirements.txt
```

Select a CUDA-enabled PyTorch 2.5.1 installation for GPU execution and the PIHCQNN CUDA-graph flow continuation. Most supplementary neural scripts also select CPU when CUDA is unavailable. Their recorded GPU elapsed times refer to the original runs.

`original_code/README.md` describes the historical scripts, including their legacy `default.qubit.torch` device and author-specific paths. The original archive is preserved byte-for-byte; its runtime configuration is documented separately from the current supplementary environment.

## Inspect results and run experiments

From the repository root:

```console
python verify_release.py
python supplementary/run_experiments.py --list
python supplementary/export_results.py
python supplementary/verify_flow_release.py
```

Select a suite and seed to display its commands. Add `--execute` to start the selected training runs:

```console
python supplementary/run_experiments.py --suite modern_poisson --seed 0
python supplementary/run_experiments.py --suite modern_poisson --seed 0 --execute
python supplementary/run_experiments.py --suite elastostatic --seed 0 --execute
python supplementary/run_experiments.py --suite fixed_weights --seed 0 --execute
python supplementary/run_flow_final.py --model all --execute
```

Fresh runs write to `supplementary/rerun_results/`. Archived run records remain in their respective experiment directories. All seeds, iteration budgets, learning rates, feature scales and output paths are explicit in the job files. Module-specific instructions are in `supplementary/README_baselines.md`, `README_training.md` and `README_flow.md`.

The flow verifier can regenerate predictions from the three final model states:

```console
python supplementary/verify_flow_release.py --checkpoints
```

## Results and provenance

The original archive covers heat, Poisson and inverse elastostatics. The supplementary flow implementation supplies the shared-protocol comparison added during revision. The manuscript retains the original flow result and Figure 10, alongside that supplementary comparison.

The flow reference files and mixed-variable source come from [Raocp/PINN-laminar-flow](https://github.com/Raocp/PINN-laminar-flow), commit `d34fc037f16a8e79dd8c01a0e3dd6389297634ee`. Their upstream README, file URLs and hashes are retained under `supplementary/flow_reproduction_20261007/source/`. The author checkpoint in that source directory is part of the upstream record; the final supplementary training states are identified in `README_flow.md`.

Raw result JSON files retain their recorded numerical precision. Exported tables identify percentage conversions and the standard-deviation convention of each experiment group. The source mappings and checksums provide a trace from each final result to its underlying files.

## Verification

`validation/` records syntax and file-integrity checks, brief supplementary training checks, the quantum-circuit comparison against PennyLane, and regeneration of the final flow predictions. The brief runs check execution and output generation; the full numerical results are preserved in the archived experiment records.
