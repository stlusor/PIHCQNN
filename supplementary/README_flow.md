# Supplementary steady-flow comparison

This directory preserves the shared-protocol comparison added during revision, together with its final checkpoints, predictions, sampling indices and training stages.

## Source and sampling

The reference velocity field is `flow_reproduction_20261007/source/FluentReferenceMu002/FluentSol.mat`, from [Raocp/PINN-laminar-flow](https://github.com/Raocp/PINN-laminar-flow), commit `d34fc037f16a8e79dd8c01a0e3dd6389297634ee`. The copied upstream README and `source_manifest.json` identify the individual files and their hashes.

The fixed observation subset contains 193 points: `floor(0.01 × 19340)`, sampled without replacement with NumPy `RandomState(1234)`. All three models use these observations throughout training. The saved held-out set contains 19,145 rows and excludes every row at a sampled coordinate, including two repeated reference rows. The package contains 1,024 residual points, 256 wall points, 64 inlet points and 64 outlet points. `protocol/points.npz` and the `.npy` indices preserve the exact arrays used.

The supplementary models return five fields: streamfunction, pressure and three stresses. Velocity is computed from derivatives of the streamfunction. Training uses the 193 observed `u,v` pairs, normalized by their RMS values, together with the mixed-variable flow residuals. Coordinates are normalized; the shared analytical ansatz imposes the geometric boundary conditions.

## Final results

| Model | Parameters | Held-out u error (%) | Held-out v error (%) |
|---|---:|---:|---:|
| 5-layer PINN | 391 | 1.9775786193 | 7.9447425268 |
| 6-layer PINN | 521 | 1.4257834467 | 5.5139345753 |
| PIHCQNN | 536 | 2.1567991087 | 7.9453935416 |

Each model receives 1,500 data-initialization iterations followed by 2,600 physics-training iterations of L-BFGS. The optimizer is reset at the phase boundary. The PIHCQNN physics phase consists of 600 eager iterations followed by 2,000 CUDA-graph iterations, with the optimizer state carried into the continuation. The final configured checkpoint supplies the reported predictions.

The classical controls ran on CPU in float64; PIHCQNN ran on RTX 4070 with float64/complex128 simulation. This comparison measures accuracy under the shared outer-iteration budget. The archived per-run elapsed times identify the corresponding execution environments.

## Files and commands

Run these commands from `supplementary/`:

```console
python verify_flow_release.py
python verify_flow_release.py --checkpoints
python run_flow_final.py --model all
python run_flow_final.py --model all --execute
```

`verify_flow_release.py` checks the fixed sampling and recalculates errors from the archived predictions. `--checkpoints` additionally regenerates predictions from all three final model states on CPU. `run_flow_final.py` first displays the explicit commands; `--execute` starts training and writes to `rerun_results/flow/`. A CUDA-enabled PyTorch installation runs the PIHCQNN graph continuation. `--model pinn5` or `--model pinn6` selects a CPU control.

The historical stage names are preserved in `flow_reproduction_20261007/convergence/`. The classical stages are in `reviewer_supplement_20261007/flow_controls/`. Raw configurations retain the paths recorded by the original runs; the released runner resolves paths relative to its own location. `release_sources.json` records the byte-identical copies.

The original paper's flow result and Figure 10 remain historical results. The archived PIHCQNN source package contains the original heat, Poisson and inverse scripts; the flow implementation here supplies the newly added shared-protocol experiment.
