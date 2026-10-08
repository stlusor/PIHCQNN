# Original PIHCQNN author code

This directory preserves the original author package recovered from `2024-QC/Submission/CZW/PIHCQNN.rar`. The RAR archive and all 13 Python files are included byte-for-byte. `inventory.json` records their source locations, sizes and SHA-256 checksums.

## Contents

- `archive/PIHCQNN.rar`: the complete original archive.
- `src/`: the 13 Python files extracted from that archive.
- `inventory.json`: source and checksum inventory.

The package contains six training scripts and three plotting scripts, plus four duplicate aliases. Its training examples cover the heat equation, the Poisson equation and inverse one-dimensional elastostatics. The supplementary experiments are provided in the release's separate `supplementary/` directory.

## Nine independent script entry points

Run these as standalone Python scripts. They execute at module level.

| Task | Entry point in `src/` |
| --- | --- |
| Heat equation, PIHCQNN | `PIHCQNN_Forward_Problem_Heat.py` |
| Heat equation, PINN | `PINN_Forward_Problem_Heat.py` |
| Poisson equation, PIHCQNN | `PIHCQNN_Forward_Problem_Poisson.py` |
| Poisson equation, PINN | `PINN_Forward_Problem_Poisson.py` |
| Inverse elastostatics, PIHCQNN | `PIHCQNN_Inverse_Problem.py` |
| Inverse elastostatics, PINN | `PINN_Inverse_Problem.py` |
| Heat equation plots | `Plot_Forward_Problem_Heat.py` |
| Poisson equation plots | `Plot_Forward_Problem_Poisson.py` |
| Inverse elastostatics plots | `Plot_Inverse_Problem.py` |

The original archive also contains four byte-identical aliases:

| Alias | Corresponding entry point |
| --- | --- |
| `PIHCQNN_Forward Problem.py` | `PIHCQNN_Forward_Problem_Heat.py` |
| `PINN_Forward Problem.py` | `PINN_Forward_Problem_Heat.py` |
| `PIHCQNN_Inverse Problem.py` | `PIHCQNN_Inverse_Problem.py` |
| `PINN_Inverse Problem.py` | `PINN_Inverse_Problem.py` |

## Dependencies and recorded runtime settings

The imports require PyTorch, PennyLane, NumPy, Matplotlib and Seaborn. The plotting scripts additionally import Pandas and Plotly. The original scripts use CUDA and include the legacy PennyLane device identifier `default.qubit.torch`; the PINN heat script also creates a `default.qubit` device.

Dependency version numbers were absent from the recovered archive. A historically compatible PennyLane installation is needed for the legacy device identifier. The source files are preserved as historical code rather than converted to a newer backend.

The heat scripts select CUDA device 0. The two Poisson scripts select device 1 at the start and move the quantum layer to device 0. These settings are preserved in the source files. For a new execution environment, create a working copy and configure the CUDA device consistently for that machine.

## Inputs and outputs

The six training scripts construct their sampling grids, analytic reference functions and boundary data internally. They write loss histories, predictions, relative errors and figures to paths set in the scripts; the forward PIHCQNN scripts also save model parameters.

The three plotting scripts read historical `loss.txt`, `l2error.txt` and, for Poisson, `prediction.txt` files from paths set in the scripts. The recovered archive contains the 13 Python source files. Historical output arrays, saved weights and the original Navier–Stokes training implementation are outside this archive's contents.

The scripts retain the original Windows font and author-machine paths. Configure a local font and output/input directories in a working copy before running a script. The training loops and CUDA operations start as soon as a training script executes; importing the module also starts that execution.

## Verification of this release copy

All 13 Python files passed an AST syntax parse. Every copied source file and the RAR archive match their source bytes and SHA-256 checksums. The four alias pairs were checked for byte identity. The verification performed for packaging consists of these file and syntax checks; training execution and historical numerical results are documented separately in the manuscript and supplementary experiment records.
