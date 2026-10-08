"""Export supplementary manuscript data from archived JSON, without training.

Usage: python supplementary/export_results.py [--out PATH]
Uses Python's standard library. Paths are resolved relative to this script.
Both sample and population SD columns are provided. The sd_ddof and
*_sd_manuscript columns identify the convention used by each manuscript
table: ddof=0 for Tables 4 and 6, and ddof=1 for the repeated-run tables.
"""
from __future__ import annotations

import argparse
import cmath
import csv
import json
import math
import statistics
from pathlib import Path


BASE = Path(__file__).resolve().parent
MISSING: list[str] = []
GROUPS: list[dict] = []


def read_json(relative: str):
    path = BASE / relative
    if not path.is_file():
        MISSING.append(relative)
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_write(out: Path, name: str, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    with (out / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float], prefix: str, scale: float = 1.0) -> dict:
    values = [float(value) * scale for value in values]
    if not values:
        return {prefix + "_mean": "", prefix + "_sd_sample": "",
                prefix + "_sd_population": ""}
    return {prefix + "_mean": statistics.mean(values),
            prefix + "_sd_sample": statistics.stdev(values) if len(values) > 1 else "",
            prefix + "_sd_population": statistics.pstdev(values) if len(values) > 1 else ""}


def aggregate(table: str, label: str, sources: list[str],
              metrics: dict[str, tuple[str, float]], **settings) -> dict | None:
    loaded = [(source, read_json(source)) for source in sources]
    loaded = [(source, data) for source, data in loaded if data is not None]
    if not loaded:
        return None
    row = {"manuscript_location": table, "case_model": label,
           "run_count": len(loaded), "expected_run_count": len(sources),
           "sd_ddof": 1,
           "seeds": ";".join(str(data.get("seed", data.get("config", {}).get("seed", "")))
                             for _, data in loaded),
           **settings}
    for output, (field, scale) in metrics.items():
        row.update(stats([data[field] for _, data in loaded if field in data], output, scale))
        row[output + "_sd_manuscript"] = row[output + ("_sd_sample" if row["sd_ddof"] == 1 else "_sd_population")]
    first = loaded[0][1]
    row["parameters"] = first.get("params", first.get("config", {}).get("parameter_count", ""))
    row["updates"] = first.get("steps", first.get("completed_steps", ""))
    row.update(stats([data["elapsed_s"] for _, data in loaded if "elapsed_s" in data], "elapsed_s"))
    row["elapsed_s_sd_manuscript"] = row["elapsed_s_sd_sample" if row["sd_ddof"] == 1 else "elapsed_s_sd_population"]
    row["source_files"] = ";".join(source for source, _ in loaded)
    row["provenance"] = "supplementary archived individual runs"
    GROUPS.append(row)
    return row


def seeds(prefix: str, suffix: str = "", count: int = 3) -> list[str]:
    return [f"{prefix}{seed}{suffix}.json" for seed in range(count)]


def repeated_tables(out: Path):
    heat, poisson = [], []
    for case, dest in [("heat", heat), ("poisson", poisson)]:
        for model in ["tanh5", "tanh6"]:
            row = aggregate("Table 1" if case == "heat" else "Table 2", f"{case}/{model}",
                            seeds(f"reviewer2_experiments/results/{case}_pinn/{model}_seed", count=5),
                            {"relative_l2_percent": ("metric", 100)},
                            evaluation="101x201 heat grid; 201x201 Poisson grid")
            if row:
                dest.append(row)
    row = aggregate("Table 2", "poisson/PIHCQNN", seeds("reviewer2_experiments/results/poisson_qpinn/seed", count=5),
                    {"relative_l2_percent": ("metric", 100)}, evaluation="201x201 independent grid")
    if row:
        poisson.append(row)
    csv_write(out, "table01_heat_supplementary_repeats.csv", heat)
    csv_write(out, "table02_poisson_repeats.csv", poisson)


def modern_baselines(out: Path):
    rows = []
    for case in ["heat", "poisson", "inverse"]:
        models = ["tanh2", "siren", "adaptive", "fourier05", "lossbalance"] if case == "inverse" else ["tanh5", "siren", "adaptive", "fourier", "lossbalance"]
        if case == "poisson":
            models = ["tanh5", "tanh6", "siren", "adaptive", "fourier05", "lossbalance"]
        for model in models:
            directory = f"{case}_variants" if model == "fourier05" else case
            row = aggregate("Table 4", f"{case}/{model}", seeds(f"reviewer1_experiments/results/{directory}/{model}_seed"),
                            {"error_percent": ("metric", 100)},
                            metric="EA relative error" if case == "inverse" else "relative L2 of solution",
                            sd_ddof=0,
                            manuscript_sd="population (ddof=0); use *_sd_population for Table 4")
            if row:
                rows.append(row)
    csv_write(out, "table04_modern_baselines.csv", rows)


def target_high_amplitude(target: str) -> float:
    """DFT reference on the stated 201-point [-pi,pi] grid (double precision)."""
    freqs, amplitudes = ([1, 3, 5], [1, 1, 1]) if target == "f135" else ([1, 7, 15], [.8, .4, .2])
    signal = [sum(a * math.sin(k * (-math.pi + 2 * math.pi * i / 200))
                  for k, a in zip(freqs, amplitudes)) for i in range(201)]
    return abs(sum(y * cmath.exp(-2j * math.pi * freqs[-1] * i / 201)
                   for i, y in enumerate(signal)))


def frequency_table(out: Path):
    rows = []
    # Recover the exact archived reference-amplitude convention from its
    # recorded ratio and the individual final amplitudes. This preserves the
    # float32 target convention of the published frequency diagnostics.
    archived = {}
    summary = BASE / "reviewer1_experiments/fourier_summary.csv"
    if summary.is_file():
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            archived = {(r["target"], r["model"]): r for r in csv.DictReader(handle)}
    for target in ["f135", "f1715"]:
        for model in ["pih1", "pih2", "pih3", "fcnn5", "fcnn7", "fourier"]:
            sources = seeds(f"reviewer1_experiments/results/fourier/{target}_{model}_seed")
            row = aggregate("Table 6 / frequency diagnostics", f"{target}/{model}", sources,
                            {"relative_l2_percent": ("relative_l2", 100)},
                            sd_ddof=0,
                            manuscript_sd="population (ddof=0); use *_sd_population for Table 6")
            if row is None:
                continue
            records = [json.loads((BASE / p).read_text(encoding="utf-8-sig"))
                       for p in sources if (BASE / p).is_file()]
            old = archived.get((target, model))
            if old and float(old["high_amp_ratio_mean"]) > 0:
                reference = statistics.mean(r["final_amplitudes"][-1] for r in records) / float(old["high_amp_ratio_mean"])
                row["amplitude_reference"] = "recovered from archived float32 target ratio"
            else:
                reference = target_high_amplitude(target)
                row["amplitude_reference"] = "double-precision DFT reconstruction"
            ratios = [r["final_amplitudes"][-1] / reference for r in records]
            hit_steps, censored = [], 0
            for record in records:
                hit = next((h["step"] for h in record["history"]
                            if h["amplitudes"][-1] >= reference / 2), None)
                censored += hit is None
                hit_steps.append(record["steps"] if hit is None else hit)
            row.update(stats(ratios, "highest_frequency_amplitude_ratio"))
            row.update(stats(hit_steps, "steps_to_half_amplitude_capped_at_budget"))
            for prefix in ["highest_frequency_amplitude_ratio", "steps_to_half_amplitude_capped_at_budget"]:
                row[prefix + "_sd_manuscript"] = row[prefix + "_sd_population"]
            row["runs_below_half_at_budget"] = censored
            row["threshold_reporting"] = ">1500 when all runs remain below half; censored runs otherwise capped at 1500"
            rows.append(row)
    csv_write(out, "table06_frequency_diagnostics.csv", rows)


def elastostatic_tables(out: Path):
    rows = []
    for model in ["pinn", "qpinn"]:
        for bc, sampling, weighting in [("soft", "uniform", "fixed"), ("hard", "uniform", "fixed"),
                                        ("soft", "uniform", "gradnorm"), ("soft", "adaptive", "fixed"),
                                        ("hard", "adaptive", "fixed")]:
            label = f"{model}_{bc}_{sampling}_{weighting}"
            row = aggregate("Table 7", label, seeds(f"reviewer11_experiments/results/elastostatic_core/{label}_seed"),
                            {"relative_l2_percent": ("rel_l2", 100), "boundary_l2": ("boundary_l2", 1)},
                            bc=bc, sampling=sampling, weighting=weighting, timing_device="RTX 4070", learning_rate=.005)
            if row:
                rows.append(row)
    csv_write(out, "table07_elastostatic_controls.csv", rows)
    rows = []
    for name in ["lr1e-4", "lr5e-4", "lr1e-3", "lr2e-3", "lr5e-3", "lr1e-2",
                 "q2", "q4", "q6", "q10", "w10", "w20", "n2", "n3"]:
        row = aggregate("Table 8", name, seeds(f"reviewer11_experiments/results/sensitivity/{name}_seed", count=2),
                        {"relative_l2_percent": ("rel_l2", 100)})
        if row:
            first = read_json(f"reviewer11_experiments/results/sensitivity/{name}_seed0.json")
            for key in ["lr", "qubits", "width", "n", "bc", "sampling", "weighting"]:
                row[key] = first[key]
            rows.append(row)
    for name in ["full", "ry", "ryrz"]:
        row = aggregate("Table 8 / ansatz", name, seeds(f"reviewer11_experiments/results/ansatz/{name}_seed", count=2),
                        {"relative_l2_percent": ("rel_l2", 100)})
        if row:
            rows.append(row)
    csv_write(out, "table08_sensitivity_and_ansatz.csv", rows)


def gradient_table(out: Path):
    proxy = read_json("reviewer11_experiments/results/gradient_scaling.json")
    rows = []
    for qubits in [2, 4, 6, 10]:
        init = next((r for r in (proxy or []) if r["n_rep"] == 3 and r["qubits"] == qubits), {})
        records = [read_json(f"reviewer11_experiments/results/sensitivity/q{qubits}_seed{s}.json") for s in [0, 1]]
        records = [r for r in records if r]
        if not records and not init:
            continue
        row = {"manuscript_location": "Table 9", "qubits": qubits,
               "sd_ddof": 1,
               "proxy_n_rep": 3, "proxy_initializations": init.get("repeats", ""),
               "proxy_log10_quantum_gradient_mean": init.get("log10_grad_mean", ""),
               "proxy_log10_quantum_gradient_sd_sample": init.get("log10_grad_sd", ""),
               "proxy_output_variance_mean": init.get("output_variance_mean", ""),
               "training_n_rep": 1, "training_seed_count": len(records),
               "training_checkpoint": "stored checkpoint immediately before final (500th) update"}
        for group, field in [("quantum", "grad_q"), ("classical", "grad_classical")]:
            row.update(stats([r["history"][-1][field] for r in records], f"training_{group}_gradient"))
        row["source_files"] = "reviewer11_experiments/results/gradient_scaling.json;" + ";".join(
            f"reviewer11_experiments/results/sensitivity/q{qubits}_seed{s}.json" for s in [0, 1])
        rows.append(row)
    csv_write(out, "table09_gradient_diagnostics.csv", rows)
    audit = read_json("reviewer_supplement_20261007/gradient_audit/audit_summary.json")
    if audit:
        (out / "full_training_gradient_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def numerical_solvers(out: Path):
    data = read_json("reviewer11_experiments/results/classical_baselines.json")
    rows = []
    for name, record in (data or {}).items():
        rows.append({"manuscript_location": "Table 10", "solver_case": name,
                     "sd_ddof": "single deterministic solution",
                     "relative_l2_percent": record["rel_l2"] * 100, "elapsed_s": record["seconds"],
                     "grid_or_elements": record.get("grid", record.get("n", "")),
                     "time_steps": record.get("time_steps", ""),
                     "evaluation": "final time" if name == "heat_fd" else "solution grid",
                     "source_file": "reviewer11_experiments/results/classical_baselines.json"})
    csv_write(out, "table10_classical_solvers.csv", rows)
    row = aggregate("Section 3.5.6 / DEM", "classical_DEM", seeds("reviewer11_experiments/results/dem_1d_seed"),
                    {"relative_l2_percent": ("rel_l2", 100), "boundary_l2": ("boundary_l2", 1)},
                    energy_points=401, evaluation_points=201, learning_rate=.005)
    csv_write(out, "dem_energy_method.csv", [row] if row else [])


def reconstruct_fem(theta: list[float], true_theta: list[float], n: int = 100):
    """Reconstruct the archived adjoint solution with the same linear FEM.

    Midpoint coefficient, endpoint trapezoidal element load, homogeneous
    Dirichlet values and linear interpolation. Solve the interior tridiagonal
    system in standard-library double precision; no optimization is rerun.
    """
    h = 1 / n
    def coeff(x, t):
        return 1 + sum(t[j] * math.sin((j + 1) * 2 * math.pi * x) for j in range(3))
    def load(x):
        a = coeff(x, true_theta)
        ax = sum((j + 1) * 2 * math.pi * true_theta[j] * math.cos((j + 1) * 2 * math.pi * x) for j in range(3))
        return -(ax * math.pi * math.cos(math.pi * x) - a * math.pi ** 2 * math.sin(math.pi * x))
    element_k = [coeff((e + .5) * h, theta) / h for e in range(n)]
    diagonal = [element_k[i - 1] + element_k[i] for i in range(1, n)]
    lower = [-element_k[i] for i in range(1, n - 1)]
    upper = list(lower)
    rhs = [h * load(i * h) for i in range(1, n)]
    for i in range(1, n - 1):
        factor = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]
    interior = [0.] * (n - 1)
    interior[-1] = rhs[-1] / diagonal[-1]
    for i in range(n - 3, -1, -1):
        interior[i] = (rhs[i] - upper[i] * interior[i + 1]) / diagonal[i]
    nodes = [0.] + interior + [0.]
    field = []
    for i in range(201):
        x = i / 200
        element = min(int(x * n), n - 1)
        weight = x * n - element
        u = nodes[element] * (1 - weight) + nodes[element + 1] * weight
        field.append({"x": x, "u_fem": u, "u_exact": math.sin(math.pi * x)})
    error = math.sqrt(sum((r["u_fem"] - r["u_exact"]) ** 2 for r in field) /
                      sum(r["u_exact"] ** 2 for r in field))
    return error, field


def inverse_table(out: Path):
    rows = []
    for model in ["pinn", "qpinn"]:
        row = aggregate("Table 11", model, seeds(f"reviewer11_experiments/results/variable_inverse_{model}_seed"),
                        {"coefficient_error_percent": ("theta_rel_error", 100),
                         "displacement_l2_percent": ("u_rel_l2", 100)},
                        coefficient_count=3, observations=50, residual_points=101, evaluation_points=201)
        if row:
            rows.append(row)
    source = "reviewer11_experiments/results/variable_inverse_fem_adjoint.json"
    fem = read_json(source)
    if fem:
        error, field = reconstruct_fem(fem["theta_hat"], fem["theta_true"], fem["mesh_elements"])
        rows.append({"manuscript_location": "Table 11", "case_model": fem["model"], "run_count": 1,
                     "sd_ddof": "single deterministic solution",
                     "coefficient_error_percent_mean": fem["theta_rel_error"] * 100,
                     "coefficient_error_percent_sd_sample": "", "displacement_l2_percent_mean": error * 100,
                     "displacement_l2_percent_sd_sample": "", "elapsed_s_mean": fem["elapsed_s"],
                     "coefficient_count": 3, "observations": fem["observations"], "evaluation_points": 201,
                     "mesh_elements": fem["mesh_elements"], "objective_evaluations": fem["nfev"],
                     "source_files": source, "provenance": "archived coefficients; reconstructed linear FEM field"})
        csv_write(out, "table11_fem_reconstructed_displacement.csv", field)
    csv_write(out, "table11_variable_inverse.csv", rows)


def fixed_weights(out: Path):
    rows = []
    for model in ["pinn", "qpinn"]:
        for weight in ["0.1", "1", "10"]:
            sources = [f"reviewer_supplement_20261007/fixed_weights/runs/{model}_lambdaBC{weight}_seed{s}/result.json" for s in range(3)]
            row = aggregate("Section 3.5.4 / fixed-weight study", f"{model}/lambdaBC={weight}", sources,
                            {"relative_l2_percent": ("relative_l2", 100), "boundary_l2": ("boundary_l2", 1),
                             "residual_mse": ("final_residual_loss", 1)},
                            lambda_boundary=float(weight), timing_device="CPU", dtype="float32",
                            learning_rate=.005, residual_points=100, evaluation_points=201)
            if row:
                rows.append(row)
    csv_write(out, "fixed_weight_study_cpu.csv", rows)


def flow_table(out: Path):
    rows = []
    q_root = "flow_reproduction_20261007/convergence/"
    stages = {
        "PINN5": ["reviewer_supplement_20261007/flow_controls/pinn5/data_initialization/result.json",
                  "reviewer_supplement_20261007/flow_controls/pinn5/physics_training/result.json"],
        "PINN6": ["reviewer_supplement_20261007/flow_controls/pinn6/data_initialization/result.json",
                  "reviewer_supplement_20261007/flow_controls/pinn6/physics_training/result.json"],
        "PIHCQNN": [q_root + "pihcqnn_capacity_193/result.json",
                    q_root + "pihcqnn_rms100_physics/history.json",
                    q_root + "pihcqnn_rms100_physics_graph/result.json"],
    }
    for model, sources in stages.items():
        raw = [read_json(source) for source in sources]
        # The first physics segment stopped after 600 updates and preserved
        # history.json plus a snapshot. Its last history row records 1264
        # closure calls; the graph continuation performs the other 2000.
        records = [{"objective_calls": r[-1]["objective_calls"],
                    "completed_outer_updates": r[-1]["step"]}
                   if isinstance(r, list) and r else r for r in raw]
        final = records[-1]
        if final is None:
            continue
        config, metrics = final["config"], final["final_metrics"]["heldout"]
        protocol = config["protocol"]
        row = {"manuscript_location": "Section 3.4 supplementary common-observation comparison",
               "sd_ddof": "single seed",
               "model": model, "seed": config["seed"], "run_count": 1,
               "parameters": config["parameters"], "u_relative_l2_percent": metrics["u_relative_l2"] * 100,
               "v_relative_l2_percent": metrics["v_relative_l2"] * 100,
               "observations": protocol["data_points"], "reference_rows": protocol["reference_points"],
               "heldout_rows": protocol["heldout_points"], "collocation_points": protocol["experiment_counts"]["colloc"],
               "dtype": config.get("dtype", "float64" if config.get("float64") else ""),
               "observed_fields": ";".join(protocol["observed_fields"]), "data_weight": config["data_weight"],
               "total_closure_evaluations": sum(r["objective_calls"] for r in records if r),
               "stage_records_complete": all(r is not None for r in records),
               "physics_segments": "600+2000 with optimizer state resumed" if model == "PIHCQNN" else "2600",
               "data_initialization_outer_updates": 1500, "physics_outer_updates": 2600,
               "source_files": ";".join(sources),
               "provenance": "single-seed supplementary comparison; final held-out velocity metrics",
               "timing_scope": "mixed execution modes; omitted from comparative elapsed-time summary"}
        rows.append(row)
    csv_write(out, "flow_common_observation_final.csv", rows)


def recorded_costs(out: Path):
    rows = []
    for row in GROUPS:
        if row.get("elapsed_s_mean", "") == "":
            continue
        rows.append({k: row[k] for k in ["manuscript_location", "case_model", "run_count", "parameters", "updates",
                                         "sd_ddof", "elapsed_s_mean", "elapsed_s_sd_sample", "elapsed_s_sd_population",
                                         "elapsed_s_sd_manuscript", "source_files"]})
    csv_write(out, "table05_and_supplementary_recorded_costs.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "publication_results",
                        help="CSV output directory (default: alongside this script)")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repeated_tables(out)
    modern_baselines(out)
    frequency_table(out)
    elastostatic_tables(out)
    gradient_table(out)
    numerical_solvers(out)
    inverse_table(out)
    fixed_weights(out)
    flow_table(out)
    recorded_costs(out)
    coverage = {"source_root": "supplementary/", "output_scope": "supplementary results and archived repeated runs",
                "sd_manuscript": "Table 4 and 6: ddof=0; repeated Table 1/2 and Table 7/8/9/11 and fixed weights: ddof=1",
                "both_sd_conventions_provided": True,
                "single_run_sd": "blank", "error_units": "percent; original JSON ratios multiplied by 100",
                "historical_original_numbers": "Original heat PIHCQNN, original inverse Table 3, original flow and approximate historical timings remain manuscript-reported values; they are not regenerated here.",
                "missing_source_files": sorted(set(MISSING)),
                "csv_files": sorted(p.name for p in out.glob("*.csv")), "training_executed": False}
    (out / "export_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(coverage['csv_files'])} CSV files to {out}")
    print(f"Missing source files: {len(coverage['missing_source_files'])}")


if __name__ == "__main__":
    main()
