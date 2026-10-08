"""Audit existing full-physics-loss gradient logs without running training.

Only writes beside this script. Source logs and training code are read only.
The archived optimizer-update ordering is explicitly retained in every row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE_ROOT = ROOT / "reviewer11_experiments"
TRAINING_SCRIPT = SOURCE_ROOT / "elastostatic_forward.py"
METRICS = (
    "grad_q", "grad_classical", "grad_total", "loss_logged",
    "objective_reconstructed", "lf", "lb", "rel_l2_51",
    "pred_mean", "pred_std", "pred_norm", "wf", "wb",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_audit(value, location=""):
    problems = []
    if isinstance(value, dict):
        for key, child in value.items():
            problems += finite_audit(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            problems += finite_audit(child, f"{location}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            problems.append({"path": location, "value": str(value)})
    return problems


def describe(values, count_label="n_checkpoints"):
    return {
        count_label: len(values), "min": min(values), "max": max(values),
        "mean": statistics.mean(values), "median": statistics.median(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def write_readable_notes(summary, runs, group_rows):
    def grouped(suite, model, bc, sampling, weighting, qubits, metric, location):
        return next(row for row in group_rows if
                    (row["suite"], row["model"], row["bc"], row["sampling"], row["weighting"], row["qubits"], row["metric"], row["checkpoint_location"])
                    == (suite, model, bc, sampling, weighting, qubits, metric, location))

    def number(value):
        return f"{value:.4g}"

    qrange = summary["quantum_gradient_range_all"]
    core_example = [run for run in runs if run["suite"] == "core" and run["model"] == "qpinn" and run["bc"] == "hard" and run["sampling"] == "uniform" and run["weighting"] == "fixed"]
    initial_loss = statistics.mean(run["objective_reconstructed_first"] for run in core_example)
    end_loss = statistics.mean(run["objective_reconstructed_end"] for run in core_example)
    table = ["| qubits | quantum gradient at initialization | quantum gradient before update 500 | classical gradient before update 500 | physics loss before update 500 |", "|---|---|---|---|---|"]
    for qubits in (2, 4, 5, 6, 10):
        cells = []
        for metric, location in (("grad_q", "first"), ("grad_q", "end"), ("grad_classical", "end"), ("objective_reconstructed", "end")):
            item = grouped("sensitivity", "qpinn", "hard", "uniform", "fixed", qubits, metric, location)
            cells.append(f"{number(item['mean'])} ± {number(item['sample_sd'])}")
        table.append(f"| {qubits} | " + " | ".join(cells) + " |")
    table_text = "\n".join(table)
    cn = f"""# 已有完整物理损失训练梯度的核对结果

本次只分析既有训练记录，没有重训练。先前回复只用了单独量子电路的初始化诊断，但现有一维弹性静力训练历史已经保存了完整加权物理损失的量子、经典和总参数梯度，可直接用于补充E.3、R11.3和R11.8。

- 数据：核心30次训练，每次11个存储检查点；qubits=2、4、5、6、10敏感性训练各2种子，每次6个检查点。qubits=5使用已有lr5e-3记录。总计40次训练、390个检查点，其中225个属于量子模型。
- 所有原始数值字段均有限；225个量子检查点的量子梯度范数均为正，范围{number(qrange['min'])}–{number(qrange['max'])}。这说明这些短训练中，存储检查点没有出现量子梯度完全消失，不能据此证明任意规模的模型都没有barren plateau。
- 梯度分组一致性：grad_total²=grad_q²+grad_classical²，最大相对平方误差{summary['gradient_partition_identity_max_relative_squared_error']:.3g}。源代码本身由两个分组构造总范数，因此这项是内部算术核对，不能称独立梯度验证。
- hard BC、uniform、fixed权重的3次量子核心训练，其更新前物理损失均值从{number(initial_loss)}降至{number(end_loss)}；可说明训练中的梯度保持活动并伴随损失下降。
- 完整训练中的量子梯度随qubits的变化不单调，不能将单独电路n=3、代理损失的两数量级下降结论套到这些n=1、完整物理损失训练中。下表对每种qubits仅有2种子，描述实际记录即可。

{table_text}

表中为均值±样本标准差，采用hard BC、uniform配点、固定权重、width=5、n=1、Adam lr=5×10⁻³、500次更新。梯度和物理损失为对应更新之前的数值；初始化列是第一次更新前。

时序核对：源码先计算lf/lb及参数梯度，再opt.step()，最后记录预测指标。同一history行的grad/loss对应第k次更新之前（已完成k−1次），rel_l2_51及pred_mean/std/norm对应第k次更新之后。CSV明确两种状态，避免把它们当成同一组参数下的观测。mid定义为最接近总更新预算一半的已存储检查点，若同距取更早的点。

记录质量：6条gradnorm训练在第501个检查点的原始loss字段沿用了上一次迭代的变量。当前加权梯度正确；分析保留loss_logged，同时以同次保存的wf×lf+wb×lb重算objective_reconstructed，并标记该6条记录。其余原始loss与重算物理损失的最大相对差为{summary['maximum_current_branch_loss_relative_difference']:.3g}。这些检查不改变历史训练过程或最终误差。

未记录的指标：40条记录都没有参数norm；pred_norm是201点评价网格上的预测场范数，不是参数范数；没有逐参数梯度分布或每一步梯度，不能补造这些值。PINN的grad_q=0表示其没有量子参数，不表示该模型发生量子梯度消失。

输出：checkpoints.csv保存全部检查点及时间状态；run_statistics.csv与metric_statistics.csv包含每次运行各指标的first/mid/end/min/max；group_checkpoint_statistics.csv包含同设置跨种子的汇总；audit_summary.json包含核对结果、原始文件SHA-256和记录异常。
"""
    (OUT / "中文结论与记录说明.md").write_text(cn, encoding="utf-8")
    en = f"""# Proposed replacement text based on existing training histories

These passages report completed archived experiments and the new read-only analysis. Insert them when the analysis table or supplementary audit is incorporated into the manuscript; no figure or line numbers are asserted here.

## Combined response for Associate Editor comment 3 and Reviewer 11 comments 3 and 8

Thank you. In addition to the learning-rate sweep and the circuit-initialization diagnostic, we analyzed the saved gradients of the full weighted physics-informed objective during the elastostatic training runs. The audit covers 30 core runs and ten sensitivity runs, with 390 stored checkpoints. At each stored checkpoint, the quantum and classical parameter gradients were recorded separately before the labeled optimizer update, while the prediction errors were evaluated after that update. All recorded numerical values are finite, and the quantum-gradient norm is positive at all 225 stored checkpoints from hybrid runs, ranging from {qrange['min']:.4g} to {qrange['max']:.4g}. In the three hard-boundary, uniform-collocation hybrid runs, the mean physics loss decreases from {initial_loss:.4g} at initialization to {end_loss:.4g} before the final update. The full-loss gradient norms do not decrease monotonically with circuit size in the two-seed sweep from two to ten qubits. These unnormalized group norms characterize the tested training trajectories rather than establishing the presence or absence of barren plateaus. The separate initialization diagnostic uses a circuit-only proxy objective and provides distinct evidence about gradient attenuation as that circuit grows.

## Manuscript Section 3.5.5 addition

The archived training histories also contain gradients of the full physics-informed objective with respect to the quantum and classical parameters. We analyzed 30 elastostatic core runs and ten circuit-size sensitivity runs, comprising 390 stored checkpoints. Gradients were recorded at initialization, before updates 101, 201 and subsequent intervals of 100 updates, and immediately before the final update; the corresponding prediction metrics were evaluated after the labeled update. All recorded values are finite. Across the 225 checkpoints from hybrid models, the quantum-gradient norm ranges from {qrange['min']:.4g} to {qrange['max']:.4g} and remains positive. For hard boundary enforcement with uniform collocation and fixed weights, the mean physics loss over three seeds decreases from {initial_loss:.4g} to {end_loss:.4g} over 1,000 updates. The two-seed circuit-size sweep does not show a monotonic decline of the full-loss quantum-gradient norm with the number of qubits. These are unnormalized group norms, and the number of parameters changes with circuit size. This training analysis complements the separate circuit-initialization diagnostic; its finite budgets and sampled checkpoints do not establish the absence of barren plateaus in larger hybrid models.

## Replacement for the conclusion placeholder

The hybrid training is sensitive to the learning rate. In the recorded elastostatic runs, the quantum gradients of the full physics-informed loss remain finite and nonzero at the stored checkpoints, without a monotonic decline across the tested circuit sizes. The separate circuit-initialization diagnostic shows attenuation for its proxy objective; its trend does not directly characterize the full-loss training trajectories.

## Compact table for incorporation

{table_text}

Values are mean ± sample standard deviation over two seeds. The complete model uses hard boundary enforcement, uniform collocation, fixed weights, classical width 5, one data-reuploading repetition, and Adam with a learning rate of 5×10⁻³ for 500 updates. Gradient norms are unnormalized Euclidean norms over the indicated parameter group. A checkpoint labeled k records the gradient and objective before update k, and its prediction metrics after update k. Qubit count 5 is represented by the lr5e-3 sensitivity records.

## Internal audit notes, for authors

The archived source computes grad_total from grad_q and grad_classical; checking their squared-norm identity is an arithmetic consistency check. Parameter norms are absent from all logs. The six adaptive-weighting records at checkpoint 501 contain a stale raw loss value because the source does not refresh that scalar in the weight-update branch. The current objective is reconstructed from the contemporaneously saved wf, lf, wb and lb. The parameter gradients in that branch are computed from the current weighted components. Preserve the source logs, use the audited objective for a loss table or plot, and keep the pre-update/post-update distinction. No statistical claim about gradient variance scaling should be made from two training seeds or from these aggregate group norms.
"""
    (OUT / "English_response_and_manuscript_text.md").write_text(en, encoding="utf-8")


def main():
    core = sorted((SOURCE_ROOT / "results" / "elastostatic_core").glob("*.json"))
    sensitivity = []
    for stem in ("q2", "q4", "q6", "q10", "lr5e-3"):
        sensitivity.extend(
            SOURCE_ROOT / "results" / "sensitivity" / f"{stem}_seed{seed}.json"
            for seed in (0, 1)
        )
    assert len(core) == 30, f"Expected 30 core runs, got {len(core)}"
    assert all(path.exists() for path in sensitivity)
    files = [("core", path) for path in core] + [("sensitivity", path) for path in sensitivity]
    checkpoints, runs, statistics_rows, manifests, problems = [], [], [], [], []
    missing_parameter_norms = []
    for suite, path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        run_id = path.stem
        source = path.relative_to(ROOT).as_posix()
        common = {"suite": suite, "run_id": run_id, "source": source}
        common.update({key: record.get(key) for key in (
            "model", "bc", "sampling", "weighting", "seed", "qubits", "width", "lr", "steps", "params"
        )})
        common["n_rep"] = record.get("n")
        bad = finite_audit(record)
        problems.extend({"source": source, **item} for item in bad)
        history = record["history"]
        assert history and history[-1]["step"] == record["steps"]
        assert [item["step"] for item in history] == sorted({item["step"] for item in history})
        if not any("param" in key and "norm" in key for row in history for key in row):
            missing_parameter_norms.append(source)
        run_rows = []
        for entry in history:
            step = entry["step"]
            total_sq = entry["grad_total"] ** 2
            groups_sq = entry["grad_q"] ** 2 + entry["grad_classical"] ** 2
            rel_sq_error = abs(total_sq - groups_sq) / max(total_sq, groups_sq, 1e-300)
            objective = entry["wf"] * entry["lf"] + entry["wb"] * entry["lb"]
            logged_objective = entry["loss"]
            rel_loss_error = abs(logged_objective-objective) / max(abs(objective), abs(logged_objective), 1e-30)
            stale_branch = record["weighting"] == "gradnorm" and record["bc"] == "soft" and step > 1 and (step - 1) % 500 == 0
            row = {
                **common, "checkpoint_step": step,
                "loss_and_gradient_parameter_state": f"before update {step}; after {step-1} updates",
                "prediction_parameter_state": f"after update {step}",
                "loss_logged": logged_objective,
                "objective_reconstructed": objective,
                "logged_loss_state": "previous iteration (archived stale-variable branch)" if stale_branch else "current pre-update objective",
                "logged_loss_stale_branch": stale_branch,
                "logged_vs_reconstructed_relative_difference": rel_loss_error,
                "gradient_partition_relative_squared_error": rel_sq_error,
                "parameter_norm": "", "parameter_norm_status": "not recorded",
            }
            row.update({key: entry.get(key) for key in METRICS if key not in ("loss_logged", "objective_reconstructed")})
            row["quantum_gradient_status"] = "absent quantum block; recorded zero is structural" if record["model"] == "pinn" else "full weighted physics loss quantum-parameter gradient"
            checkpoints.append(row)
            run_rows.append(row)
        first, last = run_rows[0], run_rows[-1]
        middle = min(run_rows, key=lambda row: (abs(row["checkpoint_step"] - record["steps"] / 2), row["checkpoint_step"]))
        stats_run = {
            **common, "checkpoint_count": len(run_rows),
            "first_checkpoint_step": first["checkpoint_step"],
            "mid_checkpoint_step": middle["checkpoint_step"],
            "mid_target_step": record["steps"] / 2,
            "end_checkpoint_step": last["checkpoint_step"],
            "final_rel_l2_201": record["rel_l2"],
            "elapsed_s": record.get("elapsed_s"),
            "finite_raw_record": not bad,
            "max_gradient_partition_relative_squared_error": max(row["gradient_partition_relative_squared_error"] for row in run_rows),
            "stale_loss_checkpoints": sum(row["logged_loss_stale_branch"] for row in run_rows),
            "parameter_norm_status": "not recorded",
        }
        for metric in METRICS:
            available = [row for row in run_rows if isinstance(row.get(metric), (int, float))]
            if not available:
                continue
            minimum = min(available, key=lambda row: row[metric])
            maximum = max(available, key=lambda row: row[metric])
            values = [row[metric] for row in available]
            descriptor = describe(values)
            stats_row = {
                **common, "metric": metric,
                "first": first.get(metric), "first_step": first["checkpoint_step"],
                "mid": middle.get(metric), "mid_step": middle["checkpoint_step"],
                "end": last.get(metric), "end_step": last["checkpoint_step"],
                "min_step": minimum["checkpoint_step"], "max_step": maximum["checkpoint_step"],
                **descriptor,
                "state_note": "pre-update" if metric in ("grad_q", "grad_classical", "grad_total", "objective_reconstructed", "lf", "lb", "wf", "wb") else "post-update" if metric.startswith("pred_") or metric == "rel_l2_51" else "preserve logged branch status in checkpoints.csv",
            }
            statistics_rows.append(stats_row)
            for key in ("first", "mid", "end", "min", "max"):
                stats_run[f"{metric}_{key}"] = stats_row[key]
        runs.append(stats_run)
        manifests.append({"source": source, "sha256": digest(path), "numeric_record_finite": not bad, "checkpoints": len(run_rows)})

    group_rows = []
    for suite in ("core", "sensitivity"):
        suite_runs = [run for run in runs if run["suite"] == suite]
        keys = sorted({(run["model"], run["bc"], run["sampling"], run["weighting"], run["qubits"], run["lr"]) for run in suite_runs})
        for key in keys:
            matches = [run for run in suite_runs if (run["model"], run["bc"], run["sampling"], run["weighting"], run["qubits"], run["lr"]) == key]
            base = dict(zip(("model", "bc", "sampling", "weighting", "qubits", "lr"), key))
            for metric in ("grad_q", "grad_classical", "grad_total", "objective_reconstructed", "lf", "lb", "rel_l2_51", "pred_std", "pred_norm"):
                for location in ("first", "mid", "end"):
                    values = [run[f"{metric}_{location}"] for run in matches]
                    group_rows.append({"suite": suite, **base, "n_rep": matches[0]["n_rep"], "metric": metric, "checkpoint_location": location, **describe(values, count_label="n_seeds")})

    q_rows = [row for row in checkpoints if row["model"] == "qpinn"]
    core_q = [row for row in q_rows if row["suite"] == "core"]
    summary = {
        "scope": "Read-only audit of archived full weighted physics-loss training histories; no retraining",
        "core_runs": len(core), "sensitivity_runs": len(sensitivity), "all_runs": len(runs),
        "stored_checkpoints": len(checkpoints), "core_checkpoints": sum(row["suite"] == "core" for row in checkpoints),
        "quantum_checkpoints": len(q_rows), "core_quantum_checkpoints": len(core_q),
        "all_raw_numeric_fields_finite": not problems, "nonfinite_fields": problems,
        "quantum_gradients_strictly_positive_at_all_recorded_quantum_checkpoints": all(row["grad_q"] > 0 for row in q_rows),
        "quantum_gradient_range_all": describe([row["grad_q"] for row in q_rows]),
        "quantum_gradient_range_core": describe([row["grad_q"] for row in core_q]),
        "classical_gradient_range_all": describe([row["grad_classical"] for row in checkpoints]),
        "gradient_partition_identity_max_relative_squared_error": max(row["gradient_partition_relative_squared_error"] for row in checkpoints),
        "gradient_partition_identity_pass_tolerance": 1e-12,
        "gradient_partition_identity_all_pass": all(row["gradient_partition_relative_squared_error"] <= 1e-12 for row in checkpoints),
        "stale_logged_loss_checkpoint_count": sum(row["logged_loss_stale_branch"] for row in checkpoints),
        "stale_logged_loss_checkpoints": [{key: row[key] for key in ("suite", "run_id", "checkpoint_step", "loss_logged", "objective_reconstructed", "logged_vs_reconstructed_relative_difference")} for row in checkpoints if row["logged_loss_stale_branch"]],
        "maximum_current_branch_loss_relative_difference": max(row["logged_vs_reconstructed_relative_difference"] for row in checkpoints if not row["logged_loss_stale_branch"]),
        "parameter_norm_recorded": False,
        "parameter_norm_missing_in_runs": missing_parameter_norms,
        "prediction_norm_is_not_parameter_norm": True,
        "checkpoint_time_definition": {
            "step_label": "one-based number of optimizer updates completed at record emission",
            "gradients_lf_lb_weights": "evaluated at parameters immediately before this labeled update",
            "prediction_metrics": "evaluated immediately after this labeled update",
            "first_checkpoint": "step 1 contains initialization-state loss/gradients and one-update prediction metrics",
            "mid_selection": "stored checkpoint nearest to half the configured update budget; ties use the earlier step",
        },
        "interpretation_limits": [
            "Only stored checkpoints (every 100 updates plus the last update) were checked, not every optimization step.",
            "Gradient-total partition consistency is an internal arithmetic audit because the source computes grad_total from the two groups.",
            "Positive finite gradients in these short runs do not establish absence of barren plateaus for arbitrary circuits or physics-informed objectives.",
            "No parameter norm or individual parameter-gradient distribution was recorded.",
            "The independently archived circuit-initialization proxy-loss test is a separate diagnostic and its scaling trend must not be assigned to the full training loss.",
        ],
        "source_training_script": {"path": TRAINING_SCRIPT.relative_to(ROOT).as_posix(), "sha256": digest(TRAINING_SCRIPT)},
        "source_manifest": manifests,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "checkpoints.csv", checkpoints)
    write_csv(OUT / "run_statistics.csv", runs)
    write_csv(OUT / "metric_statistics.csv", statistics_rows)
    write_csv(OUT / "group_checkpoint_statistics.csv", group_rows)
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_readable_notes(summary, runs, group_rows)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("source_manifest", "parameter_norm_missing_in_runs")}, ensure_ascii=False, indent=2))
    assert not problems
    assert summary["gradient_partition_identity_all_pass"]


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-root",default=str(SOURCE_ROOT),help="Folder containing archived results/elastostatic_core and results/sensitivity.")
    ap.add_argument("--training-script",default=str(TRAINING_SCRIPT))
    ap.add_argument("--outdir",default=str(ROOT/"rerun_results"/"gradient_audit"))
    a=ap.parse_args()
    SOURCE_ROOT=Path(a.source_root).resolve()
    TRAINING_SCRIPT=Path(a.training_script).resolve()
    OUT=Path(a.outdir).resolve()
    main()
