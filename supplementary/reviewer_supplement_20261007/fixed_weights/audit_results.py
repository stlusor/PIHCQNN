"""Check matrix completeness and export actual recorded parameter gradients."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SCRIPT_ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('--root',default=str(SCRIPT_ROOT),help='Folder containing the 18 runs and their summary.json.')
ap.add_argument('--outdir',default=str(SCRIPT_ROOT.parents[1]/'rerun_results'/'fixed_weights_audit'))
a=ap.parse_args()
ROOT=Path(a.root).resolve()
OUT=Path(a.outdir).resolve();OUT.mkdir(parents=True,exist_ok=True)
rows=[];details=[];point_hashes=set();max_prediction_delta=0.;max_composition_delta=0.
files=sorted((ROOT/'runs').glob('*/result.json'))
assert len(files)==18,len(files)
for path in files:
    r=json.loads(path.read_text(encoding='utf-8'));c=r['config']
    assert r['status']=='completed' and r['completed_steps']==1000,path
    assert c['environment']['device']=='cpu' and c['environment']['dtype']=='float32',path
    assert c['environment']['torch_threads']==1,path
    assert c['cold_start'] and c['lr']==.005 and c['lambda_residual']==1.,path
    assert c['parameter_count']==(106 if c['model']=='qpinn' else 76),path
    assert c['quantum_parameters']==(30 if c['model']=='qpinn' else 0),path
    checks=r['finite_checks']
    assert checks['loss_checks']==1000 and checks['gradient_checks']==1000 and checks['parameter_checks']==2000,path
    assert all(checks[k]==0 for k in ['nonfinite_loss_steps','nonfinite_gradient_steps','nonfinite_parameter_steps']),path
    point_hashes.add(c['training_points_sha256'])
    assert [h['state_step'] for h in r['history']]==list(range(0,1001,100)),path
    a=np.load(path.parent/'predictions.npz')
    independent=float(np.linalg.norm(a['u_pred'].astype(np.float64)-a['u_ref'])/np.linalg.norm(a['u_ref'].astype(np.float64)))
    delta=abs(independent-r['relative_l2']);max_prediction_delta=max(max_prediction_delta,delta)
    assert delta<1e-6,path
    for h in r['history']:
        row=dict(run=path.parent.name,model=c['model'],lambda_boundary=c['lambda_boundary'],seed=c['seed'],
                 state_step=h['state_step'],residual_loss=h['residual_loss'],
                 boundary_loss_unweighted=h['boundary_loss_unweighted'],
                 boundary_loss_weighted=h['boundary_loss_weighted'],total_loss=h['total_loss'],
                 relative_l2_percent=100*h['relative_l2_201'],
                 gradient_component_cosine=h['gradient_component_cosine'],
                 gradient_composition_max_abs_error=h['gradient_composition_max_abs_error'])
        max_composition_delta=max(max_composition_delta,h['gradient_composition_max_abs_error'])
        for component in ['total','residual','boundary_unweighted']:
            for group in ['quantum','classical','all']:
                for stat in ['parameters','l2','rms','unused_parameters']:
                    row[f'{component}_{group}_{stat}']=h[f'gradients_{component}'][group][stat]
        assert all(v is None or not isinstance(v,float) or math.isfinite(v) for v in row.values()),path
        rows.append(row)
    details.append(dict(run=path.parent.name,
                        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        checkpoint_sha256=hashlib.sha256((path.parent/'checkpoint.pt').read_bytes()).hexdigest()))
assert len(point_hashes)==1,point_hashes
with (OUT/'gradients_history.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
audit=dict(passed=True,runs=18,gradient_checkpoints=len(rows),training_point_sets=len(point_hashes),
           shared_training_point_sha256=next(iter(point_hashes)),
           prediction_relative_l2_max_recalculation_abs_difference=max_prediction_delta,
           gradient_composition_max_abs_error=max_composition_delta,
           finite_loss_checks=18000,finite_gradient_checks=18000,finite_parameter_checks=36000,
           nonfinite_event_count=0,files=details)
(OUT/'integrity_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
summary=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
lines=['固定边界权重补充实验（2026-10-07）','',
       '本目录包含18组全新CPU实验：PINN/PIHCQNN × λBC=0.1/1/10 × seeds=0/1/2。',
       '统一设置：soft BC；100个等距内部配点；201点评价；冷启动；1000次Adam更新；lr=0.005；CPU float32；每个进程1个PyTorch线程，驱动同时运行最多3个进程。',
       'PIHCQNN：5 qubits, n=1, hidden width=5, 106 trainable parameters（30 quantum + 76 classical）；PINN：hidden width=5, 76 parameters。',
       'L=Lres+λBC Lbc，Lres=mean[(u_xx+4π²sin(2πx))²]，Lbc=u(0)²+u(1)²。',
       '目标精确解 u=sin(2πx)。λBC明确指边界项权重，残差项权重始终为1。',
       '',
       'model | λBC | u relative L2 (%) mean ± sample SD | boundary norm mean ± SD | training residual MSE mean ± SD']
for g in summary['groups']:
    lines.append(f"{g['model']} | {g['lambda_boundary']:g} | {g['relative_l2_percent_mean']:.4f} ± {g['relative_l2_percent_sd']:.4f} | {g['boundary_l2_mean']:.6g} ± {g['boundary_l2_sd']:.6g} | {g['residual_loss_mean']:.6g} ± {g['residual_loss_sd']:.6g}")
lines += ['','boundary norm = sqrt(u(0)^2+u(1)^2)，为绝对端点误差；residual MSE在100个固定训练配点计算。',
          '标准差采用样本SD（ddof=1），所有表值均从本目录全新CPU三seed结果计算。',
          '固定训练预算下权重与初值均影响误差，此矩阵用于固定权重敏感性与训练梯度诊断。',
          '时间记录包含实际训练、检查点诊断及每步有限性检查。',
          '',
          '数值核验：六种 model/λBC 组合均与原elastostatic函数公式、梯度和一步Adam更新逐参数一致（最大差值均0）。',
          '18组1000步均完成；18,000次loss检查、18,000次gradient检查与36,000次parameter检查中未记录非有限事件。',
          '每组在状态step=0,100,...,1000记录实际完整PINN损失的量子/经典参数梯度L2与RMS、两项分量梯度及cos角。所有记录与随后一次更新前的参数状态一致。',
          '',
          '文件：summary.csv/summary.json为三seed汇总；all_runs.csv为逐seed结果；gradients_history.csv为198条真实训练梯度检查点；runs/*保存训练点、checkpoint、predictions与详细JSON；verification.json及integrity_checks.json保存一致性核验。',
          '',
          '运行：',
          'python reviewer_supplement_20261007/fixed_weights/train_fixed_weights.py all --workers 3',
          'python reviewer_supplement_20261007/fixed_weights/audit_results.py',
          'all命令从冷启动重新生成矩阵，覆盖相应run目录。']
(OUT/'README_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k!='files'},indent=2))
