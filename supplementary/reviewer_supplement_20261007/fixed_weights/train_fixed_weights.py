"""Controlled boundary-weight sensitivity on the real elastostatic PINN loss.

Every run starts afresh on CPU, float32, one PyTorch thread. The original
ManualQVC and elastostatic functions are imported without modifying them.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / 'reviewer11_experiments'))
from elastostatic_forward import build_model, make_u, sample_uniform, residual, exact

ROOT = Path(__file__).resolve().parent


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)


def environment():
    return dict(python=sys.version, python_executable=sys.executable,
                pytorch=str(torch.__version__), numpy=str(np.__version__),
                platform=platform.platform(), processor=platform.processor(),
                device='cpu', dtype='float32', torch_threads=torch.get_num_threads(),
                quantum_backend='original reviewer1_experiments.quantum_layer.ManualQVC',
                cuda_build=torch.version.cuda,
                gpu_available_but_not_used=torch.cuda.is_available())


def losses(u, xf):
    lf = residual(u, xf).square().mean()
    # Preserve the original two separate scalar boundary evaluations.
    lb = u(torch.zeros(1, 1)).square().mean() + u(torch.ones(1, 1)).square().mean()
    return lf, lb


def vector(grads, params):
    return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


def group_measurements(grads, params, qmask):
    result = {}
    for group, mask in [('quantum', qmask), ('classical', [not x for x in qmask]),
                        ('all', [True] * len(params))]:
        selected = [(g, p) for g, p, keep in zip(grads, params, mask) if keep]
        count = sum(p.numel() for _, p in selected)
        if not count:
            result[group] = dict(parameters=0, l2=0., rms=0., unused_parameters=0)
            continue
        norm_sq = sum(float(g.detach().square().sum()) for g, _ in selected if g is not None)
        result[group] = dict(parameters=count, l2=math.sqrt(norm_sq),
                             rms=math.sqrt(norm_sq / count),
                             unused_parameters=sum(p.numel() for g, p in selected if g is None))
    return result


def diagnostics(u, params, qmask, xf, lf, lb, loss, weight, state_step):
    gf = torch.autograd.grad(lf, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(lb, params, retain_graph=True, allow_unused=True)
    # This full objective gradient is the gradient used by Adam below.
    loss.backward()
    gs = [p.grad for p in params]
    vf, vb, vs = vector(gf, params), vector(gb, params), vector(gs, params)
    expected = vf + weight * vb
    composed_error = float((vs - expected).abs().max())
    denom = float(vf.norm() * vb.norm())
    cosine = float(torch.dot(vf, vb)) / denom if denom > 0 else None
    with torch.no_grad():
        xe = torch.linspace(0, 1, 201).reshape(-1, 1)
        pred, target = u(xe), exact(xe)
        rel = float(torch.linalg.vector_norm(pred-target) / torch.linalg.vector_norm(target))
    return dict(state_step=state_step, residual_loss=float(lf.detach()),
                boundary_loss_unweighted=float(lb.detach()),
                boundary_loss_weighted=weight * float(lb.detach()),
                total_loss=float(loss.detach()), relative_l2_201=rel,
                gradients_total=group_measurements(gs, params, qmask),
                gradients_residual=group_measurements(gf, params, qmask),
                gradients_boundary_unweighted=group_measurements(gb, params, qmask),
                gradient_component_cosine=cosine,
                gradient_composition_max_abs_error=composed_error,
                timing='all losses, gradients and predictions evaluated before the following update')


def finite_grads(params):
    return all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)


def finite_params(params):
    return all(bool(torch.isfinite(p).all()) for p in params)


def name_for(model, weight, seed):
    return f'{model}_lambdaBC{weight:g}_seed{seed}'


def train(model, weight, seed, steps=1000, out_path=None):
    setup()
    name = name_for(model, weight, seed)
    out = Path(out_path).resolve().parent if out_path else ROOT / 'runs' / name
    out.mkdir(parents=True, exist_ok=True)
    base = build_model(model, 5, 5, 1, seed, 'cpu')
    u = make_u(base, 'soft')
    params = list(u.parameters())
    qids = {id(p) for p in base.q.parameters()} if model == 'qpinn' else set()
    qmask = [id(p) in qids for p in params]
    xf = sample_uniform(100, 'cpu')
    np.save(out / 'training_points.npy', xf.detach().numpy())
    point_sha = hashlib.sha256(xf.detach().numpy().tobytes()).hexdigest()
    config = dict(case='1D forward elastostatic', model=model, seed=seed,
                  width=5, qubits=5 if model == 'qpinn' else None,
                  n_rep=1 if model == 'qpinn' else None, boundary='soft',
                  residual_points=100, sampling='fixed equally spaced interior points',
                  evaluation_points=201, optimizer='Adam', lr=.005, updates=steps,
                  lambda_residual=1., lambda_boundary=weight,
                  objective='L = mean((u_xx + 4*pi^2*sin(2*pi*x))^2) + lambda_boundary*(u(0)^2 + u(1)^2)',
                  cold_start=True, parameter_count=sum(p.numel() for p in params),
                  quantum_parameters=sum(p.numel() for p, mask in zip(params, qmask) if mask),
                  training_points_sha256=point_sha, environment=environment())
    write_json(out/'config.json', config)
    optimizer = torch.optim.Adam(params, lr=.005)
    history=[]
    finite_checks = dict(loss_checks=0, gradient_checks=0, parameter_checks=0,
                         nonfinite_loss_steps=0, nonfinite_gradient_steps=0,
                         nonfinite_parameter_steps=0)
    start = time.perf_counter()
    completed_steps=0
    status='completed'
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        xf.grad=None
        lf, lb = losses(u, xf)
        loss = lf + weight*lb
        finite_checks['loss_checks']+=1
        if not all(bool(torch.isfinite(v)) for v in [lf, lb, loss]):
            finite_checks['nonfinite_loss_steps']+=1;status='nonfinite_loss';break
        if step % 100 == 0:
            history.append(diagnostics(u, params, qmask, xf, lf, lb, loss, weight, step))
        else:
            loss.backward()
        finite_checks['gradient_checks']+=1
        if not finite_grads(params):
            finite_checks['nonfinite_gradient_steps']+=1;status='nonfinite_gradient';break
        finite_checks['parameter_checks']+=1
        if not finite_params(params):
            finite_checks['nonfinite_parameter_steps']+=1;status='nonfinite_parameter_before_update';break
        optimizer.step()
        completed_steps = step+1
        finite_checks['parameter_checks']+=1
        if not finite_params(params):
            finite_checks['nonfinite_parameter_steps']+=1;status='nonfinite_parameter_after_update';break
        if (step+1)%250==0:
            print(json.dumps(dict(run=name, completed_steps=step+1,
                                  elapsed_s=time.perf_counter()-start)), flush=True)
    elapsed = time.perf_counter()-start
    if status=='completed':
        optimizer.zero_grad(set_to_none=True);xf.grad=None
        lf,lb=losses(u,xf);loss=lf+weight*lb
        history.append(diagnostics(u,params,qmask,xf,lf,lb,loss,weight,completed_steps))
        final=history[-1]
        with torch.no_grad():
            xe=torch.linspace(0,1,201).reshape(-1,1)
            pred=u(xe);target=exact(xe)
            boundary_norm=float(torch.sqrt(u(torch.zeros(1,1)).square().sum()+u(torch.ones(1,1)).square().sum()))
        np.savez(out/'predictions.npz', x=xe.numpy(), u_ref=target.numpy(), u_pred=pred.numpy())
    else:
        final=None;boundary_norm=None
    checkpoint=dict(model=u.state_dict(),optimizer=optimizer.state_dict(),completed_steps=completed_steps,
                    config=config,status=status)
    torch.save(checkpoint,out/'checkpoint.pt')
    result=dict(config=config,status=status,completed_steps=completed_steps,
                elapsed_s=elapsed,finite_checks=finite_checks,history=history,
                relative_l2=final['relative_l2_201'] if final else None,
                boundary_l2=boundary_norm,
                final_residual_loss=final['residual_loss'] if final else None,
                final_boundary_loss=final['boundary_loss_unweighted'] if final else None)
    write_json(Path(out_path) if out_path else out/'result.json',result)
    print(json.dumps(dict(run=name,status=status,relative_l2=result['relative_l2'],elapsed_s=elapsed)),flush=True)
    return result


def verify():
    setup(); checks=[]
    for kind in ['pinn','qpinn']:
        for weight in [.1,1.,10.]:
            base=build_model(kind,5,5,1,0,'cpu')
            a=make_u(base,'soft');b=copy.deepcopy(a)
            pa,pb=list(a.parameters()),list(b.parameters())
            xa,xb=sample_uniform(100,'cpu'),sample_uniform(100,'cpu')
            lf,lb=losses(a,xa);loss=lf+weight*lb
            # Reference explicitly uses the original elastostatic-forward expression.
            lf_ref=torch.mean(residual(b,xb)**2)
            x0=torch.zeros(1,1,requires_grad=True);x1=torch.ones(1,1,requires_grad=True)
            lb_ref=b(x0).pow(2).mean()+b(x1).pow(2).mean()
            total_ref=lf_ref+weight*lb_ref
            oa,ob=torch.optim.Adam(pa,lr=.005),torch.optim.Adam(pb,lr=.005)
            loss.backward()
            gb=torch.autograd.grad(total_ref,pb,allow_unused=True)
            for p,g in zip(pb,gb):p.grad=g
            grad_error=max(float((p.grad-q.grad).abs().max()) for p,q in zip(pa,pb))
            oa.step();ob.step()
            parameter_error=max(float((p-q).abs().max()) for p,q in zip(pa,pb))
            rec=dict(model=kind,lambda_boundary=weight,parameter_count=sum(p.numel() for p in pa),
                     loss_difference=abs(float(loss)-float(total_ref)),
                     gradient_max_abs_difference=grad_error,
                     one_adam_update_parameter_max_abs_difference=parameter_error)
            assert rec['loss_difference']<1e-6 and grad_error<1e-6 and parameter_error<1e-7,rec
            checks.append(rec)
    write_json(ROOT/'verification.json',dict(environment=environment(),checks=checks,
                                             passed=True,actual_original_functions_imported=True))
    print(json.dumps(checks),flush=True)


def summarize():
    rows=[]
    for p in sorted((ROOT/'runs').glob('*/result.json')):
        r=json.loads(p.read_text(encoding='utf-8'));c=r['config']
        row=dict(run=p.parent.name,model=c['model'],lambda_boundary=c['lambda_boundary'],seed=c['seed'],
                 status=r['status'],updates=r['completed_steps'],parameters=c['parameter_count'],
                 quantum_parameters=c['quantum_parameters'],relative_l2_percent=100*r['relative_l2'] if r['relative_l2'] is not None else None,
                 residual_loss=r['final_residual_loss'],boundary_l2=r['boundary_l2'],elapsed_s=r['elapsed_s'],
                 nonfinite_losses=r['finite_checks']['nonfinite_loss_steps'],
                 nonfinite_gradients=r['finite_checks']['nonfinite_gradient_steps'],
                 nonfinite_parameters=r['finite_checks']['nonfinite_parameter_steps'])
        rows.append(row)
    if rows:
        with (ROOT/'all_runs.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[]
    for kind in ['pinn','qpinn']:
        for weight in [.1,1.,10.]:
            group=[r for r in rows if r['model']==kind and r['lambda_boundary']==weight and r['status']=='completed']
            if not group:continue
            g=dict(model=kind,lambda_boundary=weight,seeds=[r['seed'] for r in group],n_runs=len(group),
                   parameters=group[0]['parameters'],quantum_parameters=group[0]['quantum_parameters'])
            for key in ['relative_l2_percent','residual_loss','boundary_l2','elapsed_s']:
                vals=[r[key] for r in group]
                g[key+'_mean']=statistics.mean(vals)
                g[key+'_sd']=statistics.stdev(vals) if len(vals)>1 else 0.
            g['nonfinite_event_count']=sum(r[k] for r in group for k in ['nonfinite_losses','nonfinite_gradients','nonfinite_parameters'])
            groups.append(g)
    write_json(ROOT/'summary.json',dict(expected_runs=18,completed_runs=len(rows),standard_deviation='sample SD (ddof=1)',groups=groups,all_runs=rows))
    if groups:
        with (ROOT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
    print(json.dumps(groups,indent=2),flush=True)


def all_runs(workers):
    # Verification runs before any matrix job is launched.
    p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output-root',str(ROOT),'verify'],check=True)
    tasks=[(m,w,s) for m in ['pinn','qpinn'] for w in [.1,1.,10.] for s in range(3)]
    (ROOT/'logs').mkdir(exist_ok=True)
    def run_one(task):
        m,w,s=task;name=name_for(m,w,s)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--output-root',str(ROOT),'run','--model',m,'--weight',str(w),'--seed',str(s)]
        with (ROOT/'logs'/f'{name}.out.log').open('w',encoding='utf-8') as out, (ROOT/'logs'/f'{name}.err.log').open('w',encoding='utf-8') as err:
            p=subprocess.run(cmd,stdout=out,stderr=err)
        return name,p.returncode
    failed=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for future in as_completed(futures):
            name,code=future.result();print(json.dumps(dict(finished=name,exit_code=code)),flush=True)
            if code:failed.append(name)
    summarize()
    if failed:raise RuntimeError(f'Failed runs: {failed}')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-root',default=str(WORKSPACE/'rerun_results'/'reviewer_supplement_20261007'/'fixed_weights'),help='Directory for freshly generated runs and summaries.')
    sp=ap.add_subparsers(dest='command',required=True)
    sp.add_parser('verify');sp.add_parser('summarize')
    allp=sp.add_parser('all');allp.add_argument('--workers',type=int,default=3)
    runp=sp.add_parser('run');runp.add_argument('--model',choices=['pinn','qpinn'],required=True)
    runp.add_argument('--weight',type=float,required=True);runp.add_argument('--seed',type=int,required=True)
    runp.add_argument('--steps',type=int,default=1000)
    runp.add_argument('--out',help='Result JSON path; checkpoint and arrays are saved beside it.')
    a=ap.parse_args()
    ROOT=Path(a.output_root).resolve()
    if a.command=='verify':verify()
    elif a.command=='summarize':summarize()
    elif a.command=='all':all_runs(a.workers)
    else:train(a.model,a.weight,a.seed,a.steps,a.out)
