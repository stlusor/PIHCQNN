"""Summarize the published training results without retraining any model."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
import numpy as np
from variable_inverse import fem_forward


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def describe(values, factor=1.0):
    values=[factor*float(v) for v in values]
    return {'n':len(values),'mean':statistics.mean(values),
            'sample_sd':statistics.stdev(values) if len(values)>1 else 0.0}


def main(a):
    root=Path(a.results)
    out={'error_unit':'Derived *_percent fields use percent; source rel_l2 fields use a fraction','standard_deviation':'sample SD (ddof=1)',
         'results_root':str(root),'table7':[],'table8_sensitivity':[],
         'table8_ansatz':[],'table9_initialization':[],
         'table9_full_training':[],'dem':{},'table10':{},'table11':[]}
    groups={}
    for p in sorted((root/'elastostatic_core').glob('*.json')):
        r=load(p);groups.setdefault((r['model'],r['bc'],r['sampling'],r['weighting']),[]).append(r)
    for (model,bc,sampling,weighting),rows in sorted(groups.items()):
        out['table7'].append({'model':model,'boundary':bc,'sampling':sampling,'weighting':weighting,
            'relative_l2_percent':describe([r['rel_l2'] for r in rows],100),
            'boundary_norm':describe([r['boundary_l2'] for r in rows]),
            'elapsed_seconds':describe([r['elapsed_s'] for r in rows])})
    groups={}
    for p in sorted((root/'sensitivity').glob('*.json')):
        r=load(p);groups.setdefault((r['lr'],r['qubits'],r['width'],r['n']),[]).append(r)
    for (lr,qubits,width,n),rows in sorted(groups.items()):
        out['table8_sensitivity'].append({'lr':lr,'qubits':qubits,'width':width,'n':n,
            'relative_l2_percent':describe([r['rel_l2'] for r in rows],100)})
    groups={}
    for p in sorted((root/'ansatz').glob('*.json')):
        r=load(p);groups.setdefault(r['ansatz'],[]).append(r)
    for ansatz,rows in sorted(groups.items()):
        out['table8_ansatz'].append({'ansatz':ansatz,'updates':rows[0]['steps'],
            'parameters':rows[0]['params'],'relative_l2_percent':describe([r['rel_l2'] for r in rows],100)})
    out['table9_initialization']=[r for r in load(root/'gradient_scaling.json') if r['n_rep']==3]
    for nq,label in [(2,'q2'),(4,'q4'),(5,'lr5e-3'),(6,'q6'),(10,'q10')]:
        rows=[load(root/'sensitivity'/f'{label}_seed{s}.json') for s in (0,1)]
        out['table9_full_training'].append({'qubits':nq,'n_rep':1,'checkpoint':'immediately before update 500',
            'quantum_gradient_norm':describe([r['history'][-1]['grad_q'] for r in rows]),
            'classical_gradient_norm':describe([r['history'][-1]['grad_classical'] for r in rows])})
    out['dem']={'updates':5000,'relative_l2_percent':describe([load(root/f'dem_1d_seed{s}.json')['rel_l2'] for s in range(3)],100)}
    out['table10']=load(root/'classical_baselines.json')
    fem=load(root/'variable_inverse_fem_adjoint.json')
    x,u=fem_forward(np.asarray(fem['theta_hat']),n=fem['mesh_elements'])
    xe=np.linspace(0,1,201);ue=np.interp(xe,x,u);ref=np.sin(np.pi*xe)
    err=float(np.linalg.norm(ue-ref)/np.linalg.norm(ref))
    out['table11'].append({'model':'FEM_two_level_adjoint','coefficient_error_percent':100*fem['theta_rel_error'],
        'displacement_error_percent':100*err,'elapsed_seconds':fem['elapsed_s'],
        'field_evaluation':'Reconstruct 100-element linear FEM solution using the saved converged coefficients; linearly interpolate to 201 points and compare with sin(pi*x).'})
    for model in ('pinn','qpinn'):
        rows=[load(root/f'variable_inverse_{model}_seed{s}.json') for s in range(3)]
        out['table11'].append({'model':model,'coefficient_error_percent':describe([r['theta_rel_error'] for r in rows],100),
            'displacement_error_percent':describe([r['u_rel_l2'] for r in rows],100),
            'elapsed_seconds':describe([r['elapsed_s'] for r in rows])})
    target=Path(a.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'saved':str(target),'FEM_adjoint_displacement_error_percent':100*err}))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',default=str(Path(__file__).resolve().parent/'results'))
    ap.add_argument('--out',default=str(Path(__file__).resolve().parent.parent/'rerun_results'/'final_training_summary.json'))
    main(ap.parse_args())
