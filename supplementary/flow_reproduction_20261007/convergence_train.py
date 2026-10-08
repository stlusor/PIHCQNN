"""Controlled convergence trials using the fixed 193 velocity observations.

Rao weights and unobserved reference labels never enter the training objective.
Each modified protocol is recorded; a successful new trial does not recover
historical experiments. Validation coordinates are fixed independently of
training observations and are used to inspect convergence.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from flow_core import ROOT,make_model,fields,residuals,evaluate,set_runtime,L,H,load_torch_checkpoint
from convergence_quantum import replace_quantum

class NormalizedModel(nn.Module):
    def __init__(self,model,normalize=False,hard_geometry=False,stress_pressure_lift=False):
        super().__init__();self.model=model;self.normalize=normalize;self.hard_geometry=hard_geometry;self.stress_pressure_lift=stress_pressure_lift
        self.register_buffer('_coord_span',torch.tensor([L,H],dtype=torch.float64),persistent=False)
        if hard_geometry:
            self.cylinder_psi=nn.Parameter(torch.tensor(2*.2**2/H-4*.2**3/(3*H**2)))
    def forward(self,x):
        coord=x
        if self.normalize:x=2*x/self._coord_span-1
        raw=self.model(x)
        if self.hard_geometry:
            xx,yy=coord[:,0:1],coord[:,1:2]
            f=2*yy.square()/H-4*yy.pow(3)/(3*H**2)
            b=(xx/.2).square()*(yy*(H-yy)/(.2*(H-.2))).square()
            d=((xx-.2).square()+(yy-.2).square()-.05**2)/(2*.05*.05)
            a=b/(b+d.square())
            t=yy/H
            g=16*H*(xx/L).square()*t.square()*(1-t).square()*d.square()/(1+d.square())
            psi=f+a*(self.cylinder_psi-f)+g*raw[:,0:1]
            raw=torch.cat([psi,(1-xx/L)*raw[:,1:2],raw[:,2:]],1)
        if self.stress_pressure_lift:
            raw=torch.cat([raw[:,:2],raw[:,2:4]-raw[:,1:2],raw[:,4:5]],1)
        return raw

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='pinn6',choices=['pinn5','pinn6','pihcqnn','rao'])
    ap.add_argument('--optimizer',default='lbfgs',choices=['adam','lbfgs'])
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--lr',type=float,default=1.)
    ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--data-weight',type=float,default=1.)
    ap.add_argument('--physics-weight',type=float,default=1.)
    ap.add_argument('--boundary-weight',type=float,default=2.)
    ap.add_argument('--data-scale',choices=['physical','rms'],default='physical')
    ap.add_argument('--data-fields',choices=['uv','uvp'],default='uv')
    ap.add_argument('--normalize',action='store_true')
    ap.add_argument('--hard-geometry',action='store_true')
    ap.add_argument('--dense-quantum',action='store_true')
    ap.add_argument('--jet',action='store_true')
    ap.add_argument('--float64',action='store_true')
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--init',type=Path)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--reset-optimizer',action='store_true')
    ap.add_argument('--eval-every',type=int,default=250)
    ap.add_argument('--physics-chunk',type=int,default=1024)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();set_runtime(args.seed)
    global fields,residuals
    if args.jet:
        if args.model!='pihcqnn' or not args.dense_quantum:raise ValueError('jets require PIHCQNN dense backend')
        from convergence_jets import fields,residuals
    dtype=torch.float64 if args.float64 else torch.float32
    if args.float64 and args.model=='pihcqnn' and not args.dense_quantum:
        raise ValueError('float64 needs precision-aware dense VQC')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    pts=np.load(ROOT/'protocol/points.npz')
    protocol=json.loads((ROOT/'protocol/protocol.json').read_text())
    base=make_model(args.model)
    if args.init:base.load_state_dict(torch.load(args.init,weights_only=True,map_location='cpu'))
    if args.dense_quantum:base=replace_quantum(base)
    model=NormalizedModel(base,args.normalize,args.hard_geometry).to(device=args.device,dtype=dtype)
    resume=None
    if args.resume:
        resume=load_torch_checkpoint(args.resume,args.device)
        model.load_state_dict(resume['model'])
    if args.optimizer=='adam':opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    else:opt=torch.optim.LBFGS(model.parameters(),lr=args.lr,max_iter=1,max_eval=25,
            history_size=50,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    restore_optimizer=bool(resume and resume['optimizer_name']==args.optimizer and not args.reset_optimizer)
    if restore_optimizer:opt.load_state_dict(resume['optimizer'])
    tensors={k:torch.tensor(pts[k],dtype=dtype,device=args.device)
             for k in ['colloc','wall','inlet','inlet_uv','outlet','data_xy','data_uv']}
    observed=pts['uvp_ref'][pts['data_idx']] if args.data_fields=='uvp' else pts['data_uv']
    target_data=torch.tensor(observed,dtype=dtype,device=args.device)
    scales=(target_data.square().mean(0).sqrt() if args.data_scale=='rms'
            else torch.ones(target_data.shape[1],dtype=dtype,device=args.device))
    valid=np.sort(np.random.RandomState(20261007).choice(pts['test_idx'],2000,replace=False))
    reserved=np.setdiff1d(pts['test_idx'],valid)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(parameters=sum(p.numel() for p in model.parameters()),protocol=protocol,
        training_observations=f'only the fixed 193 {args.data_fields} reference values; no teacher or other reference labels',
        observed_fields=['u','v','p'] if args.data_fields=='uvp' else ['u','v'],
        validation_seed=20261007,validation_count=len(valid),reserved_test_count=len(reserved),
        data_scales=scales.detach().cpu().tolist(),initial_optimizer_restored=restore_optimizer)
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    np.save(out/'validation_indices.npy',valid);np.save(out/'reserved_test_indices.npy',reserved)
    calls=0;last={};start=time.perf_counter();history=[];best_score=float('inf')
    def objective():
        nonlocal calls,last
        opt.zero_grad(set_to_none=True);calls+=1
        pl=bl=dl=0.
        if args.physics_weight:
            for chunk in tensors['colloc'].split(args.physics_chunk):
                r,_=residuals(model,chunk.detach().clone().requires_grad_(True))
                loss=r.square().mean(0).sum()*len(chunk)/len(tensors['colloc'])
                (args.physics_weight*loss).backward();pl+=float(loss.detach())
        for k in (['wall','inlet','outlet'] if args.boundary_weight else []):
            x=tensors[k].detach().clone().requires_grad_(True)
            if k=='outlet':loss=model(x)[:,1].square().mean()
            else:
                _,f=fields(model,x)
                target=tensors['inlet_uv'] if k=='inlet' else torch.zeros_like(f[:,:2])
                loss=(f[:,:2]-target).square().mean(0).sum()
            (args.boundary_weight*loss).backward();bl+=float(loss.detach())
        if args.data_weight:
            _,f=fields(model,tensors['data_xy'].detach().clone().requires_grad_(True))
            loss=((f[:,:target_data.shape[1]]-target_data)/scales).square().mean(0).sum()
            (args.data_weight*loss).backward();dl=float(loss.detach())
        total=args.physics_weight*pl+args.boundary_weight*bl+args.data_weight*dl
        if not np.isfinite(total) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'Non-finite objective/gradient, call {calls}')
        last=dict(physics_loss=pl,boundary_loss=bl,data_loss=dl,total_loss=total)
        return torch.tensor(total,dtype=dtype,device=args.device)
    def checkpoint(step):
        nonlocal best_score
        metric,_=evaluate(model,pts['xy_ref'][valid],pts['uvp_ref'][valid],device=args.device)
        rec=dict(step=step,objective_calls=calls,elapsed_s=time.perf_counter()-start,validation=metric,**last)
        history.append(rec);(out/'history.json').write_text(json.dumps(history,indent=2))
        saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),optimizer_name=args.optimizer,
                        step=step,objective_calls=calls,config=config)
        torch.save(saved,out/'checkpoint.pt')
        score=max(metric['u_relative_l2']/.0274,metric['v_relative_l2']/.0959)
        if score<best_score:
            best_score=score;torch.save(saved,out/'best_validation_checkpoint.pt')
            (out/'best_validation.json').write_text(json.dumps(rec,indent=2))
        print(json.dumps(rec),flush=True)
    checkpoint(0)
    for step in range(1,args.steps+1):
        if args.optimizer=='adam':objective();opt.step()
        else:opt.step(objective)
        if step==1 or step%args.eval_every==0 or step==args.steps:checkpoint(step)
    # Final reserved-set evaluation happens only once after the configured run.
    scores,pred=evaluate(model,pts['xy_ref'],pts['uvp_ref'],device=args.device)
    def errors(ix):return dict(zip(['u_relative_l2','v_relative_l2','p_relative_l2'],
        map(float,np.sqrt(np.sum((pred[ix]-pts['uvp_ref'][ix])**2,0)/np.sum(pts['uvp_ref'][ix]**2,0)))))
    result=dict(config=config,history=history,objective_calls=calls,elapsed_s=time.perf_counter()-start,
                final_metrics=dict(all=scores,heldout=errors(pts['test_idx']),validation=errors(valid),
                                   reserved_test=errors(reserved),training_data=errors(pts['data_idx'])))
    np.savez(out/'predictions.npz',xy=pts['xy_ref'],uvp_ref=pts['uvp_ref'],uvp_pred=pred,
             data_idx=pts['data_idx'],heldout_idx=pts['test_idx'],validation_idx=valid,reserved_test_idx=reserved)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(completed=True,metrics=result['final_metrics'],elapsed_s=result['elapsed_s'])),flush=True)

if __name__=='__main__':main()
