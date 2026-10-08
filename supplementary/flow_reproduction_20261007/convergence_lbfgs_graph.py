"""Same L-BFGS objective with captured GPU forward/gradient operations."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from flow_core import ROOT,make_model,fields,residuals,evaluate,set_runtime,load_torch_checkpoint
from convergence_train import NormalizedModel
from convergence_quantum import replace_quantum

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=3000)
    ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--data-weight',type=float,default=100.)
    ap.add_argument('--normalize',action='store_true')
    ap.add_argument('--hard-geometry',action='store_true')
    ap.add_argument('--stress-pressure-lift',action='store_true')
    ap.add_argument('--reset-normal-stresses',action='store_true')
    ap.add_argument('--data-scale',choices=['physical','rms'],default='rms')
    ap.add_argument('--data-fields',choices=['uv','uvp'],default='uv')
    ap.add_argument('--float32',action='store_true')
    ap.add_argument('--boundary-weight',type=float,default=0.)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--reset-optimizer',action='store_true')
    ap.add_argument('--eval-every',type=int,default=250)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();set_runtime(args.seed);out=args.out;out.mkdir(parents=True,exist_ok=True)
    pts=np.load(ROOT/'protocol/points.npz')
    dtype=torch.float32 if args.float32 else torch.float64
    model=NormalizedModel(replace_quantum(make_model('pihcqnn')),args.normalize,args.hard_geometry,args.stress_pressure_lift).to(device='cuda',dtype=dtype)
    state=None
    if args.resume:
        state=load_torch_checkpoint(args.resume,'cuda');model.load_state_dict(state['model'])
    if args.reset_normal_stresses:
        if not args.stress_pressure_lift:raise ValueError('stress reset is for the explicit pressure lifting trial')
        with torch.no_grad():
            model.model.ron_out.weight[2:4].zero_();model.model.ron_out.bias[2:4].zero_()
    opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=1,max_eval=25,history_size=50,
        tolerance_grad=1e-7 if args.float32 else 1e-9,tolerance_change=1e-9 if args.float32 else 1e-12,line_search_fn='strong_wolfe')
    if state and not args.reset_optimizer:
        if state['optimizer_name']!='lbfgs':raise ValueError('Optimizer mismatch')
        opt.load_state_dict(state['optimizer'])
    p={k:torch.tensor(pts[k],dtype=dtype,device='cuda') for k in
       ['colloc','wall','inlet','inlet_uv','outlet','data_xy','data_uv']}
    for k in ['colloc','wall','inlet','data_xy']:p[k].requires_grad_(True)
    observed=pts['uvp_ref'][pts['data_idx']] if args.data_fields=='uvp' else pts['data_uv']
    target_data=torch.tensor(observed,dtype=dtype,device='cuda')
    scales=target_data.square().mean(0).sqrt() if args.data_scale=='rms' else torch.ones(target_data.shape[1],device='cuda',dtype=dtype)
    valid=np.sort(np.random.RandomState(20261007).choice(pts['test_idx'],2000,replace=False))
    test=np.setdiff1d(pts['test_idx'],valid)
    def objective(m):
        r,_=residuals(m,p['colloc']);pl=r.square().mean(0).sum();bl=pl.new_zeros(())
        if args.boundary_weight:
            for k in ['wall','inlet']:
                _,f=fields(m,p[k]);target=p['inlet_uv'] if k=='inlet' else torch.zeros_like(f[:,:2])
                bl=bl+(f[:,:2]-target).square().mean(0).sum()
            bl=bl+m(p['outlet'])[:,1].square().mean()
        _,f=fields(m,p['data_xy']);dl=((f[:,:target_data.shape[1]]-target_data)/scales).square().mean(0).sum()
        return pl+args.boundary_weight*bl+args.data_weight*dl,torch.stack([pl,bl,dl])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            for k in ['colloc','wall','inlet','data_xy']:p[k].grad=None
            loss,_=objective(model);loss.backward()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    opt.zero_grad(set_to_none=True)
    for k in ['colloc','wall','inlet','data_xy']:p[k].grad=None
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss,components=objective(model);static_loss.backward()
    # Compare fresh gradients after capture at two distinct parameter settings.
    ref=copy.deepcopy(model)
    original=[z.detach().clone() for z in model.parameters()]
    errors=[]
    for trial in range(2):
        if trial:
            with torch.no_grad():
                for a,b in zip(model.parameters(),ref.parameters()):
                    perturb=torch.randn_like(a)*1e-5;a.add_(perturb);b.copy_(a)
        graph.replay();torch.cuda.synchronize()
        ref.zero_grad(set_to_none=True)
        for k in ['colloc','wall','inlet','data_xy']:p[k].grad=None
        ref_loss,_=objective(ref);ref_loss.backward();torch.cuda.synchronize()
        max_grad=max(float((a.grad-b.grad).abs().max()) for a,b in zip(model.parameters(),ref.parameters()))
        max_ref=max(float(b.grad.abs().max()) for b in ref.parameters())
        errors.append(dict(loss_diff=abs(float(static_loss)-float(ref_loss)),gradient_max_abs_diff=max_grad,
                           gradient_scaled_diff=max_grad/(1+max_ref)))
    assert all(e['gradient_scaled_diff']<(3e-6 if args.float32 else 1e-10) for e in errors),errors
    with torch.no_grad():
        for a,b in zip(model.parameters(),original):a.copy_(b)
    del ref,original
    (out/'graph_checks.json').write_text(json.dumps(errors,indent=2))
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(parameters=sum(z.numel() for z in model.parameters()),dtype=str(dtype),optimizer='lbfgs',
        protocol=json.loads((ROOT/'protocol/protocol.json').read_text()),execution='captured objective and gradients',
        source_checkpoint_step=int(state['step']) if state else 0,optimizer_restored=bool(state and not args.reset_optimizer),
        training_observations=f'193 fixed {args.data_fields} values only; no teacher or other reference labels',
        observed_fields=['u','v','p'] if args.data_fields=='uvp' else ['u','v'],
        status='new_method_search_not_historical_reproduction',
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    (out/'config.json').write_text(json.dumps(config,indent=2))
    np.save(out/'validation_indices.npy',valid);np.save(out/'reserved_test_indices.npy',test)
    calls=0;start=time.perf_counter();history=[];best_score=float('inf')
    def closure():
        nonlocal calls
        calls+=1;graph.replay()
        # Parameter .grad tensors allocated during capture stay attached.
        return static_loss
    def checkpoint(step):
        nonlocal best_score
        torch.cuda.synchronize()
        if calls and not torch.isfinite(static_loss).item():raise RuntimeError('Nonfinite captured objective')
        scores,_=evaluate(model,pts['xy_ref'][valid],pts['uvp_ref'][valid],device='cuda')
        rec=dict(step=step,objective_calls=calls,elapsed_s=time.perf_counter()-start,validation=scores)
        if calls:rec.update(loss=float(static_loss),components=components.cpu().tolist())
        saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),optimizer_name='lbfgs',step=step,config=config)
        torch.save(saved,out/'checkpoint.pt')
        history.append(rec);(out/'history.json').write_text(json.dumps(history,indent=2))
        val=max(scores['u_relative_l2']/.0274,scores['v_relative_l2']/.0959)
        if val<best_score:
            best_score=val;torch.save(saved,out/'best_validation_checkpoint.pt')
            (out/'best_validation.json').write_text(json.dumps(rec,indent=2))
        print(json.dumps(rec),flush=True)
    checkpoint(0)
    for step in range(1,args.steps+1):
        opt.step(closure)
        if step==1 or step%args.eval_every==0 or step==args.steps:checkpoint(step)
    scores,pred=evaluate(model,pts['xy_ref'],pts['uvp_ref'],device='cuda')
    def e(ix):return dict(zip(['u_relative_l2','v_relative_l2','p_relative_l2'],map(float,
        np.sqrt(np.sum((pred[ix]-pts['uvp_ref'][ix])**2,0)/np.sum(pts['uvp_ref'][ix]**2,0)))))
    result=dict(config=config,checks=errors,history=history,objective_calls=calls,elapsed_s=time.perf_counter()-start,
        final_metrics=dict(all=scores,heldout=e(pts['test_idx']),validation=e(valid),reserved_test=e(test),training_data=e(pts['data_idx'])))
    np.savez(out/'predictions.npz',xy=pts['xy_ref'],uvp_ref=pts['uvp_ref'],uvp_pred=pred,
        data_idx=pts['data_idx'],heldout_idx=pts['test_idx'],validation_idx=valid,reserved_test_idx=test)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(completed=True,metrics=result['final_metrics'])),flush=True)

if __name__=='__main__':main()
