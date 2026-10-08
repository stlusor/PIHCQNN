"""Short timing checks of the exact existing objective; no training claims."""
import json, time
import numpy as np
import torch
from flow_core import ROOT, make_model, residuals, fields, set_runtime

def objective(model, pts, device):
    p = {k: torch.tensor(pts[k], device=device, dtype=next(model.parameters()).dtype)
         for k in ['colloc','wall','inlet','inlet_uv','outlet','data_xy','data_uv']}
    r,_ = residuals(model, p['colloc'].clone().requires_grad_(True))
    loss = r.square().mean(0).sum()
    for k in ['wall','inlet']:
        _,f = fields(model,p[k].clone().requires_grad_(True))
        target = p['inlet_uv'] if k=='inlet' else torch.zeros_like(f[:,:2])
        loss = loss + 2*(f[:,:2]-target).square().mean(0).sum()
    loss = loss + 2*model(p['outlet'])[:,1].square().mean()
    _,f = fields(model,p['data_xy'].clone().requires_grad_(True))
    return loss+(f[:,:2]-p['data_uv']).square().mean(0).sum()

if __name__ == '__main__':
    pts = np.load(ROOT/'protocol'/'points.npz')
    for device in ['cpu','cuda']:
        set_runtime(1234)
        model = make_model('pihcqnn').to(device)
        model.load_state_dict(torch.load(ROOT/'runs/pihcqnn_seed1234_steps1000/model.pt',weights_only=True,map_location=device))
        opt = torch.optim.Adam(model.parameters(),lr=4e-4)
        start = time.perf_counter()
        losses=[]
        for i in range(6):
            t=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=objective(model,pts,device);loss.backward();opt.step()
            if device=='cuda':torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            print(json.dumps(dict(device=device,step=i,seconds=time.perf_counter()-t,loss=losses[-1])),flush=True)
        print(json.dumps(dict(device=device,seconds=time.perf_counter()-start,losses=losses)),flush=True)
