"""Steady mixed-variable cylinder flow, ported from the pinned Rao source.

Coordinates and fields retain their physical units. This module does not
change the downloaded source or claim recovery of the authors' random draws.
"""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
L, H, RHO, MU, RADIUS = 1.1, 0.41, 1.0, 0.02, 0.05
CENTER = np.array([0.2, 0.2])

def load_torch_checkpoint(path, map_location='cpu'):
    # Our earlier checkpoints included torch.__version__, a str subclass.
    # Keep weights-only loading and allow only this known PyTorch value class.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path,weights_only=True,map_location=map_location)

class NumpyWeightsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('numpy.core.multiarray', '_reconstruct'):
            from numpy._core.multiarray import _reconstruct
            return _reconstruct
        if (module, name) == ('numpy', 'ndarray'): return np.ndarray
        if (module, name) == ('numpy', 'dtype'): return np.dtype
        raise pickle.UnpicklingError(f'Unexpected global: {module}.{name}')

def checkpoint_arrays():
    with (SOURCE/'PINN_steady'/'uvNN.pickle').open('rb') as stream:
        w,b = NumpyWeightsUnpickler(stream).load()
    assert len(w) == len(b) == 9
    assert [a.shape for a in w] == [(2,40)] + [(40,40)]*7 + [(40,5)]
    assert all(a.dtype == np.float32 for a in w+b)
    return w,b

class RaoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [2]+[40]*8+[5]
        self.layers = nn.ModuleList(nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:]))
        for layer in self.layers:
            std = np.sqrt(2/(layer.in_features+layer.out_features))
            nn.init.trunc_normal_(layer.weight, std=std, a=-2*std, b=2*std)
            nn.init.zeros_(layer.bias)
    def forward(self,x):
        for layer in self.layers[:-1]: x=torch.tanh(layer(x))
        return self.layers[-1](x)
    def load_author_weights(self):
        w,b=checkpoint_arrays()
        with torch.no_grad():
            for layer, wi,bi in zip(self.layers,w,b):
                layer.weight.copy_(torch.from_numpy(wi.T.copy()))
                layer.bias.copy_(torch.from_numpy(bi.reshape(-1).copy()))
        return self

class PaperPINN5(nn.Module):
    """Five layers including input: 2 -> 20 -> 5 -> 20 -> 5."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,20),nn.Tanh(),nn.Linear(20,5),nn.Tanh(),
                               nn.Linear(5,20),nn.Tanh(),nn.Linear(20,5))
    def forward(self,x): return self.net(x)

class PaperPINN6(nn.Module):
    """The quantum block is replaced by a 5 -> 5 FC layer, with the skip."""
    def __init__(self):
        super().__init__()
        self.enn_in=nn.Linear(2,20); self.enn_out=nn.Linear(20,5)
        self.middle=nn.Linear(5,5); self.ron_in=nn.Linear(10,20); self.ron_out=nn.Linear(20,5)
    def forward(self,x):
        shallow=torch.tanh(self.enn_out(torch.tanh(self.enn_in(x))))
        deep=torch.tanh(self.middle(shallow))
        return self.ron_out(torch.tanh(self.ron_in(torch.cat([shallow,deep],-1))))

def make_model(name):
    if name=='rao': return RaoMLP()
    if name=='rao_checkpoint': return RaoMLP().load_author_weights()
    if name=='pinn5': return PaperPINN5()
    if name=='pinn6': return PaperPINN6()
    if name=='pihcqnn':
        sys.path.insert(0,str(ROOT/'hybrid_backend'))
        from common import PIHCQNN, PIHConfig
        return PIHCQNN(PIHConfig(2,20,5,2,out_dim=5,concat_shallow=True))
    raise ValueError(name)

def derivative(y,xy,create_graph=True):
    return torch.autograd.grad(y,xy,torch.ones_like(y),create_graph=create_graph,
                               retain_graph=True)[0]

def fields(model,xy,create_graph=True):
    raw=model(xy)
    dpsi=derivative(raw[:,0:1],xy,create_graph)
    uvp=torch.cat([dpsi[:,1:2],-dpsi[:,0:1],raw[:,1:2]],1)
    return raw,uvp

def residuals(model,xy):
    raw,f=fields(model,xy)
    u,v,p=f.split(1,1)
    du=derivative(u,xy); dv=derivative(v,xy)
    s11,s22,s12=raw[:,2:3],raw[:,3:4],raw[:,4:5]
    ds11=derivative(s11,xy); ds22=derivative(s22,xy); ds12=derivative(s12,xy)
    ru=RHO*(u*du[:,0:1]+v*du[:,1:2])-ds11[:,0:1]-ds12[:,1:2]
    rv=RHO*(u*dv[:,0:1]+v*dv[:,1:2])-ds12[:,0:1]-ds22[:,1:2]
    r11=-p+2*MU*du[:,0:1]-s11
    r22=-p+2*MU*dv[:,1:2]-s22
    r12=MU*(du[:,1:2]+dv[:,0:1])-s12
    rp=p+(s11+s22)/2
    return torch.cat([ru,rv,r11,r22,r12,rp],1),du[:,0:1]+dv[:,1:2]

def numpy_author_forward(xy):
    """Independent analytic first derivative of the downloaded tanh MLP."""
    weights,biases=checkpoint_arrays()
    x=np.asarray(xy,dtype=np.float64)
    jac=np.broadcast_to(np.eye(2),(len(x),2,2)).copy()
    for k,(w,b) in enumerate(zip(weights,biases)):
        w=w.astype(np.float64); b=b.astype(np.float64)
        z=x@w+b; jac=jac@w
        if k < len(weights)-1:
            x=np.tanh(z); jac*=1-x[:,None,:]**2
        else: x=z
    return x,np.stack([jac[:,1,0],-jac[:,0,0],x[:,1]],1)

def reference_data():
    m=loadmat(SOURCE/'FluentReferenceMu002'/'FluentSol.mat')
    xy=np.column_stack([m['x'].ravel(),m['y'].ravel()])
    uvp=np.column_stack([m['vx'].ravel(),m['vy'].ravel(),m['p'].ravel()])
    assert xy.shape == (19340,2) and uvp.shape == (19340,3)
    assert np.isfinite(xy).all() and np.isfinite(uvp).all()
    return xy,uvp

def lhs_classic(n,samples,rng):
    cut=np.linspace(0,1,samples+1)
    rd=rng.rand(samples,n)
    rd=rd*(cut[1:]-cut[:-1])[:,None]+cut[:-1,None]
    out=np.empty_like(rd)
    for j in range(n): out[:,j]=rd[rng.permutation(samples),j]
    return out

def author_point_recipe(seed=1234):
    """LHS sizes, order and cylinder filter follow the actual public script."""
    rng=np.random.RandomState(seed)
    upper=np.array([0,H])+np.array([L,0])*lhs_classic(2,441,rng)
    lower=np.array([0,0])+np.array([L,0])*lhs_classic(2,441,rng)
    inlet=np.array([0,0])+np.array([0,H])*lhs_classic(2,201,rng)
    inlet_uv=np.column_stack([4*inlet[:,1]*(H-inlet[:,1])/H**2,np.zeros(len(inlet))])
    outlet=np.array([L,0])+np.array([0,H])*lhs_classic(2,201,rng)
    theta=2*np.pi*lhs_classic(1,251,rng).ravel()
    cyl=np.column_stack([RADIUS*np.cos(theta)+.2,RADIUS*np.sin(theta)+.2])
    wall=np.concatenate([cyl,upper,lower])
    bulk=np.array([L,H])*lhs_classic(2,40000,rng)
    near=np.array([.1,.1])+np.array([.2,.2])*lhs_classic(2,10000,rng)
    interior=np.concatenate([bulk,near])
    interior=interior[np.linalg.norm(interior-CENTER,axis=1)>RADIUS]
    # Public source appends CYLD a second time as well as including it in WALL.
    colloc=np.concatenate([interior,wall,cyl,outlet,inlet])
    return dict(colloc=colloc,wall=wall,inlet=inlet,inlet_uv=inlet_uv,outlet=outlet)

def build_protocol(path, *, collocation_count=1024, wall_count=256, inlet_count=64,
                   outlet_count=64, seed=1234):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    xy,uvp=reference_data()
    idx=np.sort(np.random.RandomState(seed).choice(len(xy),int(.01*len(xy)),replace=False))
    unique_xy,groups=np.unique(xy,axis=0,return_inverse=True)
    assert len(np.unique(groups[idx]))==len(idx), 'The sampled subset contains duplicate coordinates'
    # The CFD file repeats 140 spatial coordinates. Exclude matching locations,
    # not just sampled row numbers, when defining the held-out evaluation.
    test=np.flatnonzero(~np.isin(groups,groups[idx]))
    assert not np.intersect1d(groups[idx],groups[test]).size
    all_points=author_point_recipe(seed)
    rng=np.random.RandomState(seed+1)
    points={}
    counts=dict(colloc=collocation_count,wall=wall_count,inlet=inlet_count,outlet=outlet_count)
    selected={}
    for key in counts:
        n=len(all_points[key]) if counts[key]==0 else min(counts[key],len(all_points[key]))
        selected[key]=np.sort(rng.choice(len(all_points[key]),n,replace=False))
        points[key]=all_points[key][selected[key]]
    points['inlet_uv']=all_points['inlet_uv'][selected['inlet']]
    points.update(xy_ref=xy,uvp_ref=uvp,data_idx=idx,test_idx=test,
                  data_xy=xy[idx],data_uv=uvp[idx,:2])
    np.savez(path/'points.npz',**points)
    np.save(path/'data_indices.npy',idx)
    np.save(path/'heldout_indices.npy',test)
    metadata=dict(status='new_reproduction_protocol_not_historical_sampling',
                  reference_points=len(xy),data_points=len(idx),heldout_points=len(test),
                  data_fraction=float(len(idx)/len(xy)),sampling_rule='floor(0.01 * N), without replacement',
                  sampling_rng='numpy.random.RandomState MT19937',sampling_seed=seed,
                  fixed_during_training=True,shared_across_models=True,
                  observed_fields=['u','v'],evaluations=['all_reference_points','heldout_reference_points'],
                  heldout_rule='exclude every row whose exact coordinate equals a sampled coordinate',
                  extra_duplicate_rows_excluded=int(len(xy)-len(idx)-len(test)),
                  sampled_distinct_coordinates=int(len(np.unique(groups[idx]))),
                  author_recipe_counts={k:len(v) for k,v in all_points.items() if k!='inlet_uv'},
                  experiment_counts={k:len(points[k]) for k in counts},
                  full_reference_unique_coordinates=int(len(unique_xy)),
                  points_sha256=hashlib.sha256((path/'points.npz').read_bytes()).hexdigest(),
                  data_indices_sha256=hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest(),
                  reference_min_distance_to_cylinder_center=float(np.linalg.norm(xy-CENTER,axis=1).min()))
    (path/'protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata

def evaluate(model,xy,uvp,idx=None,batch=512,device='cuda'):
    if idx is not None: xy,uvp=xy[idx],uvp[idx]
    prediction=[]
    for start in range(0,len(xy),batch):
        x=torch.tensor(xy[start:start+batch],dtype=next(model.parameters()).dtype,device=device,requires_grad=True)
        with torch.enable_grad(): _,p=fields(model,x,False)
        prediction.append(p.detach().cpu().numpy())
    pred=np.concatenate(prediction)
    errors=np.sqrt(np.sum((pred-uvp)**2,axis=0)/np.sum(uvp**2,axis=0))
    return dict(zip(['u_relative_l2','v_relative_l2','p_relative_l2'],map(float,errors))),pred

def set_runtime(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
