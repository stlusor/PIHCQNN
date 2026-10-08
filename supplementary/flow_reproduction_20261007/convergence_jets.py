"""Exact value/first/second spatial derivatives propagated by the chain rule.

Jet order is value, x, y, xx, xy, yy. Network/circuit parameters still use
ordinary PyTorch backpropagation. This only accelerates the same objective.
"""
import torch
from flow_core import L,H,RHO,MU
from convergence_quantum import rot_matrix

def product(a,b):
    return torch.stack([a[:,0]*b[:,0],
        a[:,1]*b[:,0]+a[:,0]*b[:,1],
        a[:,2]*b[:,0]+a[:,0]*b[:,2],
        a[:,3]*b[:,0]+2*a[:,1]*b[:,1]+a[:,0]*b[:,3],
        a[:,4]*b[:,0]+a[:,1]*b[:,2]+a[:,2]*b[:,1]+a[:,0]*b[:,4],
        a[:,5]*b[:,0]+2*a[:,2]*b[:,2]+a[:,0]*b[:,5]],1)

def compose(x,value,first,second):
    g=first[:,None]*x[:,1:3]
    q=torch.stack([x[:,1].square(),x[:,1]*x[:,2],x[:,2].square()],1)
    h=first[:,None]*x[:,3:6]+second[:,None]*q
    return torch.cat([value[:,None],g,h],1)

def tanh(x):
    v=torch.tanh(x[:,0]);d=1-v.square()
    return compose(x,v,d,-2*v*d)

def linear(x,layer):
    out=x@layer.weight.T
    return torch.cat([out[:,0:1]+layer.bias,out[:,1:]],1)

def shift(x,constant):
    return torch.cat([x[:,0:1]+constant,x[:,1:]],1)

def reciprocal(x):
    v=1/x[:,0]
    return compose(x,v,-v.square(),2*v.pow(3))

def geometry_jets(raw,xy,cylinder_psi):
    n=len(xy);eye=torch.eye(2,dtype=xy.dtype,device=xy.device)
    seed=torch.cat([xy[:,None],eye.expand(n,-1,-1),xy.new_zeros(n,3,2)],1)
    x,y=seed[:,:,:1],seed[:,:,1:2]
    y2=product(y,y);f=2*y2/H-4*product(y2,y)/(3*H**2)
    xx=x/.2;wy=product(y,shift(-y,H))/(.2*(H-.2))
    b=product(product(xx,xx),product(wy,wy))
    xc=shift(x,-.2);yc=shift(y,-.2)
    d=shift(product(xc,xc)+product(yc,yc),-.05**2)/(2*.05*.05)
    d2=product(d,d);a=product(b,reciprocal(b+d2))
    t=y/H;one_t=shift(-t,1);xn=x/L
    g=16*H*product(product(product(product(xn,xn),product(t,t)),product(one_t,one_t)),product(d2,reciprocal(shift(d2,1))))
    psi=f+product(a,shift(-f,cylinder_psi))+product(g,raw[:,:,:1])
    pressure=product(shift(-x/L,1),raw[:,:,1:2])
    return torch.cat([psi,pressure,raw[:,:,2:]],2)

def qvc(x,q):
    nq=q.n_qubits
    cdtype=torch.complex128 if x.dtype==torch.float64 else torch.complex64
    mats=[]
    for w in q.weights:
        matrix=rot_matrix(w[-1],cdtype)
        for wi in reversed(w[:-1]):matrix=torch.kron(matrix,rot_matrix(wi,cdtype))
        mats.append(matrix)
    state=torch.zeros((1,1,1<<nq),dtype=cdtype,device=x.device)
    state[:,:,0]=1
    state=(state@mats[0].T).index_select(2,q.permutation)
    state=torch.cat([state,torch.zeros((1,5,1<<nq),dtype=cdtype,device=x.device)],1).expand(len(x),-1,-1)
    for k in range(q.n_rep):
        for wire in range(nq):
            angle=x[:,:,wire]/2
            c=torch.cos(angle[:,0]);s=torch.sin(angle[:,0])
            cj=compose(angle,c,-s,-c)[:,:,None,None]
            sj=compose(angle,s,c,-s)[:,:,None,None]
            low=1<<wire;high=1<<(nq-wire-1)
            z=state.reshape(len(x),6,high,2,low)
            z0,z1=z[:,:,:,0,:],z[:,:,:,1,:]
            y0=product(cj,z0)-product(sj,z1)
            y1=product(sj,z0)+product(cj,z1)
            state=torch.stack([y0,y1],3).reshape(len(x),6,-1)
        state=(state@mats[k+1].T).index_select(2,q.permutation)
    probs=product(state.real,state.real)+product(state.imag,state.imag)
    return probs@q.signs.to(dtype=x.dtype)

def raw_jets(model,xy):
    base=model.model
    x=xy.detach()
    scale=x.new_tensor([2/L,2/H]) if model.normalize else x.new_ones(2)
    value=2*x/x.new_tensor([L,H])-1 if model.normalize else x
    first=torch.eye(2,dtype=x.dtype,device=x.device)*scale
    seed=torch.cat([value[:,None],first.expand(len(x),-1,-1),x.new_zeros(len(x),3,2)],1)
    shallow=tanh(linear(tanh(linear(seed,base.enn_in)),base.enn_out))
    deep=tanh(qvc(shallow,base.q))
    z=torch.cat([shallow,deep],2)
    raw=linear(tanh(linear(z,base.ron_in)),base.ron_out)
    raw=geometry_jets(raw,xy.detach(),model.cylinder_psi) if model.hard_geometry else raw
    if model.stress_pressure_lift:
        raw=torch.cat([raw[:,:,:2],raw[:,:,2:4]-raw[:,:,1:2],raw[:,:,4:5]],2)
    return raw

def fields(model,xy,create_graph=True):
    j=raw_jets(model,xy)
    return j[:,0],torch.stack([j[:,2,0],-j[:,1,0],j[:,0,1]],1)

def residuals(model,xy):
    j=raw_jets(model,xy)
    u=j[:,2,0];v=-j[:,1,0];p=j[:,0,1]
    ux=j[:,4,0];uy=j[:,5,0];vx=-j[:,3,0];vy=-j[:,4,0]
    s11,s22,s12=j[:,0,2:].unbind(1)
    ru=RHO*(u*ux+v*uy)-j[:,1,2]-j[:,2,4]
    rv=RHO*(u*vx+v*vy)-j[:,1,4]-j[:,2,3]
    return torch.stack([ru,rv,-p+2*MU*ux-s11,-p+2*MU*vy-s22,
                        MU*(uy+vx)-s12,p+(s11+s22)/2],1),ux+vy

def verify_and_benchmark():
    import copy,json,time,numpy as np
    from flow_core import ROOT,make_model,fields as af,residuals as ar,set_runtime
    from convergence_quantum import replace_quantum
    from convergence_train import NormalizedModel
    records=[]
    for device in ['cuda','cpu']:
        set_runtime(1234)
        m=NormalizedModel(replace_quantum(make_model('pihcqnn')),True).to(device).double()
        ref=copy.deepcopy(m)
        x=torch.rand(17,2,dtype=torch.float64,device=device,requires_grad=True)
        raw,f=fields(m,x);raw2,f2=af(ref,x)
        r,_=residuals(m,x);r2,_=ar(ref,x)
        (r.square().sum()+f.square().sum()).backward()
        (r2.square().sum()+f2.square().sum()).backward()
        errs=dict(device=device,raw_max=float((raw-raw2).abs().max().detach()),
                  uvp_max=float((f-f2).abs().max().detach()),
                  residual_max=float((r-r2).abs().max().detach()),
                  parameter_grad_max=max(float((p.grad-p2.grad).abs().max()) for p,p2 in zip(m.parameters(),ref.parameters())))
        assert errs['raw_max']<1e-12 and errs['uvp_max']<1e-11 and errs['residual_max']<1e-10 and errs['parameter_grad_max']<1e-8
        pts=np.load(ROOT/'protocol/points.npz');times=[]
        tensors={k:torch.tensor(pts[k],device=device,dtype=torch.float64) for k in ['colloc','wall','inlet','inlet_uv','outlet','data_xy','data_uv']}
        opt=torch.optim.Adam(m.parameters(),lr=4e-4)
        for i in range(4):
            t=time.perf_counter();opt.zero_grad(set_to_none=True)
            rr,_=residuals(m,tensors['colloc']);loss=rr.square().mean(0).sum()
            for k in ['wall','inlet']:
                _,f=fields(m,tensors[k]);target=tensors['inlet_uv'] if k=='inlet' else torch.zeros_like(f[:,:2])
                loss=loss+2*(f[:,:2]-target).square().mean(0).sum()
            loss=loss+2*m(tensors['outlet'])[:,1].square().mean()
            _,f=fields(m,tensors['data_xy']);loss=loss+(f[:,:2]-tensors['data_uv']).square().mean(0).sum()
            loss.backward();opt.step()
            if device=='cuda':torch.cuda.synchronize()
            times.append(time.perf_counter()-t)
        errs['float64_step_seconds']=times
        print(json.dumps(errs),flush=True);records.append(errs)
    (ROOT/'convergence/jet_backend_checks.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':verify_and_benchmark()
