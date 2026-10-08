"""Algebraically identical VQC using dense shared rotation matrices.

This changes the simulator, not the circuit or its trainable parameters.
The original backend remains untouched and is used as the numerical reference.
"""
import torch
from torch import nn
from flow_core import make_model, fields, residuals, ROOT, set_runtime

def rot_matrix(w, cdtype):
    phi,theta,omega=w.unbind()
    c=torch.cos(theta/2);s=torch.sin(theta/2)
    a=c*torch.exp(-.5j*(phi+omega))
    b=-s*torch.exp(.5j*(phi-omega))
    c2=s*torch.exp(-.5j*(phi-omega))
    d=c*torch.exp(.5j*(phi+omega))
    return torch.stack([a,b,c2,d]).reshape(2,2).to(cdtype)

class DenseQVC(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.n_qubits=original.n_qubits;self.n_rep=original.n_rep
        self.weights=original.weights
        from quantum_layer import entangle_permutation
        self.register_buffer('permutation',entangle_permutation(self.n_qubits,torch.device('cpu')))
        base=torch.arange(1<<self.n_qubits)
        self.register_buffer('signs',torch.stack([1-2*((base>>i)&1) for i in range(self.n_qubits)],1))

    def forward(self, inputs):
        from quantum_layer import apply_angle_y
        cdtype=torch.complex128 if inputs.dtype==torch.float64 else torch.complex64
        mats=[]
        for w in self.weights:
            matrix=rot_matrix(w[-1],cdtype)
            for wi in reversed(w[:-1]):matrix=torch.kron(matrix,rot_matrix(wi,cdtype))
            mats.append(matrix)
        state=torch.zeros((1,1<<self.n_qubits),dtype=cdtype,device=inputs.device)
        state[:,0]=1
        state=(state@mats[0].T).index_select(1,self.permutation)
        state=state.expand(len(inputs),-1)
        for k in range(self.n_rep):
            state=apply_angle_y(state,inputs,self.n_qubits)
            state=(state@mats[k+1].T).index_select(1,self.permutation)
        probs=state.real.square()+state.imag.square()
        return probs@self.signs.to(dtype=inputs.dtype)

def replace_quantum(model):
    model.q=DenseQVC(model.q)
    return model

def verify_and_benchmark():
    import copy,json,time,numpy as np
    from benchmark_convergence import objective
    records=[]
    for device in ['cuda','cpu']:
        set_runtime(1234)
        reference=make_model('pihcqnn').to(device)
        fast=replace_quantum(copy.deepcopy(reference)).to(device)
        x=torch.rand(19,2,device=device,requires_grad=True)
        x2=x.detach().clone().requires_grad_(True)
        raw,uvp=fields(reference,x);raw2,uvp2=fields(fast,x2)
        r,_=residuals(reference,x);r2,_=residuals(fast,x2)
        loss=r.square().sum()+uvp.square().sum()
        loss2=r2.square().sum()+uvp2.square().sum()
        loss.backward();loss2.backward()
        grad_diff=max(float((p.grad-p2.grad).abs().max()) for p,p2 in zip(reference.parameters(),fast.parameters()))
        checks=dict(device=device,raw_max=float((raw-raw2).abs().max().detach()),
                    uvp_max=float((uvp-uvp2).abs().max().detach()),
                    residual_max=float((r-r2).abs().max().detach()),parameter_grad_max=grad_diff)
        assert checks['raw_max']<2e-6 and checks['uvp_max']<2e-6 and checks['residual_max']<2e-6 and grad_diff<2e-4
        pts=np.load(ROOT/'protocol/points.npz')
        opt=torch.optim.Adam(fast.parameters(),lr=4e-4)
        timings=[]
        for i in range(5):
            start=time.perf_counter();opt.zero_grad(set_to_none=True)
            loss=objective(fast,pts,device);loss.backward();opt.step()
            if device=='cuda':torch.cuda.synchronize()
            timings.append(time.perf_counter()-start)
        checks.update(step_seconds=timings)
        print(json.dumps(checks),flush=True);records.append(checks)
    path=ROOT/'convergence'
    path.mkdir(exist_ok=True)
    (path/'dense_backend_checks.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':verify_and_benchmark()
