from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import math
import time
import torch
import torch.nn as nn

from quantum_layer import ManualQVC


class SineAct(nn.Module):
    def __init__(self, w0: float = 5.0):
        super().__init__(); self.w0 = float(w0)
    def forward(self, x): return torch.sin(self.w0 * x)


class AdaptiveSineAct(nn.Module):
    def __init__(self, init: float = 0.5):
        super().__init__(); self.a = nn.Parameter(torch.tensor(float(init)))
    def forward(self, x): return torch.sin(self.a * x)


def mlp(in_dim: int, width: int, depth: int, out_dim: int = 1, activation: str = 'tanh', w0: float = 5.0):
    layers=[]; d=in_dim
    for _ in range(depth):
        layers.append(nn.Linear(d,width))
        if activation == 'tanh': layers.append(nn.Tanh())
        elif activation == 'sin': layers.append(SineAct(w0))
        elif activation == 'adaptive_sin': layers.append(AdaptiveSineAct(w0))
        else: raise ValueError(activation)
        d=width
    layers.append(nn.Linear(d,out_dim))
    return nn.Sequential(*layers)


class FourierFeatureMLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int, n_features: int = 32, sigma: float = 4.0):
        super().__init__()
        self.register_buffer('B', torch.randn(in_dim, n_features) * sigma)
        self.net = mlp(2*n_features, width, depth, 1, 'tanh')
    def forward(self, x):
        z = 2*math.pi*x @ self.B
        return self.net(torch.cat([torch.sin(z), torch.cos(z)], dim=-1))


def siren_mlp(in_dim: int, width: int, depth: int, out_dim: int = 1, w0: float = 5.0):
    model = mlp(in_dim, width, depth, out_dim, 'sin', w0)
    linears = [m for m in model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        first = linears[0]
        first.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
        first.bias.uniform_(-1.0, 1.0)
        for lin in linears[1:]:
            bound = math.sqrt(6.0 / lin.in_features) / w0
            lin.weight.uniform_(-bound, bound)
    return model


@dataclass
class PIHConfig:
    in_dim: int
    hidden_width: int
    n_qubits: int
    n_rep: int
    out_dim: int = 1
    concat_shallow: bool = False


class PIHCQNN(nn.Module):
    def __init__(self, cfg: PIHConfig):
        super().__init__(); self.cfg=cfg
        self.enn_in = nn.Linear(cfg.in_dim,cfg.hidden_width)
        self.enn_out = nn.Linear(cfg.hidden_width,cfg.n_qubits)
        self.q = ManualQVC(cfg.n_qubits,cfg.n_rep)
        ron_in = cfg.n_qubits*(2 if cfg.concat_shallow else 1)
        self.ron_in = nn.Linear(ron_in,cfg.hidden_width)
        self.ron_out = nn.Linear(cfg.hidden_width,cfg.out_dim)
    def forward(self,x):
        shallow=torch.tanh(self.enn_out(torch.tanh(self.enn_in(x))))
        q=torch.tanh(self.q(shallow))
        z=torch.cat([shallow,q],dim=-1) if self.cfg.concat_shallow else q
        return self.ron_out(torch.tanh(self.ron_in(z)))


class ClassicalMLP(nn.Module):
    def __init__(self,in_dim,width,depth,out_dim=1,kind='tanh'):
        super().__init__(); self.kind=kind; self.net=mlp(in_dim,width,depth,out_dim,'tanh')
    def forward(self,x): return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def derivative(y, x, order=1):
    out=y
    for k in range(order):
        out=torch.autograd.grad(out,x,torch.ones_like(out),create_graph=True,retain_graph=True,allow_unused=True)[0]
        if out is None: raise RuntimeError('derivative is disconnected')
    return out


@dataclass
class TrainResult:
    model: str
    case: str
    seed: int
    steps: int
    params: int
    metric: float
    metric_name: str
    elapsed_s: float
    history: list
    extra: dict


def train_supervised(model, x, y, *, steps, lr, metric_fn, model_name, case, seed, eval_every=250, device='cuda'):
    model=model.to(device); x=x.to(device); y=y.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    hist=[]; start=time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred=model(x)
        loss=torch.mean((pred-y)**2)
        loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps-1:
            with torch.no_grad(): metric=float(metric_fn(model))
            hist.append({'step':step+1,'loss':float(loss.detach().cpu()),'metric':metric})
    with torch.no_grad(): metric=float(metric_fn(model))
    return TrainResult(model_name,case,seed,steps,count_parameters(model),metric,'relative_l2',time.perf_counter()-start,hist,{})
