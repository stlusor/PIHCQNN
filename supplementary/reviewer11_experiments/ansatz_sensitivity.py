from __future__ import annotations
import argparse, json, math, time, sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'reviewer1_experiments'))
from quantum_layer import apply_rot, entangle_permutation

class VariantQVC(nn.Module):
    def __init__(self, n_qubits: int, n_rep: int, ansatz: str):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_rep = n_rep
        self.ansatz = ansatz
        if ansatz in ('full', 'ryrz'):
            shape = (n_qubits, 3 if ansatz == 'full' else 2)
        elif ansatz == 'ry':
            shape = (n_qubits,)
        else:
            raise ValueError(ansatz)
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(shape)) for _ in range(n_rep + 1)])
        for p in self.weights:
            nn.init.uniform_(p, 0.0, 2.0 * math.pi)

    @staticmethod
    def _apply_full_layer(state, w, nq):
        for wire in range(nq):
            state = apply_rot(state, w[wire], wire, nq)
        return state.index_select(1, entangle_permutation(nq, state.device))

    @staticmethod
    def _apply_ry_layer(state, w, nq):
        for wire in range(nq):
            z = torch.zeros(2, dtype=w.dtype, device=w.device)
            angles = torch.cat([z[:1], w[wire].reshape(1), z[1:]])
            state = apply_rot(state, angles, wire, nq)
        return state.index_select(1, entangle_permutation(nq, state.device))

    @staticmethod
    def _apply_ryrz_layer(state, w, nq):
        for wire in range(nq):
            angles = torch.stack([w[wire, 0], w[wire, 1], torch.zeros((), dtype=w.dtype, device=w.device)])
            state = apply_rot(state, angles, wire, nq)
        return state.index_select(1, entangle_permutation(nq, state.device))

    def forward(self, inputs):
        state = torch.zeros((inputs.shape[0], 1 << self.n_qubits), dtype=torch.complex64, device=inputs.device)
        state[:, 0] = 1.0 + 0.0j
        for k in range(self.n_rep + 1):
            if self.ansatz == 'full':
                state = self._apply_full_layer(state, self.weights[k], self.n_qubits)
            elif self.ansatz == 'ry':
                state = self._apply_ry_layer(state, self.weights[k], self.n_qubits)
            else:
                state = self._apply_ryrz_layer(state, self.weights[k], self.n_qubits)
            if k < self.n_rep:
                x = inputs
                for wire in range(self.n_qubits):
                    c = torch.cos(x[:, wire] / 2).to(state.dtype)
                    s = torch.sin(x[:, wire] / 2).to(state.dtype)
                    low = 1 << wire
                    high = 1 << (self.n_qubits - wire - 1)
                    z = state.reshape(state.shape[0], high, 2, low)
                    z0, z1 = z[:, :, 0, :], z[:, :, 1, :]
                    cb = c.reshape(c.shape + (1,) * (z0.ndim - 1))
                    sb = s.reshape(s.shape + (1,) * (z0.ndim - 1))
                    y0, y1 = cb * z0 - sb * z1, sb * z0 + cb * z1
                    state = torch.stack((y0, y1), dim=2).reshape(state.shape[0], -1)
        probs = state.real ** 2 + state.imag ** 2
        base = torch.arange(1 << self.n_qubits, device=inputs.device)
        out = []
        for wire in range(self.n_qubits):
            bit = ((base >> wire) & 1) == 0
            p0 = probs[:, bit].sum(dim=1)
            p1 = probs[:, ~bit].sum(dim=1)
            out.append(p0 - p1)
        return torch.stack(out, dim=1)

class Hybrid(nn.Module):
    def __init__(self, width, nq, nrep, ansatz):
        super().__init__()
        self.enn_in = nn.Linear(1, width)
        self.enn_out = nn.Linear(width, nq)
        self.q = VariantQVC(nq, nrep, ansatz)
        self.ron_in = nn.Linear(nq, width)
        self.ron_out = nn.Linear(width, 1)
    def forward(self, x):
        shallow = torch.tanh(self.enn_out(torch.tanh(self.enn_in(x))))
        q = torch.tanh(self.q(shallow))
        return self.ron_out(torch.tanh(self.ron_in(q)))

class HardBC(nn.Module):
    def __init__(self, base): super().__init__(); self.base = base
    def forward(self, x): return x * (1.0 - x) * self.base(x)

def source(x): return 4.0 * math.pi ** 2 * torch.sin(2.0 * math.pi * x)
def exact(x): return torch.sin(2.0 * math.pi * x)

def rel_l2(model, device):
    x = torch.linspace(0, 1, 201, device=device).reshape(-1, 1)
    with torch.no_grad(): p = model(x); y = exact(x)
    return float(torch.linalg.norm(p - y) / torch.linalg.norm(y))

def train(ansatz, seed, steps, out):
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = HardBC(Hybrid(width=5, nq=5, nrep=1, ansatz=ansatz)).to(device)
    x = torch.linspace(0, 1, 102, device=device)[1:-1].reshape(-1, 1).requires_grad_()
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=5e-3)
    hist = []
    start = time.perf_counter()
    for step in range(steps):
        pred = net(x)
        ux = torch.autograd.grad(pred, x, torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True, retain_graph=True)[0]
        loss = torch.mean((uxx + source(x)) ** 2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 25 == 0 or step == steps - 1:
            pct = 100.0 * (step + 1) / steps
            bar = '#' * int(pct // 5) + '-' * (20 - int(pct // 5))
            print(f'PROGRESS ansatz={ansatz} seed={seed} [{bar}] {pct:5.1f}% step={step+1}/{steps}', flush=True)
        if step % 50 == 0 or step == steps - 1:
            hist.append({'step': step + 1, 'loss': float(loss.detach().cpu()), 'rel_l2': rel_l2(net, device)})
    result = {
        'case': 'elastostatic_forward_hard_bc',
        'ansatz': ansatz,
        'seed': seed,
        'steps': steps,
        'qubits': 5,
        'n_rep': 1,
        'width': 5,
        'params': sum(p.numel() for p in net.parameters()),
        'rel_l2': rel_l2(net, device),
        'elapsed_s': time.perf_counter() - start,
        'history': hist,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}, indent=2), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ansatz', choices=['full', 'ryrz', 'ry'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    train(a.ansatz, a.seed, a.steps, a.out)
