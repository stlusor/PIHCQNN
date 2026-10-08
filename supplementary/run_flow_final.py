"""Run the staged flow comparison with the released fixed observation set."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FLOW = ROOT / 'flow_reproduction_20261007'

def commands(model, output):
    base = [sys.executable, '-X', 'utf8', str(FLOW / 'convergence_train.py'),
            '--model', model, '--optimizer', 'lbfgs', '--lr', '1', '--seed', '1234',
            '--data-fields', 'uv', '--data-scale', 'rms', '--normalize',
            '--hard-geometry', '--float64', '--boundary-weight', '0',
            '--device', 'cuda' if model == 'pihcqnn' else 'cpu']
    if model == 'pihcqnn':
        base += ['--dense-quantum']
    initial = output / model / 'data_initialization'
    physics = output / model / 'physics_training'
    phases = [('data_initialization', base + ['--steps', '1500', '--physics-weight', '0',
        '--data-weight', '1', '--eval-every', '500', '--out', str(initial)])]
    if model != 'pihcqnn':
        phases.append(('physics_training', base + ['--steps', '2600', '--physics-weight', '1',
            '--data-weight', '100', '--eval-every', '500', '--resume', str(initial / 'checkpoint.pt'),
            '--reset-optimizer', '--out', str(physics)]))
    else:
        eager = output / model / 'physics_training_eager'
        phases.append(('physics_training_eager', base + ['--steps', '600', '--physics-weight', '1',
            '--data-weight', '100', '--eval-every', '100', '--resume', str(initial / 'checkpoint.pt'),
            '--reset-optimizer', '--out', str(eager)]))
        # Keep the L-BFGS history when continuing the same physics stage.
        phases.append(('physics_training', [sys.executable, '-X', 'utf8',
            str(FLOW / 'convergence_lbfgs_graph.py'), '--steps', '2000', '--seed', '1234',
            '--data-weight', '100', '--normalize', '--hard-geometry', '--data-fields', 'uv',
            '--data-scale', 'rms', '--boundary-weight', '0', '--eval-every', '250',
            '--resume', str(eager / 'checkpoint.pt'), '--out', str(physics)]))
    return phases

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['pinn5', 'pinn6', 'pihcqnn', 'all'], default='all')
    parser.add_argument('--out', type=Path, default=ROOT / 'rerun_results' / 'flow')
    parser.add_argument('--execute', action='store_true', help='Start the selected full training runs.')
    args = parser.parse_args()
    output = args.out.resolve()
    archived = [FLOW / 'convergence', ROOT / 'reviewer_supplement_20261007' / 'flow_controls']
    if any(output == p.resolve() or p.resolve() in output.parents for p in archived):
        parser.error('Choose a rerun output directory outside the archived result directories.')
    models = ['pinn5', 'pinn6', 'pihcqnn'] if args.model == 'all' else [args.model]
    plans = [(model, phase, cmd) for model in models for phase, cmd in commands(model, output)]
    for model, phase, cmd in plans:
        print(json.dumps({'model': model, 'phase': phase, 'command': cmd}, ensure_ascii=False), flush=True)
    if args.execute:
        if 'pihcqnn' in models:
            import torch
            if not torch.cuda.is_available():
                parser.error('The PIHCQNN continuation uses CUDA; select a classical model or a CUDA environment.')
        for model, phase, cmd in plans:
            target = Path(cmd[cmd.index('--out') + 1])
            if target.exists() and any(target.iterdir()):
                parser.error(f'Choose a new output directory; {target} already contains files.')
        output.mkdir(parents=True, exist_ok=True)
        (output / 'commands.json').write_text(json.dumps(plans, indent=2, ensure_ascii=False), encoding='utf-8')
        for _, _, cmd in plans:
            subprocess.run(cmd, cwd=FLOW, check=True)

if __name__ == '__main__':
    main()
