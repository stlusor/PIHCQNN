"""List or run the final supplementary experiment recipes."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def load_jobs():
    jobs = []
    for path in sorted(ROOT.glob('jobs_*.json')):
        jobs.extend(json.loads(path.read_text(encoding='utf-8'))['jobs'])
    labels = [job['label'] for job in jobs]
    if len(labels) != len(set(labels)):
        raise ValueError('Experiment labels must be unique.')
    for job in jobs:
        if not (ROOT / job['script']).is_file():
            raise FileNotFoundError(job['script'])
    return jobs

def make_command(job, output_root=None):
    output = ROOT / job['output']
    if output_root is not None:
        relative = Path(job['output']).relative_to('rerun_results')
        output = output_root / relative
    output = output.resolve()
    is_directory = job.get('output_kind') == 'directory'
    replacements = {'output': str(output), 'output_dir': str(output if is_directory else output.parent),
                    'root': str(ROOT)}
    command = [sys.executable, '-X', 'utf8', str(ROOT / job['script'])]
    command += [str(arg).format(**replacements) for arg in job['args']]
    return command, output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List suite names and experiment counts.')
    parser.add_argument('--suite', nargs='+', help='Select one or more suites.')
    parser.add_argument('--label', nargs='+', help='Select one or more individual experiment labels.')
    parser.add_argument('--seed', type=int, help='Select a stored seed.')
    parser.add_argument('--output-root', type=Path, help='Write reruns under this directory.')
    parser.add_argument('--execute', action='store_true', help='Run the displayed full experiments.')
    args = parser.parse_args()
    jobs = load_jobs()
    if args.list or (not args.suite and not args.label):
        print(json.dumps({'suites': dict(sorted(Counter(j['suite'] for j in jobs).items())),
                          'total_jobs': len(jobs), 'flow_entry': 'run_flow_final.py'}, indent=2))
        if args.execute and not args.suite and not args.label:
            parser.error('Select a suite or label to run.')
        if not args.suite and not args.label:
            return
    if args.suite:
        unknown = set(args.suite) - {j['suite'] for j in jobs}
        if unknown:
            parser.error('Unknown suites: ' + ', '.join(sorted(unknown)))
    if args.label:
        unknown = set(args.label) - {j['label'] for j in jobs}
        if unknown:
            parser.error('Unknown labels: ' + ', '.join(sorted(unknown)))
    selected = [j for j in jobs if (not args.suite or j['suite'] in args.suite)
                and (not args.label or j['label'] in args.label)
                and (args.seed is None or j.get('seed') == args.seed)]
    if not selected:
        parser.error('The selection contains no experiments.')
    outroot = args.output_root.resolve() if args.output_root else None
    plans = [(job, *make_command(job, outroot)) for job in selected]
    for job, command, output in plans:
        print(json.dumps({'label': job['label'], 'command': command, 'output': str(output)}), flush=True)
    if args.execute:
        archive_roots = [ROOT / 'reviewer1_experiments/results', ROOT / 'reviewer2_experiments/results',
                         ROOT / 'reviewer11_experiments/results',
                         ROOT / 'reviewer_supplement_20261007/fixed_weights/runs',
                         ROOT / 'flow_reproduction_20261007/convergence',
                         ROOT / 'reviewer_supplement_20261007/flow_controls']
        for _, _, output in plans:
            if any(output == p.resolve() or p.resolve() in output.parents for p in archive_roots):
                parser.error('Choose an output directory outside the archived result directories.')
            if output.exists():
                parser.error(f'Choose a new rerun directory; {output} already exists.')
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        for job, command, output in plans:
            (output if job.get('output_kind') == 'directory' else output.parent).mkdir(parents=True, exist_ok=True)
            (output.parent / (output.stem + '_command.json')).write_text(json.dumps(command, indent=2), encoding='utf-8')
            subprocess.run(command, cwd=ROOT / job.get('cwd', '.'), env=env, check=True)

if __name__ == '__main__':
    main()
