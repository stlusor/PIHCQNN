"""Check the release manifest, file hashes and author Python syntax."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text(encoding='utf-8'))
    checked = 0
    for entry in manifest['files']:
        path = ROOT / entry['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError(f'File integrity mismatch: {entry["path"]}')
        checked += 1
    parsed = 0
    # The upstream TensorFlow-1 source is preserved as an attributed reference.
    for directory in [ROOT / 'original_code/src', ROOT / 'supplementary']:
        for path in sorted(directory.rglob('*.py')):
            if 'source' in path.parts:
                continue
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            parsed += 1
    print(json.dumps({'verified_files': checked, 'python_syntax_files': parsed,
                      'manifest_format': manifest['format_version']}, indent=2))

if __name__ == '__main__':
    main()
