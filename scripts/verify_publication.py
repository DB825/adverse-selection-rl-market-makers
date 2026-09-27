"""Verify every published evidence file against its publication checksum."""
import hashlib
import json
from pathlib import Path


def verify(folder=Path('published-results')):
    folder = Path(folder)
    manifest = json.loads((folder / 'publication_manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('files'):
        raise ValueError('Empty publication manifest')
    for rel, record in manifest['files'].items():
        path = (folder / rel).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError(f'Unsafe artifact path: {rel}')
        data = path.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['public_sha256']:
            raise ValueError(f'Published artifact mismatch: {rel}')
    return len(manifest['files'])


if __name__ == '__main__':
    print(f'Verified {verify()} published evidence files')
