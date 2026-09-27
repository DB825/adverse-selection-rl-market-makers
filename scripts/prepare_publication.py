"""Create a portable text-results snapshot; never modify original run records."""
from pathlib import Path
import hashlib
import json
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clean_string(value, root):
    for prefix in (str(root), root.as_posix()):
        value = value.replace(prefix + '\\', '').replace(prefix + '/', '').replace(prefix, '.')
    # Temp/test/runtime paths outside the project are machine-specific too.
    return re.sub(r'[A-Za-z]:[\\/]Users[\\/][^\\/\s"<>]+', '<USER_HOME>', value)


def clean_json(value, root):
    if isinstance(value, str):
        return clean_string(value, root)
    if isinstance(value, list):
        return [clean_json(v, root) for v in value]
    if isinstance(value, dict):
        return {clean_string(k, root): clean_json(v, root) for k, v in value.items()}
    return value


def main():
    root = Path.cwd().resolve()
    source, output = root / 'results', root / 'published-results'
    if not source.is_dir():
        raise FileNotFoundError('No original results archive; run experiments first')
    if (output / 'publication_manifest.json').exists():
        raise FileExistsError('Publication snapshot already exists; preserve it and use an explicit new release')
    entries = {}
    omitted = {'repository_inventory.json', 'entropy_delivery_manifest.json'}
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if not path.is_file() or path.suffix not in {'.json', '.csv', '.xml'} or path.name in omitted:
            continue
        if 'deprecated_seed_overlap' in rel.parts or 'tensorboard' in rel.parts:
            continue
        original = path.read_bytes()
        if path.suffix == '.json':
            data = json.loads(original)
            sanitized = clean_json(data, root)
            public = original if data == sanitized else (json.dumps(sanitized, indent=2, allow_nan=False) + '\n').encode('utf-8')
        else:
            text = original.decode('utf-8')
            sanitized = clean_string(text, root)
            public = original if text == sanitized else sanitized.encode('utf-8')
        target = output / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(public)
        entries[rel.as_posix()] = {'original_sha256': digest(original), 'public_sha256': digest(public),
                                  'bytes': len(public), 'local_paths_sanitized': original != public}
    manifest = {'schema_version': 1, 'files': entries,
                'scope': 'Text evidence snapshot. Models, activations, tensorboard files, superseded overlapping-seed runs and local inventory/delivery records are excluded.',
                'provenance': 'Original result files remain untouched and untracked under results/. Historical experiment hashes identify original bytes. This manifest maps those hashes to public bytes after path sanitization; it does not claim omitted binaries are present.'}
    (output / 'publication_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(entries)} files; {sum(v["local_paths_sanitized"] for v in entries.values())} sanitized; {sum(v["bytes"] for v in entries.values())/1e6:.1f} MB')


if __name__ == '__main__':
    main()
