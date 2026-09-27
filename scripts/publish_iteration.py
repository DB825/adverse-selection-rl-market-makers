"""Append one immutable, sanitized public evidence release after complete audits."""
import hashlib
import json
from pathlib import Path

from scripts.prepare_publication import clean_json, clean_string
from scripts.verify_publication import verify
from scripts.make_followup_manifest import check_junit
from src.run_store import verify_run


def main():
    root = Path.cwd().resolve()
    output = Path('published-results/iteration-v1')
    if output.exists():
        raise FileExistsError('Version already exists; preserve it and use an explicit new release')
    verify()
    for arm in ('control','treatment'):
        for seed in range(21,26):
            verify_run(Path('results/replication_v1')/arm/f'seed_{seed}')
            verify_run(Path('results/replication_v1/evaluation')/arm/f'seed_{seed}')
    for seed in range(11,16):
        verify_run(Path('results/nonlinear_v1')/f'seed_{seed}')
    for name in ('replication_summary.json','nonlinear_summary.json'):
        if not (Path('results/iteration_v1')/name).is_file():
            raise ValueError('Both audited analyses must exist')
    replication = json.loads(Path('results/iteration_v1/replication_summary.json').read_text())
    for path, expected in replication['run_manifest_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Replication analysis is stale')
    nonlinear = json.loads(Path('results/iteration_v1/nonlinear_summary.json').read_text())
    for record in nonlinear['audits']:
        path = Path('results/nonlinear_v1')/f'seed_{record["seed"]}'/'run_manifest.json'
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['manifest_sha256']:
            raise ValueError('Nonlinear analysis is stale')
    tests = check_junit('results/iteration_tests.xml')
    report = Path('results/iteration_tests.xml').read_text(encoding='utf-8')
    required = ('test_managed_runs','test_run_store','test_nonlinear_probe','test_iteration_analysis')
    if tests['passed'] < 157 or tests['skipped'] or not all(name in report for name in required):
        raise ValueError('Missing complete iteration test suite')
    files = []
    for label, source in [('replication',Path('results/replication_v1')),
                          ('nonlinear',Path('results/nonlinear_v1')),('analysis',Path('results/iteration_v1'))]:
        for path in sorted(source.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json','.csv','.xml'}:
                continue
            if any(part.startswith('.') for part in path.relative_to(source).parts):
                continue
            files.append((Path(label)/path.relative_to(source),path))
    files.append((Path('tests.xml'),Path('results/iteration_tests.xml')))
    records = {}
    for relative,path in files:
        original = path.read_bytes()
        text = original.decode('utf-8')
        if path.suffix == '.json':
            value = json.loads(text)
            cleaned = clean_json(value,root)
            public = original if value == cleaned else (json.dumps(cleaned,indent=2,allow_nan=False)+'\n').encode('utf-8')
        else:
            clean = clean_string(text,root)
            public = original if clean == text else clean.encode('utf-8')
        target = output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(public)
        records[relative.as_posix()] = {'original_sha256':hashlib.sha256(original).hexdigest(),
                'public_sha256':hashlib.sha256(public).hexdigest(),'bytes':len(public),
                'local_paths_sanitized':public != original}
    manifest = {'schema_version':1,'files':records,'tests':tests,
                'scope':'Completed replication and nonlinear-control text evidence; models, decoder weights, arrays, event logs and failed attempts excluded.',
                'provenance':'Run manifests describe full original artifacts including omitted binaries. This publication manifest verifies the public text copies; original files remain unchanged locally.'}
    (output/'publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Published {verify(output)} verified evidence files in {output}')


if __name__ == '__main__':
    main()
