"""Immutable completed jobs, atomic publication, and OS-released process locks.

Recovery restarts incomplete computations from their declared seeds. It does
not claim to resume an optimizer or its RNG state from a partial checkpoint.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

from .artifacts import json_default


SCHEMA_VERSION = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,
                      default=json_default).encode('utf-8')


def identity(spec):
    return hashlib.sha256(canonical(spec)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(canonical(data) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def job_lock(output):
    """Nonblocking advisory lock; the OS releases it even after process death."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name('.' + output.name + '.lock')
    stream = lock.open('a+b')
    locked = False
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError(f'Job is locked by another process: {output}') from exc
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def verify_run(output, spec=None):
    output = Path(output)
    path = output / 'run_manifest.json'
    if not path.is_file():
        raise ValueError(f'Unmanaged or incomplete output; choose a new destination: {output}')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('state') != 'complete'
            or manifest.get('run_id') != identity(manifest.get('spec'))):
        raise ValueError('Invalid completed run manifest')
    if spec is not None and canonical(manifest['spec']) != canonical(spec):
        raise ValueError('Completed output has different inputs, code or dependencies; use a new destination')
    records = manifest['artifacts']
    if not records:
        raise ValueError('Completed run has no artifacts')
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()
              and p != path}
    if actual != set(records):
        raise ValueError('Completed artifact set changed')
    for rel, record in records.items():
        p = output / rel
        if not p.resolve().is_relative_to(output.resolve()):
            raise ValueError('Artifact escapes its run directory')
        if p.stat().st_size != record['bytes'] or file_hash(p) != record['sha256']:
            raise ValueError(f'Completed artifact changed: {rel}')
    return manifest


def run_job(output, spec, compute, result_name, required=()):
    """Compute privately, then rename a complete directory on the same volume."""
    output = Path(output)
    spec = json.loads(canonical(spec))
    if Path(result_name).name != result_name:
        raise ValueError('Result name must be a plain filename')
    with job_lock(output):
        if output.exists():
            verify_run(output, spec)
            return json.loads((output / result_name).read_text(encoding='utf-8'))
        attempts = output.parent / '.attempts'
        attempts.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=output.name + '-', dir=attempts))
        atomic_json(work / 'attempt.json', {'run_id': identity(spec), 'spec': spec,
                    'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()})
        try:
            # Fresh and reused jobs expose the same portable JSON types.
            result = json.loads(canonical(compute(work)))
            atomic_json(work / result_name, result)
            for name in required:
                if not (work / name).is_file():
                    raise ValueError(f'Missing required output: {name}')
            atomic_json(work / 'attempt.json', {'run_id': identity(spec), 'state': 'complete'})
            artifacts = {p.relative_to(work).as_posix(): {'sha256': file_hash(p), 'bytes': p.stat().st_size}
                         for p in work.rglob('*') if p.is_file()}
            manifest = {'schema_version': SCHEMA_VERSION, 'state': 'complete', 'run_id': identity(spec),
                        'spec': spec, 'artifacts': artifacts,
                        'completed_utc': datetime.now(timezone.utc).isoformat()}
            atomic_json(work / 'run_manifest.json', manifest)
            verify_run(work, spec)
            os.rename(work, output)
            return result
        except BaseException as exc:
            # Preserve failed artifacts for diagnosis. Do not publish them.
            if work.exists():
                atomic_json(work / 'failure.json', {'run_id': identity(spec), 'state': 'failed',
                            'exception_type': type(exc).__name__,
                            'recovery': 'Rerun the same job to start a fresh attempt; partial optimizer state is not resumed.'})
            raise
