import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.run_store import atomic_json, identity, job_lock, run_job, verify_run


def complete(work):
    (work / 'model.zip').write_bytes(b'checkpoint')
    return {'objective': 1.25}


def test_run_identity_ignores_mapping_order():
    assert identity({'seed': 1, 'params': {'b': 2, 'a': 3}}) == identity({'params': {'a': 3, 'b': 2}, 'seed': 1})
    assert identity({'seed': 1}) != identity({'seed': 2})


def test_completed_job_reuses_only_matching_immutable_outputs(tmp_path):
    folder = tmp_path / 'run'
    assert run_job(folder, {'seed': 1}, complete, 'summary.json', ('model.zip',))['objective'] == 1.25
    assert run_job(folder, {'seed': 1}, lambda _: pytest.fail('must reuse'), 'summary.json')['objective'] == 1.25
    with pytest.raises(ValueError, match='different inputs'):
        run_job(folder, {'seed': 2}, complete, 'summary.json')
    (folder / 'model.zip').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='artifact changed'):
        run_job(folder, {'seed': 1}, complete, 'summary.json')


def test_failure_keeps_attempt_private_and_next_attempt_restarts(tmp_path):
    folder = tmp_path / 'run'
    def fail(work):
        (work / 'partial.zip').write_bytes(b'partial')
        raise RuntimeError('simulated interruption')
    with pytest.raises(RuntimeError):
        run_job(folder, {'seed': 1}, fail, 'summary.json')
    assert not folder.exists()
    assert len(list((tmp_path / '.attempts').glob('*/failure.json'))) == 1
    run_job(folder, {'seed': 1}, complete, 'summary.json')
    assert not (folder / 'partial.zip').exists()
    verify_run(folder)


def test_missing_output_and_unmanaged_folders_rejected(tmp_path):
    with pytest.raises(ValueError, match='Missing required'):
        run_job(tmp_path / 'run', {}, lambda _: {}, 'summary.json', ('model.zip',))
    (tmp_path / 'legacy').mkdir()
    with pytest.raises(ValueError, match='Unmanaged'):
        run_job(tmp_path / 'legacy', {}, complete, 'summary.json')


def test_completed_run_can_move_without_identity_change(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'moved'
    run_job(source, {'seed': 1}, complete, 'summary.json')
    before = verify_run(source)['run_id']
    shutil.move(source, target)
    assert verify_run(target)['run_id'] == before


def test_lock_excludes_other_process_and_recovers_after_process_exit(tmp_path):
    folder = tmp_path / 'run'
    code = 'from src.run_store import job_lock; import sys,os\nwith job_lock(sys.argv[1]):\n os._exit(0)'
    with job_lock(folder):
        attempt = subprocess.run([sys.executable, '-c', code, str(folder)], capture_output=True, timeout=20)
        assert attempt.returncode != 0 and b'locked' in attempt.stderr
    assert subprocess.run([sys.executable, '-c', code, str(folder)], timeout=20).returncode == 0
    with job_lock(folder):
        pass  # An abrupt child exit did not leave a stale ownership lock.


def test_failed_json_serialization_preserves_previous_file(tmp_path):
    p = tmp_path / 'record.json'
    atomic_json(p, {'state': 'complete'})
    with pytest.raises(ValueError):
        atomic_json(p, {'bad': float('nan')})
    assert json.loads(p.read_text()) == {'state': 'complete'}
