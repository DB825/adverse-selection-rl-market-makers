import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_publication import clean_json
from scripts.verify_publication import verify


def test_sanitization_preserves_scientific_values():
    root = Path.cwd().resolve()
    value = {'checkpoint': str(root / 'results' / 'model.zip'), 'effect': .715, 'seeds': [11, 12]}
    cleaned = clean_json(value, root)
    assert cleaned['checkpoint'].replace('\\', '/') == 'results/model.zip'
    assert cleaned['effect'] == value['effect'] and cleaned['seeds'] == value['seeds']
    assert value['checkpoint'].startswith(str(root))


def test_publication_detects_modified_results(tmp_path):
    data = b'{"effect": 0.715}'
    (tmp_path / 'evidence.json').write_bytes(data)
    manifest = {'files': {'evidence.json': {'bytes': len(data), 'public_sha256': hashlib.sha256(data).hexdigest()}}}
    (tmp_path / 'publication_manifest.json').write_text(json.dumps(manifest))
    assert verify(tmp_path) == 1
    (tmp_path / 'evidence.json').write_bytes(b'{"effect": 0.999}')
    with pytest.raises(ValueError, match='mismatch'):
        verify(tmp_path)


def test_publication_rejects_path_escape(tmp_path):
    (tmp_path / 'publication_manifest.json').write_text(json.dumps({'files': {'../secret': {}}}))
    with pytest.raises(ValueError, match='Unsafe'):
        verify(tmp_path)
