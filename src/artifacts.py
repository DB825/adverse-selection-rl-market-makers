"""Small explicit, portable artifact helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    return {str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src", "configs", "scripts", "tests")
            for p in sorted(Path(folder).glob("**/*"))
            if p.is_file() and p.suffix in {".py", ".yaml"}}
