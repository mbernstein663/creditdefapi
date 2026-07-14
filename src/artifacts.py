from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
from typing import Any

import joblib

from .config import ARTIFACT_DIR

"""
Defines model saving protocol: saves, loads, versions, and fingerprints model artifacts.
Important for moving models between files and API integration.
"""
@dataclass
class ModelBundle:
    model: Any
    calibrator: Any
    feature_columns: list[str]
    model_type: str
    metadata: dict = field(default_factory=dict)
    required_input_schema: dict = field(default_factory=dict)

    def with_timestamp(self):
        self.metadata.setdefault("training_timestamp", datetime.now(timezone.utc).isoformat())
        return self


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.with_timestamp(), output)
    return output


def load_model_bundle(path: str | Path):
    return joblib.load(Path(path))


def bundle_path(name: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / name


def package_versions() -> dict[str, str]:
    names = ["fastapi", "pandas", "numpy", "scikit-learn", "matplotlib", "joblib"]
    versions = {}
    versions["python"] = platform.python_version()
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    stat = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(p.resolve()),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }
