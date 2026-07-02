from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib

from .config import ARTIFACT_DIR


@dataclass
class ModelBundle:
    model: Any
    calibrator: Any
    feature_columns: list[str]
    model_type: str
    metadata: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
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
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    stat = p.stat()
    return {
        "path": str(p.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
