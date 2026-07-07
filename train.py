from __future__ import annotations

from src import train as _train
from src.config import ARTIFACT_DIR, DEFAULT_ACCEPTED_BUNDLE, DEFAULT_FRONTEND_BUNDLE, DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE, REPORT_DIR

def _sync_train_state() -> None:
    _train.ARTIFACT_DIR = ARTIFACT_DIR
    _train.DEFAULT_ACCEPTED_BUNDLE = DEFAULT_ACCEPTED_BUNDLE
    _train.DEFAULT_FRONTEND_BUNDLE = DEFAULT_FRONTEND_BUNDLE
    _train.DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE = DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE
    _train.REPORT_DIR = REPORT_DIR


def _paths(*args, **kwargs):
    _sync_train_state()
    return _train._paths(*args, **kwargs)


def _frontend_output_path(*args, **kwargs):
    _sync_train_state()
    return _train._frontend_output_path(*args, **kwargs)


def train_accepted_model(*args, **kwargs):
    _sync_train_state()
    return _train.train_accepted_model(*args, **kwargs)


def main(*args, **kwargs):
    _sync_train_state()
    return _train.main(*args, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
