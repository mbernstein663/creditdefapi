from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.artifacts import file_fingerprint
from src.config import ACCEPTED_CSV, DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE
from src.preprocessing import feature_columns, prepare_accepted_loans, split_chronological, split_manifest, split_summary


@dataclass(slots=True)
class PreprocessedAcceptedLoans:
    source_fingerprint: dict
    target_summary: dict
    accepted: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    manifest: dict
    split_summary: list[dict]
    selected_features: list[str]


def _read_accepted(path, selected_features, sample=None):
    needed = set(selected_features) | {"id", "loan_status", "issue_d", "term"}
    return pd.read_csv(path, usecols=lambda col: col in needed, low_memory=False, nrows=sample)


def preprocess_accepted_loans(csv_path=ACCEPTED_CSV, sample=None, selected_features: list[str] | None = None) -> PreprocessedAcceptedLoans:
    selected_features = list(selected_features or feature_columns())
    source = _read_accepted(csv_path, selected_features, sample=sample)
    accepted, target_summary = prepare_accepted_loans(source, return_summary=True)
    splits = split_chronological(accepted)
    return PreprocessedAcceptedLoans(
        source_fingerprint=file_fingerprint(csv_path),
        target_summary=target_summary,
        accepted=accepted,
        splits=splits,
        manifest=split_manifest(splits),
        split_summary=split_summary(splits),
        selected_features=selected_features,
    )


def save_preprocessed_accepted_loans(bundle: PreprocessedAcceptedLoans, path: str | Path = DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_preprocessed_accepted_loans(path: str | Path = DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE) -> PreprocessedAcceptedLoans:
    return joblib.load(Path(path))


def main():
    parser = argparse.ArgumentParser(description="Prepare accepted LendingClub loans for default-risk training.")
    parser.add_argument("--csv", default=ACCEPTED_CSV)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output", default=DEFAULT_PREPROCESSED_ACCEPTED_BUNDLE)
    args = parser.parse_args()
    result = preprocess_accepted_loans(args.csv, sample=args.sample)
    saved = save_preprocessed_accepted_loans(result, args.output)
    print(
        json.dumps(
            {
                "output": str(saved),
                "source_fingerprint": result.source_fingerprint,
                "target_summary": result.target_summary,
                "split_manifest": result.manifest,
                "split_summary": result.split_summary,
                "selected_features": result.selected_features,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
