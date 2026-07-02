from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.artifacts import load_model_bundle
from src.config import DEFAULT_ACCEPTED_BUNDLE
from src.scorer import score_frame


def score_csv(input_csv, output_csv, bundle_path=DEFAULT_ACCEPTED_BUNDLE, chunksize=50_000):
    bundle = load_model_bundle(bundle_path)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for chunk in pd.read_csv(input_csv, chunksize=chunksize):
        scored = score_frame(chunk, bundle)
        scored.to_csv(output, mode="w" if first else "a", index=False, header=first)
        first = False
    return output


def main():
    parser = argparse.ArgumentParser(description="Batch score loans with a saved bundle.")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--bundle", default=DEFAULT_ACCEPTED_BUNDLE)
    parser.add_argument("--chunksize", type=int, default=50_000)
    args = parser.parse_args()
    print(score_csv(args.input_csv, args.output_csv, args.bundle, args.chunksize))


if __name__ == "__main__":
    main()
