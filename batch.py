from __future__ import annotations

import argparse
from pathlib import Path

import httpx

DEFAULT_API_URL = "http://127.0.0.1:8000"


def score_csv(input_csv, output_csv, api_url=DEFAULT_API_URL):
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_csv).open("rb") as handle, httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{api_url.rstrip('/')}/score-batch",
            files={"file": (Path(input_csv).name, handle, "text/csv")},
        )
    response.raise_for_status()
    output.write_bytes(response.content)
    return output


def main():
    parser = argparse.ArgumentParser(description="Batch score loans by uploading a CSV to the FastAPI batch endpoint.")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    args = parser.parse_args()
    print(score_csv(args.input_csv, args.output_csv, args.api_url))


if __name__ == "__main__":
    main()
