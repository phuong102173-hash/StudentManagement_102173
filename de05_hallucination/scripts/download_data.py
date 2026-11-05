"""Download official text datasets with TLS verification and a local provenance manifest."""

import argparse
import hashlib
import json
import os
import re
import ssl
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "halueval-qa": ("RUCAIBox/HaluEval", "data/qa_data.json"),
    "halueval-dialogue": ("RUCAIBox/HaluEval", "data/dialogue_data.json"),
    "halueval-summarization": ("RUCAIBox/HaluEval", "data/summarization_data.json"),
    "halueval-general": ("RUCAIBox/HaluEval", "data/general_data.json"),
    "truthfulqa": ("sylinrl/TruthfulQA", "TruthfulQA.csv"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", choices=list(SOURCES))
    parser.add_argument("--output", type=Path, default=Path("data/downloads"))
    parser.add_argument(
        "--revision", default="main", help="Prefer immutable 40-character commit for reproduction"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.revision):
        parser.error("Invalid revision")
    args.output.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=os.getenv("SSL_CERT_FILE") or None)
    for dataset in args.datasets:
        repository, relative = SOURCES[dataset]
        target = args.output / f"{dataset}{Path(relative).suffix}"
        manifest = target.with_suffix(target.suffix + ".source.json")
        if target.exists() or manifest.exists():
            parser.error(f"{target} already exists; preserve the old dataset or choose a new output path")
        url = f"https://raw.githubusercontent.com/{repository}/{args.revision}/{relative}"
        digest = hashlib.sha256()
        temp_name = None
        try:
            with urllib.request.urlopen(url, context=context, timeout=90) as response:
                with tempfile.NamedTemporaryFile(dir=args.output, delete=False) as stream:
                    temp_name = stream.name
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 512 * 1024 * 1024:
                            raise ValueError("Unexpectedly large dataset download")
                        digest.update(chunk)
                        stream.write(chunk)
            Path(temp_name).replace(target)
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "url": url,
                        "revision": args.revision,
                        "sha256": digest.hexdigest(),
                        "bytes": total,
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "note": "Checksum records this download; it is not a publisher-signed digest.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Downloaded {dataset}: {total} bytes, sha256={digest.hexdigest()}")
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
