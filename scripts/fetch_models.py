"""Download (optional) and unpack the model-B checkpoints, verified against artifacts/model_manifest.json.

  python scripts/fetch_models.py --zip models_2024320020.zip           # file downloaded by hand
  python scripts/fetch_models.py --url <direct or Google Drive link>   # needs `pip install gdown` for Drive

Only the paths listed in the manifest are extracted, and every file must match its SHA-256.
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256


def download(url, target):
    if "drive.google.com" in url:
        subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", url, "-O", str(target)], check=True)
    else:
        with urllib.request.urlopen(url) as response, target.open("wb") as out:
            shutil.copyfileobj(response, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/model_manifest.json")
    parser.add_argument("--zip", type=Path, help="Local archive (default: models_<id>.zip in the project root)")
    parser.add_argument("--url", help="Download the archive from this link first")
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    archive = args.zip or ROOT / manifest["archive"]
    if args.url and not archive.exists():
        print(f"Downloading to {archive} ...", flush=True)
        download(args.url, archive)
    if not archive.exists():
        raise SystemExit(f"{archive} not found: download it from the link in docs/reproduction.md")
    if sha256(archive) != manifest["archive_sha256"]:
        raise SystemExit(f"{archive} does not match the manifest SHA-256")
    with zipfile.ZipFile(archive) as z:
        names = set(z.namelist())
        if names != set(manifest["files"]):
            raise SystemExit("The archive contents differ from the manifest")
        for name in sorted(names):  # only manifest paths, so nothing is written outside outputs/
            z.extract(name, ROOT)
    for name, info in manifest["files"].items():
        if sha256(ROOT / name) != info["sha256"]:
            raise SystemExit(f"{name} does not match the manifest SHA-256")
    print(f"Verified and unpacked {len(names)} files: {', '.join(manifest['checkpoints'])}", flush=True)


if __name__ == "__main__":
    main()
