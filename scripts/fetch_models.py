"""Download (optional) and unpack the model-B checkpoints, verified against artifacts/model_manifest.json.

  python scripts/fetch_models.py --zip models_2024320020.zip           # file downloaded by hand
  python scripts/fetch_models.py --url <direct or Google Drive link>   # plain Python; gdown only as fallback

Only the paths listed in the manifest are extracted, and every file must match its SHA-256.
"""

import argparse
import re
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from review_retrieval.common import read_json, sha256


def drive_id(url):
    """File ID of a Google Drive link (.../file/d/<id>/... or ...?id=<id>), else None."""
    match = re.search(r"/file/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    return match.group(1) if match and "drive" in url else None


def fetch(url, target):
    """Stream url to target; returns False (and writes nothing) if the server sends an HTML page."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            return False
        total, done = int(response.headers.get("Content-Length") or 0), 0
        with target.open("wb") as out:
            while chunk := response.read(1 << 22):
                out.write(chunk)
                done += len(chunk)
                if done % (1 << 28) < (1 << 22) or done == total:
                    print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)
    return True


def download(url, target):
    file_id = drive_id(url)
    if file_id is None:
        if not fetch(url, target):
            raise SystemExit(f"{url} returned a web page, not the archive")
        return
    # Public Drive file: direct download with plain Python first, gdown (any version) as a fallback.
    direct = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    if not fetch(direct, target):
        subprocess.run([sys.executable, "-m", "gdown", file_id, "-O", str(target)], check=True)


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
        partial = archive.with_name(archive.name + ".part")  # renamed only when complete
        download(args.url, partial)
        partial.replace(archive)
    if not archive.exists():
        raise SystemExit(f"{archive} not found: download it from the link in docs/reproduction.md")
    if sha256(archive) != manifest["archive_sha256"]:
        raise SystemExit(f"{archive} does not match the manifest SHA-256 (incomplete or wrong file): delete it and run again")
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
