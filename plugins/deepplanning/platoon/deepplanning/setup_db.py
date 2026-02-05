# Script to download and extract the DeepPlanning databases
# Make sure you set up HF_TOKEN in your environment variables

# Run from platoon/plugins/deepplanning/platoon/deepplanning


from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/DeepPlanning"

def download(repo_id: str, filename: str, cache_dir: str | None = None) -> Path:
    p = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=cache_dir,   # optional: set to control where files land
        local_dir=None,
    )
    return Path(p)

def extract_tar_gz(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest_dir)

def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

def main():
    # Choose where to put extracted databases in your repo
    repo_root = Path.cwd()
    shopping_db_root = repo_root / "shoppingplanning" / "database"
    travel_db_root = repo_root / "travelplanning" / "database"

    for level in (1, 2, 3):
        tar_name = f"database_level{level}.tar.gz"
        tar_path = download(REPO_ID, tar_name)
        extract_tar_gz(tar_path, shopping_db_root)

    for zip_name in ("database_zh.zip", "database_en.zip"):
        zip_path = download(REPO_ID, zip_name)
        extract_zip(zip_path, travel_db_root)

if __name__ == "__main__":
    main()