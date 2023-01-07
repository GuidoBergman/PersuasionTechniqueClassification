import requests
import os
import shutil
from tqdm import tqdm
from pathlib import Path
from functools import partial
from zipfile import ZipFile

DATA_FOLDER = Path(__file__).parent.resolve()
SRC_DATA_FOLDER = DATA_FOLDER / "src"
DATASET_FOLDER = DATA_FOLDER / "datasets"

SRC_URL = "https://pmb.let.rug.nl/releases/pmb-4.0.0.zip"


def get_source(force_redownload: bool = False) -> None:

    fname = SRC_URL.split("/")[-1]

    _make_dir_if_not_exists(SRC_DATA_FOLDER)

    if not Path(SRC_DATA_FOLDER / fname).exists() or force_redownload:
        # clean up the folder before recreating it
        for path in Path(SRC_DATA_FOLDER).iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

        _download_file(SRC_URL, fname)
        _extract_file(fname, SRC_DATA_FOLDER)
    
    else:
        print("### data already downloaded, skipping download ###")


def _extract_file(src_file: str, path: Path) -> None:
    """extracts the zipfile into the src-folder"""

    src_path = SRC_DATA_FOLDER / src_file
    with ZipFile(src_path, "r") as zip_f:
        for file in tqdm(iterable=zip_f.namelist(), total=len(zip_f.namelist()), desc=f"unpacking {src_file}"):
            #trg_path = path / file
            zip_f.extract(member=file)


def _download_file(url: str, filename: str) -> None:
    """downloads the (compressed) source file"""

    with requests.get(url, stream=True, timeout=5) as res:
        if res.status_code != 200:
            res.raise_for_status()
            raise RuntimeError(f"{url} returned {res.status_code} status")

        size = int(res.headers.get("Content-Length", 0))

        res.raw.read = partial(res.raw.read, decode_content=True)

        desc = f"downloading {filename}"
        with tqdm.wrapattr(res.raw, "read", total=size, desc=desc) as raw_res:
            with open(SRC_DATA_FOLDER / filename, "wb") as file:
                shutil.copyfileobj(raw_res, file)


def _make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    get_source()
