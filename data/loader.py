import requests
import os
import shutil
from tqdm import tqdm
from pathlib import Path
from functools import partial
from zipfile import ZipFile
import lxml.etree as ET
import yaml
from datasets import Dataset, load_from_disk, Features, Value, Sequence, ClassLabel

DATA_FOLDER = Path(__file__).parent.resolve()
SRC_DATA_FOLDER = DATA_FOLDER / "src"
DATASET_FOLDER = DATA_FOLDER / "datasets"
ROLE_PATH = DATA_FOLDER / "pmb_roles.yaml"

SRC_URL = "https://pmb.let.rug.nl/releases/pmb-4.0.0.zip"


def load_data(langs: tuple, quality: tuple, force_regen: bool = False) -> Dataset:
    """load the pmb data as dataset in the specified configuration"""
    get_source(force_regen)
    ds = create_dataset(langs, quality, force_regen)

    return ds


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


def create_dataset(langs: tuple, quality: tuple, force_regen: bool = False) -> Dataset:
    """creates a dataset from the PMB data files"""

    if len(langs) == 0:
        raise RuntimeError(
            "at least one language should be specified when creating the dataset!")

    if len(quality) == 0:
        raise RuntimeError(
            "at least one quality standard has to be specified when creating the dataset!")

    _make_dir_if_not_exists(DATASET_FOLDER)

    ds_name = ".".join(langs) + "_" + ".".join(quality)

    ds_path = DATASET_FOLDER / ds_name
    if not Path(ds_path).exists() or force_regen:

        with open(ROLE_PATH, "r", encoding="utf-8") as file:
            pmb_roles = yaml.safe_load(file)

        verbnet_roles = ["0"] + pmb_roles["verbnet"]["event"] + \
            pmb_roles["verbnet"]["concept"] + \
            pmb_roles["verbnet"]["time"] + pmb_roles["verbnet"]["other"]

        ds_features = Features({
            "tok": Sequence(feature=Value(dtype="string")),
            "verbnet": Sequence(feature=Sequence(feature=ClassLabel(num_classes=len(verbnet_roles), names=verbnet_roles))),
            "sem": Sequence(feature=Value(dtype="string")),
            "cat": Sequence(feature=Value(dtype="string")),
            "lang": ClassLabel(num_classes=len(pmb_roles["language"]), names=pmb_roles["language"]),
            "quality": ClassLabel(num_classes=len(pmb_roles["quality"]), names=pmb_roles["quality"]),
            "id": Value(dtype="string")
        })

        dataset = Dataset.from_generator(_dataset_gen, features=ds_features, gen_kwargs={
                                         "languages": langs, "standards": quality}, config_name=ds_name)

        dataset = dataset.train_test_split(test_size=0.1, seed=42)

        dataset.save_to_disk(ds_path)
    else:
        dataset = load_from_disk(ds_path)

    return dataset


def _dataset_gen(languages: tuple, standards: tuple) -> dict:
    """generator function to create the dataset from the pmb data"""

    data_path = SRC_DATA_FOLDER / "pmb-4.0.0" / "data"

    for lang in languages:

        for stand in standards:

            path = data_path / lang / stand

            for part_dir in path.iterdir():
                part = part_dir.stem

                for doc_dir in part_dir.iterdir():
                    doc = doc_dir.stem

                    xml_path = doc_dir / f"{lang}.drs.xml"

                    tok_dict = _parse_drs_xml(xml_path)

                    tok_dict["lang"] = lang
                    tok_dict["quality"] = stand
                    tok_dict["id"] = f"{part}/{doc}"

                    yield tok_dict


def _parse_drs_xml(filepath: Path) -> dict:
    """parses the contents of a single DRS file"""

    drs_dict = {
        "tok": [],
        "verbnet": [],
        "sem": [],
        "cat": []
    }

    with open(filepath, "r", encoding="utf-8") as file:
        tree = ET.parse(file)
        root = tree.getroot()

        taggedtokens = root.find("xdrs/taggedtokens")

        for tagtoken in taggedtokens:
            verbnet = []
            tok = ""
            sem = ""
            cat = ""
            for tag in tagtoken.iter("tag"):
                tag_type = tag.get("type")
                if tag_type == "verbnet":
                    if tag.get("n") != "0":
                        verbnet = tag.text[1:-1].split(",")
                    else:
                        verbnet = ["0"]
                elif tag_type == "tok":
                    tok = tag.text.replace("~", " ")
                elif tag_type == "sem":
                    sem = tag.text
                elif tag_type == "cat":
                    cat = tag.text

            if tok == "ø":
                continue
            else:
                drs_dict["verbnet"].append(verbnet)
                drs_dict["tok"].append(tok)
                drs_dict["sem"].append(sem)
                drs_dict["cat"].append(cat)

    return drs_dict


def _extract_file(src_file: str, path: Path) -> None:
    """extracts the zipfile into the src-folder"""

    src_path = SRC_DATA_FOLDER / src_file
    with ZipFile(src_path, "r") as zip_f:
        for file in tqdm(iterable=zip_f.namelist(), total=len(zip_f.namelist()), desc=f"unpacking {src_file}"):
            zip_f.extract(member=file, path=path)


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
    ds = load_data(("en",), ("gold", "silver"))

    print(ds.shape)
    print(ds.column_names)
    print(ds["train"][0])
