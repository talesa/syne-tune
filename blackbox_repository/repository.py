from pathlib import Path
from typing import List

import s3fs as s3fs
import sagemaker

from blackbox_repository import BlackboxOffline
from blackbox_repository.blackbox_offline import deserialize as deserialize_offline
from blackbox_repository.blackbox_tabular import deserialize as deserialize_tabular

# where the blackbox repository is stored on s3
s3_blackbox_folder = f"{sagemaker.Session().default_bucket()}/blackbox-repository"

repository_path = Path("~/.blackbox-repository/").expanduser()


def blackbox_list() -> List[str]:
    """
    :return: list of blackboxes available
    """
    fs = s3fs.S3FileSystem()
    return sorted(list(set([
        Path(x).parent.name
        for x in fs.glob(f"s3://{s3_blackbox_folder}/*/*.parquet")
    ])))


def upload(name: str):
    """
    Uploads a blackbox locally present in repository_path to S3.
    :param name: folder must be available in repository_path/name
    """
    # test that blackbox can be retrieved before uploading it
    load(name)

    fs = s3fs.S3FileSystem()
    for src in Path(repository_path / name).glob("*"):
        tgt = f"s3://{s3_blackbox_folder}/{name}/{src.name}"
        print(f"copy {src} to {tgt}")
        fs.put(str(src), tgt)


def load(name: str, skip_if_present: bool = True) -> BlackboxOffline:
    """
    :param name: name of a blackbox present in the repository, see list() to get list of available blackboxes
    :param skip_if_present: skip the download if the file locally exists
    :return: blackbox with the given name, download it if not present.
    """
    tgt_folder = Path(repository_path) / name
    tgt_folder.mkdir(exist_ok=True)
    if tgt_folder.exists() and skip_if_present:
        print(f"skipping download of {name} as {tgt_folder} already exists, change skip_if_present to redownload")
    else:
        # download files from s3 to repository_path
        fs = s3fs.S3FileSystem()
        for src in fs.glob(f"{s3_blackbox_folder}/{name}/*"):
            tgt = tgt_folder / Path(src).name
            print(f"copying {src} to {tgt}")
            fs.get(src, str(tgt))
    if (tgt_folder / "hyperparameters.parquet").exists():
        return deserialize_tabular(tgt_folder)
    else:
        return deserialize_offline(tgt_folder)


if __name__ == '__main__':
    # list all blackboxes available
    blackboxes = blackbox_list()
    print(blackboxes)

    # download an existing blackbox
    blackbox = load(blackboxes[0])

    # upload a blackbox stored locally to the repository
    # upload(blackboxes[0])