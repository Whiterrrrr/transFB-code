# pylint: disable=protected-access

"""Reformats the exorl dataset to a single .npz file."""

from argparse import ArgumentParser
from os import listdir
import numpy as np
from utils import BASE_DIR
from tqdm import tqdm
from loguru import logger
import subprocess
import os

# Overwrite default config using argparse
# parser = ArgumentParser()
# parser.add_argument("domain_algorithm_pair", type=str)
# args = parser.parse_args()

# domain_algorithm_pair = [args.domain_algorithm_pair.rsplit("_", 1)]

# print(domain_algorithm_pair)
# download from exorl bucket
domains = os.listdir(BASE_DIR / "datasets")
for domain in domains:
    algos = os.listdir(BASE_DIR / "datasets" / domain)
    # subprocess.call(["bash", "download.sh", domain, algorithm])
    for algo in algos:
        data_dir = BASE_DIR / f"datasets/{domain}/{algo}/buffer"
        video_dir = BASE_DIR / f"datasets/{domain}/{algo}/video"
        try:
            data_fnames = [f for f in listdir(data_dir) if f[-4:] == ".npz"]
            video_fnames = [f for f in listdir(video_dir) if f[-4:] == ".mp4"]
            new_dataset_path = BASE_DIR / f"datasets/{domain}/{algo}/dataset.npz"

            dataset = {}
            logger.info(f"Reformatting {domain} {algo} exorl dataset.")
            for fname in tqdm(data_fnames, desc="Reformatting dataset"):
                data = np.load(f"{data_dir}/{fname}")
                episode = {fname[:-4]: dict(data)}
                dataset = {**dataset, **episode}

            logger.info(f"Reformatting complete. Saving dataset to {new_dataset_path}.")
            np.savez(new_dataset_path, **dataset)

            # delete old files
            for fname in tqdm(data_fnames, desc="Deleting old buffer files"):
                (data_dir / fname).unlink()

            for fname in tqdm(video_fnames, desc="Deleting old video files"):
                (video_dir / fname).unlink()
        except Exception:
            continue
        # delete old directories
        # data_dir.rmdir()
        # video_dir.rmdir()
