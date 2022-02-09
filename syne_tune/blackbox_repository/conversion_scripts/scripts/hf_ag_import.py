"""
Dataset generated using examples/launch_huggingface_sweep_ag.py
"""
from typing import Optional
import pandas as pd
import numpy as np
from blackbox_repository.blackbox_offline import serialize, BlackboxOffline
from blackbox_repository.conversion_scripts.utils import repository_path, upload
import syne_tune.search_space as sp
import syne_tune.experiments

SOURCE_SYNE_TUNE_JOB_NAMES = (
    'loss-lr-wd-bs-2022-02-04-12-09-26-911',
    'loss-lr-wd-bs-2-2022-02-07-23-13-30-781',
)

def serialize_hf_ag():
    blackbox = 'hf-ag'
    df = pd.concat(tuple(syne_tune.experiments.load_experiment(job_name).results
                         for job_name in SOURCE_SYNE_TUNE_JOB_NAMES))

    # Rename some columns
    # TODO if we regenerate the dataset some renaming will change due to updates in the generation code
    columns_to_rename = {
        'loss': 'metric_training_loss',
        'config_train_batch_size': 'per_device_train_batch_size',
        'config_learning_rate': 'learning_rate',
        'config_weight_decay': 'weight_decay',
    }
    df = df.rename(columns=columns_to_rename)

    configuration_space = dict(
        per_device_train_batch_size=sp.choice([2, 4, 8, 12, 16]),
        learning_rate=sp.loguniform(1e-7, 1e-4),
        weight_decay=sp.loguniform(1e-6, 1e-2),
    )

    serialize({
        'imdb': BlackboxOffline(
            df_evaluations=df,
            configuration_space=configuration_space,
            objectives_names=['metric_training_loss'],
        )
        },
        path=repository_path / "hf-ag"
    )


def generate_hf_ag(s3_root: Optional[str] = None):
    serialize_hf_ag()
    upload(name="hf-ag", s3_root=s3_root)


if __name__ == '__main__':
    generate_hf_ag()