"""
Dataset generated using examples/launch_huggingface_sweep_ag.py
"""
from typing import Optional
import pandas as pd
import numpy as np
from syne_tune.blackbox_repository.blackbox_offline import serialize, BlackboxOffline
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path, upload
import syne_tune.search_space as sp
import syne_tune.experiments

SOURCE_SYNE_TUNE_JOB_NAMES = (
    # 'loss-lr-wd-bs-2022-02-04-12-09-26-911',  # Old schema, i.e. names of the fields, don't use.
    'loss-lr-wd-bs-2-2022-02-07-23-13-30-781',
)

BLACKBOX_NAME = 'hf-ag'

def serialize_hf_ag():
    df = pd.concat(tuple(syne_tune.experiments.load_experiment(job_name).results
                         for job_name in SOURCE_SYNE_TUNE_JOB_NAMES))

    # Rename some columns
    # TODO if we regenerate the dataset some renaming will change due to updates in the generation code
    columns_to_rename = {
        'loss': 'metric_training_loss',
        'st_worker_time': 'metric_train_runtime',  # TODO change this to train_runtime outputed by huggingface
        'config_per_device_train_batch_size': 'per_device_train_batch_size',
        'config_learning_rate': 'learning_rate',
        'config_weight_decay': 'weight_decay',
    }
    df = df.rename(columns=columns_to_rename)
    # df = df.dropna(subset=['config_per_device_train_batch_size'])

    configuration_space = dict(
        per_device_train_batch_size=sp.choice([2, 4, 8, 12, 16]),
        learning_rate=sp.loguniform(1e-7, 1e-4),
        weight_decay=sp.loguniform(1e-6, 1e-2),
    )

    # TODO shouldn't fidelity_values be implied given fidelity_space?
    N = 14
    fidelity_values = [i*100 for i in range(1, N+1)]
    fidelity_space = dict(
        step=sp.choice(fidelity_values),
        # TODO What to do since “step” (number of gradient updates) implicitly defines different fidelity measures for different batch_sizes?
        # step=sp.finrange(lower=100, upper=500, size=5),  # [100, 200, 300, 400, 500]
    )

    serialize(
        {
            'imdb': BlackboxOffline(
                df_evaluations=df,
                configuration_space=configuration_space,
                fidelity_space=fidelity_space,
                fidelity_values=fidelity_values,
                objectives_names=[col for col in df.columns if col.startswith("metric_")],
            )
        },
        path=repository_path / BLACKBOX_NAME
    )


def generate_hf_ag(s3_root: Optional[str] = None):
    serialize_hf_ag()
    upload(name=BLACKBOX_NAME, s3_root=s3_root)


if __name__ == '__main__':
    generate_hf_ag()