from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

# from benchmarking.blackbox_repository import load, BlackboxOffline, add_surrogate, serialize
from benchmarking.blackbox_repository.blackbox import Blackbox
import syne_tune.config_space as sp
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.util import s3_experiment_path
from benchmarking.blackbox_repository.blackbox_offline import serialize, BlackboxOffline
from benchmarking.blackbox_repository.conversion_scripts.utils import repository_path, upload
import syne_tune.experiments


SOURCE_SYNE_TUNE_JOB_NAMES = (
    'speed-bs-it-nw-new-2022-02-21-18-05-01-921',  # big sweep
    'speed-bs-it-nw-new-2022-02-28-13-37-54-540',  # ml.g5.xlarge, bs=48, num_data_workers>1
    'speed-bs-it-nw-new-2022-02-28-14-43-39-336',  # ml.g5.12xlarge, bs=48 num_data_workers>1
    'speed-bs-it-nw-g5xlarge-bs52-2022-03-11-14-41-19-715',  # ml.g5.xlarge, bs=52, num_data_workers=0,1,2
    'speed-bs-it-nw-p3-16-2022-03-10-22-49-26-353',  # ml.p3.16xlarge
    'speed-bs-it-nw-g5-48-2022-03-11-10-05-28-531',  # ml.g5.48xlarge
    'speed-bs-it-nw-p3dn-24-2022-03-11-09-52-19-358',  # ml.p3dn.24xlarge
    'speed-bs-it-nw-p4d-24-2022-03-11-12-54-38-398',  # ml.p4d.24xlarge
    'speed-bs-it-nw-g5Xxlarge-bs52-2022-03-11-15-02-39-559',  # ml.g5.*xlarge, bs=52
)

BLACKBOX_NAME = 'hf-cloud-speed'


def serialize_hf_cloud_speed():
    # TODO account for the failed attempts

    dfs_to_concat = list()
    trial_id_max = -1
    for tuner_job_name in SOURCE_SYNE_TUNE_JOB_NAMES:
        df_temp = syne_tune.experiments.load_experiment(tuner_job_name).results
        df_temp['trial_id'] += trial_id_max + 1
        trial_id_max = df_temp['trial_id'].max()
        dfs_to_concat.append(df_temp)
    df = pd.concat(dfs_to_concat).reset_index()

    # Drop trials which have duplicate entries.
    # The reason for why some trials have these duplicates is not understood.
    # Doing this to ensure correctness.
    temp = df.groupby(['trial_id', 'step']).loss.count().reset_index()
    trial_ids_to_be_deleted = temp[temp.loss > 1].trial_id.unique()

    df.drop(df.index[df['trial_id'].isin(trial_ids_to_be_deleted)], inplace=True)

    # Compute time per samples
    dfg = df.groupby(['trial_id'])

    instance_info = InstanceInfos()

    number_of_samples_processed = \
        (dfg.step.max() - dfg.step.min()) * \
        dfg.config_per_device_train_batch_size.max() * \
        dfg.config_st_instance_type.max().map(lambda x: instance_info(x).num_gpu)

    samples_processed_per_second = number_of_samples_processed / (dfg.st_worker_time.max() - dfg.st_worker_time.min())

    b = pd.concat([
        samples_processed_per_second,
        dfg.config_st_instance_type.max(),
        dfg.config_per_device_train_batch_size.max(),
        dfg.config_dataloader_num_workers.max(),
        dfg.st_worker_time.max(),
    ], axis=1)
    b.columns = ['samples_processed_per_second'] + list(b.columns)[1:]
    # TODO investigate where are the NaNs coming from
    #  - probably some of the runs not having the measurement for at least two steps - having too large batch size to
    #    record 200.

    b = b.dropna(subset=['samples_processed_per_second'], how='all')

    c = b.groupby([
        'config_st_instance_type',
        'config_per_device_train_batch_size',
        'config_dataloader_num_workers']).agg(
        {
            'samples_processed_per_second': ['mean'],
            'st_worker_time': ['mean'],
        })

    dfrt = c.reset_index()
    dfrt.columns = [v[0] for v in list(c.reset_index().columns)]

    dfrt['training-runtime-per-sample'] = 1. / dfrt['samples_processed_per_second']
    dfrt['training-cost-per-sample'] = (
            dfrt['training-runtime-per-sample'] *
            dfrt.config_st_instance_type.map(lambda x: instance_info(x).cost_per_hour))

    dfrt['step'] = 1

    configuration_space = dict(
        config_st_instance_type=sp.choice(dfrt.config_st_instance_type.unique().tolist()),
        config_per_device_train_batch_size=sp.finrange(4.0, 88.0, 22),  # [4, 8, ..., 88]
        config_dataloader_num_workers=sp.finrange(0, 2, 3),  # [0, 1, 2]
    )

    serialize(
        {
            'imdb': BlackboxOffline(
                df_evaluations=dfrt,
                configuration_space=configuration_space,
                fidelity_space={'step': sp.choice([1, ])},
                fidelity_values={'step': [1, ]},
                objectives_names=['training-runtime-per-sample', 'training-cost-per-sample'],
            )
        },
        path=repository_path / BLACKBOX_NAME
    )


def generate_hf_cloud_speed(s3_root: Optional[str] = None):
    serialize_hf_cloud_speed()
    upload(name=BLACKBOX_NAME, s3_root=s3_root)


if __name__ == '__main__':
    generate_hf_cloud_speed()
