from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from syne_tune.blackbox_repository import load, add_surrogate
from syne_tune.blackbox_repository.blackbox import Blackbox
import syne_tune.config_space as sp
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.util import s3_experiment_path
from syne_tune.blackbox_repository.blackbox_offline import serialize, BlackboxOffline
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path, upload
import syne_tune.experiments

SOURCE_SYNE_TUNE_JOB_NAMES = (
    'speed-bs-it-nw-new-2022-02-21-18-05-01-921',  # big sweep
    # 'speed-bs-it-nw-new-2022-02-28-13-37-54-540',  # ml.g5.xlarge, bs=48, num_data_workers>1
    # 'speed-bs-it-nw-new-2022-02-28-14-43-39-336',  # ml.g5.12xlarge, bs=48 num_data_workers>1
    # 'speed-bs-it-nw-g5xlarge-bs52-2022-03-11-14-41-19-715',  # ml.g5.xlarge, bs=52, num_data_workers=0,1,2
    # 'speed-bs-it-nw-p3-16-2022-03-10-22-49-26-353',  # ml.p3.16xlarge, largest_bs_possible
    # 'speed-bs-it-nw-g5-48-2022-03-11-10-05-28-531',  # ml.g5.48xlarge, largest_bs_possible
    # 'speed-bs-it-nw-p3dn-24-2022-03-11-09-52-19-358',  # ml.p3dn.24xlarge, largest_bs_possible
    # 'speed-bs-it-nw-p4d-24-2022-03-11-12-54-38-398',  # ml.p4d.24xlarge, largest_bs_possible
    # 'speed-bs-it-nw-g5Xxlarge-bs52-2022-03-11-15-02-39-559',  # ml.g5.*xlarge, bs=52
)

BLACKBOX_NAME = 'hf-cloud-speed'

# Syne Tune's st_worker_time and st_tuner_cost do not account for either:
# a) the EC2 instance startup overhead time (customers don't pay for this)
# b) the time at the beginning of the running of the script (starting the script, setting up the dataloader etc)
# We assume an idealized scenario where we have each instance type already running and with dataloader prepared,
# such that they're all ready to start our jobs. Hence we can omit both effects a) and b).
# However, we still assume the attempts that fail due to OOM, incur the cost equivalent to SCRIPT_SETUP_OVERHEAD_TIME.
INSTANCE_STARTUP_OVERHEAD_TIME = 0  # in seconds
# This was estimated as the median of "BillableTimeInSeconds-st_worker_time" for runs of a given training script
# (gluonts-on-electricity or hugging-distil-bert-finetunes-on-imdb).
# For gluonts on 'deepar-speed-bs-32-2022-04-21-14-48-44-131': SCRIPT_SETUP_OVERHEAD_TIME = 65
# For distill-bert-on-imdb on 'loss-lr-wd-bs-2-2022-02-07-23-13-30-781': SCRIPT_SETUP_OVERHEAD_TIME = 410
# SCRIPT_SETUP_OVERHEAD_TIME = 410  # in seconds
SCRIPT_SETUP_OVERHEAD_TIME = 0.  # in seconds

# The speed benchmarking experiments were run for max_run=5min timeout setting of the HuggingFace estimator.
# If we are more efficient about software engineering of the solution we could run only a few batches
# and do it cheaper.
ST_WORKER_TIME_DISCOUNT = 0.1


class HFCloudSpeedBlackbox(Blackbox):
    """
    Dataset generated using adam_scripts/launch_huggingface_sweep_ag.py
    """

    def __init__(self, bb):
        super(HFCloudSpeedBlackbox, self).__init__(
            configuration_space=bb.configuration_space,
            fidelity_space=bb.fidelity_space,
            fidelity_values=bb.fidelity_values,
            objectives_names=bb.objectives_names,
        )
        self.bb = add_surrogate(bb, surrogate=KNeighborsRegressor(n_neighbors=2, weights='distance'))

        self.metrics = ['training-runtime-per-sample', 'training-cost-per-sample']
        self.instance_info = InstanceInfos()

        #       GPU  RAM max_bs_imdb
        # g4dn T4   16GB          32
        # g5   A10G 24GB          52
        # p2   K80  12GB          24
        # p3   V100 16GB          32
        # p3dn V100 32GB          68
        # p4d  A100 40GB          88

        self.per_device_train_batch_size_limits = {
            'g4dn': 32,
            'g5': 52,
            'p2': 24,
            'p3': 32,
            'p3dn': 68,
            'p4d': 88,
        }

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
        failed_attempt = np.array([[
            1.,  # training-runtime-per-sample
            1.,  # training-cost-per-sample
            # This is how long it would take before the script would fail
            INSTANCE_STARTUP_OVERHEAD_TIME + SCRIPT_SETUP_OVERHEAD_TIME,  # st_worker_time
            # This is how much the customer would be billed for it
            (SCRIPT_SETUP_OVERHEAD_TIME / 60. / 60. *
             self.instance_info(configuration['config_st_instance_type']).cost_per_hour),  # st_tuner_cost
        ]])

        if self.per_device_train_batch_size_limits[configuration['config_st_instance_type'].split('.')[1]] < \
                configuration['config_per_device_train_batch_size']:
            return failed_attempt

        res = self.bb.objective_function(configuration=configuration, fidelity=fidelity, seed=seed)

        # Please see the comments on line ~31, above the definition of class HFCloudSpeedBlackbox.
        res[:, 2] = (res[:, 2] * ST_WORKER_TIME_DISCOUNT + INSTANCE_STARTUP_OVERHEAD_TIME + SCRIPT_SETUP_OVERHEAD_TIME)
        res[:, 3] = (res[:, 3] * ST_WORKER_TIME_DISCOUNT +
                     SCRIPT_SETUP_OVERHEAD_TIME / 60. / 60. *
                     self.instance_info(configuration['config_st_instance_type']).cost_per_hour)

        return res


def serialize_hf_cloud_speed():
    dfs_to_concat = list()
    trial_id_max = -1
    for tuner_job_name in SOURCE_SYNE_TUNE_JOB_NAMES:
        df_temp = syne_tune.experiments.load_experiment(tuner_job_name).results
        df_temp['trial_id'] += trial_id_max + 1
        trial_id_max = df_temp['trial_id'].max()
        dfs_to_concat.append(df_temp)
    df = pd.concat(dfs_to_concat).reset_index()

    # Drop duplicates resulting from what is understood is the following issue
    # https://github.com/awslabs/syne-tune/issues/214
    temp = df.groupby(['trial_id', 'step']).loss.count().reset_index()
    trial_ids_to_be_deleted = temp[temp.loss > 1].trial_id.unique()

    df.drop(df.index[df['trial_id'].isin(trial_ids_to_be_deleted)], inplace=True)

    # Compute time_per_sample
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
        dfg.st_worker_cost.max(),
    ], axis=1)
    b.columns = ['samples_processed_per_second'] + list(b.columns)[1:]
    # TODO investigate where are the NaNs coming from
    #  - probably some of the runs not having the measurement for at least two steps - having too large batch size to
    #    record 200.
    #  - do I have NaNs also in the notebook?

    b = b.dropna(subset=['samples_processed_per_second'], how='all')

    c = b.groupby([
        'config_st_instance_type',
        'config_per_device_train_batch_size',
        'config_dataloader_num_workers']).agg(
        {
            'samples_processed_per_second': ['mean'],
            'st_worker_time': ['mean'],
            'st_worker_cost': ['mean'],
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
        # config_per_device_train_batch_size=sp.finrange(8.0, 56.0, 7),  # {8.0, 16.0, 24.0, 32.0, 40.0, 48.0}
        config_per_device_train_batch_size=sp.finrange(8.0, 48.0, 6),  # {8.0, 16.0, 24.0, 32.0, 40.0, 48.0}
        config_dataloader_num_workers=sp.finrange(0, 1, 2),  # [0, 1]
        # config_per_device_train_batch_size=sp.finrange(4.0, 88.0, 22),  # [4, 8, ..., 88]
        # config_dataloader_num_workers=sp.finrange(0, 2, 3),  # [0, 1, 2]
    )

    serialize(
        {
            'imdb': BlackboxOffline(
                df_evaluations=dfrt,
                configuration_space=configuration_space,
                fidelity_space={'step': sp.choice([1, ])},
                fidelity_values=[1, ],
                objectives_names=['training-runtime-per-sample', 'training-cost-per-sample',
                                  'st_worker_time', 'st_worker_cost', ],
            )
        },
        path=repository_path / BLACKBOX_NAME
    )


def import_hf_cloud_speed():
    bb = load("hf-cloud-speed")
    bb_dict = {'imdb': HFCloudSpeedBlackbox(bb=bb['imdb'])}
    return bb_dict


def generate_hf_cloud_speed(s3_root: Optional[str] = None):
    serialize_hf_cloud_speed()
    upload(name=BLACKBOX_NAME, s3_root=s3_root)


if __name__ == '__main__':
    generate_hf_cloud_speed()
