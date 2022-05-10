from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from syne_tune.blackbox_repository import load, add_surrogate
from syne_tune.blackbox_repository.blackbox import Blackbox
import syne_tune.config_space as sp
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.blackbox_repository.blackbox_offline import BlackboxOffline

from adam_scripts.utils import (
    concatenate_syne_tune_experiment_results,
    upload_df_to_team_bucket,
)

SPEED_SYNE_TUNE_JOB_NAMES = (
    'speed-bs-it-nw-new-2022-02-21-18-05-01-921',
)

BLACKBOX_SPEED_S3_PATH = ('s3://mnemosyne-team-bucket/dataset/'
                          'hf-distilbert-on-imdb-static-blackbox/hf-distilbert-on-imdb-static-blackbox-speed.csv.zip')

BLACKBOX_NAME = 'hf-distilbert-on-imdb-static'

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


class HFDistilbertOnImdbStaticBlackbox(Blackbox):
    """
    Dataset generated using adam_scripts/launch_huggingface_sweep_ag.py
    """
    per_device_train_batch_size_limits = {
        'g4dn': 32,
        'g5': 52,
        'p2': 24,
        'p3': 32,
        'p3dn': 68,
        'p4d': 88,
    }

    #       GPU  RAM max_bs_imdb
    # g4dn T4   16GB          32
    # g5   A10G 24GB          52
    # p2   K80  12GB          24
    # p3   V100 16GB          32
    # p3dn V100 32GB          68
    # p4d  A100 40GB          88

    def __init__(self, reload_from_syne_tune_reports: bool = True):
        """
            :param: reload_from_syne_tune_reports: when True the data is reloaded from *_SYNE_TUNE_JOB_NAMES sources
            and uploaded to the Syne team S3 bucket at BLACKBOX_*_S3_PATH
        """
        if reload_from_syne_tune_reports:
            df = concatenate_syne_tune_experiment_results(SPEED_SYNE_TUNE_JOB_NAMES)
            upload_df_to_team_bucket(df, BLACKBOX_SPEED_S3_PATH)
        df = pd.read_csv(BLACKBOX_SPEED_S3_PATH)

        # Drop duplicates resulting from what is understood is the following issue
        # https://github.com/awslabs/syne-tune/issues/214
        temp = df.groupby(['trial_id', 'step']).loss.count().reset_index()
        trial_ids_to_be_deleted = temp[temp.loss > 1].trial_id.unique()

        df.drop(df.index[df['trial_id'].isin(trial_ids_to_be_deleted)], inplace=True)

        # Compute time_per_sample
        dfg = df.groupby(['trial_id'])

        instance_info = InstanceInfos()

        samples_processed_per_second = (
                dfg.step.max()
                * dfg.config_per_device_train_batch_size.max()
                * dfg.config_st_instance_type.max().map(lambda x: instance_info(x).num_gpu)
                / dfg.st_worker_time.max())

        b = pd.concat([
            samples_processed_per_second,
            dfg.config_st_instance_type.max(),
            dfg.config_per_device_train_batch_size.max(),
            dfg.config_dataloader_num_workers.max(),
            dfg.st_worker_time.max(),
            dfg.st_worker_cost.max(),
        ], axis=1)
        b.columns = ['samples_processed_per_second'] + list(b.columns)[1:]

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
            config_per_device_train_batch_size=sp.finrange(8.0, 48.0, 6),  # {8.0, 16.0, 24.0, 32.0, 40.0, 48.0}
            config_dataloader_num_workers=sp.finrange(0, 1, 2),  # [0, 1]
        )

        bb = BlackboxOffline(
                df_evaluations=dfrt,
                configuration_space=configuration_space,
                fidelity_space={'step': sp.choice([1, ])},
                fidelity_values=[1, ],
                objectives_names=['training-runtime-per-sample', 'training-cost-per-sample',
                                  'st_worker_time', 'st_worker_cost',],
            )

        super(HFDistilbertOnImdbStaticBlackbox, self).__init__(
            configuration_space=bb.configuration_space,
            fidelity_space=bb.fidelity_space,
            fidelity_values=bb.fidelity_values,
            objectives_names=bb.objectives_names,
        )
        self.bb = add_surrogate(bb, surrogate=KNeighborsRegressor(n_neighbors=2, weights='distance'))

        self.metrics = ['training-runtime-per-sample', 'training-cost-per-sample']
        self.instance_info = InstanceInfos()

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


def import_hf_cloud_speed():
    bb_dict = {'imdb': HFDistilbertOnImdbStaticBlackbox()}
    return bb_dict