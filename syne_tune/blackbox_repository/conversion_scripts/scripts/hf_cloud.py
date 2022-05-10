from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from syne_tune.blackbox_repository import load, add_surrogate, serialize
from syne_tune.blackbox_repository.blackbox import Blackbox
import syne_tune.config_space as cs
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.blackbox_repository.blackbox_offline import serialize, BlackboxOffline

from adam_scripts.utils import (
    concatenate_syne_tune_experiment_results,
    upload_df_to_team_bucket,
)


METRIC_VALID_ERROR = 'metric_training_loss'

# This is cumulative time required for consuming the resource up until this point in training.
METRIC_TIME_CUMULATIVE_RESOURCE = 'metric_train_runtime'

LEARNING_CURVE_SYNE_TUNE_JOB_NAMES = (
    'loss-lr-wd-bs-2-2022-02-07-23-13-30-781',
)
SPEED_SYNE_TUNE_JOB_NAMES = (
    'speed-bs-it-2022-02-07-23-12-47-916',
)

BLACKBOX_SPEED_S3_PATH = ('s3://mnemosyne-team-bucket/dataset/'
                          'hf-distilbert-on-imdb-blackbox/hf-distilbert-on-imdb-blackbox-speed.csv.zip')
BLACKBOX_ERROR_S3_PATH = ('s3://mnemosyne-team-bucket/dataset/'
                          'hf-distilbert-on-imdb-blackbox/hf-distilbert-on-imdb-blackbox-error.csv.zip')

BLACKBOX_NAME = 'hf-cloud'


class HFCloudBlackbox(Blackbox):
    """
    Dataset generated using adam_scripts/launch_huggingface_sweep_ag.py
    """
    def __init__(self, reload_from_syne_tune_reports: bool = False):
        """
            :param: reload_from_syne_tune_reports: when True the data is reloaded from *_SYNE_TUNE_JOB_NAMES sources
            and uploaded to the Syne team S3 bucket at BLACKBOX_*_S3_PATH
        """
        if reload_from_syne_tune_reports:
            df = concatenate_syne_tune_experiment_results(LEARNING_CURVE_SYNE_TUNE_JOB_NAMES)
            upload_df_to_team_bucket(df, BLACKBOX_ERROR_S3_PATH)
        df = pd.read_csv(BLACKBOX_ERROR_S3_PATH)

        # Drop trials with duplicate entries, most likely due to this https://github.com/awslabs/syne-tune/issues/214
        temp = df.groupby(['trial_id', 'step']).loss.count().reset_index()
        trial_ids_to_be_deleted = temp[temp.loss > 1].trial_id.unique()
        df.drop(df.index[df['trial_id'].isin(trial_ids_to_be_deleted)], inplace=True)

        assert len(df.config_st_instance_type.unique()) == 1, \
            "All of the trials should have been performed on the same instance type."

        # Rename some columns
        columns_to_rename = {k: k.replace('config_', '') for k in df.columns if k.startswith('config_')}
        columns_to_rename.update({
            'loss': 'metric_training_loss',
            'st_worker_time': 'metric_train_runtime',
        })
        df = df.rename(columns=columns_to_rename)

        if reload_from_syne_tune_reports:
            df_speed = concatenate_syne_tune_experiment_results(SPEED_SYNE_TUNE_JOB_NAMES)
            upload_df_to_team_bucket(df_speed, BLACKBOX_SPEED_S3_PATH)
        df_speed = pd.read_csv(BLACKBOX_SPEED_S3_PATH)

        configuration_space = dict(
            # We are setting batch_size to the values [4, 8, 12, 16] because that's the overlap of the
            # per_device_train_batch_size field in the search spaces used for A) training curves/loss function values
            # generation, and B) relative training speed generation.
            per_device_train_batch_size=cs.finrange(4.0, 16.0, 4),  # [4, 8, 12, 16]
            learning_rate=cs.loguniform(1e-7, 1e-4),
            weight_decay=cs.loguniform(1e-6, 1e-2),
            st_instance_type=cs.choice(df_speed.config_st_instance_type.unique())
        )

        # We set the maximum fidelity to be the minimum final fidelity across all learning curves gathered.
        df['step'] = df.st_worker_iter + 1
        Nmax = df.reset_index().groupby('trial_id').step.max().min()
        Nmin = df.reset_index().groupby('trial_id').step.min().min()
        fidelity_values = list(range(Nmin, Nmax + 1))
        fidelity_space = dict(
            step=cs.finrange(Nmin, Nmax, (Nmax - Nmin) + 1, cast_int=True),
        )

        bb = BlackboxOffline(
                    df_evaluations=df,
                    configuration_space=configuration_space,
                    fidelity_space=fidelity_space,
                    fidelity_values=fidelity_values,
                    objectives_names=['metric_training_loss', 'metric_train_runtime'],
                )

        super(HFCloudBlackbox, self).__init__(
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            fidelity_values=fidelity_values,
            objectives_names=['metric_training_loss', 'metric_train_runtime', 'metric_cost'],
        )
        self.bb = add_surrogate(bb, surrogate=KNeighborsRegressor(n_neighbors=3, weights='distance'),)

        assert len(bb.df.reset_index().st_instance_type.unique()) == 1, \
            "All of the trials should have been performed on the same instance type."
        # Sets the baseline_instance_type to the single instance type all of the loss values were collected for.
        baseline_instance_type = bb.df.reset_index().st_instance_type.unique()[0]
        self.instance_speed_cost_dict = instance_speed_cost(baseline_instance_type=baseline_instance_type)

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
        if 'st_instance_type' not in configuration:
            raise ValueError(f'No st_instance_type provided in the configuration: {configuration}')
        instance_type = configuration['st_instance_type']
        relative_time_factor, cost_per_second = self.instance_speed_cost_dict[(instance_type, configuration['per_device_train_batch_size'])]

        res = self.bb.objective_function(configuration=configuration, fidelity=fidelity, seed=seed)

        # TODO should 3 metrics be hardcoded here?
        if fidelity is not None:
            adjusted_time = res[METRIC_TIME_CUMULATIVE_RESOURCE] * relative_time_factor
            return {
                METRIC_VALID_ERROR: res[METRIC_VALID_ERROR],
                "metric_train_runtime": adjusted_time,
                "metric_cost": adjusted_time * cost_per_second,
            }
        else:
            index_time = [i for i, x in enumerate(self.objectives_names) if x == METRIC_TIME_CUMULATIVE_RESOURCE][0]
            res[:, index_time] *= relative_time_factor

            # add cost which runtime seconds time second cost
            cost_per_second = res[:, index_time:index_time+1] * cost_per_second
            res = np.hstack([res, cost_per_second])

            return res


def instance_speed_cost(baseline_instance_type: str) -> Dict[Tuple[str, float], Tuple[float, float]]:
    """
    :param baseline_instance_type: The baseline instance type that was used to collect the loss functions for the
      blackbox.
    :return: dictionary from tuple (instance_type, batch_size) to the tuple
      (relative-time, cost-per-second of the chosen instance type).
    """
    # gets instance dollar-cost
    instance_info = InstanceInfos()
    instance_hourly_cost = {instance: instance_info(instance).cost_per_hour for instance in instance_info.instances}

    df = pd.read_csv(BLACKBOX_SPEED_S3_PATH)
    time_col = 'time_per_gradient_step'
    df[time_col] = df.st_worker_time / df.step

    # gets time per batch relative to the time for a baseline instance
    # taking the shortest train_runtime per instance-type which most of the time corresponds to the largest batch_size
    time_per_instance = (
        df.groupby(['config_st_instance_type', 'config_per_device_train_batch_size'])[time_col].mean())
    relative_time_per_instance = time_per_instance / time_per_instance.loc[baseline_instance_type]
    output = {
        # gets the cost per second
        (instance, batch_size): (relative_time_per_instance.loc[(instance, batch_size)],
                                 instance_hourly_cost[instance] / 3600.)
        for (instance, batch_size) in relative_time_per_instance.index
    }

    # A provisional fix because these two records for some reason are not present in the SPEED_SYNE_TUNE_JOB_NAME
    # results, most likely the jobs executing these failed and the results were never recorded
    output[('ml.g4dn.4xlarge', 4.)] = (1.0, 0.0004683333333333333)
    output[('ml.g4dn.4xlarge', 16.)] = (1.0, 0.0004683333333333333)

    return output


def import_hf_cloud():
    bb_dict = {'imdb': HFCloudBlackbox()}
    return bb_dict