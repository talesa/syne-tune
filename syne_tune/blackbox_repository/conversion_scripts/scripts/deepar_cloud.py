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
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path, upload
import syne_tune.experiments


METRIC_VALID_ERROR = 'metric_training_loss'

# This is cumulative time required for consuming the resource up until this point in training.
METRIC_TIME_CUMULATIVE_RESOURCE = 'metric_train_runtime'

LEARNING_CURVE_SOURCE_SYNE_TUNE_JOB_NAMES = (
    'deepar-curves-2022-04-25-10-46-32-616',
    'deepar-curves-2-2022-04-25-12-31-30-154',
    'deepar-curves-3-2022-04-25-12-32-14-346',
#     'deepar-curves-4-2022-05-04-08-53-23-418',
#     # There was no deepar-curves-5
#     'deepar-curves-6-2022-05-04-08-54-35-102',
)
SPEED_SYNE_TUNE_JOB_NAME = 'speed-bs-it-2022-02-07-23-12-47-916'

BLACKBOX_NAME = 'deepar-cloud'


class DeepARCloudBlackbox(Blackbox):
    """
    Dataset generated using adam_scripts/
    """
    def __init__(self, bb):
        self.configuration_space = bb.configuration_space
        self.objectives_names = bb.objectives_names + ["cost"]
        super(DeepARCloudBlackbox, self).__init__(
            configuration_space=self.configuration_space,
            fidelity_space=bb.fidelity_space,
            fidelity_values=bb.fidelity_values,
            objectives_names=self.objectives_names,
        )
        self.bb = add_surrogate(bb, surrogate=KNeighborsRegressor(n_neighbors=3, weights='distance'),)

        assert len(bb.df.config_st_instance_type.unique()) == 1, \
            "All of the trials should have been performed on the same instance type."
        # Sets the baseline_instance_type to the single instance type all of the loss values were collected for.
        baseline_instance_type = bb.df.config_st_instance_type.unique()[0]
        self.instance_speed_cost_dict = instance_speed_cost(baseline_instance_type=baseline_instance_type)
        self.configuration_space["instance_type"] = sp.choice(
            np.unique(np.array(list(self.instance_speed_cost_dict.keys()))[:, 0]).tolist())

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
        if 'instance_type' not in configuration:
            raise ValueError(f'No instance_type provided in the configuration: {configuration}')
        instance_type = configuration.pop('instance_type')
        relative_time_factor, cost_per_second = self.instance_speed_cost_dict[(instance_type, configuration['per_device_train_batch_size'])]

        res = self.bb.objective_function(configuration=configuration, fidelity=fidelity, seed=seed)

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


def instance_speed_cost(baseline_instance_type: str) -> Dict[str, Tuple[float, float]]:
    """
    :param baseline_instance_type: The baseline instance type that was used to collect the loss functions for the
      blackbox.
    :return: dictionary from tuple (instance_type, batch_size) to the tuple
      (relative-time, cost-per-second of the chosen instance type).
    """
    # gets instance dollar-cost
    instance_info = InstanceInfos()
    instance_hourly_cost = {instance: instance_info(instance).cost_per_hour for instance in instance_info.instances}

    df = syne_tune.experiments.load_experiment(SPEED_SYNE_TUNE_JOB_NAME).results
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


def import_deepar_cloud():
    bb = load("deepar-cloud")
    bb_dict = {'electricity': DeepARCloudBlackbox(bb=bb['electricity'])}
    return bb_dict


def serialize_deepar_cloud():
    dfs_to_concat = list()
    trial_id_max = -1
    for tuner_job_name in LEARNING_CURVE_SOURCE_SYNE_TUNE_JOB_NAMES:
        df_temp = syne_tune.experiments.load_experiment(tuner_job_name).results
        df_temp['trial_id'] += trial_id_max + 1
        trial_id_max = df_temp['trial_id'].max()
        dfs_to_concat.append(df_temp)
    df = pd.concat(dfs_to_concat).reset_index()

    # Drop trials with duplicate entries, most likely due to this https://github.com/awslabs/syne-tune/issues/214
    temp = df.groupby(['trial_id', 'st_worker_iter']).mean_wQuantileLoss.count().reset_index()

    trial_ids_with_duplicates = set(temp[temp.mean_wQuantileLoss > 1].trial_id.unique())
    trial_ids_with_all_iters = set(df[df.epoch_no == 100].trial_id.unique())

    trial_ids_to_keep = (trial_ids_with_all_iters.difference(trial_ids_with_duplicates))

    df.drop(df.index[~df['trial_id'].isin(trial_ids_to_keep)], inplace=True)

    # Rename some columns
    columns_to_rename = {k: k.replace('config_', '') for k in df.columns if k.startswith('config_')}
    columns_to_rename.update({
        'st_worker_time': 'metric_train_runtime',
        'st_worker_iter': 'step',
    })
    df = df.rename(columns=columns_to_rename)

    configuration_space = {
        "lr": cs.loguniform(1e-4, 1e-1),
        "batch_size": cs.logfinrange(8, 128, 5, cast_int=True),  # cs.choice([8, 16, 32, 64, 128]),
        "num_cells": cs.randint(lower=1, upper=200),
        "num_layers": cs.randint(lower=1, upper=4),
    }

    # We set the maximum fidelity to be the minimum final fidelity across all learning curves gathered.
    N = df.reset_index().groupby('trial_id').step.max().min()
    fidelity_values = list(range(N+1))
    fidelity_space = dict(
        step=cs.choice(fidelity_values),
    )

    serialize(
        {
            'electricity': BlackboxOffline(
                df_evaluations=df,
                configuration_space=configuration_space,
                fidelity_space=fidelity_space,
                fidelity_values=fidelity_values,
                objectives_names=[col for col in df.columns if col.startswith("metric_")],
            )
        },
        path=repository_path / BLACKBOX_NAME
    )


def generate_deepar_cloud(s3_root: Optional[str] = None):
    serialize_deepar_cloud()
    upload(name=BLACKBOX_NAME, s3_root=s3_root)


if __name__ == '__main__':
    generate_deepar_cloud()
