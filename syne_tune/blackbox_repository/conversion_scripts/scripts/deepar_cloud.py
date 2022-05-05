import copy
from numbers import Number
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from syne_tune.blackbox_repository import load, add_surrogate, serialize
from syne_tune.blackbox_repository.blackbox import Blackbox
import syne_tune.config_space as cs
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.blackbox_repository.blackbox_offline import serialize, BlackboxOffline
from syne_tune.blackbox_repository.conversion_scripts.utils import repository_path, upload
import syne_tune.experiments


METRIC_VALID_ERROR = 'mean_wQuantileLoss'

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
SPEED_SYNE_TUNE_JOB_NAMES = (
    'deepar-speed-bs-32-2022-04-21-16-25-04-045',
    'deepar-speed-bs-32-2022-04-21-14-48-44-131',
)

BLACKBOX_NAME = 'deepar-cloud'

INSTANCE_TYPES = [
        # CPU
        'ml.c4.2xlarge',
        'ml.c4.4xlarge',
        'ml.c4.8xlarge',
        'ml.c4.xlarge',
        'ml.c5.18xlarge',
        'ml.c5.2xlarge',
        'ml.c5.4xlarge',
        'ml.c5.9xlarge',
        'ml.c5.xlarge',
        'ml.c5n.18xlarge',
        'ml.c5n.2xlarge',
        'ml.c5n.4xlarge',
        'ml.c5n.9xlarge',
        'ml.c5n.xlarge',
        'ml.m4.10xlarge',
        'ml.m4.16xlarge',
        'ml.m4.2xlarge',
        'ml.m4.4xlarge',
        'ml.m4.xlarge',
        'ml.m5.12xlarge',
        'ml.m5.24xlarge',
        'ml.m5.2xlarge',
        'ml.m5.4xlarge',
        'ml.m5.large',
        'ml.m5.xlarge',
]


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

        self.instance_info = InstanceInfos()

        # TODO rewrite this properly creating Blackbox objects rather than hacking it like this
        # Prepare the two blackboxes, self.bb_instance_type and self.bb_training_runtime, which return the
        # st_instance_type the metric_training_runtime for consecutive fidelities.
        # We need these such that we can then adjust the st_worker_time/train_runtime using information from the
        # speed-benchmarking-runs.
        bb_instance_type = copy.deepcopy(bb)
        bb_instance_type.configuration_space = {
            k: v for k, v in bb_instance_type.configuration_space.items()
            if k not in ['lr', 'st_instance_type']}
        df = bb_instance_type.df
        df = df.reset_index().set_index(['batch_size', 'num_cells', 'num_layers', 'step'])

        # The same combination of ['batch_size', 'num_cells', 'num_layers'] might point to multiple trial_id
        # (with different st_instance_type and/or lr)
        # Here, we pick remove some of the trial_ids such that we have only one trial_id corresponding to each
        # combination of ['batch_size', 'num_cells', 'num_layers'], such that we can query that runs st_worker_time
        # for consecutive fidelities and its st_instance_type.
        trials_ids_per_groupby = df.groupby(['batch_size', 'num_cells', 'num_layers']).trial_id.unique()
        trials_ids_to_remove = []
        for trials_ids in trials_ids_per_groupby:
            trials_ids_to_remove += list(trials_ids)[1:]
        df = df.query('trial_id not in @trials_ids_to_remove')
        bb_instance_type.df = df
        bb_instance_type.index_cols = ['batch_size', 'num_cells', 'num_layers', 'step']

        bb_instance_type.metric_cols = ['st_instance_type']
        self.bb_instance_type = add_surrogate(bb_instance_type, surrogate=KNeighborsClassifier(n_neighbors=1), )

        bb_training_runtime = copy.deepcopy(bb_instance_type)
        bb_training_runtime.metric_cols = ['metric_train_runtime']
        self.bb_training_runtime = add_surrogate(bb_training_runtime, surrogate=KNeighborsRegressor(n_neighbors=1), )

        # Set up KNN-surrogate-based Blackbox for the speed measurements
        dfs_to_concat = list()
        trial_id_max = -1
        for tuner_job_name in SPEED_SYNE_TUNE_JOB_NAMES:
            df_temp = syne_tune.experiments.load_experiment(tuner_job_name).results
            df_temp['trial_id'] += trial_id_max + 1
            trial_id_max = df_temp['trial_id'].max()
            dfs_to_concat.append(df_temp)
        df = pd.concat(dfs_to_concat).reset_index()

        columns_to_rename = {k: k.replace('config_', '') for k in df.columns if k.startswith('config_')}
        columns_to_rename.update({
            'st_worker_time': 'metric_train_runtime',
        })
        df = df.rename(columns=columns_to_rename)

        configuration_space_speed = copy.deepcopy(self.configuration_space)
        configuration_space_speed.pop('lr')
        configuration_space_speed = {
            k: v for k, v in configuration_space_speed.items()
            if k not in ['lr']}
        # configuration_space_speed.update({
        #     'st_instance_type': cs.choice(self.configuration_space['st_instance_type']),
        # })
        speed_bb = BlackboxOffline(
                df_evaluations=df,
                configuration_space=configuration_space_speed,
                objectives_names=["metric_train_runtime"],
            )
        # TODO think about the below
        # The blackbox below is currently interpolating between categorical st_instance_type variable.
        # It might make sense to either create separate blackboxes for each instance type (such that the interpolation
        # is happening only within a single st_instance_type value, or specify a tailored distance function for the
        # KNeighborsRegressor object that prevents interpolation across instance types.
        self.speed_bb = add_surrogate(speed_bb, surrogate=KNeighborsRegressor(n_neighbors=3, weights='distance'),)

        # self.bb_instance_type.objective_function({'batch_size': 32, 'num_cells': 18, 'num_layers': 2})
        # self.bb_training_runtime.objective_function({'batch_size': 32, 'num_cells': 18, 'num_layers': 2})
        # self.speed_bb.objective_function({'st_instance_type': 'ml.m5.xlarge', 'batch_size': 32, 'num_cells': 18, 'num_layers': 2})

    def instance_speed(self, configuration: Dict, baseline_instance_type: str, fidelity: Union[Dict, Number] = None) -> float:
        """
        :param configuration: The configuration queried.
        :param baseline_instance_type: The baseline instance type that was used to collect the loss function curve for
          a particular configuration.
        :param fidelity:
        :return: relative-time of training , cost-per-second of the chosen instance type).
        """
        configuration_baseline = configuration.copy()
        configuration_baseline['st_instance_type'] = baseline_instance_type
        target_time = self.speed_bb.objective_function(configuration=configuration, fidelity=fidelity)['metric_train_runtime']
        baseline_time = self.speed_bb.objective_function(configuration=configuration_baseline, fidelity=fidelity)['metric_train_runtime']
        output = target_time/baseline_time

        return output

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
        if 'st_instance_type' not in configuration:
            raise ValueError(f'No st_instance_type provided in the configuration: {configuration}')
        target_instance_type = configuration['st_instance_type']

        res = self.bb.objective_function(configuration=configuration, fidelity=fidelity, seed=seed)
        learning_curve = res[:, 0:1]
        baseline_runtimes = self.bb_training_runtime.objective_function(
            configuration=configuration, fidelity=fidelity, seed=seed)
        baseline_instance_type_array = self.bb_instance_type.objective_function(
            configuration=configuration, fidelity=fidelity, seed=seed)
        assert len(np.unique(baseline_instance_type_array)) == 1
        baseline_instance_type = np.unique(baseline_instance_type_array)[0]
        relative_time_factor = self.instance_speed(configuration, baseline_instance_type, fidelity)

        target_runtimes = baseline_runtimes * relative_time_factor

        cost_per_second = self.instance_info(target_instance_type).cost_per_hour / 60. / 60.

        if fidelity is not None:
            return {
                METRIC_VALID_ERROR: learning_curve,
                "metric_train_runtime": target_runtimes,
                "metric_cost": target_runtimes * cost_per_second,
            }
        else:
            return np.hstack([learning_curve, target_runtimes, target_runtimes * cost_per_second])


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
        # 'mean_wQuantileLoss': 'metric_training_loss',
        'st_worker_time': 'metric_train_runtime',
    })
    df = df.rename(columns=columns_to_rename)

    configuration_space = {
        "lr": cs.loguniform(1e-4, 1e-1),
        "batch_size": cs.logfinrange(8, 128, 5, cast_int=True),  # cs.choice([8, 16, 32, 64, 128]),
        "num_cells": cs.randint(lower=1, upper=200),
        "num_layers": cs.randint(lower=1, upper=4),
        "st_instance_type": cs.choice(INSTANCE_TYPES),
    }

    # We set the maximum fidelity to be the minimum final fidelity across all learning curves gathered.
    df['step'] = df.st_worker_iter + 1
    Nmax = df.reset_index().groupby('trial_id').step.max().min()
    Nmin = df.reset_index().groupby('trial_id').step.min().min()
    fidelity_values = list(range(Nmin, Nmax + 1))
    fidelity_space = dict(
        step=cs.finrange(Nmin, Nmax, (Nmax - Nmin) + 1, cast_int=True),
    )

    serialize(
        {
            'electricity': BlackboxOffline(
                df_evaluations=df,
                configuration_space=configuration_space,
                fidelity_space=fidelity_space,
                fidelity_values=fidelity_values,
                objectives_names=[METRIC_VALID_ERROR] + [col for col in df.columns if col.startswith("metric_")],
            )
        },
        path=repository_path / BLACKBOX_NAME
    )


def import_deepar_cloud():
    bb = load("deepar-cloud")
    bb_dict = {'electricity': DeepARCloudBlackbox(bb=bb['electricity'])}
    return bb_dict


def generate_deepar_cloud(s3_root: Optional[str] = None):
    serialize_deepar_cloud()
    upload(name=BLACKBOX_NAME, s3_root=s3_root)


if __name__ == '__main__':
    generate_deepar_cloud()
