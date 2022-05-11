import copy
from numbers import Number
from typing import Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from syne_tune.blackbox_repository import add_surrogate
from syne_tune.blackbox_repository.blackbox import Blackbox
import syne_tune.config_space as cs
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.blackbox_repository.blackbox_offline import BlackboxOffline

from adam_scripts.utils import (
    concatenate_syne_tune_experiment_results,
    upload_df_to_team_bucket,
)


METRIC_VALID_ERROR = 'mean_wQuantileLoss'

# This is cumulative time required for consuming the resource up until this point in training.
METRIC_TIME_CUMULATIVE_RESOURCE = 'metric_train_runtime'

LEARNING_CURVE_SYNE_TUNE_JOB_NAMES = (
    'deepar-curves-2022-04-25-10-46-32-616',
    'deepar-curves-2-2022-04-25-12-31-30-154',
    'deepar-curves-3-2022-04-25-12-32-14-346',
    'deepar-curves-4-2022-05-04-08-53-23-418',
#     # There was no deepar-curves-5
    'deepar-curves-6-2022-05-04-08-54-35-102',
)
SPEED_SYNE_TUNE_JOB_NAMES = (
    'deepar-speed-bs-128-2022-05-06-11-50-30-088',
    'deepar-speed-bs-64-2022-05-06-11-45-25-787',
    'deepar-speed-bs-32-2022-05-10-10-07-22-265',
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

BLACKBOX_SPEED_S3_PATH = ('s3://mnemosyne-team-bucket/dataset/'
                          'deepar-on-electricity-blackbox/deepar-on-electricity-blackbox-speed.csv.zip')
BLACKBOX_ERROR_S3_PATH = ('s3://mnemosyne-team-bucket/dataset/'
                          'deepar-on-electricity-blackbox/deepar-on-electricity-blackbox-error.csv.zip')


class DeepARCloudBlackbox(Blackbox):
    """
    Dataset generated using adam_scripts/launch_deepar_gluonts_dataset_generation_sweep.py
    """
    def __init__(self, reload_from_syne_tune_reports: bool = False):
        """
            :param: reload_from_syne_tune_reports: when True the data is reloaded from *_SYNE_TUNE_JOB_NAMES sources
            and uploaded to the Syne team S3 bucket at BLACKBOX_*_S3_PATH
        """
        # In __init__ we set up 3 surrogates:
        # 1. Blackbox for the error, KNN(n_neighbors=3)
        # 2. Blackbox for the baseline training_runtime and instance_type, KNN(n_neighbors=1)
        # 3. Blackbox for the training_runtime correction for the target instance_type, KNN(n_neighbors=2)

        # 1. Blackbox for the error, KNN(n_neighbors=3)
        if reload_from_syne_tune_reports:
            df = concatenate_syne_tune_experiment_results(LEARNING_CURVE_SYNE_TUNE_JOB_NAMES)
            upload_df_to_team_bucket(df, BLACKBOX_ERROR_S3_PATH)
        df = pd.read_csv(BLACKBOX_ERROR_S3_PATH)

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

        objectives_names = [METRIC_VALID_ERROR, 'metric_train_runtime', "metric_cost"]
        super(DeepARCloudBlackbox, self).__init__(
            configuration_space=configuration_space,
            fidelity_space=fidelity_space,
            fidelity_values=fidelity_values,
            objectives_names=objectives_names,
        )

        bb_error = BlackboxOffline(
                df_evaluations=df,
                configuration_space=configuration_space,
                fidelity_space=fidelity_space,
                fidelity_values=fidelity_values,
                objectives_names=[METRIC_VALID_ERROR] + [col for col in df.columns if col.startswith("metric_")],
            )

        self.bb_error = add_surrogate(bb_error, surrogate=KNeighborsRegressor(n_neighbors=2, weights='distance'), )

        # 2. Blackbox for the baseline training_runtime and instance_type, KNN(n_neighbors=1)
        # TODO rewrite this properly creating Blackbox objects rather than hacking it like this
        # Prepare the two blackboxes, self.bb_instance_type and self.bb_training_runtime, which return the
        # st_instance_type the metric_training_runtime for consecutive fidelities.
        # We need these such that we can then adjust the st_worker_time/train_runtime using information from the
        # speed-benchmarking-runs.
        bb_instance_type = copy.deepcopy(bb_error)
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

        # 3. Blackbox for the training_runtime correction for the target instance type, KNN(n_neighbors=2)
        if reload_from_syne_tune_reports:
            df = concatenate_syne_tune_experiment_results(SPEED_SYNE_TUNE_JOB_NAMES)
            upload_df_to_team_bucket(df, BLACKBOX_SPEED_S3_PATH)
        df = pd.read_csv(BLACKBOX_SPEED_S3_PATH)

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
        self.bb_speed = add_surrogate(speed_bb, surrogate=KNeighborsRegressor(n_neighbors=3, weights='distance'), )

        # self.bb_instance_type.objective_function({'batch_size': 32, 'num_cells': 18, 'num_layers': 2})
        # self.bb_training_runtime.objective_function({'batch_size': 32, 'num_cells': 18, 'num_layers': 2})
        # self.speed_bb.objective_function({'st_instance_type': 'ml.m5.xlarge', 'batch_size': 32, 'num_cells': 18, 'num_layers': 2})

        self.instance_info = InstanceInfos()

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
        target_time = self.bb_speed.objective_function(configuration=configuration, fidelity=fidelity)['metric_train_runtime']
        baseline_time = self.bb_speed.objective_function(configuration=configuration_baseline, fidelity=fidelity)['metric_train_runtime']
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

        res = self.bb_error.objective_function(configuration=configuration, fidelity=fidelity, seed=seed)
        learning_curve = res[:, 0:1]
        # TODO check what parameters are passed here
        baseline_runtimes = self.bb_training_runtime.objective_function(
            configuration=configuration, fidelity=fidelity, seed=seed)
        baseline_instance_type_array = self.bb_instance_type.objective_function(
            configuration=configuration, fidelity=fidelity, seed=seed)
        assert len(np.unique(baseline_instance_type_array)) == 1
        assert res.shape[0] == 100
        assert baseline_runtimes.shape[0] == 100
        assert baseline_instance_type_array.shape[0] == 100
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


def import_deepar_cloud():
    bb_dict = {'electricity': DeepARCloudBlackbox()}
    return bb_dict
