from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from benchmarking.blackbox_repository import load, BlackboxOffline, add_surrogate, serialize
from benchmarking.blackbox_repository.blackbox import Blackbox
import syne_tune.search_space as sp
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.util import s3_experiment_path
from benchmarking.blackbox_repository.blackbox_offline import serialize, BlackboxOffline
from benchmarking.blackbox_repository.conversion_scripts.utils import repository_path, upload
import syne_tune.experiments


METRIC_VALID_ERROR = 'metric_training_loss'
METRIC_TIME_THIS_RESOURCE = 'metric_train_runtime'
# RESOURCE_ATTR = 'hp_epoch'
BASELINE_INSTANCE_TYPE = 'ml.g4dn.xlarge'

SOURCE_SYNE_TUNE_JOB_NAMES = (
    # 'loss-lr-wd-bs-2022-02-04-12-09-26-911',  # Old schema, i.e. names of the fields, don't use.
    'loss-lr-wd-bs-2-2022-02-07-23-13-30-781',
)

BLACKBOX_NAME = 'hf-cloud'


class HFCloudBlackbox(Blackbox):
    """
    Dataset generated using examples/launch_huggingface_sweep_ag.py
    """
    def __init__(self, bb):
        self.instance_speed_cost_dict = instance_speed_cost(baseline_instance_type=BASELINE_INSTANCE_TYPE)
        self.configuration_space = bb.configuration_space
        self.configuration_space["instance_type"] = sp.choice(list(self.instance_speed_cost_dict.keys()))
        self.objectives_names = bb.objectives_names + ["cost"]
        # FIXME HACK
        # TODO N is the minimum value of any of the runs, so setting this ensures that none fail
        N = bb.df.reset_index().groupby('trial_id').step.max().min()
        bb.fidelity_values = list(range(1, N + 1))  # FIXME HACK
        super(HFCloudBlackbox, self).__init__(
            configuration_space=self.configuration_space,
            fidelity_space=bb.fidelity_space,
            fidelity_values=bb.fidelity_values,
            objectives_names=self.objectives_names,
        )
        self.bb = add_surrogate(bb,
                                surrogate=KNeighborsRegressor(n_neighbors=3),
                                hps_to_exclude=('instance_type',))

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
        if 'instance_type' not in configuration:
            raise ValueError(f'No instance_type provided in the configuration: {configuration}')
        instance_type = configuration.pop('instance_type')
        relative_time_factor, cost_per_second = self.instance_speed_cost_dict[instance_type]

        res = self.bb.objective_function(configuration=configuration, fidelity=fidelity, seed=seed)

        if fidelity is not None:
            adjusted_time = res[METRIC_TIME_THIS_RESOURCE] * relative_time_factor
            return {
                METRIC_VALID_ERROR: res[METRIC_VALID_ERROR],
                "metric_train_runtime": adjusted_time,
                "metric_cost": adjusted_time * cost_per_second,
            }
        else:
            index_time = [i for i, x in enumerate(self.objectives_names) if x == METRIC_TIME_THIS_RESOURCE][0]
            # add relative time
            res[:, index_time] *= relative_time_factor

            # add cost which runtime seconds time second cost
            cost_per_second = res[:, index_time:index_time+1] * cost_per_second
            res = np.hstack([res, cost_per_second])

            return res

    def hyperparameter_objectives_values(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # TODO somehow avoid redefining it this way and handle this via inheritance?
        return self.bb.hyperparameter_objectives_values()


def instance_speed_cost(baseline_instance_type: str, max_instance: int = None) -> Dict[str, Tuple[float, float]]:
    """
    :param type:
    :return: dictionary from instance to relative-time and cost per-second instance.
    """
    # gets instance dollar-cost
    instance_info = InstanceInfos()
    instance_hourly_cost = {instance: instance_info(instance).cost_per_hour for instance in instance_info.instances}

    # gets instance speed
    csv_path = Path(__file__).parent / f"hf-cloud-instance-speed.csv.zip"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(s3_experiment_path(tuner_name='speed-bs-it-2022-02-07-23-12-47-916') + '/train_runtime.csv')
        # df = syne_tune.experiments.load_experiment(tuner_job_name).results
        df.to_csv(csv_path)

    time_col = 'train_runtime'

    # gets time per batch relative to the time for a baseline instance
    # taking the shortest train_runtime per instance-type which most of the time corresponds to the largest batch_size
    # TODO account for different batch_sizes here
    relative_time_per_instance = df.groupby('config_st_instance_type')[time_col].min().sort_values()
    relative_time_per_instance /= relative_time_per_instance[baseline_instance_type]
    if max_instance is None:
        max_instance = len(relative_time_per_instance)
    return {
        # gets the cost per second
        instance: (relative_time_per_instance[instance], instance_hourly_cost[instance] / 3600.)
        for instance in relative_time_per_instance.index[:max_instance]
    }


def import_hf_cloud():
    bb = load("hf-cloud")
    bb_dict = {'imdb': HFCloudBlackbox(bb=bb['imdb'])}
    return bb_dict


def serialize_hf_cloud():
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

    # Changing steps to contiguous integers allows us to run multi-fidelity algorithms like ASHA easily.
    df.step = (df.step / 100).astype(np.int64)

    configuration_space = dict(
        per_device_train_batch_size=sp.choice([2, 4, 8, 12, 16]),
        learning_rate=sp.loguniform(1e-7, 1e-4),
        weight_decay=sp.loguniform(1e-6, 1e-2),
    )

    # TODO shouldn't fidelity_values be implied given fidelity_space?
    N = df.reset_index().groupby('trial_id').step.max().min()
    fidelity_values = list(range(1, N+1))
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


def generate_hf_cloud(s3_root: Optional[str] = None):
    serialize_hf_cloud()
    upload(name=BLACKBOX_NAME, s3_root=s3_root)


if __name__ == '__main__':
    generate_hf_cloud()
