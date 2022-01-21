from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np

from blackbox_repository import load, Blackbox
import syne_tune.search_space as sp
from blackbox_repository.conversion_scripts.scripts.nasbench201_import import METRIC_VALID_ERROR, \
    METRIC_TIME_THIS_RESOURCE
from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos


class BlackboxModified(Blackbox):
    def __init__(self, bb):
        self.instance_speed_cost_dict = instance_speed_cost()
        self.configuration_space = bb.configuration_space
        self.configuration_space["instance_type"] = sp.choice(list(self.instance_speed_cost_dict.keys()))
        self.objectives_names = bb.objectives_names + ["cost"]
        self.bb = bb
        super(BlackboxModified, self).__init__(
            configuration_space=self.configuration_space,
            fidelity_space=bb.fidelity_space,
            fidelity_values=bb.fidelity_values,
            objectives_names=self.objectives_names,
        )

    def _objective_function(
            self,
            configuration: Dict,
            fidelity: Optional[Dict] = None,
            seed: Optional[int] = None
    ) -> Dict:
        instance_type = configuration['instance_type']
        res = self.bb.objective_function(configuration=configuration, fidelity=fidelity, seed=seed)
        relative_time, cost_per_second = self.instance_speed_cost_dict[instance_type]

        if fidelity is not None:
            adjusted_time = res[METRIC_TIME_THIS_RESOURCE] * relative_time
            return {
                "error": res[METRIC_VALID_ERROR],
                "runtime": adjusted_time,
                "cost": adjusted_time * cost_per_second,
            }
        else:
            index_time = [i for i, x in enumerate(self.objectives_names) if x == METRIC_TIME_THIS_RESOURCE][0]
            # add relative time
            res[:, index_time] *= relative_time

            # add cost which runtime seconds time second cost
            cost_per_second = res[:, index_time:index_time+1] * cost_per_second
            res = np.hstack([res, cost_per_second])

            return res


def instance_speed_cost(max_instance: int = None) -> Dict[str, Tuple[float, float]]:
    """
    :param type:
    :return: dictionary from instance to relative-time and cost per-second instance.
    """
    # gets instance dollar-cost
    instance_info = InstanceInfos()
    instance_hourly_cost = {instance: instance_info(instance).cost_per_hour for instance in instance_info.instances}

    # gets instance speed, relative to performance on p2.xlarge
    df = pd.read_csv(Path(__file__).parent / f"resnet-instance-speed.csv.zip")

    time_col = 'time'

    # gets time per batch relative to the time for a baseline instance
    baseline = "ml.p2.xlarge"
    # p2.xlarge are ~4 times slower than 1080ti that were used to acquire nas201
    df[time_col] *= 4
    relative_time_per_instance = df.groupby("InstanceType").mean()[time_col].sort_values()
    relative_time_per_instance /= relative_time_per_instance[baseline]
    if max_instance is None:
        max_instance = len(relative_time_per_instance)
    return {
        # gets the cost per second
        instance: (relative_time_per_instance[instance], instance_hourly_cost[instance] / 3600)
        for instance in relative_time_per_instance.index[:max_instance]
    }


def generate_nas201_cloud():
    bb_dict = load("nasbench201")
    bb_dict = {
        task: BlackboxModified(bb=bb)
        for task, bb in bb_dict.items()
    }
    return bb_dict