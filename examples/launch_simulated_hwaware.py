import logging
import numpy as np
import pandas as pd
import syne_tune.search_space as sp

from blackbox_repository import load, add_surrogate
from blackbox_repository.blackbox_tabular import BlackboxTabular
from blackbox_repository.conversion_scripts.scripts.nasbench201_cloud import generate_nas201_cloud
from blackbox_repository.tabulated_benchmark import BlackboxRepositoryBackend, UserBlackboxBackend

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import ASHA
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner


def simulate_benchmark(backend, metric):
    # Random search without stopping
    scheduler = ASHA(
        backend.blackbox.configuration_space,
        max_t=max(blackbox.fidelity_values),
        resource_attr=next(iter(blackbox.fidelity_space.keys())),
        mode='min',
        metric=metric,
        random_seed=31415927
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=7200)

    # It is important to set `sleep_time` to 0 here (mandatory for simulator backend)
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )
    tuner.run()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # loading data and querying stuff
    bb_dict = generate_nas201_cloud()
    blackbox = bb_dict["cifar10"]
    config = {f'hp_x{i}': 'avg_pool_3x3' for i in range(6)}
    for instance_type in ['ml.c5.2xlarge', 'ml.g4dn.8xlarge']:
        config['instance_type'] = instance_type
        print(f"performance for {instance_type}: {blackbox(configuration=config, fidelity=200, seed=0)}")

    # simulating HPO
    n_workers = 4
    metric = "metric_valid_error"
    time_this_resource_attr = 'metric_runtime'
    elapsed_time_attr = 'metric_elapsed_time'

    backend = UserBlackboxBackend(
        blackbox=blackbox,
        elapsed_time_attr="runtime",
        time_this_resource_attr=time_this_resource_attr,
    )
    simulate_benchmark(backend=backend, metric=metric)
