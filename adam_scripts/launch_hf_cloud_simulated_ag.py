import logging
import numpy as np
import pandas as pd
import syne_tune.search_space as sp

from benchmarking.blackbox_repository import load, add_surrogate
from benchmarking.blackbox_repository.blackbox_tabular import BlackboxTabular
from benchmarking.blackbox_repository.conversion_scripts.scripts.hf_cloud import import_hf_cloud
from benchmarking.blackbox_repository.simulated_tabular_backend import BlackboxRepositoryBackend, UserBlackboxBackend

from sklearn.neighbors import KNeighborsRegressor

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

    n_workers = 1

    # It is important to set `sleep_time` to 0 here (mandatory for simulator backend)
    tuner = Tuner(
        trial_backend=backend,
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
    bb_dict = import_hf_cloud()
    blackbox = bb_dict["imdb"]

    config = {
        'per_device_train_batch_size': 10,
        'weight_decay': 0.0001,
        'learning_rate': 0.000003,
    }
    for instance_type in ['ml.g5.xlarge', 'ml.g4dn.8xlarge']:
        config['instance_type'] = instance_type
        print(f"performance for {instance_type}: {blackbox(configuration=config, fidelity=199, seed=0)}")

    # simulating HPO
    n_workers = 1
    metric = "metric_training_loss"
    elapsed_time_attr = 'metric_train_runtime'

    backend = UserBlackboxBackend(
        blackbox=blackbox,
        elapsed_time_attr=elapsed_time_attr,
    )
    simulate_benchmark(backend=backend, metric=metric)
