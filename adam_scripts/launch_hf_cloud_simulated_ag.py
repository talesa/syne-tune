import logging
import numpy as np
import pandas as pd
import syne_tune.config_space as sp

from benchmarking.blackbox_repository import load, add_surrogate
from benchmarking.blackbox_repository.blackbox_tabular import BlackboxTabular
from benchmarking.blackbox_repository.conversion_scripts.scripts.hf_cloud import import_hf_cloud
from benchmarking.blackbox_repository.simulated_tabular_backend import BlackboxRepositoryBackend, UserBlackboxBackend

from sklearn.neighbors import KNeighborsRegressor

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import ASHA, RandomSearch
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

import syne_tune.experiments


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # loading data and querying stuff
    bb_dict = import_hf_cloud()
    blackbox = bb_dict["imdb"]

    # simulating HPO
    n_workers = 1
    metric = "metric_training_loss"
    elapsed_time_attr = 'metric_train_runtime'

    backend = UserBlackboxBackend(
        blackbox=blackbox,
        elapsed_time_attr=elapsed_time_attr,
    )

    backend.blackbox.configuration_space['per_device_train_batch_size'] = sp.choice([4, 8, 12, 16])

    # Random search without stopping
    # scheduler = RandomSearch(
    #     backend.blackbox.configuration_space,
    #     # max_t=max(blackbox.fidelity_values),
    #     # resource_attr=next(iter(blackbox.fidelity_space.keys())),
    #     mode='min',
    #     metric=metric,
    #     random_seed=31415927
    # )

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
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps.
        callbacks=[SimulatorCallback()],
    )
    tuner.run()

    print(f"dfff = syne_tune.experiments.load_experiment('{tuner.name}')")

