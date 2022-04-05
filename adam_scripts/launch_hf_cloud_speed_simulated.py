import logging
from pathlib import Path

import numpy as np
import pandas as pd
import syne_tune.config_space as sp

from benchmarking.blackbox_repository import load, add_surrogate
from benchmarking.blackbox_repository.blackbox_tabular import BlackboxTabular
from benchmarking.blackbox_repository.conversion_scripts.scripts.hf_cloud_speed import import_hf_cloud_speed
from benchmarking.blackbox_repository.simulated_tabular_backend import BlackboxRepositoryBackend, UserBlackboxBackend

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import ASHA, RandomSearch
from syne_tune.optimizer.schedulers.multiobjective.botorch_mo_gp import BotorchMOGP
from syne_tune.remote.remote_launcher import RemoteLauncher
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

import tqdm

import syne_tune.experiments


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.CRITICAL)

    # loading data and querying stuff
    bb_dict = import_hf_cloud_speed()
    blackbox = bb_dict["imdb"]

    # simulating HPO
    n_workers = 1

    elapsed_time_attr = 'st_worker_time'

    temp = []
    for i in tqdm.trange(100):
        backend = UserBlackboxBackend(
            blackbox=blackbox,
            elapsed_time_attr=elapsed_time_attr,
        )

        scheduler = BotorchMOGP(
            config_space=backend.blackbox.configuration_space,
            mode='min',
            metrics=blackbox.metrics,
            ref_point=blackbox.ref_point,
        )

        stop_criterion = StoppingCriterion(max_cost=90.)

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
        temp.append(tuner.name)

    print(temp)

    # tuner_job_name = 'test-hf-cloud-speed-remotelauncher'
    # root = Path(syne_tune.__path__[0]).parent
    # remote_launcher = RemoteLauncher(
    #     tuner=tuner,
    #     instance_type='ml.m5.large',
    #     tuner_name=tuner_job_name,
    #     dependencies=[str(root / "benchmarking")],
    #     sleep_time=0.0,
    # )
    # remote_launcher.run(wait=False)
    #
    # print(f"dfff = syne_tune.experiments.load_experiment('{tuner.name}')")

