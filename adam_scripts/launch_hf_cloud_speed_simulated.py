import logging
from argparse import ArgumentParser

from syne_tune.blackbox_repository.conversion_scripts.scripts.hf_cloud_speed import import_hf_cloud_speed
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.optimizer.schedulers.multiobjective.botorch_mo_gp import BotorchMOGP
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

import tqdm

import torch
import random
import numpy as np
import botorch

fixed_seed = False
if fixed_seed:
    seed = 1
    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    botorch.utils.sampling.manual_seed(seed=seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--features', nargs='+', required=True)
    parser.add_argument('-i', '--iters', type=int, default=10)
    parser.add_argument('-mc', '--max_cost', type=float, default=20.)
    parser.add_argument('-dt', '--deterministic_transform', type=int, default=0)
    parser.add_argument('-s', '--searcher', type=str, default='mobo')

    args, _ = parser.parse_known_args()

    logging.getLogger().setLevel(logging.CRITICAL)

    # loading data and querying stuff
    bb_dict = import_hf_cloud_speed()
    blackbox = bb_dict["imdb"]

    # simulating HPO
    n_workers = 1

    elapsed_time_attr = 'st_worker_time'

    if args.searcher not in ['random', 'mobo']:
        raise ValueError(f"Unknown setting for --searcher: {args.searcher}. Should be one of: random, mobo.")
    if args.searcher == 'random':
        assert args.deterministic_transform == 0, "If args.searcher==random, args.deterministic_transform should be 0."

    print(args.features)

    tuners_names = []
    for i in tqdm.trange(args.iters):
        backend = UserBlackboxBackend(
            blackbox=blackbox,
            elapsed_time_attr=elapsed_time_attr,
        )

        if args.searcher == 'mobo':
            scheduler = BotorchMOGP(
                config_space=blackbox.configuration_space,
                mode='min',
                metrics=blackbox.metrics,
                ref_point=[5., 5.],  # since the objectives are standardized (mean=0, std=1)
                features=args.features,
                deterministic_transform=args.deterministic_transform,
            )
        elif args.searcher == 'random':
            scheduler = RandomSearch(
                config_space=backend.blackbox.configuration_space,
                mode='min',
                metric='training-cost-per-sample',
            )

        stop_criterion = StoppingCriterion(
            max_cost=args.max_cost,
            max_num_trials_finished=168,
        )

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
        tuners_names.append(tuner.name)

    print(f'({args.features}, {tuners_names})')
