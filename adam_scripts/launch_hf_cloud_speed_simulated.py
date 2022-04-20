import logging
from argparse import ArgumentParser

from syne_tune.blackbox_repository.conversion_scripts.scripts.hf_cloud_speed import import_hf_cloud_speed
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.multiobjective.botorch_mo_gp import BotorchMOGP
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--features', nargs='+', required=True)
    parser.add_argument('-i', '--iters', type=int, default=10)
    parser.add_argument('-mc', '--max_cost', type=float, default=20.)
    args, _ = parser.parse_known_args()

    logging.getLogger().setLevel(logging.CRITICAL)

    # loading data and querying stuff
    bb_dict = import_hf_cloud_speed()
    blackbox = bb_dict["imdb"]

    # simulating HPO
    n_workers = 1

    elapsed_time_attr = 'st_worker_time'

    print(args.features)

    tuners_names = []
    for i in tqdm.trange(args.iters):
        backend = UserBlackboxBackend(
            blackbox=blackbox,
            elapsed_time_attr=elapsed_time_attr,
        )

        scheduler = BotorchMOGP(
            config_space=blackbox.configuration_space,
            mode='min',
            metrics=blackbox.metrics,
            ref_point=[5., 5.],  # since the objectives are standardized (mean=0, std=1)
            features=args.features,
            deterministic_transform=False,
        )

        stop_criterion = StoppingCriterion(max_cost=args.max_cost)

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
