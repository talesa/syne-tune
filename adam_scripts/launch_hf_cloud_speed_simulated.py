import logging

from syne_tune.blackbox_repository.conversion_scripts.scripts.hf_cloud_speed import import_hf_cloud_speed
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.multiobjective.botorch_mo_gp import BotorchMOGP
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner

import tqdm

import syne_tune.experiments

import sagemaker
import boto3

from syne_tune.util import s3_experiment_path

if __name__ == '__main__':
    # boto3.setup_default_session(region_name='us-west-2')
    # tuner_name = 'simulated-tabular-backend-2022-04-11-17-33-57-906'
    # print(f's3_experiment_path(tuner_name=tuner_name): {s3_experiment_path(tuner_name=tuner_name)}')
    # print(syne_tune.experiments.load_experiment(tuner_name=tuner_name))

    logging.getLogger().setLevel(logging.CRITICAL)

    # loading data and querying stuff
    bb_dict = import_hf_cloud_speed()
    blackbox = bb_dict["imdb"]

    # simulating HPO
    n_workers = 1

    elapsed_time_attr = 'st_worker_time'

    # ('GPUFP32TFLOPS', 'cost_per_hour', 'num_cpu', 'num_gpu', 'GPUMemory', 'GPUFP32TFLOPS*num_gpu')
    # instance_type_features = ('GPUFP32TFLOPS*num_gpu', 'cost_per_hour')
    instance_type_features = ('GPUFP32TFLOPS',)
    # instance_type_features = []

    temp = []
    # try:
    for i in tqdm.trange(10):
        backend = UserBlackboxBackend(
            blackbox=blackbox,
            elapsed_time_attr=elapsed_time_attr,
        )

        scheduler = BotorchMOGP(
            config_space=blackbox.configuration_space,
            mode='min',
            metrics=blackbox.metrics,
            ref_point=[5., 5.],
            instance_type_features=instance_type_features if len(instance_type_features) > 0 else tuple(),
            deterministic_transform=False,
        )

        stop_criterion = StoppingCriterion(max_cost=50.)

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
    # except AssertionError as e:
    #     print(e)
    #     print(temp)
    #     print(tuner.name)

    print(tuner.name)
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