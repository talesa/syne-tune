import logging

from blackbox_repository import load, add_surrogate
from blackbox_repository.blackbox_tabular import BlackboxTabular
from blackbox_repository.tabulated_benchmark import TabulatedBenchmark

from sagemaker_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from sagemaker_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from sagemaker_tune.stopping_criterion import StoppingCriterion
from sagemaker_tune.tuner import Tuner


def example_blackbox():
    import numpy as np
    import pandas as pd
    import sagemaker_tune.search_space as sp
    n = 100
    x1 = np.arange(n)
    x2 = np.arange(n)[::-1]
    hyperparameters = pd.DataFrame(data=np.stack([x1, x2]).T, columns=["hp_x1", "hp_x2"])
    config_space = {
        "hp_x1": sp.randint(0, n),
        "hp_x2": sp.randint(0, n),
    }
    n_epochs = 100
    cs_fidelity = {
        'hp_epoch': sp.randint(0, n_epochs),
    }
    num_seeds = 1
    num_objectives = 3

    objectives_evaluations = np.random.rand(
        len(hyperparameters),
        num_seeds,
        n_epochs,
        num_objectives
    )
    return add_surrogate(BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=config_space,
        fidelity_space=cs_fidelity,
        objectives_evaluations=objectives_evaluations,
        objectives_names=["metric_error", "runtime", "gpu_usage"]
    ))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARNING)

    n_workers = 4

    blackbox_name, dataset, metric, elapsed_time_attr = "nas201", "cifar100", "metric_error", 'metric_runtime'
    blackbox = load(blackbox_name)[dataset]

    # blackbox_name, dataset, metric, elapsed_time_attr = "fcnet", "protein_structure", "metric_valid_loss", "metric_runtime"
    # blackbox = load(blackbox_name)[dataset]

    # blackbox = example_blackbox()
    # elapsed_time_attr, metric = "runtime", "metric_error"

    backend = TabulatedBenchmark(
        blackbox=blackbox,
        elapsed_time_attr=elapsed_time_attr,
    )

    # Random search without stopping
    scheduler = HyperbandScheduler(
        backend.blackbox.configuration_space,
        searcher="random",
        max_t=max(backend.fidelities),
        resource_attr=backend.ressource_attr,
        mode='min',
        metric=metric,
        random_seed=31415927
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=72000)

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