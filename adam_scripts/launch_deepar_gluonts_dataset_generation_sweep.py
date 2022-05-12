# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
This launches an HPO tuning several hyperparameters of a gluonts model.
To run this example locally, you need to have installed dependencies in `requirements.txt` in your current interpreter.
"""
import logging
from pathlib import Path

import numpy as np

from sagemaker.mxnet import MXNet

import syne_tune
from syne_tune.backend import LocalBackend, SageMakerBackend
from syne_tune.backend.sagemaker_backend.instance_info import select_instance_type
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.optimizer.baselines import ASHA, RandomSearch
from syne_tune import Tuner, StoppingCriterion
import syne_tune.config_space as cs
from syne_tune.remote.remote_launcher import RemoteLauncher

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # np.random.seed(0)

    # instance_types = select_instance_type(min_gpu=0, max_cost_per_hour=20.0)
    instance_types = [
        # CPU
        'ml.c4.2xlarge',
        'ml.c4.4xlarge',
        'ml.c4.8xlarge',
        'ml.c4.xlarge',
        'ml.c5.18xlarge',
        'ml.c5.2xlarge',
        'ml.c5.4xlarge',
        'ml.c5.9xlarge',
        'ml.c5.xlarge',
        'ml.c5n.18xlarge',
        'ml.c5n.2xlarge',
        'ml.c5n.4xlarge',
        'ml.c5n.9xlarge',
        'ml.c5n.xlarge',
        'ml.m4.10xlarge',
        'ml.m4.16xlarge',
        'ml.m4.2xlarge',
        'ml.m4.4xlarge',
        'ml.m4.xlarge',
        'ml.m5.12xlarge',
        'ml.m5.24xlarge',
        'ml.m5.2xlarge',
        'ml.m5.4xlarge',
        'ml.m5.large',
        'ml.m5.xlarge',

        # GPUs
        # 'ml.g4dn.xlarge',
        # # 'ml.g4dn.2xlarge',
        # # 'ml.g4dn.4xlarge',
        # # 'ml.g4dn.8xlarge',
        # 'ml.g4dn.12xlarge',
        # # 'ml.g4dn.16xlarge',
        #
        # 'ml.g5.xlarge',
        # # 'ml.g5.2xlarge',
        # # 'ml.g5.4xlarge',
        # # 'ml.g5.8xlarge',
        # 'ml.g5.12xlarge',
        # # 'ml.g5.16xlarge',
        # # 'ml.g5.24xlarge',
        # 'ml.g5.48xlarge',
        #
        # 'ml.p2.xlarge',
        # 'ml.p2.8xlarge',
        # 'ml.p2.16xlarge',
        #
        # 'ml.p3.2xlarge',
        # 'ml.p3.8xlarge',
    ]

    config_space = {
        "lr": 1e-4,
        "epochs": 100,
        "num_cells": cs.choice([1, 50, 100, 150, 200]),
        "num_layers": cs.choice([1, 2, 3, 4]),
        "batch_size": 32,
        "dataset": "electricity",
        "st_instance_type": cs.choice(instance_types),
        "only_benchmark_speed": 1,
    }
    tuner_name = f'deepar-speed-bs-{config_space["batch_size"]}'

    # config_space = {
    #     "epochs": 100,
    #     "lr": cs.loguniform(1e-4, 1e-1),
    #     # "batch_size": cs.choice([8, 16, 32, 64, 128]),
    #     "num_cells": cs.randint(lower=1, upper=200),
    #     "num_layers": cs.randint(lower=1, upper=4),
    #     # "batch_size": cs.choice([32, 64, 128]),
    #     "batch_size": cs.choice([8, 16]),
    #     # "batch_size": cs.choice([64, 128]),
    #     "dataset": "electricity",
    #     "st_instance_type": 'ml.c5.xlarge',
    #     "only_benchmark_speed": 0,
    # }
    # tuner_name = 'deepar-curves-6'

    n_workers = 1
    # n_workers = int(len(instance_types) * 20 / 2)

    wallclock_time_budget = 3600 * 128
    dollar_cost_budget = 400.0
    max_num_trials_completed = 40
    # max_num_trials_started = 1

    stop_criterion = StoppingCriterion(
        # max_wallclock_time=wallclock_time_budget,
        # max_cost=dollar_cost_budget,
        # max_num_trials_completed=max_num_trials_completed,
        # max_num_trials_started=max_num_trials_started,
    )

    mode = "min"
    metric = "mean_wQuantileLoss"
    entry_point = Path(__file__).parent.parent / "examples" / "training_scripts" / "gluonts" / "train_gluonts.py"

    trial_backend = SageMakerBackend(
        sm_estimator=MXNet(
            entry_point=entry_point.name,
            source_dir=str(entry_point.parent),
            instance_type="ml.m5.large",
            instance_count=1,
            role=get_execution_role(),
            max_run=60 * 60 * 72,
            framework_version='1.7',
            py_version='py3',
            base_job_name='hpo-gluonts',
            disable_profiler=True,
            debugger_hook_config=False,
        ),
        # names of metrics to track. Each metric will be detected by Sagemaker if it is written in the
        # following form: "[RMSE]: 1.2", see in train_main_example how metrics are logged for an example
        metrics_names=[metric],
        # inputs={'train': 's3://sagemaker-us-west-2-640549960621/gluon-ts/datasets'},
        inputs={'train': 's3://mnemosyne-team-bucket/AdamG/sweeps_resources/gluonts/datasets'},
    )

    scheduler = RandomSearch(
        config_space,
        mode='min',
        metric=metric
    )

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        # stops if wallclock time or dollar-cost exceeds budget,
        # dollar-cost is only available when running on Sagemaker
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        # some failures may happen when SGD diverges with NaNs
        max_failures=10000000,
        tuner_name=tuner_name,
        trial_throttling_sleep_time=15.0,
    )

    # launch the tuning
    tuner.run()

    # root = Path(syne_tune.__path__[0]).parent
    # remote_launcher = RemoteLauncher(
    #     tuner=tuner,
    #     # instance_type='ml.g4dn.xlarge',
    #     instance_type='ml.m5.large',
    #     tuner_name=tuner_name,
    #     dependencies=[str(root / "benchmarking"), str(root / "syne_tune")],
    # )
    # remote_launcher.run(wait=False)
