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
from pathlib import Path
import pytest

import numpy as np

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune import Tuner
from syne_tune.search_space import uniform
from syne_tune import StoppingCriterion


if __name__ == '__main__':
    max_steps = 27
    n_workers = 4

    config_space = {
        "steps": max_steps,
        "theta": uniform(0, np.pi / 2),
        "sleep_time": 0.01,
    }

    entry_point = Path(__file__).parent.parent / "examples" / "training_scripts" / "mo_artificial" / "mo_artificial.py"
    trial_backend = LocalBackend(entry_point=str(entry_point))

    # Multi objective scalarization BayesOpt
    searcher = 'mo_scalar_bayesopt'
    # When using mo_scalar_bayesopt searcher, for now we need to set metric to 'scalarization', it's a dummy value that
    #  is ignored in pracitce.
    # TODO this is not a good design choice but good enough for now.
    metric = 'scalarization'
    search_options = {
        'num_init_random': n_workers,
        # Since FIFOScheduler takes only one metric, we provide the names of multiple metrics in search_options.
        'metrics': ['y1', 'y2'],
    }
    stop_criterion = StoppingCriterion(max_wallclock_time=30)

    myscheduler = FIFOScheduler(
        config_space,
        searcher=searcher,
        metric=metric,  # TODO this is not a good design choice but good enough for now
        search_options=search_options,
        mode='min',
    )

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=myscheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
    )

    tuner.run()
