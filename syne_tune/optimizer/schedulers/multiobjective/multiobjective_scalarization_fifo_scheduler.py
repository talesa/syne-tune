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
import numpy as np
from typing import Type, Optional, Dict, List
import logging
import copy
import time

from syne_tune.backend.trial_status import Trial

from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

logger = logging.getLogger(__name__)


# TODO this implementation scalarizes the objective before it is presented to the GP, and the GP is only learning
#  about the scalarized which is clearly wrong
class WrongMOScalarFIFOScheduler(FIFOScheduler):
    """Multiobjective Scalarization FIFOScheduler"""

    def __init__(self, config_space: Dict, metrics: List[str], **kwargs):
        """
        Available scalarization_method: "fixed-ones", "fixed", "random-weights", "golovin".
        Default scalarization_method: "golovin".

        If you use scalarization_method "fixed", you need to provide a field scalarization_weights inside argument
        search_options which is a List[float] of the same length as the argument metrics.
        """
        self._metrics = metrics
        # TODO add a way to initialize weigths
        self._weights = [1. for _ in range(len(self._metrics))]

        searcher = kwargs['searcher']
        # assert searcher in ['bayesopt', 'random'], 'Searcher can only be bayesopt or random.'
        if isinstance(searcher, str):  # I know this is redundant at the moment
            search_options = kwargs.get('search_options')
            if search_options is None:
                search_options = dict()
            else:
                search_options = search_options.copy()
            search_options.update({
                'metrics': self._metrics,
                'scheduler': 'mofifo',
            })
            if 'scalarization_method' not in search_options:
                search_options['scalarization_method'] = 'golovin'

        super().__init__(config_space, metric='scalarized_objective', **kwargs)

    def metric_names(self) -> List[str]:
        # TODO what to do in this context?
        #  https://github.com/awslabs/syne-tune/blob/1245d185340e54c5e6af05544cf08f70b750a0f5/syne_tune/experiments.py#L240-L245
        return self._metrics

    # TODO the logic below pretty much duplicated between MOScalarGPFIFOSearcher and MOScalarGPFIFOScheduler
    def _scalarize_objectives(self, result: Dict, scalarization_method: str=None):
        return sum(weight * objective_value for weight, objective_value in
                   zip(self._weights, [result[k] for k in self._metrics]))

    def on_trial_result(self, trial: Trial, result: Dict) -> str:
        for key in self._metrics:
            self._check_key_of_result(result, key)
        result[self.metric] = self._scalarize_objectives(result)
        return super().on_trial_result(trial, result)