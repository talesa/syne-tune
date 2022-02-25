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

from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import TrialEvaluations, Configuration, MetricValues, \
    dictionarize_objective, INTERNAL_METRIC_NAME, INTERNAL_COST_NAME, \
    ConfigurationFilter

from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import GPFIFOSearcher

logger = logging.getLogger(__name__)


class MOScalarGPFIFOSearcher(GPFIFOSearcher):
    """Multiobjective Scalarization GPFIFOSearcher"""
    def __init__(self, metrics: List[str], **kwargs):
        super().__init__(metric='scalarized_objective', **kwargs)
        self._metrics = metrics
        # TODO add a way to initialize weigths
        self._weights = [1. for _ in range(len(self._metrics))]

    # TODO the logic below pretty much duplicated between MOScalarGPFIFOSearcher and MOScalarGPFIFOScheduler
    def _scalarize_objectives(self, result: Dict):
        return sum(weight * objective_value for weight, objective_value in
                   zip(self._weights, [result[k] for k in self._metrics]))

    def _update(self, trial_id: str, config: Dict, result: Dict):
        result[self._metric] = self._scalarize_objectives(result)
        super()._update(trial_id, config, result)
