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

from syne_tune.optimizer.schedulers.searchers.cost_aware_gp_fifo_searcher import MultiModelGPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import \
    GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    cost_aware_gp_fifo_searcher_defaults, \
    cost_aware_coarse_gp_fifo_searcher_factory, \
    cost_aware_fine_gp_fifo_searcher_factory
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import \
    decode_state
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer \
    import ModelStateTransformer
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import INTERNAL_METRIC_NAME

from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import GPFIFOSearcher

# from syne_tune.optimizer.schedulers.searchers.cost_aware_gp_fifo_searcher \
#     import MultiModelGPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    constrained_gp_fifo_searcher_defaults, constrained_gp_fifo_searcher_factory
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import \
    decode_state
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import TrialEvaluations, INTERNAL_CONSTRAINT_NAME

logger = logging.getLogger(__name__)

__all__ = ['MOScalarGPFIFOSearcher']


# TODO this implementation scalarizes the objective before it is presented to the GP, and the GP is only learning
#  about the scalarized which is clearly wrong
class WrongMOScalarGPFIFOSearcher(GPFIFOSearcher):
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


# TODO for now just copied CostAwareGPFIFOSearcher
class MOScalarGPFIFOSearcher(MultiModelGPFIFOSearcher):
    """
    Gaussian process-based cost-aware hyperparameter optimization (to be used
    with a FIFO scheduler). The searcher requires a cost metric, which is
    given by `cost_attr`.

    Implements two different variants. If `resource_attr` is given, cost values
    are read from each report and cost is modeled as c(x, r), the cost model
    being given by `kwargs['cost_model']`.
    If `resource_attr` is not given, cost values are read only at the end (just
    like the primary metric) and cost is modeled as c(x), using a default GP
    surrogate model.

    """
    def __init__(self, configspace, metric, **kwargs):
        assert kwargs.get('cost_attr') is not None, \
            "This searcher needs a cost attribute. Please specify its " +\
            "name in search_options['cost_attr']"
        super().__init__(configspace, metric, **kwargs)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *cost_aware_gp_fifo_searcher_defaults(),
            dict_name='search_options')
        # If `resource_attr` is specified, we do fine-grained, otherwise
        # coarse-grained
        if kwargs.get('resource_attr') is not None:
            logger.info("Fine-grained: Modelling cost values c(x, r) " +\
                        "obtained at every resource level r")
            kwargs_int = cost_aware_fine_gp_fifo_searcher_factory(**_kwargs)
        else:
            logger.info("Coarse-grained: Modelling cost values c(x) " +\
                        "obtained together with metric values")
            kwargs_int = cost_aware_coarse_gp_fifo_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state['state'], self._hp_ranges_in_state())
        output_skip_optimization = state['skip_optimization']
        output_model_factory = self.state_transformer.model_factory
        # Call internal constructor
        new_searcher = MOScalarGPFIFOSearcher(
            **self._new_searcher_kwargs_for_clone(),
            output_model_factory=output_model_factory,
            init_state=init_state,
            output_skip_optimization=output_skip_optimization)
        new_searcher._restore_from_state(state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
