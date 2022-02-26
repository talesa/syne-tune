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
    multiobjective_scalarized_gp_fifo_searcher_defaults, \
    multiobjective_scalarized_gp_fifo_searcher_factory
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


# TODO for now mostly just copied from CostAwareGPFIFOSearcher
class MOScalarGPFIFOSearcher(MultiModelGPFIFOSearcher):
    """
    Gaussian process-based scalarization-based multiobjective hyperparameter
    optimization to be used with a FIFO scheduler.
    """
    def __init__(self, configspace, metric, **kwargs):
        assert kwargs.get('metrics') is not None, \
            "This searcher needs the parameter 'metrics'."
        assert kwargs.get('scalarization_method') is not None, \
            "This searcher needs the parameter 'scalarization_method'."
        super().__init__(configspace, metric, **kwargs)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *multiobjective_scalarized_gp_fifo_searcher_defaults(),
            dict_name='search_options')
        kwargs_int = multiobjective_scalarized_gp_fifo_searcher_factory(**_kwargs)
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
