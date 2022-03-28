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
from typing import Dict, Optional, List
import logging

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.utils.errors import NotPSDError

from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples

# from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples

from botorch.optim.optimize import optimize_acqf, optimize_acqf_list, optimize_acqf_mixed
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement

from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.backend.trial_status import Trial
import syne_tune.config_space as cs

__all__ = ['BotorchMOGP']

from syne_tune.optimizer.schedulers.botorch.encode_decode_utils import encode_config, decode_config

logger = logging.getLogger(__name__)


class BotorchMOGP(TrialScheduler):

    def __init__(
            self,
            config_space: Dict,
            metrics: List[str],
            num_init_random_draws: int = 5,
            mode: str = "min",
            points_to_evaluate: Optional[List[Dict]] = None,
            fantasising: bool = True,
            cat_dims: int = 0,
            k: int = 0,
    ):
        """
        :param config_space:
        :param metrics: List of metrics to optimize.
        :param num_init_random_draws: number of initial random draw, after this number the suggestion are obtained
        using the posterior of a GP built on available observations.
        :param mode: 'min' or 'max'
        :param points_to_evaluate: points to evaluate first
        :param fantasising: boolean whether to use fantasizing of the pending evaluations
        :param cat_dims: number of categorical variables, at the moment limited ot 1. Those are the dimensions at the
        end of the tensor.
        :param k: number of categories for the categorical variable, if cat_dims==1
        """
        super().__init__(config_space)
        assert num_init_random_draws >= 2
        assert mode in ['min', 'max']
        self.mode = mode
        self.metrics = metrics
        self.num_evaluations = 0
        self.num_minimum_observations = num_init_random_draws
        self.points_to_evaluate = points_to_evaluate
        self.x = []
        self.y = []
        self.categorical_maps = {
            k: {cat: i for i, cat in enumerate(v.categories)}
            for k, v in config_space.items()
            if isinstance(v, cs.Categorical)
        }
        self.inv_categorical_maps = {
            hp: dict(zip(map.values(), map.keys())) for hp, map in self.categorical_maps.items()
        }
        self.pending_trials = {}
        self.fantasising = fantasising
        assert cat_dims in [0, 1], "At the moment, only cat_dims = 0 or 1 is implemented."
        self.cat_dims = cat_dims
        self.k = k
        # TODO set these bounds appropriately from the config_space
        self.bounds = config_space

        # TODO make sure this ref_point makes sense
        self.ref_point = [1. for _ in range(len(self.metrics))]

    def on_trial_complete(self, trial: Trial, result: Dict):
        # update available observations with final result.
        self.x.append(encode_config(
            config=trial.config,
            config_space=self.config_space,
            categorical_maps=self.categorical_maps,
        ))
        # TODO check the format of this encoded config from above
        self.y.append(result[self.metric_name])
        self.pending_trials.pop(trial.trial_id)

    def _suggest(self, trial_id: int) -> Optional[TrialSuggestion]:
        if self.points_to_evaluate is not None and self.num_evaluations < len(self.points_to_evaluate):
            # if we are not done yet with points_to_evaluate, we pick the next one from this list
            suggestion = self.points_to_evaluate[self.num_evaluations]
        else:
            enough_suggestion = len(self.y) < self.num_minimum_observations
            if enough_suggestion:
                # if not enough suggestion made, sample randomly
                suggestion = self.sample_random()
            else:
                suggestion = self.sample_gp()

        self.num_evaluations += 1
        self.pending_trials[trial_id] = suggestion
        return TrialSuggestion.start_suggestion(config=suggestion)

    def sample_random(self) -> Dict:
        return {
            k: v.sample()
            if isinstance(v, cs.Domain) else v
            for k, v in self.config_space.items()
        }

    def sample_gp(self) -> Dict:
        try:
            # First updates GP and compute its posterior, then maximum acquisition function to find candidate.
            x = self.x
            y = self.y

            # if self.fantasising:
            #     # when fantasising we draw observations for pending observations according to a unit prior
            #     # TODO sample from the previous posterior instead from a standard normal?
            #     x += [
            #         encode_config(config=config, config_space=self.config_space, categorical_maps=self.categorical_maps)
            #         for config in self.pending_trials.values()
            #     ]
            #     y += list(np.random.normal(size=(len(self.pending_trials))))

            # from here making use of botorch utils
            # TODO make sure these bounds are set correctly from the config space
            x = normalize(torch.Tensor(x), bounds=self.bounds)
            y = standardize(torch.Tensor(y).reshape(-1, 1))

            MC_SAMPLES = 100

            if self.cat_dims > 0:
                models = [MixedSingleTaskGP(x, y[:, i:i + 1], cat_dims=[-1]) for i in range(len(self.metrics))]
            else:
                models = [SingleTaskGP(x, y[:, i:i + 1]) for i in range(len(self.metrics))]
            model = ModelListGP(*models)

            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            bounds = torch.stack([x.min(dim=0).values, y.max(dim=0).values])

            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.ref_point,
                X_baseline=x,
                prune_baseline=True,
                # prune baseline points that have estimated zero probability of being Pareto optimal
                sampler=sampler,
                maximize=(self.mode == 'max'),
            )

            if self.cat_dims > 0:
                candidate, acq_value = optimize_acqf_mixed(
                    acq_function=acq_func,
                    bounds=bounds,
                    q=1,
                    num_restarts=20,
                    fixed_features_list=[{-1: v} for v in range(self.k)],  # x.shape[-1]-1
                    raw_samples=100,
                )
            else:
                candidate, acq_value = optimize_acqf(
                    acq_function=acq_func,
                    bounds=bounds,
                    q=1,
                    num_restarts=20,
                    raw_samples=100,
                )

            return decode_config(
                config_space=self.config_space,
                encoded_vector=candidate.detach().numpy()[0],
                inv_categorical_maps=self.inv_categorical_maps,
            )

        except NotPSDError as e:
            # In case Cholesky inversion fails, we sample randomly.
            logger.info("Chlolesky failed, sampling randomly.")
            return self.sample_random()

    def metric_names(self) -> List[str]:
        return self.metrics

    def metric_mode(self) -> str:
        return self.mode