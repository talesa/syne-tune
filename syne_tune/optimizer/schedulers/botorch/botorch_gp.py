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

from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.backend.trial_status import Trial
import syne_tune.config_space as cs

__all__ = ['BotorchGP']

from syne_tune.optimizer.schedulers.botorch.encode_decode_utils import encode_config, decode_config

logger = logging.getLogger(__name__)


class BotorchGP(TrialScheduler):

    def __init__(
            self,
            config_space: Dict,
            metric: str,
            num_init_random_draws: int = 5,
            mode: str = "min",
            points_to_evaluate: Optional[List[Dict]] = None,
            fantasising: bool = True,
    ):
        """
        :param config_space:
        :param metric: metric to optimize.
        :param num_init_random_draws: number of initial random draw, after this number the suggestion are obtained
        using the posterior of a GP built on available observations.
        :param mode: 'min' or 'max'
        :param points_to_evaluate: points to evaluate first
        :param fantasising:
        """
        super().__init__(config_space)
        assert num_init_random_draws >= 2
        assert mode in ['min', 'max']
        self.mode = mode
        self.metric_name = metric
        self.num_evaluations = 0
        self.num_minimum_observations = num_init_random_draws
        self.points_to_evaluate = points_to_evaluate
        self.X = []
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

    def on_trial_complete(self, trial: Trial, result: Dict):
        # update available observations with final result.
        self.X.append(encode_config(
            config=trial.config,
            config_space=self.config_space,
            categorical_maps=self.categorical_maps,
        ))
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
            # todo normalize input data
            X = self.X
            y = self.y
            if self.fantasising:
                # when fantasising we draw observations for pending observations according to a unit prior
                X += [
                    encode_config(config=config, config_space=self.config_space, categorical_maps=self.categorical_maps)
                    for config in self.pending_trials.values()
                ]
                y += list(np.random.normal(size=(len(self.pending_trials))))

            X_tensor = torch.Tensor(X)
            Y_tensor = standardize(torch.Tensor(y).reshape(-1, 1))
            self.gp = SingleTaskGP(X_tensor, Y_tensor)
            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_model(mll)

            # ask candidate to GP by maximizing its acquisition function.
            UCB = UpperConfidenceBound(
                self.gp,
                beta=0.1,
                maximize=self.mode == 'max'
            )

            bounds = torch.stack([X_tensor.min(axis=0).values, X_tensor.max(axis=0).values])
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
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
        return [self.metric_name]

    def metric_mode(self) -> str:
        return self.mode