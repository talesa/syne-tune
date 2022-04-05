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
import itertools
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
            ref_point: List[float],
            num_init_random_draws: int = 5,
            mode: str = "min",
            points_to_evaluate: Optional[List[Dict]] = None,
            fantasising: bool = True,
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
        # TODO make sure this mode actually changes something
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

        self.cat_dims = [i for i, v in enumerate(self.config_space.values()) if isinstance(v, cs.Categorical)]

        ks = [len(v) for i, v in enumerate(self.config_space.values()) if isinstance(v, cs.Categorical)]
        self.fixed_feature_lists = list(
            map(lambda vs: {self.cat_dims[i]: v for i, v in enumerate(vs)},
                itertools.product(*[list(range(k)) for k in ks]))
        )

        # Bounded domains (integer and continuous) will be normalized by encode_config(.) function
        bounds = []
        for name, domain in self.config_space.items():
            if isinstance(domain, cs.Categorical):
                bound = (0, len(domain)-1)
            elif hasattr(domain, "lower") and hasattr(domain, "upper"):
                bound = (0., 1.)
            else:
                raise Exception(f"Cannot add a bound for parameter: {name} {domain}")
            bounds.append(bound)
        self.bounds = torch.Tensor(bounds).to(dtype=torch.double).T
        assert (self.bounds.shape[0] == 2), "self.bounds should have shape [2, number_of_hparams]"

        # BOTorch assumes minimization so we negate the ref_point
        self.ref_point = [-v for v in ref_point]

        all_possible_configs = set(tuple(v.sample() for v in self.config_space.values()) for _ in range(10000))
        all_configs_dicts = [{k: v for k, v in zip(self.config_space.keys(), config)} for config in all_possible_configs]
        self.all_configs_vector = {
            tuple(encode_config(
                config=config,
                config_space=self.config_space,
                categorical_maps=self.categorical_maps,
                cat_to_onehot=False,
                normalize_bounded_domains=True,
            ))
            for config in all_configs_dicts}
        # self.all_configs_tensor = torch.Tensor(np.stack(all_configs_vector).reshape(-1, 1, 3))

    def on_trial_complete(self, trial: Trial, result: Dict):
        # update available observations with final result.
        self.x.append(encode_config(
            config=trial.config,
            config_space=self.config_space,
            categorical_maps=self.categorical_maps,
            cat_to_onehot=False,
            normalize_bounded_domains=True,
        ))

        self.y.append([result[k] for k in self.metrics])
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

            # TODO enable fantasizing
            # if self.fantasising:
            #     # when fantasising we draw observations for pending observations according to a unit prior
            #     # TODO sample from the previous posterior instead from a standard normal?
            #     x += [
            #         encode_config(config=config, config_space=self.config_space, categorical_maps=self.categorical_maps)
            #         for config in self.pending_trials.values()
            #     ]
            #     y += list(np.random.normal(size=(len(self.pending_trials))))

            # From here on we are making use of BOTorch

            # We don't need to normalize x here using BOTorch utils because encode_config takes care of it in
            #  on_trial_complete(.)
            # x = normalize(torch.Tensor(x).to(dtype=torch.double), bounds=self.bounds)
            x = torch.Tensor(np.array(x)).to(dtype=torch.double)
            # BOTorch assumes maximization while we want to minimize so we negate the y tensor
            y = standardize(-torch.Tensor(y).to(dtype=torch.double))

            MC_SAMPLES = 100

            if len(self.cat_dims) > 0:
                models = [MixedSingleTaskGP(x, y[:, i:i + 1], cat_dims=self.cat_dims) for i in range(len(self.metrics))]
            else:
                models = [SingleTaskGP(x, y[:, i:i + 1]) for i in range(len(self.metrics))]
            model = ModelListGP(*models)

            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.ref_point,
                X_baseline=x,
                prune_baseline=True,
                # prune baseline points that have estimated zero probability of being Pareto optimal
                sampler=sampler,
                # maximize=(self.mode == 'max'),
            )

            xs_to_consider = list(self.all_configs_vector.difference(set(tuple(v) for v in self.x)))
            idxmax = acq_func(torch.Tensor(np.stack(xs_to_consider).reshape(-1, 1, 3))).argmax()

            candidate = np.array(xs_to_consider[idxmax])

            # if len(self.cat_dims) > 0:
            #     candidate, acq_value = optimize_acqf_mixed(
            #         acq_function=acq_func,
            #         bounds=self.bounds,
            #         q=1,
            #         num_restarts=20,
            #         fixed_features_list=self.fixed_feature_lists,
            #         raw_samples=100,
            #     )
            #     candidate = candidate.detach().numpy()[0]
            # else:
            #     candidate, acq_value = optimize_acqf(
            #         acq_function=acq_func,
            #         bounds=self.bounds,
            #         q=1,
            #         num_restarts=20,
            #         raw_samples=100,
            #     )
            #     candidate = candidate.detach().numpy()[0]

            return decode_config(
                config_space=self.config_space,
                encoded_vector=candidate,
                inv_categorical_maps=self.inv_categorical_maps,
                cat_to_onehot=False,
                normalize_bounded_domains=True,
            )

        except NotPSDError as e:
            # In case Cholesky inversion fails, we sample randomly.
            logger.info("Chlolesky failed, sampling randomly.")
            return self.sample_random()

    def metric_names(self) -> List[str]:
        return self.metrics

    def metric_mode(self) -> str:
        return self.mode