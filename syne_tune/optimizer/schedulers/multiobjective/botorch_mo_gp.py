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
from collections import defaultdict
from typing import Dict, Optional, List, Union, Iterable, Sequence, Sized
import logging

import sys

from botorch.models.transforms.input import InputTransform, ChainedInputTransform, Normalize

sys.path.insert(0, '/Users/awgol/code/botorch/')

import numpy as np
import torch
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.models import SingleTaskGP, ModelList
from botorch.fit import fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.utils.errors import NotPSDError

from botorch.posteriors import Posterior

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
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.backend.trial_status import Trial
import syne_tune.config_space as cs

from torch import Tensor
from torch.nn import Module

__all__ = ['BotorchMOGP']

from syne_tune.optimizer.schedulers.botorch.encode_decode_utils import encode_config, decode_config

logger = logging.getLogger(__name__)


class MyMCObj(MCMultiOutputObjective):
    def __init__(self, y_mean, y_std, mobo_object):
        super().__init__()
        self.y_mean = y_mean
        self.y_std = y_std
        self.mobo_object = mobo_object
        self.instance_info = InstanceInfos()

    def forward(self, samples: Tensor, X: Optional[Tensor] = None, **kwargs) -> Tensor:
        X_clone = X.clone().detach()
        instance_type_int = X_clone[..., self.mobo_object.field_to_id['config_st_instance_type']]
        # instance_type_int2 = X[..., self.mobo_object.field_to_id['config_st_instance_type']]
        # assert torch.allclose(
        #     instance_type_int,
        #     instance_type_int.to(dtype=torch.long).to(dtype=torch.float64))
        instance_cost = instance_type_int.apply_(
            lambda x: self.instance_info(
                self.mobo_object.inv_categorical_maps['config_st_instance_type'][int(x)]).cost_per_hour)
        # assert torch.allclose(
        #     instance_type_int2,
        #     instance_type_int2.to(dtype=torch.long).to(dtype=torch.float64))

        # samples are from single output GP, so batch x q x 1, repeat to make batch x q x 2
        samples = samples.repeat(*((samples.dim() - 1) * [1]), 2)

        y0_unnorm = samples[..., 0] * self.y_std[0, 0] + self.y_mean[0, 0]
        # assert y0_unnorm.shape[-instance_cost.ndim:] == instance_cost.shape

        # Apply the affine transform to 2nd output dimension
        y1_unnorm = y0_unnorm * instance_cost
        # y1_norm = standardize(y1_unnorm)
        y1_norm = (y1_unnorm - self.y_mean[0, 1]) / self.y_std[0, 1]
        samples[..., 1] = y1_norm

        # if y1_unnorm.shape[-1] == len(self.mobo_object.y):
        # #     y0_true = np.array(self.mobo_object.y)[:, 0]
        #     y1_true = np.array(self.mobo_object.y)[:, 1]
        #     mask = (y1_true != 1.)
        #     assert np.allclose(
        #         mask * y0_unnorm.mean(dim=0).numpy() * (-1. if self.mobo_object.mode == 'min' else 1.),
        #         mask * y0_true,
        #         rtol=0.2)
        #     assert np.allclose(
        #         mask * y1_unnorm.mean(dim=0).numpy() * (-1. if self.mobo_object.mode == 'min' else 1.),
        #         mask * y1_true,
        #         rtol=0.2)

        return samples


class CustomFeatures(InputTransform, Module):
    def __init__(
        self,
        feature_set: Tensor,
        transform_on_train: bool = False,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = False,
    ) -> None:
        r"""Append `feature_set` to each input.

        Args:
            feature_set: An `n_f x d_f`-dim tensor denoting the features to be
                appended to the inputs.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: False.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: False.
        """
        super().__init__()
        if feature_set.dim() != 2:
            raise ValueError("`feature_set` must be an `n_f x d_f`-dim tensor!")
        self.register_buffer("feature_set", feature_set)
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize

    def transform(self, X: Tensor) -> Tensor:
        r"""Transform the inputs by appending `feature_set` to each input.

        For each `1 x d`-dim element in the input tensor, this will produce
        an `n_f x (d + d_f)`-dim tensor with `feature_set` appended as the last `d_f`
        dimensions. For a generic `batch_shape x q x d`-dim `X`, this translates to a
        `batch_shape x (q * n_f) x (d + d_f)`-dim output, where the values corresponding
        to `X[..., i, :]` are found in `output[..., i * n_f: (i + 1) * n_f, :]`.

        Note: Adding the `feature_set` on the `q-batch` dimension is necessary to avoid
        introducing additional bias by evaluating the inputs on independent GP
        sample paths.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (q * n_f) x (d + d_f)`-dim tensor of appended inputs.
        """
        expanded_X = X.unsqueeze(dim=-2).expand(
            *X.shape[:-1], self.feature_set.shape[0], -1
        )
        expanded_features = self.feature_set.expand(*expanded_X.shape[:-1], -1)
        appended_X = torch.cat([expanded_X, expanded_features], dim=-1)
        return appended_X.view(*X.shape[:-2], -1, appended_X.shape[-1])


class BotorchMOGP(TrialScheduler):
    def __init__(
            self,
            config_space: Dict,
            metrics: List[str],
            ref_point: List[float],
            num_init_random_draws: int = 5,
            mode: str = "min",
            points_to_evaluate: Optional[List[Dict]] = None,
            # fantasising: bool = True,
            num_mc_samples: int = 100,
            instance_type_features: Sequence = tuple(),
            deterministic_transform: bool = False,
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
        # BOTorch assumes minimization so we negate the ref_point
        if self.mode == 'min':
            self.ref_point = [-v for v in ref_point]

        self.deterministic_transform = deterministic_transform

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
        # self.fantasising = fantasising
        self.num_mc_samples = num_mc_samples

        self.cat_dims = [i for i, v in enumerate(self.config_space.values()) if isinstance(v, cs.Categorical)]

        # Additional per-instance_type features
        for v in instance_type_features:
            assert v in (
                'GPUFP32TFLOPS', 'cost_per_hour', 'num_cpu', 'num_gpu', 'GPUMemory', 'GPUFP32TFLOPS*num_gpu',
                'GPUMemory/batch_size',
            )
        self.instance_type_features = instance_type_features

        instance_info = InstanceInfos()
        self.instance_type_to_tuple_features_dict = {}
        per_feature_values = defaultdict(list)
        for instance_type in self.config_space['config_st_instance_type']:
            record = []
            for k in self.instance_type_features:
                fields = k.split('*')
                value = np.prod(tuple(instance_info(instance_type).__dict__[f] for f in fields))
                record.append(value)
                per_feature_values[k].append(value)
            self.instance_type_to_tuple_features_dict[instance_type] = tuple(record)

        bounds = []
        # bounds for the self.config_space variables
        for name, domain in self.config_space.items():
            # We are treating config_st_instance_type as a categorical in the BOTorch framework, it should not be
            # normalized so need to set its bounds to [0, 1].
            if isinstance(domain, cs.Categorical):
                bound = (0., 1.)
            # for domains with attributes lower and upper, they define the bounds
            elif hasattr(domain, "lower") and hasattr(domain, "upper"):
                bound = (domain.lower, domain.upper)
            else:
                raise Exception(f"Cannot add a bound for parameter: {name} {domain}")
            bounds.append(bound)
        # bounds for per-instance_type features
        for k in self.instance_type_features:
            bounds.append((min(per_feature_values[k]), max(per_feature_values[k])))
        self.bounds = torch.DoubleTensor(bounds).to(dtype=torch.double).T
        assert (self.bounds.shape[0] == 2), "self.bounds should have shape [2, number_of_hparams]"

        self.field_to_id = {k: i for i, k in enumerate(self.config_space.keys())}

        # TODO how to generate all values from a discrete hp config like finrange?
        #  Then we could use itertools.product rather than sampling which is very silly
        all_possible_configs = set(tuple(v.sample() for v in self.config_space.values()) for _ in range(10000))
        all_configs_dicts = [{k: v for k, v in zip(self.config_space.keys(), config)} for config in
                             all_possible_configs]
        self.all_configs_vector = {
            tuple(encode_config(
                config=config,
                config_space=self.config_space,
                categorical_maps=self.categorical_maps,
                cat_to_onehot=False,
                normalize_bounded_domains=False,
            ))
            for config in all_configs_dicts}

    def on_trial_complete(self, trial: Trial, result: Dict):
        # update available observations with final result
        # store them in the unnormalized form of only the variables available in the self.config_space
        self.x.append(encode_config(
            config=trial.config,
            config_space=self.config_space,
            categorical_maps=self.categorical_maps,
            cat_to_onehot=False,
            normalize_bounded_domains=False,
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
        sample = {
            k: v.sample()
            if isinstance(v, cs.Domain) else v
            for k, v in self.config_space.items()
        }
        sample_encoded = encode_config(
            config=sample,
            config_space=self.config_space,
            categorical_maps=self.categorical_maps,
            cat_to_onehot=False,
            normalize_bounded_domains=False,
        )
        if sample_encoded not in self.x:
            return sample
        else:
            return self.sample_random()

    def append_instance_type_features(self, x):
        instance_type_feature_vector = list()
        for v in x:
            instance_type_field = int(v[self.field_to_id['config_st_instance_type']])
            instance_type_string = self.inv_categorical_maps['config_st_instance_type'][instance_type_field]
            instance_type_feature_vector.append(self.instance_type_to_tuple_features_dict[instance_type_string])
        instance_type_feature_tensor = torch.DoubleTensor(instance_type_feature_vector)

        output = torch.cat([torch.DoubleTensor(np.stack(x, axis=0)), instance_type_feature_tensor], dim=1)
        return output

    def sample_gp(self) -> Union[dict, None]:
        try:
            x_unnorm = self.append_instance_type_features(self.x)
            # x_norm = normalize(x_unnorm, bounds=self.bounds)
            # x_unnorm = torch.DoubleTensor(np.stack(self.x, axis=0))

            y = torch.DoubleTensor(self.y)
            if self.mode == 'min':
                # BOTorch assumes maximization while we want to minimize so we negate the y tensor
                y = -y

            # y = standardize(y)
            stddim = -1 if y.dim() < 2 else -2
            y_std = y.std(dim=stddim, keepdim=True)
            y_std = y_std.where(y_std >= 1e-9, torch.full_like(y_std, 1.0))
            y_mean = y.mean(dim=stddim, keepdim=True)
            y = (y - y_mean) / y_std

            input_transforms = ChainedInputTransform(
                # tf1=None,
                tf2=Normalize(d=x_unnorm.shape[-1],
                              bounds=self.bounds,
                              indices=list(range(1, 3 + len(self.instance_type_features)))),
            )

            if self.deterministic_transform:
                if len(self.cat_dims) > 0:
                    models = [MixedSingleTaskGP(x_unnorm, y[:, 0:1],
                                                cat_dims=self.cat_dims,
                                                input_transform=input_transforms)]
                else:
                    raise NotImplementedError
                    # models = [SingleTaskGP(x_norm, y[:, 0:1])]
            else:
                if len(self.cat_dims) > 0:
                    models = [MixedSingleTaskGP(x_unnorm, y[:, i:i + 1],
                                                cat_dims=self.cat_dims,
                                                input_transform=input_transforms)
                              for i in range(len(self.metrics))]
                else:
                    raise NotImplementedError
                    # models = [SingleTaskGP(x_norm, y[:, i:i + 1])
                    #           for i in range(len(self.metrics))]
            model = ModelListGP(*models)

            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            sampler = SobolQMCNormalSampler(num_samples=self.num_mc_samples)
            if self.deterministic_transform:
                objective = MyMCObj(y_mean=y_mean, y_std=y_std, mobo_object=self)
            else:
                objective = None

            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.ref_point,
                # X_baseline=x_norm,
                X_baseline=x_unnorm,
                prune_baseline=True,  # False if self.deterministic_transform else True,
                sampler=sampler,
                objective=objective,
                cache_root=False if self.deterministic_transform else True,
            )

            # compute all of the points in the config_space still left to consider
            x_to_consider_unnorm = tuple(self.all_configs_vector.difference(set(self.x)))
            if len(x_to_consider_unnorm) == 0:
                # No more points in the config space to consider, the end of the optimization
                return None
            if not (len(x_to_consider_unnorm) + len(self.x) == len(self.all_configs_vector)):
                raise Exception('Value has not been removed from self.all_configs_vector correctly.')

            x_to_consider_unnorm = self.append_instance_type_features(x_to_consider_unnorm)
            x_to_consider_norm = normalize(x_to_consider_unnorm, self.bounds)
            # batch_shape below required to compute the acquisition function over different possible points rather than
            # all of those points jointly
            acq_func_values = acq_func(x_to_consider_norm.reshape(-1, 1, x_to_consider_norm.shape[-1]))
            acq_func_idxmax = acq_func_values.argmax()
            candidate = np.array(x_to_consider_unnorm[acq_func_idxmax], dtype=np.double)
            # take only the fields contained in self.config_space
            candidate = candidate[:len(x_to_consider_unnorm[0]) - len(self.instance_type_features)]

            return decode_config(
                config_space=self.config_space,
                encoded_vector=candidate,
                inv_categorical_maps=self.inv_categorical_maps,
                cat_to_onehot=False,
                normalize_bounded_domains=False,
            )

        except NotPSDError as e:
            # In case Cholesky inversion fails, we sample randomly.
            logger.info("Chlolesky failed, sampling randomly.")
            return self.sample_random()

    def metric_names(self) -> List[str]:
        return self.metrics

    def metric_mode(self) -> str:
        return self.mode
