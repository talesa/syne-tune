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
import logging
import sys
from collections import defaultdict
from typing import Dict, Optional, List, Union, Sequence

from syne_tune.blackbox_repository.conversion_scripts.scripts.hf_distilbert_on_imdb_static import (
    HFDistilbertOnImdbStaticBlackbox)

sys.path.insert(0, '/Users/awgol/code/botorch/')

from gpytorch.utils.errors import NotPSDError
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective

from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize, Log, ChainedOutcomeTransform
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import normalize

from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler

from syne_tune.backend.sagemaker_backend.instance_info import InstanceInfos
from syne_tune.optimizer.scheduler import TrialScheduler, TrialSuggestion
from syne_tune.backend.trial_status import Trial
import syne_tune.config_space as cs

from torch import Tensor

__all__ = ['HFDistilbertOnImdbStaticMOBO']

from syne_tune.optimizer.schedulers.botorch.encode_decode_utils import encode_config, decode_config

logger = logging.getLogger(__name__)

import torch
import random
import numpy as np
import botorch


class HFDistilbertOnImdbStaticMOBO(TrialScheduler):
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
            features: Sequence = tuple(),
            deterministic_transform: bool = False,
            exclude_oom_runs: bool = False,
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
        map_instance_type_family = tuple(set(
            k.split('.')[1] for k in self.categorical_maps['config_st_instance_type'].keys()))
        self.categorical_maps['instance_type_family'] = {k: i for i, k in enumerate(map_instance_type_family)}
        self.inv_categorical_maps['instance_type_family'] = {i: k for i, k in enumerate(map_instance_type_family)}
        self.pending_trials = {}
        self.num_mc_samples = num_mc_samples
        self.exclude_oom_runs = exclude_oom_runs

        self.field_to_id = {k: i for i, k in enumerate(self.config_space.keys())}

        # TODO how to generate all values from a discrete hp config like finrange?
        #  Then we could use itertools.product rather than sampling which is very silly
        all_possible_configs = set(tuple(v.sample() for v in self.config_space.values()) for _ in range(10000))
        if self.exclude_oom_runs:
            # remove configs that fail due to OOM
            all_possible_configs = set(
                x for x in all_possible_configs
                if x[1] <= HFDistilbertOnImdbStaticBlackbox.per_device_train_batch_size_limits[x[0].split('.')[1]])
        all_configs_dicts = tuple({k: v for k, v in zip(self.config_space.keys(), config)}
                                  for config in all_possible_configs)
        self.all_configs_vector = dict.fromkeys(
            tuple(encode_config(
                config=config,
                config_space=self.config_space,
                categorical_maps=self.categorical_maps,
                cat_to_onehot=False,
                normalize_bounded_domains=False,
            ))
            for config in all_configs_dicts)
        self.number_of_combinations = len(self.all_configs_vector)

        self.features = features

        instance_info = InstanceInfos()
        self.instance_info = instance_info

        self.config_to_tuple_features_dict = {}
        cat_dims = set()
        per_feature_values = defaultdict(list)
        # TODO could vectorize this
        for config_dict, config_tuple in zip(all_configs_dicts, self.all_configs_vector.keys()):
            record = []
            for i, k in enumerate(self.features):
                # This needs to be maintained
                if k in ('config_st_instance_type',):
                    value = self.categorical_maps['config_st_instance_type'][config_dict['config_st_instance_type']]
                    cat_dims.add(i)
                elif k in ('instance_type_family',):
                    value = self.categorical_maps['instance_type_family'][
                        config_dict['config_st_instance_type'].split('.')[1]]
                    cat_dims.add(i)
                elif k in ('config_per_device_train_batch_size', 'config_dataloader_num_workers'):
                    value = config_dict[k]
                elif k in ('GPUMemory/batch_size',):
                    value = (instance_info(config_dict['config_st_instance_type']).__dict__['GPUMemory'] /
                             config_dict['config_per_device_train_batch_size'])
                elif k in ('GPUFP32TFLOPS*num_gpu',):
                    fields = k.split('*')
                    value = np.prod(tuple(instance_info(config_dict['config_st_instance_type']).__dict__[f] for f in fields))
                elif k in ('GPUFP32TFLOPS', 'cost_per_hour', 'num_cpu', 'num_gpu', 'GPUMemory',):
                    value = instance_info(config_dict['config_st_instance_type']).__dict__[k]
                else:
                    raise ValueError(f'Unrecognized feature: {k}')
                record.append(value)
                per_feature_values[k].append(value)
            self.config_to_tuple_features_dict[config_tuple] = tuple(record)

        self.cat_dims = list(cat_dims)

        bounds = []
        for i, k in enumerate(self.features):
            if i in self.cat_dims:
                bounds.append((0., 1.))
            else:
                bounds.append((min(per_feature_values[k]), max(per_feature_values[k])))
        self.bounds = torch.DoubleTensor(bounds).T
        assert (self.bounds.shape[0] == 2), "self.bounds should have shape [2, number_of_hparams]"

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

        if suggestion is None:
            return None
        else:
            self.num_evaluations += 1
            self.pending_trials[trial_id] = suggestion
            return TrialSuggestion.start_suggestion(config=suggestion)

    def sample_random(self) -> Dict:
        idx = np.random.randint(0, len(self.all_configs_vector))
        sample_encoded = list(self.all_configs_vector.keys())[idx]
        self.all_configs_vector.pop(sample_encoded)
        if sample_encoded in self.x:
            raise Exception('Something went wrong with removing the values when sampling.')
        return decode_config(
            encoded_vector=sample_encoded,
            config_space=self.config_space,
            inv_categorical_maps=self.inv_categorical_maps,
            cat_to_onehot=False,
            normalize_bounded_domains=False,
        )

    def sample_gp(self) -> Union[dict, None]:
        try:
            fixed_seed = False
            if fixed_seed:
                seed = 1
                torch.manual_seed(seed=seed)
                np.random.seed(seed)
                random.seed(seed)
                botorch.utils.sampling.manual_seed(seed=seed)

            x_unnorm = torch.DoubleTensor(np.stack([self.config_to_tuple_features_dict[k] for k in self.x], axis=0))
            x_norm = normalize(x_unnorm, bounds=self.bounds)

            y = torch.DoubleTensor(self.y)
            y = y.log()

            input_transforms = None
            outcome_transform = ChainedOutcomeTransform(
             # tf1=Log(),
                tf2=Standardize(m=1),
            )

            if self.deterministic_transform:
                if len(self.cat_dims) > 0:
                    models = [MixedSingleTaskGP(x_norm, y[:, 0:1],
                                                cat_dims=self.cat_dims,
                                                input_transform=input_transforms,
                                                outcome_transform=outcome_transform)]
                else:
                    raise NotImplementedError
                    # models = [SingleTaskGP(x_norm, y[:, 0:1])]
            else:
                if len(self.cat_dims) > 0:
                    models = [MixedSingleTaskGP(x_norm, y[:, i:i + 1],
                                                cat_dims=self.cat_dims,
                                                input_transform=input_transforms,
                                                outcome_transform=outcome_transform)
                              for i in range(len(self.metrics))]
                else:
                    raise NotImplementedError
                    # models = [SingleTaskGP(x_norm, y[:, i:i + 1])
                    #           for i in range(len(self.metrics))]
            model = ModelListGP(*models)

            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            sampler = SobolQMCNormalSampler(num_samples=self.num_mc_samples, seed=1 if fixed_seed else None)

            def objective_transform(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
                # cloning because we are using an in-place operation below
                X_clone = X.clone()
                instance_type_int = X_clone[..., self.field_to_id['config_st_instance_type']]
                instance_cost = instance_type_int.apply_(
                    lambda x: self.instance_info(
                        self.inv_categorical_maps['config_st_instance_type'][int(x)]).cost_per_hour)

                y0_unnorm = Y[..., 0]
                y1_unnorm = y0_unnorm * instance_cost

                # assert (y0_unnorm < 0.).sum() == 0

                # if y1_unnorm.shape[-1] == len(self.y):
                #     y_true = np.array(self.y)
                #     mask = (y_true[:, 1] != 1.)
                #     assert np.allclose(
                #         mask * y0_unnorm.mean(dim=0).numpy(),
                #         mask * y_true[:, 0],
                #         rtol=0.5)
                #     assert np.allclose(
                #         mask * y1_unnorm.mean(dim=0).numpy(),
                #         mask * y_true[:, 1],
                #         rtol=0.5)

                return torch.stack([y0_unnorm, y1_unnorm], dim=-1)

            if self.deterministic_transform:
                objective = GenericMCMultiOutputObjective(
                    lambda Y, X: objective_transform(Y.exp(), X) * (-1. if self.mode == 'min' else 1.))
            else:
                objective = GenericMCMultiOutputObjective(
                    lambda Y, X: Y.exp() * (-1. if self.mode == 'min' else 1.))

            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=self.ref_point,
                X_baseline=x_norm,
                # X_baseline=x_unnorm,
                prune_baseline=True,  # False if self.deterministic_transform else True,
                sampler=sampler,
                objective=objective,
                cache_root=False if self.deterministic_transform else True,
            )

            # compute all of the points in the config_space still left to consider
            x_to_consider = tuple(self.all_configs_vector.keys())
            if len(x_to_consider) == 0:
                # No more points in the config space to consider, the end of the optimization
                return None
            if not (len(self.all_configs_vector) + len(self.x) == self.number_of_combinations):
                raise Exception('Value has not been removed from self.all_configs_vector correctly.')

            x_to_consider_unnorm = torch.DoubleTensor(np.stack([
                self.config_to_tuple_features_dict[k] for k in x_to_consider], axis=0))
            x_to_consider_norm = normalize(x_to_consider_unnorm, self.bounds)
            # batch_shape below required to compute the acquisition function over different possible points rather than
            # all of those points jointly
            acq_func_values = acq_func(x_to_consider_norm.reshape(-1, 1, x_to_consider_norm.shape[-1]))
            acq_func_idxmax = acq_func_values.argmax()
            candidate = np.array(x_to_consider[acq_func_idxmax], dtype=np.double)
            # remove the selected point from among the points to consider
            self.all_configs_vector.pop(x_to_consider[acq_func_idxmax], None)

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
            print("Chlolesky failed, sampling randomly.")
            return self.sample_random()

    def metric_names(self) -> List[str]:
        return self.metrics

    def metric_mode(self) -> str:
        return self.mode
