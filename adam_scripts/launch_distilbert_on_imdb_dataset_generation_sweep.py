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
"""
Example for how to fine-tune a DistilBERT model on the IMDB sentiment classification task using the Hugging Face SageMaker Framework.
"""
import logging
import math
from pathlib import Path

import numpy as np

from sagemaker.huggingface import HuggingFace

import syne_tune
from benchmarking.definitions.definition_distilbert_on_imdb import distilbert_imdb_default_params, distilbert_imdb_benchmark
from syne_tune.backend.sagemaker_backend.sagemaker_backend import SageMakerBackend
from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.remote.remote_launcher import RemoteLauncher
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion
import syne_tune.config_space as sp
from syne_tune.backend.sagemaker_backend.instance_info import select_instance_type


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # We pick the DistilBERT on IMDB benchmark
    # The 'benchmark' dict contains arguments needed by scheduler and
    # searcher (e.g., 'mode', 'metric'), along with suggested default values
    # for other arguments (which you are free to override)

    default_params = distilbert_imdb_default_params()
    benchmark = distilbert_imdb_benchmark(default_params)
    mode = benchmark['mode']
    metric = benchmark['metric']
    # config_space = benchmark['config_space']

    # EDIT THIS SECTION
    random_seed = 31415927

    # instance_types = select_instance_type(min_gpu=1, max_cost_per_hour=10.0)
    instance_types = [
        # 'ml.p3.2xlarge',
        # 'ml.p2.xlarge',
        # 'ml.p2.8xlarge',
        # 'ml.g4dn.xlarge',
        # 'ml.g4dn.2xlarge',
        # 'ml.g4dn.4xlarge',
        # 'ml.g4dn.8xlarge',
        # 'ml.g4dn.12xlarge',
        # 'ml.g5.xlarge',
        'ml.g5.2xlarge',
        'ml.g5.4xlarge',
        'ml.g5.8xlarge',
        'ml.g5.12xlarge',
        'ml.g5.24xlarge',
        # 'ml.p3.16xlarge',
        # 'ml.p3dn.24xlarge',
        # 'ml.p4d.24xlarge',
        # 'ml.g5.48xlarge',
    ]

    #       GPU  RAM max_bs_imdb
    # g4dn T4   16GB          32
    # g5   A10G 24GB          52
    # p2   K80  12GB          24
    # p3   V100 16GB          32
    # p3dn V100 32GB          68
    # p4d  A100 40GB          88

    # instance_type_config_space = dict(
    #     num_gpus=sp.logfinrange(1, 8, 3),  # [1, 4, 8]
    #     num_vcpus=sp.logfinrange(1, 8, 4),  # [1, 2, 4, 8]
    #     # num_vcpus=sp.logfinrange([1, 2, 4, 8, ...]),  # [1, 4, 8]
    # )
    # instance_type_config_space['num_gpus'].sample()

    # tuner_job_name = 'test-g4dn-bs-26'

    n_workers = 1
    # single_job_max_run = 10 * 60  # 10min

    # tuner_job_name = 'speed-bs-it-nw-p3dn-24'
    # instance_types = ['ml.p3dn.24xlarge',]
    # per_device_train_batch_size_list = [68]

    # tuner_job_name = 'speed-bs-it-nw-p4d-24'
    # instance_types = ['ml.p4d.24xlarge',]
    # per_device_train_batch_size_list = [88]
    single_job_max_run = 5 * 60  # 5min

    # tuner_job_name = 'speed-bs-it-nw-g5-48'
    # instance_types = ['ml.g5.48xlarge',]
    # per_device_train_batch_size_list = [52]

    tuner_job_name = 'speed-bs-it-nw-g5Xxlarge-bs52'
    # instance_types = ['ml.g5.xlarge',]
    per_device_train_batch_size_list = [52]

    n_workers = 100


    # per_device_train_batch_size_list = [4, 8, 16, 24, 32, 40, 48]
    # dataloader_num_workers_list = list(range(2, 7))
    # seeds = [0, 1, 2]

    # per_device_train_batch_size_list = [52, 60, 68, 76, 84, 96]
    # per_device_train_batch_size_list = [32]
    dataloader_num_workers_list = [0, 1]
    seeds = [0, 1, 2]
    # dataloader_num_workers_list = [0, 1, 2]
    # seeds = [0, 1, 2]

    config_space = dict(
        per_device_train_batch_size=sp.choice(per_device_train_batch_size_list),  # TODO select me
        # per_device_train_batch_size=26,
        # fp16=1,
        n_train_data=25000,
        epochs=100,
        per_device_eval_batch_size=1,
        n_eval_data=1,
        log_interval=100,
        eval_interval=0,
        learning_rate=1e-6,
        weight_decay=1e-6,
        dataloader_num_workers=sp.choice(dataloader_num_workers_list),
        st_instance_type=sp.choice(instance_types),  # TODO select me
        # st_instance_type='ml.g4dn.xlarge',
        seed=sp.choice(seeds),  # TODO select me
        max_resource_level=default_params['max_resource_level'])
    stop_criterion = StoppingCriterion(
        # max_wallclock_time=10 * 60 * 60,  # 10h
        max_num_trials_finished=int(np.prod(tuple(map(len, (per_device_train_batch_size_list, dataloader_num_workers_list, instance_types, seeds))))),
        # max_num_trials_completed=1000,
        # max_num_trials_completed=1,
    )

    # tuner_job_name = 'loss-lr-wd-bs-2'
    # n_workers = 5
    # config_space = dict(
    #     # per_device_train_batch_size=randint(4, 16),
    #     per_device_train_batch_size=sp.choice([2, 4, 8, 12, 16]),
    #     n_train_data=25000,
    #     epochs=3,
    #     per_device_eval_batch_size=32,
    #     n_eval_data=8192,
    #     log_interval=100,
    #     eval_interval=0,
    #     learning_rate=loguniform(1e-7, 1e-4),
    #     weight_decay=loguniform(1e-6, 1e-2),
    #     dataloader_num_workers=1,
    #     st_instance_type='ml.g4dn.xlarge',
    #     # dataset_path=default_params['dataset_path'],
    #     # keep_in_memory=0,
    #     max_resource_level=default_params['max_resource_level'])
    # single_job_max_run = 90*60  # 90min
    # stop_criterion = StoppingCriterion(
    #     max_wallclock_time=10 * 60 * 60,  # 10h
    #     max_num_trials_completed=100)

    # tuner_job_name = 'dataloader-num-workers'
    # points_to_evaluate = [dict(
    #     learning_rate=1e-6,
    #     weight_decay=1e-6,
    #     dataloader_num_workers=dataloader_num_workers,
    #     # st_instance_type=sp.choice(instance_types),
    #     epochs=epochs,
    #     # dataset_path=default_params['dataset_path'],
    #     keep_in_memory=False,
    #     log_interval=0,
    #     eval_interval=0,
    #     max_steps=default_params['max_resource_level']) for dataloader_num_workers in range(5)]

    # tuner_job_name = 'keep-in-memory-true'
    # points_to_evaluate = [dict(
    #     learning_rate=1e-6,
    #     weight_decay=1e-6,
    #     dataloader_num_workers=0,
    #     # st_instance_type=sp.choice(instance_types),
    #     epochs=1,
    #     # dataset_path=default_params['dataset_path'],
    #     keep_in_memory=True,
    #     log_interval=100,
    #     eval_interval=0,
    #     max_steps=default_params['max_resource_level']) for dataloader_num_workers in range(1)]
    #
    # n_workers = len(points_to_evaluate)

    metrics_to_be_defined = ('loss', 'learning_rate', 'epoch', 'eval_loss', 'eval_accuracy', 'eval_runtime',
                             'eval_samples_per_second', 'train_runtime', 'train_samples_per_second', 'elapsed_time')
    metric_definitions = [{'Name': k, 'Regex': f'{k}=([0-9\.]+)'} for k in metrics_to_be_defined]

    # Define Hugging Face SageMaker estimator
    root = Path(syne_tune.__path__[0]).parent
    huggingface_estimator = HuggingFace(
        entry_point=str(benchmark['script']),
        # source_dir=str(script_path.parent)
        # TODO you can try with instance-type='local'
        # TODO make use of launch_hpo.py
        base_job_name='hpo-transformer',
        instance_type='ml.g5.xlarge',  # instance-type given here are override by value sampled from `st_instance_type`.
        max_run=single_job_max_run,  # timeout after 15min
        instance_count=1,
        transformers_version='4.4',
        pytorch_version='1.6',
        py_version='py36',
        role=get_execution_role(),
        dependencies=[root / "benchmarking"],
        metric_definitions=metric_definitions,
        # environment=dict(HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',),
                         # TRANSFORMERS_CACHE='/opt/ml/input/data/hfcache'),
        disable_profiler=True,
        debugger_hook_config=False,
    )

    # SageMaker backend
    backend = SageMakerBackend(
        sm_estimator=huggingface_estimator,
        metrics_names=[metric],
        inputs={
            'train':   's3://sagemaker-us-west-2-640549960621/samples/datasets/launch_huggingface_sweep_ag/train',
            'eval':    's3://sagemaker-us-west-2-640549960621/samples/datasets/launch_huggingface_sweep_ag/eval',
            'hfcache': 's3://sagemaker-us-west-2-640549960621/samples/hfcache'},
    )

    # Random search without stopping
    scheduler = RandomSearch(
        config_space,
        mode=mode,
        metric=metric,
        random_seed=random_seed,
        # points_to_evaluate=points_to_evaluate,  # TODO dont miss this
    )

    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        tuner_name=tuner_job_name,
        max_failures=10000000,
    )
    tuner.run()

    # remote_launcher = RemoteLauncher(
    #     tuner=tuner,
    #     instance_type='ml.m5.large',  # TODO you can use instance_type='local'
    #     tuner_name=tuner_job_name,
    #     dependencies=[str(root / "benchmarking")],
    #     sleep_time=5.0,
    # )
    # remote_launcher.run(wait=False)