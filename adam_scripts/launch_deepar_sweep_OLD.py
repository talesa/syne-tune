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
Example for how to train a DeepAR model on a synthetic dataset using the SageMaker Framework.
"""
import logging
import math
from pathlib import Path

import numpy as np

import sagemaker
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

    prefix = "sagemaker/DEMO-deepar"

    sagemaker_session = sagemaker.Session()
    role = "arn:aws:iam::640549960621:role/service-role/AmazonSageMaker-ExecutionRole-20220114T195275"
    bucket = sagemaker_session.default_bucket()

    s3_data_path = f"{bucket}/{prefix}/data"
    s3_output_path = f"{bucket}/{prefix}/output"

    freq = "H"
    prediction_length = 48
    context_length = 72

    t0 = "2016-01-01 00:00:00"
    data_length = 1000
    num_ts = 400
    period = 24

    # EDIT THIS SECTION
    random_seed = 31415927

    # instance_types = select_instance_type(min_gpu=1, max_cost_per_hour=10.0)
    instance_types = [
        'ml.p3.2xlarge',
        'ml.p2.xlarge',
        'ml.p2.8xlarge',
        # 'ml.g4dn.xlarge',
        # 'ml.g4dn.2xlarge',
        # 'ml.g4dn.4xlarge',
        # 'ml.g4dn.8xlarge',
        # 'ml.g4dn.12xlarge',
        # 'ml.g5.xlarge',
        # 'ml.g5.2xlarge',
        # 'ml.g5.4xlarge',
        # 'ml.g5.8xlarge',
        # 'ml.g5.12xlarge',
        # 'ml.g5.24xlarge',
        'ml.c5n.xlarge',
        'ml.c5n.2xlarge',
        'ml.c5.xlarge',
        'ml.c5.2xlarge',
        'ml.c4.xlarge',
        'ml.c4.2xlarge',
        'ml.m4.xlarge',
        'ml.m4.2xlarge',
        'ml.m5.large',
        'ml.m5.xlarge',
        'ml.m5.2xlarge',
    ]

    # n_workers = 1
    n_workers = len(instance_types)

    # per_device_train_batch_size_list = [52, 60, 68, 76, 84, 96]
    # dataloader_num_workers_list = [1]
    # seeds = [0]

    # config_space = dict(
    #     per_device_train_batch_size=sp.choice(per_device_train_batch_size_list),  # TODO select me
    #     # per_device_train_batch_size=26,
    #     fp16=1,
    #     n_train_data=25000,
    #     epochs=3,
    #     per_device_eval_batch_size=1,
    #     n_eval_data=1,
    #     log_interval=100,
    #     eval_interval=0,
    #     learning_rate=1e-6,
    #     weight_decay=1e-6,
    #     dataloader_num_workers=sp.choice(dataloader_num_workers_list),
    #     st_instance_type=sp.choice(instance_types),  # TODO select me
    #     # st_instance_type='ml.g4dn.xlarge',
    #     seed=sp.choice(seeds),  # TODO select me
    # )
    # single_job_max_run = 60*60  # 10min
    stop_criterion = StoppingCriterion(
        # max_wallclock_time=10 * 60 * 60,  # 10h
        # max_num_trials_finished=int(np.prod(tuple(map(len, (per_device_train_batch_size_list, dataloader_num_workers_list, instance_types, seeds))))),
        # max_num_trials_completed=1000,
        max_num_trials_completed=len(instance_types),
    )

    config_space = {
        "time_freq": freq,
        "context_length": str(context_length),
        "prediction_length": str(prediction_length),
        "likelihood": "gaussian",
        "epochs": "200",
        "mini_batch_size": "128",
        "learning_rate": "0.001",
        "dropout_rate": "0.05",
        # "early_stopping_patience": "10",
        # "st_instance_type": "ml.c4.2xlarge",
        "st_instance_type": sp.choice(instance_types),
    }

    # tuner_job_name = 'deepar-small'
    # config_space.update({
    #     "num_cells": "30",
    #     "num_layers": "1",
    # })

    tuner_job_name = 'deepar-large'
    config_space.update({
        "num_cells": "100",
        "num_layers": "4",
    })

    # metrics_to_be_defined = ('loss', 'learning_rate', 'epoch', 'eval_loss', 'eval_accuracy', 'eval_runtime',
    #                          'eval_samples_per_second', 'train_runtime', 'train_samples_per_second', 'elapsed_time')
    # metric_definitions = [{'Name': k, 'Regex': f'{k}=([0-9\.]+)'} for k in metrics_to_be_defined]

    # Define DeepAR SageMaker estimator
    image_uri = sagemaker.image_uris.retrieve("forecasting-deepar", region='us-west-2', image_scope="training")
    estimator = sagemaker.estimator.Estimator(
        image_uri=image_uri,
        instance_count=1,
        role=role,
        instance_type="ml.c4.xlarge",
        base_job_name=tuner_job_name,
        output_path=f"s3://{s3_output_path}",
    )

    # SageMaker backend
    data_channels = {
        "train": f"s3://{s3_data_path}/train/",
        "test": f"s3://{s3_data_path}/test/"
    }
    backend = SageMakerBackend(
        sm_estimator=estimator,
        # metrics_names=[metric],
        inputs=data_channels,
    )

    # Random search without stopping
    scheduler = RandomSearch(
        config_space,
        # mode=mode,
        metric='train:loss',
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
    #     # dependencies=[str(root / "benchmarking")],
    #     sleep_time=5.0,
    # )
    # remote_launcher.run(wait=False)