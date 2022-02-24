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
from typing import List, Optional, Tuple

from syne_tune.blackbox_repository.blackbox import Blackbox
import numpy as np
import pandas as pd
import tqdm

import boto3

import syne_tune.experiments
from syne_tune.util import s3_experiment_path


def metrics_for_configuration(
        blackbox: Blackbox, config: dict, resource_attr: str,
        fidelity_range: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None) -> List[dict]:
    """
    Returns all results for configuration `config` at fidelities in range
    `fidelity_range`.

    :param blackbox: Blackbox
    :param config: Configuration
    :param resource_attr: Name of resource attribute
    :param fidelity_range: Range [min_f, max_f], only fidelities in this range
        (both ends inclusive) are returned. Default is no filtering
    :param seed: Seed for queries to blackbox. Drawn at random if not
        given
    :return: List of result dicts

    """
    all_fidelities = blackbox.fidelity_values
    assert all_fidelities is not None, \
        "Blackbox must come with fidelities"
    res = []
    if fidelity_range is None:
        fidelity_range = (min(all_fidelities), max(all_fidelities))
    else:
        assert len(fidelity_range) == 2 \
               and fidelity_range[0] <= fidelity_range[1], \
        f"fidelity_range = {fidelity_range} must be tuple (min, max), min <= max"
    objective_values = blackbox._objective_function(config, seed=seed)
    for fidelity, value in enumerate(all_fidelities):
        if value >= fidelity_range[0] and value <= fidelity_range[1]:
            res_dict = dict(zip(blackbox.objectives_names,
                                objective_values[fidelity]))
            res_dict[resource_attr] = value
            res.append(res_dict)
    return res


def download_cloudwatch_metrics_and_save_to_csv(
        tuner_job_name, fields_to_extract, boto3_client_sagemaker, boto3_resource_cloudwatch,
        save_csv_to_tuner_s3_folder=True, force_download=False):

    try:
        df = pd.read_csv(s3_experiment_path(tuner_name=tuner_job_name) + '/cloudwatch_metrics.csv')
        file_exists_on_s3 = True
        print(f"File found on S3")
    except FileNotFoundError:
        file_exists_on_s3 = False

    if (not file_exists_on_s3) or force_download:
        if (not file_exists_on_s3):
            print(f"File not found on S3, processing {tuner_job_name}")
        elif force_download:
            print(f"File found on S3 but force_download=True, so processing {tuner_job_name}")

        tuning_experiment_results = syne_tune.experiments.load_experiment(tuner_job_name)

        trial_ids = tuning_experiment_results.results.trial_id.unique().tolist()
        print(f'Skipping the failed or missing trial_ids: {set(range(max(trial_ids) + 1)) - set(trial_ids)}', flush=True)

        rows = []
        for trial_id in tqdm.tqdm(trial_ids):
            job_details = boto3_client_sagemaker.describe_training_job(TrainingJobName=f'{tuner_job_name}-{trial_id}')
            dimensions = [{'Name': 'Host', 'Value': f'{tuner_job_name}-{trial_id}/algo-1'}]

            row = [tuner_job_name, trial_id, ]
            for metric_name, window_aggregation_statistic, training_aggregation_function in fields_to_extract:
                metric = boto3_resource_cloudwatch.Metric("/aws/sagemaker/TrainingJobs", metric_name)
                response = metric.get_statistics(
                    Dimensions=dimensions,
                    StartTime=job_details['TrainingStartTime'],
                    EndTime=job_details['TrainingEndTime'],
                    Period=60,  # 1-min metric windows as per SageMaker default
                    Statistics=[window_aggregation_statistic, ],
                )
                row.append(training_aggregation_function(
                    [datapoint[window_aggregation_statistic] for datapoint in response['Datapoints']]))
            rows.append(row)
        df = pd.DataFrame(rows, columns=['tuner_job_name', 'trial_id'] + [v[0] for v in fields_to_extract])

    if save_csv_to_tuner_s3_folder and (not file_exists_on_s3):
        df.to_csv(s3_experiment_path(tuner_name=tuner_job_name) + '/cloudwatch_metrics.csv')

    return df


if __name__ == '__main__':
    boto3_client_sagemaker = boto3.client('sagemaker')
    boto3_resource_cloudwatch = boto3.resource('cloudwatch')

    # (metric name, 1-min-window aggregation statistic ['Average', 'Maximum'], entire-training aggregation function)
    fields_to_extract = [
        ('CPUUtilization', 'Average', np.max),
        ('MemoryUtilization', 'Average', np.max),
        ('GPUUtilization', 'Average', np.max),
        ('GPUMemoryUtilization', 'Average', np.max),
    ]

    tuner_job_name = 'speed-bs-it-nw-new-2022-02-21-18-05-01-921'

    download_cloudwatch_metrics_and_save_to_csv(
        tuner_job_name, fields_to_extract, boto3_client_sagemaker, boto3_resource_cloudwatch,
        save_csv_to_tuner_s3_folder=True)
