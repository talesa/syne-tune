from argparse import ArgumentParser
import numpy as np
import pandas as pd
import tqdm

import boto3

import syne_tune.experiments
from syne_tune.util import s3_experiment_path


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
        if not file_exists_on_s3:
            print(f"File not found on S3, processing {tuner_job_name}")
        elif force_download:
            print(f"File found on S3 but force_download=True, so processing {tuner_job_name}")

        tuning_experiment_results = syne_tune.experiments.load_experiment(tuner_job_name, force_download=force_download)

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
    parser = ArgumentParser()
    parser.add_argument('-t', '--tuner_job_name', type=str, required=True)

    args, _ = parser.parse_known_args()

    boto3_client_sagemaker = boto3.client('sagemaker')
    boto3_resource_cloudwatch = boto3.resource('cloudwatch')

    # (metric name, 1-min-window aggregation statistic ['Average', 'Maximum'], entire-training aggregation function)
    fields_to_extract = [
        ('CPUUtilization', 'Average', np.max),
        ('MemoryUtilization', 'Average', np.max),
        ('GPUUtilization', 'Average', np.max),
        ('GPUMemoryUtilization', 'Average', np.max),
    ]

    download_cloudwatch_metrics_and_save_to_csv(
        args.tuner_job_name, fields_to_extract, boto3_client_sagemaker, boto3_resource_cloudwatch,
        save_csv_to_tuner_s3_folder=True, force_download=True)
