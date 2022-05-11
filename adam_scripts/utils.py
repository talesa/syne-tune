import random
import string
import subprocess

import pandas as pd

import syne_tune.experiments


def concatenate_syne_tune_experiment_results(experiment_names):
    dfs_to_concat = list()
    trial_id_max = -1
    for tuner_job_name in experiment_names:
        df_temp = syne_tune.experiments.load_experiment(tuner_job_name).results
        df_temp['trial_id'] += trial_id_max + 1
        trial_id_max = df_temp['trial_id'].max()
        dfs_to_concat.append(df_temp)
    df = pd.concat(dfs_to_concat).reset_index()
    return df


def upload_df_to_team_bucket(df, s3_path):
    temp_path = f"/tmp/{''.join(random.choice(string.ascii_lowercase) for i in range(10))}.csv.zip"
    df.to_csv(temp_path)
    subprocess.run(
        ['aws', 's3', 'cp',
         temp_path,
         s3_path,
         '--profile', 'mnemosyne'])
