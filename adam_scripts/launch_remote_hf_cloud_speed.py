from pathlib import Path

from coolname import generate_slug
from sagemaker.pytorch import PyTorch

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path, random_string

if __name__ == '__main__':
    all_features = (
        'config_st_instance_type',
        'config_per_device_train_batch_size',
        'config_dataloader_num_workers',
        'instance_type_family',
        'GPUMemory/batch_size',
        'GPUFP32TFLOPS',
        'GPUFP32TFLOPS*num_gpu',
        'cost_per_hour',
        'num_cpu',
        'num_gpu',
        'GPUMemory',
    )
    feature_combinations = (
        ('config_st_instance_type', 'config_per_device_train_batch_size', 'config_dataloader_num_workers',),
        ('config_st_instance_type', 'GPUMemory/batch_size', 'config_dataloader_num_workers',),

        ('config_st_instance_type', 'config_per_device_train_batch_size', 'config_dataloader_num_workers', 'GPUFP32TFLOPS',),
        ('config_st_instance_type', 'GPUMemory/batch_size', 'config_dataloader_num_workers', 'GPUFP32TFLOPS',),

        ('config_st_instance_type', 'config_per_device_train_batch_size', 'config_dataloader_num_workers',
         'GPUFP32TFLOPS*num_gpu',),
        ('config_st_instance_type', 'GPUMemory/batch_size', 'config_dataloader_num_workers', 'GPUFP32TFLOPS*num_gpu',),


        ('instance_type_family', 'num_cpu', 'config_per_device_train_batch_size', 'config_dataloader_num_workers',),
        ('instance_type_family', 'cost_per_hour', 'config_per_device_train_batch_size', 'config_dataloader_num_workers',),

        ('instance_type_family', 'num_cpu', 'GPUMemory/batch_size', 'config_dataloader_num_workers',),
        ('instance_type_family', 'cost_per_hour', 'GPUMemory/batch_size', 'config_dataloader_num_workers',),
    )

    experiments_names = []
    for features in feature_combinations:
        experiment_tag = generate_slug(2)
        hash = random_string(4)

        sm_args = dict(
            entry_point="launch_hf_cloud_speed_simulated.py",
            source_dir=str(Path(__file__).parent),
            checkpoint_s3_uri=s3_experiment_path(experiment_name=experiment_tag),
            instance_type="ml.m5.large",
            instance_count=1,
            py_version="py38",
            framework_version='1.10.0',
            max_run=3600 * 72,
            role=get_execution_role(),
            dependencies=syne_tune.__path__ + benchmarking.__path__ + ['/Users/awgol/code/syne-tune/adam_scripts'],
            disable_profiler=True,
            debugger_hook_config=False,
        )

        sm_args["hyperparameters"] = {
            "features": ' '.join(features),
            'iters': 100,
            'max_cost': 60,
        }
        est = PyTorch(**sm_args)
        est.fit(job_name=f"{experiment_tag}-{hash}", wait=False)

        experiments_names.append(f"{experiment_tag}-{hash}")

    print(experiments_names)
