import asyncio
import concurrent.futures
from timeit import default_timer as timer

import syne_tune.experiments
from sagemaker import TrainingJobAnalytics

import functools

tuner_job_name = 'speed-bs-it-2022-02-07-23-12-47-916'

tuning_experiment_results = syne_tune.experiments.load_experiment(tuner_job_name)
dfr = tuning_experiment_results.results
trial_ids = dfr.trial_id.tolist()[:10]


def job(tuner_job_name, trial_id):
    return (trial_id, TrainingJobAnalytics(f'{tuner_job_name}-{trial_id}',
                                           metric_names=['train_samples_per_second']).dataframe().iloc[0, 2])


async def non_blocking(executor):
    loop = asyncio.get_event_loop()
    blocking_tasks = []
    for trial_id in trial_ids:
        blocking_tasks.append(loop.run_in_executor(
            executor,
            functools.partial(job, tuner_job_name, trial_id)))
    completed, pending = await asyncio.wait(blocking_tasks)
    results = [t.result() for t in completed]
    return results


def blocking():
    results = []
    for trial_id in trial_ids:
        results.append(job(tuner_job_name, trial_id))
    return results


if __name__ == '__main__':
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10, )
    event_loop = asyncio.get_event_loop()

    # async next
    start = timer()
    results = event_loop.run_until_complete(non_blocking(executor))
    print(results)
    elapsed = (timer() - start)
    print("Non-blocking took: {}".format(elapsed))

    start = timer()
    results = blocking()
    print(results)
    elapsed = (timer() - start)
    print("Blocking took: {}".format(elapsed))
