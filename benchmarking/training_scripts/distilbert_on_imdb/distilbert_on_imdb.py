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
DistilBERT fine-tuned on IMDB sentiment classification task
"""
import argparse
import logging
import os
import time

from syne_tune import Reporter
from syne_tune.config_space import loguniform, add_to_argparse

METRIC_ACCURACY = 'loss'

RESOURCE_ATTR = 'step'

_config_space = {
    'learning_rate': loguniform(1e-6, 1e-4),
    'weight_decay': loguniform(1e-6, 1e-4)
}


def prepare_data(config, train_dataset, eval_dataset, seed=42):
    # Subsample data
    train_dataset = train_dataset.shuffle(seed=seed).select(range(config['n_train_data']))
    eval_dataset = eval_dataset.shuffle(seed=seed).select(range(config['n_eval_data']))

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=os.environ["SM_CHANNEL_HFCACHE"])
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True)
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    return train_dataset, eval_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    metric = load_metric(f"{os.environ['SM_CHANNEL_HFCACHE']}/accuracy")
    return metric.compute(predictions=predictions, references=labels)


def objective(config):
    trial_id = config.get('trial_id')

    # Download and prepare data
    # train_dataset, eval_dataset = load_dataset(
    #     'imdb', split=['train', 'test'], keep_in_memory=config['keep_in_memory'],
    #     cache_dir=os.environ["SM_CHANNEL_HFCACHE"])
    # # else:
    train_dataset = load_from_disk(os.environ["SM_CHANNEL_TRAIN"])
    eval_dataset = load_from_disk(os.environ["SM_CHANNEL_EVAL"])

    train_dataset, eval_dataset = prepare_data(config, train_dataset, eval_dataset)

    # Download model from Hugging Face model hub
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2,
        cache_dir=os.environ["SM_CHANNEL_HFCACHE"])

    # Define training args
    training_args = TrainingArguments(
        output_dir='./',
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        dataloader_num_workers=config['dataloader_num_workers'],
        evaluation_strategy='steps' if config['eval_interval'] != 0 else "no",
        eval_steps=config['eval_interval'] // (config['per_device_train_batch_size'] * float(os.environ['SM_NUM_GPUS']))
                   if config['eval_interval'] != 0 else 1,
        logging_strategy='steps',
        logging_steps=config['log_interval'] if config['log_interval'] != 0 else
        round(config['n_train_data'] / config['per_device_train_batch_size'] / float(os.environ['SM_NUM_GPUS'])),
        learning_rate=float(config['learning_rate']),
        weight_decay=float(config['weight_decay']),
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True,
        # avoid filling disk
        save_strategy="no",
        seed=int(config['seed']),
        fp16=bool(config['fp16']),
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # add a callback so that accuracy is sent to Syne Tune whenever it is computed
    class Callback(TrainerCallback):
        # def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        #     # Feed the validation accuracy back to Tune
        #     # print('metrics', flush=True)
        #     # print(metrics, flush=True)
        #     if 'eval_loss' in metrics.keys():
        #         report_dict = {RESOURCE_ATTR: state.global_step,
        #                        'loss': metrics['eval_loss']}
        #         report(**report_dict)

        def on_log(self, args, state, control, logs=None, **kwargs):
            # print('logs', flush=True)
            # print(logs, flush=True)
            # {'loss': 0.6808, 'learning_rate': 0.0, 'epoch': 1.0}
            # {'eval_loss': 0.6770151853561401, 'eval_accuracy': 0.8125, 'eval_runtime': 1.2068, 'eval_samples_per_second': 26.516, 'epoch': 1.0}
            # {'train_runtime': 206.9605, 'train_samples_per_second': 1.45, 'epoch': 1.0}
            if 'loss' in logs.keys():
                report_dict = {RESOURCE_ATTR: state.global_step,
                               'loss': logs['loss']}
                report(**report_dict)
            logs_text = ','.join(f"{k}={v}" for k, v in logs.items())
            print(f"trial={trial_id},step={state.global_step},{logs_text}", flush=True)

    trainer.add_callback(Callback())

    # Do not want to count the time to download the dataset and the model.
    ts_start = time.time()

    report = Reporter()

    # Train model
    trainer.train()

    # Evaluate model
    # eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    # eval_accuracy = eval_result['eval_accuracy']

    elapsed_time = time.time() - ts_start

    print(f"elapsed_time={elapsed_time:.2f}", flush=True)


if __name__ == '__main__':
    # Benchmark-specific imports are done here, in order to avoid import
    # errors if the dependencies are not installed (such errors should happen
    # only when the code is really called)
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
    from transformers import TrainerCallback
    import datasets
    from datasets import load_dataset, load_metric, load_from_disk

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32)
    parser.add_argument('--n_train_data', type=int, default=300*16) #25000) # TODO change me
    parser.add_argument('--n_eval_data', type=int, default=32)  # TODO change me
    parser.add_argument('--eval_interval', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=0)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp16', type=int, default=0)
    # parser.add_argument('--keep_in_memory', type=int, default=0)
    parser.add_argument('--trial_id', type=str)
    add_to_argparse(parser, _config_space)
    
    args, _ = parser.parse_known_args()

    objective(config=vars(args))
