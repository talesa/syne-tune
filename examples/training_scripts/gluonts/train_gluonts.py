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
Trains a simple time forecasting model using GluonTS.
"""
import logging
import os
from pathlib import Path

from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.deepar import DeepAREstimator

from syne_tune import Reporter
from argparse import ArgumentParser
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.callback import Callback
from gluonts.core.component import validated
from gluonts.dataset.repository.datasets import get_dataset, default_dataset_path

# import mxnet as mx
# import torch


class GluontsTuneReporter(Callback):
    @validated()
    def __init__(self, validation_data):
        self.reporter = Reporter()
        self.val_dataset = validation_data
        # number of samples used in evaluation
        self.num_samples = 10

    def set_estimator(self, estimator):
        # since the callback does not provide all information to compute forecasting metrics, we set the estimator
        # in order to have the transformation.
        self.estimator = estimator

    def compute_metrics(self, predictor, dataset):
        forecast_it, ts_it = make_evaluation_predictions(
            dataset, predictor=predictor, num_samples=self.num_samples
        )
        # adding more than one worker throws an error, not sure why
        agg_metrics, item_metrics = Evaluator(num_workers=0)(
            ts_it, forecast_it, num_series=len(dataset),
        )
        return agg_metrics

    def on_validation_epoch_end(self, epoch_no: int, epoch_loss: float, training_network, trainer) -> bool:
        metrics = {
            "epoch_no": epoch_no + 1,
            "epoch_loss": epoch_loss,
        }
        predictor = self.estimator.create_predictor(
            transformation=self.estimator.create_transformation(),
            trained_network=training_network,
        )
        metrics["mean_wQuantileLoss"] = self.compute_metrics(predictor, self.val_dataset)["mean_wQuantileLoss"]
        self.reporter(**metrics)
        return True


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_cells', type=int, default=40)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default="electricity")
    parser.add_argument('--only_benchmark_speed', type=int, default=0)

    args, _ = parser.parse_known_args()

    if args.only_benchmark_speed != 0:
        root.info(
            "args.only_benchmark_speed=1 so overriding defaults to args.epochs=2 and args.num_batches_per_epoch=1")
        args.epochs = 2
        num_batches_per_epoch = 1
    else:
        num_batches_per_epoch = 500

    if "SM_CHANNEL_TRAIN" in os.environ:
        dataset_path = Path(os.environ["SM_CHANNEL_TRAIN"])
    else:
        dataset_path = default_dataset_path

    dataset = get_dataset(args.dataset, path=dataset_path, regenerate=False)
    prediction_length = dataset.metadata.prediction_length
    freq = dataset.metadata.freq
    context_length = prediction_length

    # print(f'mx.context.num_gpus(): {mx.context.num_gpus()}')
    # # print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    #
    # stream = os.popen('nvcc --version')
    # output = stream.read()
    # print('nvcc --version')
    # print(output)

    # TODO, we should provide a validation split in all our datasets
    #  for now we use the test as the validation.
    validation_data = dataset.test
    reporter = GluontsTuneReporter(validation_data=validation_data)
    trainer = Trainer(
        learning_rate=args.lr,
        epochs=args.epochs,
        num_batches_per_epoch=num_batches_per_epoch,
        callbacks=[reporter],
        # ctx=mx.context.gpu() if mx.context.num_gpus() > 0 else None,
    )
    estimator = DeepAREstimator(
        num_layers=args.num_layers,
        num_cells=args.num_cells,
        batch_size=args.batch_size,
        prediction_length=prediction_length,
        context_length=context_length,
        freq=freq,
        trainer=trainer,
    )
    # required to pass additional context so that the callback can compute forecasting metrics
    reporter.set_estimator(estimator)

    predictor = estimator.train(
        dataset.train,
        validation_data=validation_data,
        num_workers=None
    )
