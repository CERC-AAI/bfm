import logging
import os
import sys
from argparse import ArgumentParser

# import wandb
from hashlib import sha1
from itertools import islice
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import typer
import yaml
from chronos import ChronosPipeline
from gluonts.dataset.arrow import ArrowFile
from gluonts.dataset.common import DatasetCollection, MetaData
from gluonts.dataset.repository import dataset_names, get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MAPE, MASE, MSE, SMAPE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm

sys.path.append("./data/")
from read_dataset import get_transformation

app = typer.Typer(pretty_exceptions_enable=False)


def get_latest_checkpoint(directory, n_gpus):
    if n_gpus == 1:
        rng_path = ["rng_state.pth"]
    else:
        rng_path = ["rng_state_" + str(i) + ".pth" for i in range(0, n_gpus)]
    required_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    ] + rng_path  #'rng_state.pth',

    if os.path.isdir(directory):
        checkpoints = [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            if all(
                os.path.exists(
                    os.path.join(max(checkpoints, key=os.path.getmtime), file)
                )
                for file in required_files
            ):
                return max(checkpoints, key=os.path.getmtime)
            else:
                return min(checkpoints, key=os.path.getmtime)
    return None


def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def load_and_split_dataset(backtest_config: dict):
    test_path = backtest_config["test_path"]
    # dataset_name = backtest_config["name"]
    # offset = backtest_config["offset"]
    # prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]

    gts_dataset, _freq = get_transformation(test_path)
    pred_length = 64  # 64
    _, test_template = split(gts_dataset, offset=-pred_length)
    test_data = test_template.generate_instances(pred_length, windows=num_rolls)

    return test_data, pred_length


def generate_sample_forecasts(
    test_data_input: Iterable,
    pipeline: ChronosPipeline,
    prediction_length: int,
    batch_size: int,
    num_samples: int,
    **predict_kwargs,
):
    # Generate forecast samples
    forecast_samples = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]
        forecast_samples.append(
            pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                limit_prediction_length=False,
                **predict_kwargs,
            ).numpy()
        )
    forecast_samples = np.concatenate(forecast_samples)

    # Convert forecast samples into gluonts SampleForecast objects
    sample_forecasts = []
    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts


# @app.command()
def main(
    config_path: Path,
    directory_path: Path,
    device: str = "cuda",
    seed: int = 0,
    model_id: str = None,
    torch_dtype: str = "bfloat16",
    batch_size: int = 16,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    primary_checkpoint_dir = directory_path + "/" + str(seed)
    if model_id is not None:
        checkpoint_dir = directory_path
        primary_checkpoint_dir = directory_path
    else:
        checkpoint_dir = get_latest_checkpoint(primary_checkpoint_dir, 64)
    print("checkpoint_dir", checkpoint_dir)

    # experiment_name = directory_path.split("/")[-1]
    # full_experiment_name = experiment_name + "-seed-" + str(seed)

    # experiment_id = sha1(full_experiment_name.encode("utf-8")).hexdigest()[
    #     :8
    # ]  # Replace with your actual experiment ID

    if model_id is not None:
        print("Loading model from Hugging Face model hub", checkpoint_dir)
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            cache_dir=checkpoint_dir,
            local_files_only=True,
            device_map=device,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = ChronosPipeline.from_pretrained(
            checkpoint_dir,
            cache_dir=checkpoint_dir,
            local_files_only=True,
            device_map=device,
            torch_dtype=torch_dtype,
        )

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    for config in backtest_configs:
        dataset_name = config["name"]
        # prediction_length = config["prediction_length"]

        logger.info(f"Loading {dataset_name}")
        test_data, prediction_length = load_and_split_dataset(backtest_config=config)

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        if "fmri" in dataset_name:
            pipeline.model.config.context_length = 256

        sample_forecasts = generate_sample_forecasts(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        logger.info(f"Evaluating forecasts for {dataset_name}")
        metrics = (
            evaluate_forecasts(
                sample_forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                    MAPE(),
                    # MAE(),
                    SMAPE(),
                ],
                # batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
        result_rows.append({"dataset": dataset_name, "seed": seed, **metrics[0]})
        print(dataset_name)
        print(metrics[0])

        test_data = list(test_data)
        len_test_data = (len(test_data) // 4) * 4
        if len_test_data >= 52:
            len_test_data = 52
        start_index = 0
        if not os.path.exists(primary_checkpoint_dir + "/metrics"):
            os.makedirs(primary_checkpoint_dir + "/metrics")
        if not os.path.exists(primary_checkpoint_dir + "/forecasts"):
            os.makedirs(primary_checkpoint_dir + "/forecasts")
        if model_id is not None and not os.path.exists("zero_shot"):
            os.makedirs("zero_shot")
        for it in range(4, 52, 4):
            # print(it)
            plt.rcParams.update({"font.size": 10})
            plt.figure(figsize=(20, 25))
            plt.rcParams.update({"font.size": 15})
            item_range = [i for i in range(start_index, it)]
            start_index = it
            plt.gcf().tight_layout()
            for idx, item_id in enumerate(item_range):
                ax = plt.subplot(2, 2, idx + 1)
                pred_item = sample_forecasts[item_id].samples
                test_item = test_data[item_id]
                x = np.concatenate(
                    [
                        test_item[0]["target"][-3 * prediction_length :],
                        test_item[1]["target"],
                    ]
                )
                print(len(test_item[0]["target"][-3 * prediction_length :]))
                print(len(test_item[1]["target"]))
                low, median, high = np.quantile(
                    np.array(pred_item), [0.1, 0.5, 0.9], axis=0
                )

                plt.plot(range(len(x)), x, label="Ground Truth", color="C0")
                plt.plot(
                    range(len(x) - prediction_length, len(x)),
                    median,
                    color="tomato",
                    label="Median Forecast",
                )
                plt.fill_between(
                    range(len(x) - prediction_length, len(x)),
                    low,
                    high,
                    color="tomato",
                    alpha=0.3,
                    label="80% Interval",
                )
                ax.set_title(dataset_name)
                ax.grid()
            plt.gcf().tight_layout()
            plt.legend()
            plt.savefig(
                primary_checkpoint_dir
                + "/forecasts/"
                + dataset_name
                + str(it)
                + "prediction_large.svg"
            )
            plt.savefig(
                primary_checkpoint_dir
                + "/forecasts/"
                + dataset_name
                + str(it)
                + "prediction_large.png"
            )
        results_df = (
            pd.DataFrame(result_rows)
            .rename(
                {
                    "MASE[0.5]": "MASE",
                    "mean_weighted_sum_quantile_loss": "WQL",
                    "MAPE[0.5]": "MAPE",
                    "sMAPE[0.5]": "sMAPE",
                },
                axis="columns",
            )
            .sort_values(by="dataset")
        )
        if model_id is not None:
            results_df.to_csv("zero_shot/" + dataset_name + "metrics.csv", index=False)
        else:
            results_df.to_csv(
                primary_checkpoint_dir + "/metrics/" + dataset_name + "metrics.csv",
                index=False,
            )

    # Save results to a CSV file
    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {
                "MASE[0.5]": "MASE",
                "mean_weighted_sum_quantile_loss": "WQL",
                "MAPE[0.5]": "MAPE",
                "sMAPE[0.5]": "sMAPE",
            },
            axis="columns",
        )
        .sort_values(by="dataset")
    )

    if model_id is not None:
        results_df.to_csv("zero_shot/metrics.csv", index=False)
    else:
        results_df.to_csv(primary_checkpoint_dir + "/metrics_final.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    # get args
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="chronos/scripts/training/configs/chronos-tuab-inference.yaml",
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default="chronos/Experiments/chronos-tuab-tiny_tuab",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config_path = args.config_path
    main(
        config_path=config_path,
        directory_path=args.directory_path,
        device=args.device,
        seed=args.seed,
        model_id=args.model_id,
    )
