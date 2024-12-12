import json
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from gluonts.core.component import validated
from gluonts.dataset import DataEntry, Dataset
from gluonts.dataset.arrow import ArrowFile
from gluonts.dataset.common import DatasetCollection, MetaData
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import AbstractBaseSplitter, slice_data_entry
from gluonts.itertools import Cyclic
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    Identity,
    InstanceSampler,
    SampleTargetDim,
    SetField,
    SimpleTransformation,
    TargetDimIndicator,
    Transformation,
)


class DateTransform(SimpleTransformation):
    @validated()
    def __init__(
        self,
        output_field: str,
        freq: str,
        # Add any additional arguments here
    ):
        super().__init__()
        self.output_field = output_field
        self.freq = freq

    def transform(self, data: DataEntry) -> DataEntry:
        if self.freq == "MS":
            freq = "M"
        elif self.freq == "QS":
            freq = "3M"
        elif self.freq == "AS":
            freq = "Y"
        else:
            freq = self.freq
        data[self.output_field] = pd.Period(
            data[self.output_field], freq=freq
        )  # Replace with your transformed data
        return data


class TargetDtype(SimpleTransformation):
    @validated()
    def __init__(
        self,
        output_field: str,
    ):
        super().__init__()
        self.output_field = output_field

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = np.float32(
            data[self.output_field]
        )  # Replace with your transformed data
        return data


class SingleInstanceSampler(InstanceSampler):
    """
    Randomly pick a single valid window in the given time series.
    This fix the bias in ExpectedNumInstanceSampler which leads to varying sampling frequency
    of time series of unequal length, not only based on their length, but when they were sampled.
    """

    """End index of the history"""

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        window_size = b - a + 1
        if window_size <= 0:
            return np.array([], dtype=int)
        indices = np.random.randint(window_size, size=1)
        return indices + a


class BaseSplitter(ABC):
    """
    Base class for all other splitter.
    """

    @abstractmethod
    def training_entry(self, entry: DataEntry) -> DataEntry:
        pass

    @abstractmethod
    def validation_entry(self, entry: DataEntry) -> DataEntry:
        pass

    def split(self, dataset: Dataset) -> Tuple["TrainingDataset", "ValDataset"]:
        return (
            TrainingDataset(dataset=dataset, splitter=self),
            ValDataset(dataset=dataset, splitter=self),
        )

    def generate_training_entries(
        self, dataset: Dataset
    ) -> Generator[DataEntry, None, None]:
        yield from map(self.training_entry, dataset)

    def generate_validation_entries(
        self, dataset: Dataset
    ) -> Generator[DataEntry, None, None]:
        yield from map(self.validation_entry, dataset)


@dataclass
class TrainingDataset:
    dataset: Dataset
    splitter: BaseSplitter

    def __iter__(self) -> Generator[DataEntry, None, None]:
        return self.splitter.generate_training_entries(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class ValDataset:
    dataset: Dataset
    splitter: BaseSplitter

    def __iter__(self) -> Generator[DataEntry, None, None]:
        return self.splitter.generate_validation_entries(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class TrainValidationsplitter(BaseSplitter):
    """
    A splitter that slices training and test data based on a ``pandas.Period``.

    Training entries obtained from this class will be limited to observations
    up to (including) the given ``date``.

    Parameters
    ----------
    date
        ``pandas.Period`` determining where the training data ends.
    """

    offset: int

    def training_entry(self, entry: DataEntry) -> DataEntry:
        return slice_data_entry(entry, slice(None, self.offset))

    def validation_entry(
        self,
        entry: DataEntry,
    ) -> DataEntry:
        return slice_data_entry(entry, slice(self.offset, None))


def get_transformation(path):
    freq = "10L"  # metadata.freq
    transformation = Identity()
    transformation += DateTransform(output_field=FieldName.START, freq="10ms")
    transformation += TargetDtype(output_field=FieldName.TARGET)
    whole_dataset = DatasetCollection(
        datasets=[
            transformation.apply(ArrowFile(path=path + filename))
            for filename in os.listdir(path)
        ]
    )

    return whole_dataset, freq


# def get_transformation(path):
#     with open(path + "metadata.json", "r") as f:
#         metadata_dict = json.load(f)
#         metadata = MetaData(**metadata_dict)
#     freq = metadata.freq
#     transformation = Identity()
#     transformation += DateTransform(output_field=FieldName.START, freq=freq)
#     transformation += TargetDtype(output_field=FieldName.TARGET)
#     # transformation += SetField(output_field="item_id", value=data_id)
#     train_shards = [
#         transformation.apply(ArrowFile(path=arrow_path))
#         for arrow_path in Path(path + "train/").glob("*.arrow")
#     ]
#     whole_dataset = DatasetCollection(datasets=train_shards)

#     # val_shards = [transformation.apply(ArrowFile(path=arrow_path)) for arrow_path in Path(path+"val/").glob("*.arrow")]
#     # val_dataset = DatasetCollection(datasets=val_shards)
#     splitter = TrainValidationsplitter(
#         offset=-6000
#     )  # no of datapoints to validation set in same series
#     train_dataset, val_data = splitter.split(whole_dataset)

#     splitter = TrainValidationsplitter(
#         offset=-3000
#     )  # no of datapoints to test set in same series
#     val_dataset, test_dataset_indist = splitter.split(val_data)
#     test_shards = [
#         transformation.apply(ArrowFile(path=arrow_path))
#         for arrow_path in Path(path + "test/").glob("*.arrow")
#     ]
#     test_dataset = DatasetCollection(datasets=test_shards)

#     return train_dataset, val_dataset, test_dataset_indist, test_dataset, freq


def get_test_data(path):
    freq = "10ms"
    transformation = Identity()
    transformation += DateTransform(output_field=FieldName.START, freq=freq)
    transformation += TargetDtype(output_field=FieldName.TARGET)

    test_shards = [
        transformation.apply(ArrowFile(path=arrow_path))
        for arrow_path in Path(path).glob("*.arrow")
    ]
    test_dataset = DatasetCollection(datasets=test_shards)

    return test_dataset, freq


if __name__ == "__main__":
    # get args
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="NMT_v1/")
    args = parser.parse_args()

    train_ds, val_ds, test_data_indist, test_ds, freq = get_transformation(args.path)
    print("done")
