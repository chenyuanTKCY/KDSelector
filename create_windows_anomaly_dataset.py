import sys
import os
from tqdm import tqdm
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.config import *

def create_metric_dataset(
    name,
    save_dir,
    data_path,
    metric_path,
    window_size,
    metric,
):
    """Generates a new dataset from given dataset using metrics instead of time series values.

    :param name: the name of the experiment
    :param save_dir: directory in which to save the new dataset
    :param data_path: path to dataset to be divided
    :param metric_path: path to metrics of the dataset
    :param window_size: the size of the window timeseries will be split to
    :param metric: the specific metric to read
    """
    # Form new dataset's name
    name = '{}_Anomaly_Label_{}'.format(name, window_size)

    # Load datasets
    dataloader = DataLoader(data_path)
    datasets = dataloader.get_dataset_names()
    x, y, fnames = dataloader.load(datasets)

    # Load metrics
    metricsloader = MetricsLoader(metric_path)
    metrics_data = metricsloader.read(metric)

    # Delete any data not in metrics (some timeseries metric scores were not computed)
    idx_to_delete = [i for i, f in enumerate(fnames) if f not in metrics_data.index]
    if len(idx_to_delete) > 0:
        for idx in sorted(idx_to_delete, reverse=True):
            del x[idx]
            del y[idx]
            del fnames[idx]
    metrics_data = metrics_data.loc[fnames]

    # Create subfolder for each dataset
    for dataset in datasets:
        Path(os.path.join(save_dir, name, dataset)).mkdir(parents=True, exist_ok=True)

    # Save new dataset with metrics data
    for metric_vals, fname in tqdm(zip(metrics_data.values, fnames), total=len(metrics_data), desc='Save metrics dataset'):
        fname_split = fname.split('/')
        dataset_name = fname_split[-2]
        ts_name = fname_split[-1].split('.')[0]

        # Ensure metric_vals is two-dimensional
        if metric_vals.ndim == 1:
            metric_vals = metric_vals.reshape(1, -1)

        # Prepare column names
        col_names = ["metric_{}".format(i) for i in range(metric_vals.shape[1])]

        # Create DataFrame
        df = pd.DataFrame(metric_vals, columns=col_names)
        csv_path = os.path.join(save_dir, name, dataset_name, f"{ts_name}.csv")
        df.to_csv(csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create a dataset of metrics for time series windows.'
    )
    parser.add_argument('--name', type=str, default="TSB")
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--metric_path', type=str, default="path_to_metrics")
    parser.add_argument('--window_size', type=int, required=True)
    parser.add_argument('--metric', type=str, default='AUC_PR')

    args = parser.parse_args()
    create_metric_dataset(
        name=args.name,
        save_dir=args.save_dir,
        data_path=args.path,
        metric_path=args.metric_path,
        window_size=args.window_size,
        metric=args.metric,
    )