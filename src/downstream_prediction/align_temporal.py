import sys
import os
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import yaml
from box import Box
from loguru import logger

import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibrationDisplay


def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    #parser.add_argument('file', nargs='+', help='dataset', default=None, type=str)
    parser.add_argument('--cv', help='run cross-validation', default=False, action='store_true')
    parser.add_argument('--high', help='run cross-validation', default=False, action='store_true')
    args = parser.parse_args()
    args.cv = True

    with open('predict_config.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)
    config = Box(yaml_config)

    label_path = Path('/mnt/data_volume/apdc/study1/preprocessed/tasks/')

    for task in config.tasks:
        label_files = [label_path / f'{task}_{window}.parquet' for window in config.windows]
        df = [pl.scan_parquet(fname).select('ppn_int') for fname in label_files]

        intersection = (
            df[0].join(df[1], on='ppn_int', how='inner', coalesce=True)
                .join(df[2], on='ppn_int', how='inner', coalesce=True)
        )

        intersection = intersection.collect(streaming=True)
        intersection.write_parquet(label_path / f'{task}_window_intersection.parquet')


if __name__ == "__main__":
    main()