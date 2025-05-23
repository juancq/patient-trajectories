import fabric
from fabric import task
from invoke import run
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Optional

@task
def test(c) -> bool:
    print('hello world')

@task
def gen_dataset(c) -> bool:
    """
    Generates the downstream task datasets.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    args = '+dataset_dir="/mnt/data_volume/apdc/study1/preprocessed/esgpt_data/study1/"'

    if run(f"{sys.executable} generate_task_dataset.py {args}", warn=True).failed:
        print("Error in inclusion/exclusion criteria stage.")
        return False
    return True

@task
def gen_embeddings(c) -> bool:
    """
    Generate embeddings for downstream tasks.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    if run(
        f"{sys.executable} generate_prediction_point_embeddings.py "
        "--config-name=study1_convert_embed --config-path=configs.py", 
        warn=True).failed:
        print("Error in filter cohort stage.")
        return False
    return True
