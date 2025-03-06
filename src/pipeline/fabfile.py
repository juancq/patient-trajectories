import fabric
from fabric import task
from invoke import run
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from typing import Optional

# File to store pipeline state
STATE_FILE = 'pipeline_state.json'


def save_state(stage: str) -> None:
    """
    Save the current state of the pipeline to a file.

    Args:
        stage (str): The name of the stage that has just completed.
    """
    state = {
        'last_completed_stage': stage,
        'timestamp': datetime.now().isoformat()
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_state() -> dict | None:
    """
    Load the current state of the pipeline from a file.

    Returns:
        A dictionary containing the state of the pipeline, or None if the file does not exist.
    """
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None

@task
def inclusion(c: fabric.connection.Connection) -> bool:
    """
    Run the inclusion/exclusion criteria stage.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    if run(f"{sys.executable} inclusion_exclusion_criteria.py", warn=True).failed:
        print("Error in inclusion/exclusion criteria stage.")
        return False
    save_state("inclusion")
    return True

@task
def filter(c: fabric.connection.Connection) -> bool:
    """
    Run the filter cohort stage.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    if run(f"{sys.executable} filter_cohort.py", warn=True).failed:
        print("Error in filter cohort stage.")
        return False
    save_state("filter")
    return True

@task
def preprocess(c: fabric.connection.Connection) -> bool:
    """
    Run the preprocess cohort stage.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    if run(f"{sys.executable} preprocess_cohort.py", warn=True).failed:
        print("Error in preprocess cohort stage.")
        return False
    save_state("preprocess")
    return True


@task
def downstream_task_data_map(c: fabric.connection.Connection) -> bool:
    """
    Converts APDC/EDDC data to HALO format, but using an index date for 
    downstream tasks.

    This task can only complete if pick_prediction_point has been run first.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    if run(f"{sys.executable} map_halo_cohort_for_finetune.py", warn=True).failed:
        print("Error in mapping data to HALO format for downstream tasks.")
        return False
    save_state("downstream_task_data_map")
    return True

@task
def data_map(c: fabric.connection.Connection) -> bool:
    """
    Run the create data map stage.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    if run(f"{sys.executable} create_data_map.py", warn=True).failed:
        print("Error in create data map stage.")
        return False
    save_state("data_map")
    return True

@task
def variable_map(c: fabric.connection.Connection) -> bool:
    """
    Run the create variable map stage.

    Args:
        c (fabric.connection.Connection): The connection to use.

    Returns:
        bool: True if the stage completed successfully, False otherwise.
    """
    if run(f"{sys.executable} create_variable_map.py", warn=True).failed:
        print("Error in create variable map stage.")
        return False
    save_state("variable_map")
    return True

@task
def run_pipeline(
    c: fabric.connection.Connection, 
    start: Optional[str] = None, 
    end: Optional[str] = None, 
    resume: bool = False
) -> None:
    """
    Run the data processing pipeline.

    :param c: Fabric's Context object (automatically provided)
    :param start: Stage to start the pipeline from (default: None)
    :param end: Stage to end the pipeline at (default: None)
    :param resume: Whether to resume from the last completed stage (default: False)
    :return: None
    """
    stages = [inclusion, filter, preprocess, data_map, variable_map, downstream_task_data_map]
    stage_names = [stage.__name__ for stage in stages]

    if resume:
        state = load_state()
        if state:
            start = stage_names[stage_names.index(state['last_completed_stage']) + 1] if state['last_completed_stage'] != stage_names[-1] else stage_names[-1]
            print(f"Resuming from stage: {start}")
        else:
            print("No previous state found. Starting from the beginning.")
            start = stage_names[0]
    elif not start:
        start = stage_names[0]

    if not end:
        end = stage_names[-1]

    try:
        start_index = stage_names.index(start)
        end_index = stage_names.index(end)
    except ValueError:
        print("Error: Invalid start or end stage specified.")
        return

    if start_index > end_index:
        print("Error: Start stage cannot come after end stage.")
        return

    for stage in stages[start_index:end_index+1]:
        print(f"Running {stage.__name__} stage...")
        if not stage(c):
            print(f"Pipeline stopped at {stage.__name__} stage.")
            return

    print("Pipeline completed successfully.")

@task(default=True)
def list_stages(c: fabric.connection.Connection) -> None:
    """List all available pipeline stages.

    :param c: Fabric connection object (automatically provided)
    :return: None
    """
    print("Available pipeline stages:")
    stages = [inclusion, filter, preprocess, data_map, variable_map]
    for stage in stages:
        print(f"- {stage.__name__}")

@task
def clean(c: fabric.connection.Connection, all: bool = False) -> None:
    """
    Clean up temporary or output files.

    :param c: Fabric connection object (automatically provided)
    :param all: If True, removes all output files including the state file
    :return: None
    """
    print("Cleaning up...")
    # Add commands to remove temporary files, e.g.:
    # run("rm -f *.tmp")
    # run("rm -rf output_directory")
    if all and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("Removed pipeline state file.")
    print("Cleanup completed.")

@task
def show_state(c: fabric.connection.Connection) -> None:
    """Show the current state of the pipeline.

    :param c: Fabric connection object (automatically provided)
    :return: None
    """
    state = load_state()
    if state:
        print(f"Last completed stage: {state['last_completed_stage']}")
        print(f"Timestamp: {state['timestamp']}")
    else:
        print("No pipeline state found.")
