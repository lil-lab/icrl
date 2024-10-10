from datetime import datetime
import json
import logging
import os
from logging import Logger
import pickle
from typing import Any, Dict

import numpy as np
import pytz
import torch


def _prepare_logger(output_dir: str) -> Logger:
    """
    Prepare a logger that logs messages to both a file and the console.

    Args:
        output_dir (str): The directory where the log file will be saved.

    Returns:
        Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(filename)s - %(levelname)s - %(message)s")

    # Log to file
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def _prepare_output_dir(base_dir: str = "./runs/") -> str:
    """
    Prepare the output directory for saving logs and other outputs.

    Args:
        base_dir (str): The base directory where the output directory will be created.

    Returns:
        str: The path to the created output directory.
    """
    experiment_dir = os.path.join(
        base_dir, datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


# Prepare the output directory and logger
output_dir = _prepare_output_dir()
logger = _prepare_logger(output_dir)

def get_logger() -> Logger:
    """
    Get the configured logger instance.

    Returns:
        Logger: Configured logger instance.
    """
    return logger

def get_output_dir() -> str:
    """
    Get the path to the output directory.

    Returns:
        str: Path to the output directory.
    """
    return output_dir

def save_json(data: object, json_path: str) -> None:
    """
    Save data to a JSON file.

    Args:
        data (object): The data to be saved.
        json_path (str): The path to the JSON file.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_pickle(data: object, pickle_path: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data (object): The data to be saved.
        pickle_path (str): The path to the pickle file.
    """
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)

def save(data: object, path: str) -> None:
    """
    Save data to a file. The file format is determined by the file extension.

    Args:
        data (object): The data to be saved.
        path (str): The path to the file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path.endswith(".npy"):
        np.save(path, data)
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    elif path.endswith(".pt"):
        torch.save(data, path)
    elif path.endswith(".pickle"):
        with open(path, "wb") as f:
            pickle.dump(data, f)

def read_pickle(pickle_path: str) -> Any:
    """
    Read data from a pickle file.

    Args:
        pickle_path (str): The path to the pickle file.

    Returns:
        Any: The data read from the pickle file.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def read_json(json_path: str) -> Dict:
    """
    Read data from a JSON file.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        Dict: The data read from the JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read(path: str) -> Any:
    """
    Read data from a file. The file format is determined by the file extension.

    Args:
        path (str): The path to the file.

    Returns:
        Any: The data read from the file.
    """
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".json"):
        return read_json(path)
    elif path.endswith(".pt"):
        return torch.load(path)
    elif path.endswith(".pickle"):
        return read_pickle(path)

# Default data directory
data_dir = "./data/"

def get_data_dir() -> str:
    """
    Get the path to the data directory.

    Returns:
        str: Path to the data directory.
    """
    return data_dir

# Default plots directory
plots_dir = "./plots/"

def get_plots_dir() -> str:
    """
    Get the path to the plots directory.

    Returns:
        str: Path to the plots directory.
    """
    return plots_dir