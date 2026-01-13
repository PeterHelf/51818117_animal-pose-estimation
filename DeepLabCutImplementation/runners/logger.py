#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

def setup_file_logging(filepath: Path) -> None:
    """
    Sets up logging to a file

    Args:
        filepath: the path where logs should be saved
    """
    logging.basicConfig(
        filename=filepath,
        filemode="a",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        format="%(asctime)-15s %(message)s",
        force=True,
    )
    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(console_logger)


def destroy_file_logging() -> None:
    """Resets the logging module to log everything to the console"""
    root = logging.getLogger()
    handlers = [h for h in root.handlers]
    for handler in handlers:
        root.removeHandler(handler)


class BaseLogger(ABC):
    """Base class for logging training runs"""

    @abstractmethod
    def log_config(self, config: dict = None) -> None:
        """Logs the configuration data for a training run

        Args:
            config: the training configuration used for the run
        """

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Logs data from a training run

        Args:
            metrics: the metrics to log
            step: The global step in processing. Defaults to None.
        """

    @abstractmethod
    def save(self) -> None:
        """Saves the current training logs"""

class CSVLogger(BaseLogger):
    """Logger saving stats and metrics to a CSV file"""

    def __init__(self, train_folder: str, log_filename: str) -> None:
        """Initialize the CSVLogger class.

        Args:
            train_folder: The path of the folder containing training files.
            log_filename: The name of the file in which to store training stats
        """
        super().__init__()
        train_folder = Path(train_folder)
        self.train_folder = train_folder
        self.log_filename = log_filename
        self.log_file = train_folder / log_filename

        self._steps: list[int] = []
        self._metric_store: list[dict] = []
        self._logged_metrics: set[str] = set()

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Logs metrics from runs

        Args:
            metrics: the metrics to log
            step: The global step in processing. Defaults to None.
        """
        if step is None:
            if len(self._steps) == 0:
                step = 0
            else:
                step = self._steps[-1] + 1

        self._logged_metrics = self._logged_metrics.union(metrics.keys())
        if len(self._steps) > 0 and step == self._steps[-1]:
            self._metric_store[-1].update(metrics)
        else:
            self._steps.append(step)
            self._metric_store.append(metrics)

        self.save()

    def save(self):
        """Saves the metrics to the file system"""
        logs = self._prepare_logs()
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(logs)

    def log_config(self, config: dict = None) -> None:
        """Does not do anything as the config should already be saved

        Args:
            config: Experiment config file.
        """
        pass

    def _prepare_logs(self) -> list[list]:
        """Prepares the data to log as a list of strings"""
        if len(self._metric_store) == 0:
            return []

        metrics = list(sorted(self._logged_metrics))
        logs = [["step"] + metrics]
        for step, step_metrics in zip(self._steps, self._metric_store):
            logs.append([step] + [step_metrics.get(m) for m in metrics])

        return logs
