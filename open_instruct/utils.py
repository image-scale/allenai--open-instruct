"""Core utilities for the open_instruct project."""

import os
import shutil
import socket
from typing import Sequence, TypeVar

import torch

from open_instruct.logging_utils import setup_logger

T = TypeVar("T")

logger = setup_logger(__name__)

# Disk usage warning threshold (85% used triggers warning)
DISK_USAGE_WARNING_THRESHOLD = 0.85

# Cloud storage path prefixes that should skip disk checks
CLOUD_PATH_PREFIXES = ("gs://", "s3://", "az://", "hdfs://", "/filestore")

# GPU specifications for performance calculations
# For FLOPS, we assume bf16 and ignore sparsity.
# Memory bandwidth values are peak theoretical bandwidth.
GPU_SPECS = {
    "a100": {"flops": 312e12, "memory_size": 80e9, "memory_bandwidth": 2.0e12},
    "b200": {"flops": 2250e12, "memory_size": 192e9, "memory_bandwidth": 8e12},
    "h100": {"flops": 990e12, "memory_size": 80e9, "memory_bandwidth": 3.35e12},
    "h200": {"flops": 989e12, "memory_size": 141e9, "memory_bandwidth": 4.8e12},
    "a6000": {"flops": 155e12, "memory_size": 48e9, "memory_bandwidth": 768e9},
    "l40s": {"flops": 362e12, "memory_size": 48e9, "memory_bandwidth": 864e9},
    "pro 6000": {"flops": 503.8e12, "memory_size": 96e9, "memory_bandwidth": 1792e9},
    "6000": {"flops": 728.5e12, "memory_size": 48e9, "memory_bandwidth": 960e9},
    "4090 laptop": {"flops": 32.98e12, "memory_size": 24e9, "memory_bandwidth": 576e9},
    "gb10": {"flops": 104e12, "memory_size": 128e9, "memory_bandwidth": 273e9},
}


def repeat_each(seq: Sequence[T], k: int) -> list[T]:
    """Repeat each element in a sequence k times.

    Args:
        seq: Input sequence of elements.
        k: Number of times to repeat each element.

    Returns:
        List with each element repeated k times.

    Example:
        >>> repeat_each([1, 2, 3], 2)
        [1, 1, 2, 2, 3, 3]
    """
    return [item for item in seq for _ in range(k)]


class MetricsTracker:
    """A class to preallocate metrics in an array for efficient aggregation.

    This allows doing only one allreduce operation to get metrics mean
    across distributed processes.

    Attributes:
        metrics: Preallocated tensor for storing metric values.
        names2idx: Mapping from metric names to tensor indices.
        current_idx: Next available index for new metrics.
        max_metrics: Maximum number of metrics that can be tracked.
    """

    def __init__(self, max_metrics: int = 32, device: str = "cuda"):
        """Initialize the metrics tracker.

        Args:
            max_metrics: Maximum number of metrics to track.
            device: Device to allocate the metrics tensor on.
        """
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx: dict[str, int] = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def _maybe_register_metric(self, name: str) -> int:
        """Register a new metric if not already registered.

        Args:
            name: Name of the metric.

        Returns:
            Index of the metric in the tensor.

        Raises:
            ValueError: If maximum number of metrics is exceeded.
        """
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1
        return self.names2idx[name]

    def __getitem__(self, name: str) -> torch.Tensor:
        """Get metric value by name.

        Args:
            name: Name of the metric.

        Returns:
            Tensor containing the metric value.
        """
        idx = self._maybe_register_metric(name)
        return self.metrics[idx]

    def __setitem__(self, name: str, value: float | torch.Tensor) -> None:
        """Set metric value by name.

        Args:
            name: Name of the metric.
            value: Value to set.
        """
        idx = self._maybe_register_metric(name)
        self.metrics[idx] = value

    def update(self, metrics: dict[str, float | torch.Tensor]) -> None:
        """Update multiple metrics at once.

        Args:
            metrics: Dictionary of metric names to values.
        """
        for name, value in metrics.items():
            self[name] = value

    def get_metrics_list(self) -> dict[str, float]:
        """Get all registered metrics as a dictionary.

        Returns:
            Dictionary mapping metric names to their float values.
        """
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}


def warn_if_low_disk_space(
    path: str,
    *,
    threshold: float = DISK_USAGE_WARNING_THRESHOLD,
    send_slack_alerts: bool = False,
) -> None:
    """Warn when disk usage exceeds the provided threshold.

    Args:
        path: Filesystem path to check disk usage for.
        threshold: Usage ratio (0.0-1.0) above which to warn.
        send_slack_alerts: Whether to also send a Slack alert when warning.
    """
    if path.startswith(CLOUD_PATH_PREFIXES):
        return

    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        logger.warning(f"Skipping disk usage check for {path}, encountered OS error: {e}")
        return

    if usage.total == 0:
        return

    used_ratio = usage.used / usage.total
    if used_ratio >= threshold:
        used_percent = used_ratio * 100
        free_gib = usage.free / (1024**3)
        total_gib = usage.total / (1024**3)
        warning_message = (
            f"Disk usage near capacity for {path}: {used_percent:.1f}% used "
            f"({free_gib:.1f} GiB free of {total_gib:.1f} GiB). Checkpointing may fail."
        )
        logger.warning(warning_message)


def find_free_port() -> int:
    """Find and return an available port number.

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def max_num_processes() -> int:
    """Returns a reasonable default number of processes for multiprocessing.

    Returns:
        Number of available CPU cores for the current process.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count() or 1


def is_checkpoint_folder(dir_path: str, folder: str) -> bool:
    """Check if a folder is a checkpoint folder.

    Args:
        dir_path: Parent directory path.
        folder: Folder name to check.

    Returns:
        True if the folder is a checkpoint folder (step_* or epoch_*).
    """
    return (folder.startswith("step_") or folder.startswith("epoch_")) and os.path.isdir(
        os.path.join(dir_path, folder)
    )


def get_last_checkpoint(folder: str, incomplete: bool = False) -> str | None:
    """Get the path to the last checkpoint in a folder.

    Args:
        folder: Directory containing checkpoints.
        incomplete: If True, include checkpoints without COMPLETED marker.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    if not os.path.isdir(folder):
        return None

    content = os.listdir(folder)
    checkpoint_steps = [path for path in content if path.startswith("step_")]
    checkpoint_epochs = [path for path in content if path.startswith("epoch_")]

    if len(checkpoint_steps) > 0 and len(checkpoint_epochs) > 0:
        logger.info("Mixed step and epoch checkpoints found. Using step checkpoints.")
        checkpoints = checkpoint_steps
    elif len(checkpoint_steps) == 0:
        checkpoints = checkpoint_epochs
    else:
        checkpoints = checkpoint_steps

    if not incomplete:
        checkpoints = [
            path
            for path in checkpoints
            if os.path.exists(os.path.join(folder, path, "COMPLETED"))
        ]

    if len(checkpoints) == 0:
        return None

    return os.path.join(folder, max(checkpoints, key=lambda x: int(x.split("_")[-1])))


def clean_last_n_checkpoints(output_dir: str, keep_last_n_checkpoints: int) -> None:
    """Remove old checkpoints, keeping only the last N.

    Args:
        output_dir: Directory containing checkpoints.
        keep_last_n_checkpoints: Number of checkpoints to keep. If < 0, keep all.
    """
    if not os.path.isdir(output_dir):
        return

    folders = [f for f in os.listdir(output_dir) if is_checkpoint_folder(output_dir, f)]
    checkpoints = sorted(folders, key=lambda x: int(x.split("_")[-1]))

    if keep_last_n_checkpoints >= 0 and len(checkpoints) > keep_last_n_checkpoints:
        for checkpoint in checkpoints[: len(checkpoints) - keep_last_n_checkpoints]:
            logger.info(f"Removing checkpoint {checkpoint}")
            shutil.rmtree(os.path.join(output_dir, checkpoint), ignore_errors=True)

    logger.info("Remaining files:" + str(os.listdir(output_dir)))
