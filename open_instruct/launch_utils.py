"""Launch utilities for subprocess execution and cloud storage operations."""

import os
import subprocess

from open_instruct.logging_utils import setup_logger

logger = setup_logger(__name__)

# Directory for auto-created Beaker configs
AUTO_CREATED_BEAKER_CONFIG_DIR = "configs/beaker_configs/auto_created"

# Weka clusters available for use
WEKA_CLUSTERS = [
    "ai2/jupiter",
    "ai2/saturn",
    "ai2/titan",
    "ai2/neptune",
    "ai2/ceres",
    "ai2/triton",
    "ai2/rhea",
    "ai2/prometheus",
]

# Clusters with high-speed interconnect
INTERCONNECT_CLUSTERS = ["ai2/jupiter", "ai2/ceres", "ai2/titan"]


def live_subprocess_output(cmd: list[str]) -> str:
    """Run a subprocess and print output in real-time.

    Executes a command, streaming stdout/stderr to the console as it
    becomes available.

    Args:
        cmd: Command and arguments to execute.

    Returns:
        Complete output from the command as a string.

    Raises:
        Exception: If the command returns a non-zero exit code.
    """
    output_lines = []
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in iter(process.stdout.readline, ""):
        if line.strip():
            print(line.strip())
            output_lines.append(line.strip())
    process.wait()

    if process.returncode != 0:
        full_output = "\n".join(output_lines)
        error_message = (
            f"Command `{' '.join(cmd)}` failed with return code "
            f"{process.returncode}:\n{full_output}"
        )
        raise Exception(error_message)

    return "\n".join(output_lines)


def gs_folder_exists(path: str) -> bool:
    """Check if a Google Cloud Storage folder exists.

    Args:
        path: GCS path (e.g., gs://bucket/folder).

    Returns:
        True if the folder exists, False otherwise.
    """
    cmd = ["gsutil", "ls", path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return process.returncode == 0


def download_from_gs_bucket(src_paths: list[str], dest_path: str) -> None:
    """Download files from Google Cloud Storage.

    Args:
        src_paths: List of GCS source paths to download.
        dest_path: Local destination directory.

    Raises:
        Exception: If the download fails.
    """
    os.makedirs(dest_path, exist_ok=True)
    cmd = [
        "gsutil",
        "-o",
        "GSUtil:parallel_thread_count=1",
        "-o",
        "GSUtil:sliced_object_download_threshold=150",
        "-m",
        "cp",
        "-r",
    ]
    cmd.extend(src_paths)
    cmd.append(dest_path)
    logger.info(f"Downloading from GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def upload_to_gs_bucket(src_path: str, dest_path: str) -> None:
    """Upload files to Google Cloud Storage.

    Args:
        src_path: Local source path to upload.
        dest_path: GCS destination path.

    Raises:
        Exception: If the upload fails.
    """
    cmd = [
        "gsutil",
        "-o",
        "GSUtil:parallel_composite_upload_threshold=150M",
        "cp",
        "-r",
        src_path,
        dest_path,
    ]
    logger.info(f"Copying to GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def validate_beaker_workspace(workspace: str) -> None:
    """Validate Beaker workspace format.

    Beaker workspaces must be fully qualified as 'org/workspace'.

    Args:
        workspace: Workspace string to validate.

    Raises:
        ValueError: If the workspace format is invalid.
    """
    parts = workspace.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"--workspace must be fully qualified as '<org>/<workspace>' "
            f"(e.g., 'ai2/oe-adapt-general'). Received: '{workspace}'"
        )


def auto_created_spec_path(experiment_name: str) -> str:
    """Get the path for an auto-created Beaker spec file.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        Path where the spec file should be created.
    """
    os.makedirs(AUTO_CREATED_BEAKER_CONFIG_DIR, exist_ok=True)
    return os.path.join(AUTO_CREATED_BEAKER_CONFIG_DIR, f"{experiment_name}.yaml")
