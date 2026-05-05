"""Tests for launch utilities."""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestLiveSubprocessOutput(unittest.TestCase):
    """Tests for live_subprocess_output function."""

    def test_runs_command_and_returns_output(self):
        """live_subprocess_output runs command and returns output."""
        from open_instruct.launch_utils import live_subprocess_output

        result = live_subprocess_output(["echo", "hello"])
        self.assertEqual(result, "hello")

    def test_raises_on_nonzero_return_code(self):
        """live_subprocess_output raises exception on non-zero return code."""
        from open_instruct.launch_utils import live_subprocess_output

        with self.assertRaises(Exception) as context:
            live_subprocess_output(["false"])

        self.assertIn("failed with return code", str(context.exception))

    def test_captures_multiple_lines(self):
        """live_subprocess_output captures multiple output lines."""
        from open_instruct.launch_utils import live_subprocess_output

        result = live_subprocess_output(["printf", "line1\nline2\nline3"])
        self.assertIn("line1", result)
        self.assertIn("line2", result)
        self.assertIn("line3", result)


class TestGsFolderExists(unittest.TestCase):
    """Tests for gs_folder_exists function."""

    @patch("subprocess.Popen")
    def test_returns_true_when_exists(self, mock_popen):
        """gs_folder_exists returns True if GCS folder exists."""
        from open_instruct.launch_utils import gs_folder_exists

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"gs://bucket/path/", b"")
        mock_popen.return_value = mock_process

        result = gs_folder_exists("gs://bucket/path")
        self.assertTrue(result)

    @patch("subprocess.Popen")
    def test_returns_false_when_not_exists(self, mock_popen):
        """gs_folder_exists returns False if GCS folder does not exist."""
        from open_instruct.launch_utils import gs_folder_exists

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"CommandException: No URLs matched")
        mock_popen.return_value = mock_process

        result = gs_folder_exists("gs://bucket/nonexistent")
        self.assertFalse(result)


class TestDownloadFromGsBucket(unittest.TestCase):
    """Tests for download_from_gs_bucket function."""

    @patch("open_instruct.launch_utils.live_subprocess_output")
    def test_creates_dest_directory(self, mock_subprocess):
        """download_from_gs_bucket creates destination directory."""
        from open_instruct.launch_utils import download_from_gs_bucket

        mock_subprocess.return_value = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "new_folder")
            download_from_gs_bucket(["gs://bucket/file"], dest)
            self.assertTrue(os.path.isdir(dest))

    @patch("open_instruct.launch_utils.live_subprocess_output")
    def test_calls_gsutil_with_correct_args(self, mock_subprocess):
        """download_from_gs_bucket calls gsutil with correct arguments."""
        from open_instruct.launch_utils import download_from_gs_bucket

        mock_subprocess.return_value = ""

        with tempfile.TemporaryDirectory() as tmpdir:
            download_from_gs_bucket(["gs://bucket/file1", "gs://bucket/file2"], tmpdir)

            call_args = mock_subprocess.call_args[0][0]
            self.assertEqual(call_args[0], "gsutil")
            self.assertIn("-m", call_args)
            self.assertIn("cp", call_args)
            self.assertIn("-r", call_args)
            self.assertIn("gs://bucket/file1", call_args)
            self.assertIn("gs://bucket/file2", call_args)
            self.assertIn(tmpdir, call_args)


class TestUploadToGsBucket(unittest.TestCase):
    """Tests for upload_to_gs_bucket function."""

    @patch("open_instruct.launch_utils.live_subprocess_output")
    def test_calls_gsutil_with_correct_args(self, mock_subprocess):
        """upload_to_gs_bucket calls gsutil with correct arguments."""
        from open_instruct.launch_utils import upload_to_gs_bucket

        mock_subprocess.return_value = ""

        upload_to_gs_bucket("/local/path", "gs://bucket/dest")

        call_args = mock_subprocess.call_args[0][0]
        self.assertEqual(call_args[0], "gsutil")
        self.assertIn("cp", call_args)
        self.assertIn("-r", call_args)
        self.assertIn("/local/path", call_args)
        self.assertIn("gs://bucket/dest", call_args)


class TestValidateBeakerWorkspace(unittest.TestCase):
    """Tests for validate_beaker_workspace function."""

    def test_accepts_valid_format(self):
        """validate_beaker_workspace accepts valid 'org/workspace' format."""
        from open_instruct.launch_utils import validate_beaker_workspace

        # These should not raise
        validate_beaker_workspace("ai2/oe-adapt-general")
        validate_beaker_workspace("org/workspace")
        validate_beaker_workspace("my-org/my-workspace")

    def test_raises_for_missing_org(self):
        """validate_beaker_workspace raises ValueError for missing org."""
        from open_instruct.launch_utils import validate_beaker_workspace

        with self.assertRaises(ValueError) as context:
            validate_beaker_workspace("/workspace")

        self.assertIn("must be fully qualified", str(context.exception))

    def test_raises_for_missing_workspace(self):
        """validate_beaker_workspace raises ValueError for missing workspace."""
        from open_instruct.launch_utils import validate_beaker_workspace

        with self.assertRaises(ValueError) as context:
            validate_beaker_workspace("org/")

        self.assertIn("must be fully qualified", str(context.exception))

    def test_raises_for_no_slash(self):
        """validate_beaker_workspace raises ValueError for no slash."""
        from open_instruct.launch_utils import validate_beaker_workspace

        with self.assertRaises(ValueError) as context:
            validate_beaker_workspace("workspace")

        self.assertIn("must be fully qualified", str(context.exception))

    def test_raises_for_multiple_slashes(self):
        """validate_beaker_workspace raises ValueError for multiple slashes."""
        from open_instruct.launch_utils import validate_beaker_workspace

        with self.assertRaises(ValueError) as context:
            validate_beaker_workspace("org/sub/workspace")

        self.assertIn("must be fully qualified", str(context.exception))


class TestAutoCreatedSpecPath(unittest.TestCase):
    """Tests for auto_created_spec_path function."""

    def test_creates_directory_and_returns_path(self):
        """auto_created_spec_path creates directory and returns path."""
        from open_instruct.launch_utils import (
            AUTO_CREATED_BEAKER_CONFIG_DIR,
            auto_created_spec_path,
        )

        result = auto_created_spec_path("test_experiment")

        expected = os.path.join(AUTO_CREATED_BEAKER_CONFIG_DIR, "test_experiment.yaml")
        self.assertEqual(result, expected)
        self.assertTrue(os.path.isdir(AUTO_CREATED_BEAKER_CONFIG_DIR))


if __name__ == "__main__":
    unittest.main()
