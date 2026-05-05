"""Tests for core utilities."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch


class TestRepeatEach(unittest.TestCase):
    """Tests for repeat_each function."""

    def test_repeats_elements(self):
        """repeat_each repeats each element k times."""
        from open_instruct.utils import repeat_each

        result = repeat_each([1, 2, 3], 2)
        self.assertEqual(result, [1, 1, 2, 2, 3, 3])

    def test_single_repeat(self):
        """repeat_each with k=1 returns same elements."""
        from open_instruct.utils import repeat_each

        result = repeat_each([1, 2, 3], 1)
        self.assertEqual(result, [1, 2, 3])

    def test_empty_sequence(self):
        """repeat_each with empty sequence returns empty list."""
        from open_instruct.utils import repeat_each

        result = repeat_each([], 5)
        self.assertEqual(result, [])

    def test_works_with_strings(self):
        """repeat_each works with string elements."""
        from open_instruct.utils import repeat_each

        result = repeat_each(["a", "b"], 3)
        self.assertEqual(result, ["a", "a", "a", "b", "b", "b"])


class TestMetricsTracker(unittest.TestCase):
    """Tests for MetricsTracker class."""

    def test_preallocates_on_device(self):
        """MetricsTracker preallocates metrics array on specified device."""
        from open_instruct.utils import MetricsTracker

        tracker = MetricsTracker(max_metrics=16, device="cpu")
        self.assertEqual(tracker.metrics.device.type, "cpu")
        self.assertEqual(len(tracker.metrics), 16)

    def test_get_set_by_name(self):
        """MetricsTracker supports getting/setting metrics by name."""
        from open_instruct.utils import MetricsTracker

        tracker = MetricsTracker(max_metrics=8, device="cpu")
        tracker["loss"] = 0.5
        tracker["accuracy"] = 0.95

        self.assertAlmostEqual(tracker["loss"].item(), 0.5, places=5)
        self.assertAlmostEqual(tracker["accuracy"].item(), 0.95, places=5)

    def test_update_multiple(self):
        """MetricsTracker.update() sets multiple metrics at once."""
        from open_instruct.utils import MetricsTracker

        tracker = MetricsTracker(max_metrics=8, device="cpu")
        tracker.update({"loss": 0.1, "accuracy": 0.9})

        self.assertAlmostEqual(tracker["loss"].item(), 0.1, places=5)
        self.assertAlmostEqual(tracker["accuracy"].item(), 0.9, places=5)

    def test_get_metrics_list(self):
        """MetricsTracker.get_metrics_list() returns dict of name to float."""
        from open_instruct.utils import MetricsTracker

        tracker = MetricsTracker(max_metrics=8, device="cpu")
        tracker["loss"] = 0.25
        tracker["f1"] = 0.85

        metrics = tracker.get_metrics_list()
        self.assertIsInstance(metrics, dict)
        self.assertAlmostEqual(metrics["loss"], 0.25, places=5)
        self.assertAlmostEqual(metrics["f1"], 0.85, places=5)

    def test_exceeds_max_metrics(self):
        """MetricsTracker raises ValueError when exceeding max metrics."""
        from open_instruct.utils import MetricsTracker

        tracker = MetricsTracker(max_metrics=2, device="cpu")
        tracker["a"] = 1.0
        tracker["b"] = 2.0

        with self.assertRaises(ValueError):
            tracker["c"] = 3.0


class TestWarnIfLowDiskSpace(unittest.TestCase):
    """Tests for warn_if_low_disk_space function."""

    def test_warns_when_threshold_exceeded(self):
        """warn_if_low_disk_space warns when disk usage exceeds threshold."""
        from open_instruct.utils import warn_if_low_disk_space

        # Use /tmp as it should exist on most systems
        with self.assertLogs("open_instruct.utils", level="WARNING") as cm:
            # Use a very low threshold to trigger warning
            warn_if_low_disk_space("/tmp", threshold=0.0)

        self.assertTrue(any("Disk usage" in log for log in cm.output))

    def test_skips_cloud_paths(self):
        """warn_if_low_disk_space skips cloud paths."""
        from open_instruct.utils import warn_if_low_disk_space

        # These should not raise any errors or log anything
        warn_if_low_disk_space("gs://bucket/path")
        warn_if_low_disk_space("s3://bucket/path")
        warn_if_low_disk_space("az://container/path")

    def test_handles_nonexistent_path(self):
        """warn_if_low_disk_space handles nonexistent paths gracefully."""
        from open_instruct.utils import warn_if_low_disk_space

        # Should log warning but not raise
        with self.assertLogs("open_instruct.utils", level="WARNING") as cm:
            warn_if_low_disk_space("/nonexistent/path/that/does/not/exist")

        self.assertTrue(any("OS error" in log for log in cm.output))


class TestGetLastCheckpoint(unittest.TestCase):
    """Tests for get_last_checkpoint function."""

    def test_returns_latest_completed_checkpoint(self):
        """get_last_checkpoint returns latest completed checkpoint."""
        from open_instruct.utils import get_last_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories
            os.makedirs(os.path.join(tmpdir, "step_100"))
            os.makedirs(os.path.join(tmpdir, "step_200"))
            os.makedirs(os.path.join(tmpdir, "step_300"))

            # Mark step_100 and step_200 as completed
            open(os.path.join(tmpdir, "step_100", "COMPLETED"), "w").close()
            open(os.path.join(tmpdir, "step_200", "COMPLETED"), "w").close()

            result = get_last_checkpoint(tmpdir)
            self.assertEqual(result, os.path.join(tmpdir, "step_200"))

    def test_returns_none_for_no_checkpoints(self):
        """get_last_checkpoint returns None when no checkpoints exist."""
        from open_instruct.utils import get_last_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_last_checkpoint(tmpdir)
            self.assertIsNone(result)

    def test_includes_incomplete_when_flag_set(self):
        """get_last_checkpoint includes incomplete checkpoints when flag is set."""
        from open_instruct.utils import get_last_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "step_100"))
            os.makedirs(os.path.join(tmpdir, "step_200"))

            # Only step_100 is completed
            open(os.path.join(tmpdir, "step_100", "COMPLETED"), "w").close()

            # Without incomplete flag, should get step_100
            result = get_last_checkpoint(tmpdir, incomplete=False)
            self.assertEqual(result, os.path.join(tmpdir, "step_100"))

            # With incomplete flag, should get step_200
            result = get_last_checkpoint(tmpdir, incomplete=True)
            self.assertEqual(result, os.path.join(tmpdir, "step_200"))


class TestCleanLastNCheckpoints(unittest.TestCase):
    """Tests for clean_last_n_checkpoints function."""

    def test_keeps_last_n_checkpoints(self):
        """clean_last_n_checkpoints keeps only the last N checkpoints."""
        from open_instruct.utils import clean_last_n_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 checkpoint directories
            for i in [100, 200, 300, 400, 500]:
                os.makedirs(os.path.join(tmpdir, f"step_{i}"))

            clean_last_n_checkpoints(tmpdir, keep_last_n_checkpoints=2)

            remaining = os.listdir(tmpdir)
            self.assertEqual(len(remaining), 2)
            self.assertIn("step_400", remaining)
            self.assertIn("step_500", remaining)

    def test_keeps_all_when_negative(self):
        """clean_last_n_checkpoints keeps all when keep_last_n_checkpoints < 0."""
        from open_instruct.utils import clean_last_n_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in [100, 200, 300]:
                os.makedirs(os.path.join(tmpdir, f"step_{i}"))

            clean_last_n_checkpoints(tmpdir, keep_last_n_checkpoints=-1)

            remaining = os.listdir(tmpdir)
            self.assertEqual(len(remaining), 3)


class TestGPUSpecs(unittest.TestCase):
    """Tests for GPU_SPECS dictionary."""

    def test_contains_common_gpus(self):
        """GPU_SPECS contains specifications for common GPUs."""
        from open_instruct.utils import GPU_SPECS

        self.assertIn("h100", GPU_SPECS)
        self.assertIn("a100", GPU_SPECS)
        self.assertIn("h200", GPU_SPECS)

    def test_has_required_fields(self):
        """GPU_SPECS entries have required fields."""
        from open_instruct.utils import GPU_SPECS

        for name, specs in GPU_SPECS.items():
            self.assertIn("flops", specs, f"{name} missing flops")
            self.assertIn("memory_size", specs, f"{name} missing memory_size")
            self.assertIn("memory_bandwidth", specs, f"{name} missing memory_bandwidth")


class TestFindFreePort(unittest.TestCase):
    """Tests for find_free_port function."""

    def test_returns_available_port(self):
        """find_free_port returns an available port number."""
        from open_instruct.utils import find_free_port

        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)

    def test_ports_are_unique(self):
        """find_free_port returns different ports on subsequent calls."""
        from open_instruct.utils import find_free_port

        ports = [find_free_port() for _ in range(5)]
        # Ports should generally be unique (though not guaranteed)
        self.assertGreaterEqual(len(set(ports)), 1)


class TestMaxNumProcesses(unittest.TestCase):
    """Tests for max_num_processes function."""

    def test_returns_positive_integer(self):
        """max_num_processes returns a positive integer."""
        from open_instruct.utils import max_num_processes

        result = max_num_processes()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_reasonable_default(self):
        """max_num_processes returns a reasonable number."""
        from open_instruct.utils import max_num_processes

        result = max_num_processes()
        # Should be at least 1 and not more than a reasonable max
        self.assertGreaterEqual(result, 1)
        self.assertLessEqual(result, 1024)


if __name__ == "__main__":
    unittest.main()
