"""Tests for logging utilities."""

import logging
import unittest


class TestSetupLogger(unittest.TestCase):
    """Tests for the setup_logger function."""

    def setUp(self):
        """Clear all handlers before each test."""
        # Clear root logger handlers to ensure clean state
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def tearDown(self):
        """Clean up after each test."""
        # Clear root logger handlers after tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def test_returns_root_logger_when_name_is_none(self):
        """setup_logger(name=None) returns root logger."""
        from open_instruct.logging_utils import setup_logger

        logger = setup_logger(name=None)
        root_logger = logging.getLogger()
        self.assertIs(logger, root_logger)

    def test_returns_named_logger(self):
        """setup_logger(name='module_name') returns named logger."""
        from open_instruct.logging_utils import setup_logger

        logger = setup_logger(name="test_module")
        self.assertEqual(logger.name, "test_module")
        self.assertIsInstance(logger, logging.Logger)

    def test_format_includes_required_fields(self):
        """Logger format includes timestamp, level, filename, line number, and message."""
        from open_instruct.logging_utils import setup_logger

        setup_logger(name=None, rank=0)
        root_logger = logging.getLogger()
        self.assertTrue(len(root_logger.handlers) > 0)

        handler = root_logger.handlers[0]
        fmt = handler.formatter._fmt

        # Check that format includes required fields
        self.assertIn("%(asctime)s", fmt)
        self.assertIn("%(levelname)s", fmt)
        self.assertIn("%(filename)s", fmt)
        self.assertIn("%(lineno)d", fmt)
        self.assertIn("%(message)s", fmt)

    def test_rank_zero_logs_at_info_level(self):
        """Rank 0 logs at INFO level by default."""
        from open_instruct.logging_utils import setup_logger

        setup_logger(name=None, rank=0)
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)

    def test_non_zero_rank_logs_at_warning_level(self):
        """Non-zero ranks log at WARNING level."""
        from open_instruct.logging_utils import setup_logger

        setup_logger(name=None, rank=1)
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.WARNING)

    def test_basicconfig_only_called_once(self):
        """basicConfig is only called once (no duplicate handlers)."""
        from open_instruct.logging_utils import setup_logger

        # First call should set up handlers
        setup_logger(name=None, rank=0)
        root_logger = logging.getLogger()
        initial_handler_count = len(root_logger.handlers)

        # Second call should not add more handlers
        setup_logger(name="another_module", rank=0)
        self.assertEqual(len(root_logger.handlers), initial_handler_count)

    def test_datetime_format(self):
        """Datetime format is 'YYYY-MM-DD HH:MM:SS'."""
        from open_instruct.logging_utils import setup_logger

        setup_logger(name=None, rank=0)
        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        datefmt = handler.formatter.datefmt
        self.assertEqual(datefmt, "%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    unittest.main()
