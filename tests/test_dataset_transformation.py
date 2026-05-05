"""Tests for dataset transformation utilities."""

import unittest
from unittest.mock import MagicMock, patch

import torch


class TestChatTemplates(unittest.TestCase):
    """Tests for CHAT_TEMPLATES dictionary."""

    def test_contains_named_templates(self):
        """CHAT_TEMPLATES contains expected template names."""
        from open_instruct.dataset_transformation import CHAT_TEMPLATES

        self.assertIn("simple_concat_with_space", CHAT_TEMPLATES)
        self.assertIn("simple_chat", CHAT_TEMPLATES)
        self.assertIn("tulu", CHAT_TEMPLATES)
        self.assertIn("zephyr", CHAT_TEMPLATES)

    def test_templates_are_strings(self):
        """All templates are strings."""
        from open_instruct.dataset_transformation import CHAT_TEMPLATES

        for name, template in CHAT_TEMPLATES.items():
            self.assertIsInstance(template, str, f"Template {name} should be a string")

    def test_simple_concat_template_has_eos(self):
        """simple_concat_with_space template includes eos_token handling."""
        from open_instruct.dataset_transformation import CHAT_TEMPLATES

        template = CHAT_TEMPLATES["simple_concat_with_space"]
        self.assertIn("eos_token", template)
        self.assertIn("add_generation_prompt", template)


class TestTokenizerConfig(unittest.TestCase):
    """Tests for TokenizerConfig dataclass."""

    def test_stores_configuration(self):
        """TokenizerConfig stores configuration fields."""
        from open_instruct.dataset_transformation import TokenizerConfig

        config = TokenizerConfig(
            tokenizer_name_or_path="gpt2",
            tokenizer_revision="main",
            chat_template_name="simple_chat",
            add_bos=True,
            trust_remote_code=False,
        )

        self.assertEqual(config.tokenizer_name_or_path, "gpt2")
        self.assertEqual(config.tokenizer_revision, "main")
        self.assertEqual(config.chat_template_name, "simple_chat")
        self.assertTrue(config.add_bos)
        self.assertFalse(config.trust_remote_code)

    def test_has_default_values(self):
        """TokenizerConfig has sensible defaults."""
        from open_instruct.dataset_transformation import TokenizerConfig

        config = TokenizerConfig()

        self.assertIsNone(config.tokenizer_name_or_path)
        self.assertIsNone(config.tokenizer_revision)
        self.assertIsNone(config.chat_template_name)
        self.assertFalse(config.add_bos)
        self.assertTrue(config.use_fast)
        self.assertEqual(config.get_tokenizer_fn, "get_tokenizer_tulu_v1")

    def test_tokenizer_property_raises_without_path(self):
        """TokenizerConfig.tokenizer raises ValueError when path not set."""
        from open_instruct.dataset_transformation import TokenizerConfig

        config = TokenizerConfig()

        with self.assertRaises(ValueError) as ctx:
            _ = config.tokenizer

        self.assertIn("tokenizer_name_or_path must be set", str(ctx.exception))


class TestSFTTokenization(unittest.TestCase):
    """Tests for SFT tokenization functions."""

    def setUp(self):
        """Create mock tokenizer."""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kwargs: list(range(len(msgs) * 10))
        )
        self.mock_tokenizer.pad_token_id = 0

    def test_sft_tokenize_v1_returns_required_keys(self):
        """sft_tokenize_v1 returns input_ids, attention_mask, and labels."""
        from open_instruct.dataset_transformation import (
            ATTENTION_MASK_KEY,
            INPUT_IDS_KEY,
            INPUT_IDS_PROMPT_KEY,
            LABELS_KEY,
            sft_tokenize_v1,
        )

        row = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}

        result = sft_tokenize_v1(row, self.mock_tokenizer)

        self.assertIn(INPUT_IDS_KEY, result)
        self.assertIn(ATTENTION_MASK_KEY, result)
        self.assertIn(LABELS_KEY, result)
        self.assertIn(INPUT_IDS_PROMPT_KEY, result)

    def test_sft_tokenize_v1_attention_mask_all_ones(self):
        """sft_tokenize_v1 creates attention_mask of all 1s."""
        from open_instruct.dataset_transformation import ATTENTION_MASK_KEY, INPUT_IDS_KEY, sft_tokenize_v1

        row = {"messages": [{"role": "user", "content": "Hello"}]}

        result = sft_tokenize_v1(row, self.mock_tokenizer)

        self.assertEqual(len(result[ATTENTION_MASK_KEY]), len(result[INPUT_IDS_KEY]))
        self.assertTrue(all(x == 1 for x in result[ATTENTION_MASK_KEY]))

    def test_sft_tokenize_mask_out_prompt_v1_masks_prompt(self):
        """sft_tokenize_mask_out_prompt_v1 masks prompt tokens with -100."""
        from open_instruct.dataset_transformation import (
            INPUT_IDS_PROMPT_KEY,
            LABELS_KEY,
            MASKED_TOKEN_VALUE,
            sft_tokenize_mask_out_prompt_v1,
        )

        # Mock to return different lengths for prompt vs full
        self.mock_tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda msgs, add_generation_prompt=False, **kwargs: (
                [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]
            )
        )

        row = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}

        result = sft_tokenize_mask_out_prompt_v1(row, self.mock_tokenizer)

        prompt_len = len(result[INPUT_IDS_PROMPT_KEY])
        # First prompt_len tokens should be masked
        self.assertEqual(result[LABELS_KEY][:prompt_len], [MASKED_TOKEN_VALUE] * prompt_len)
        # Remaining tokens should not be masked
        self.assertNotIn(MASKED_TOKEN_VALUE, result[LABELS_KEY][prompt_len:])

    def test_sft_filter_v1_filters_by_max_token_length(self):
        """sft_filter_v1 filters examples exceeding max_token_length."""
        from open_instruct.dataset_transformation import (
            INPUT_IDS_KEY,
            INPUT_IDS_PROMPT_KEY,
            LABELS_KEY,
            sft_filter_v1,
        )

        row = {
            INPUT_IDS_KEY: list(range(100)),
            INPUT_IDS_PROMPT_KEY: list(range(50)),
            LABELS_KEY: list(range(100)),
        }

        # Should pass with higher limit
        self.assertTrue(sft_filter_v1(row, self.mock_tokenizer, max_token_length=200))

        # Should fail with lower limit
        self.assertFalse(sft_filter_v1(row, self.mock_tokenizer, max_token_length=50))

    def test_sft_filter_v1_filters_by_max_prompt_length(self):
        """sft_filter_v1 filters examples exceeding max_prompt_token_length."""
        from open_instruct.dataset_transformation import (
            INPUT_IDS_KEY,
            INPUT_IDS_PROMPT_KEY,
            LABELS_KEY,
            sft_filter_v1,
        )

        row = {
            INPUT_IDS_KEY: list(range(100)),
            INPUT_IDS_PROMPT_KEY: list(range(50)),
            LABELS_KEY: list(range(100)),
        }

        # Should pass with higher limit
        self.assertTrue(sft_filter_v1(row, self.mock_tokenizer, max_prompt_token_length=100))

        # Should fail with lower limit
        self.assertFalse(sft_filter_v1(row, self.mock_tokenizer, max_prompt_token_length=30))


class TestPreferenceTokenization(unittest.TestCase):
    """Tests for preference tokenization functions."""

    def setUp(self):
        """Create mock tokenizer."""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kwargs: list(range(len(msgs) * 10))
        )

    def test_preference_tokenize_v1_returns_required_keys(self):
        """preference_tokenize_v1 returns all required keys."""
        from open_instruct.dataset_transformation import (
            ATTENTION_MASK_PROMPT_KEY,
            CHOSEN_ATTENTION_MASK_KEY,
            CHOSEN_INPUT_IDS_KEY,
            INPUT_IDS_PROMPT_KEY,
            REJECTED_ATTENTION_MASK_KEY,
            REJECTED_INPUT_IDS_KEY,
            preference_tokenize_v1,
        )

        row = {
            "chosen": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            "rejected": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Bye"}],
        }

        result = preference_tokenize_v1(row, self.mock_tokenizer)

        self.assertIn(INPUT_IDS_PROMPT_KEY, result)
        self.assertIn(ATTENTION_MASK_PROMPT_KEY, result)
        self.assertIn(CHOSEN_INPUT_IDS_KEY, result)
        self.assertIn(CHOSEN_ATTENTION_MASK_KEY, result)
        self.assertIn(REJECTED_INPUT_IDS_KEY, result)
        self.assertIn(REJECTED_ATTENTION_MASK_KEY, result)

    def test_preference_filter_v1_filters_by_max_lengths(self):
        """preference_filter_v1 filters by max prompt and token lengths."""
        from open_instruct.dataset_transformation import (
            CHOSEN_INPUT_IDS_KEY,
            INPUT_IDS_PROMPT_KEY,
            REJECTED_INPUT_IDS_KEY,
            preference_filter_v1,
        )

        row = {
            INPUT_IDS_PROMPT_KEY: list(range(50)),
            CHOSEN_INPUT_IDS_KEY: list(range(80)),
            REJECTED_INPUT_IDS_KEY: list(range(90)),
        }

        # Should pass with higher limits
        self.assertTrue(
            preference_filter_v1(row, self.mock_tokenizer, max_prompt_token_length=100, max_token_length=100)
        )

        # Should fail when chosen exceeds max
        self.assertFalse(preference_filter_v1(row, self.mock_tokenizer, max_token_length=70))

        # Should fail when rejected exceeds max
        row[CHOSEN_INPUT_IDS_KEY] = list(range(50))
        self.assertFalse(preference_filter_v1(row, self.mock_tokenizer, max_token_length=80))


class TestRLVRTokenization(unittest.TestCase):
    """Tests for RLVR tokenization functions."""

    def setUp(self):
        """Create mock tokenizer."""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda msgs, **kwargs: list(range(len(msgs) * 10))
        )
        self.mock_tokenizer.pad_token_id = None

    def test_rlvr_tokenize_v1_preserves_ground_truth(self):
        """rlvr_tokenize_v1 preserves ground_truth and verifier_source."""
        from open_instruct.dataset_transformation import (
            GROUND_TRUTHS_KEY,
            INPUT_IDS_PROMPT_KEY,
            RAW_PROMPT_KEY,
            VERIFIER_SOURCE_KEY,
            rlvr_tokenize_v1,
        )

        row = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "ground_truth": "4",
            "dataset": "gsm8k",
        }

        result = rlvr_tokenize_v1(row, self.mock_tokenizer)

        self.assertIn(INPUT_IDS_PROMPT_KEY, result)
        self.assertIn(GROUND_TRUTHS_KEY, result)
        self.assertIn(VERIFIER_SOURCE_KEY, result)
        self.assertIn(RAW_PROMPT_KEY, result)

    def test_rlvr_tokenize_v1_wraps_string_verifier_in_list(self):
        """rlvr_tokenize_v1 wraps string verifier source in list."""
        from open_instruct.dataset_transformation import GROUND_TRUTHS_KEY, VERIFIER_SOURCE_KEY, rlvr_tokenize_v1

        row = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "ground_truth": "4",
            "dataset": "gsm8k",
        }

        result = rlvr_tokenize_v1(row, self.mock_tokenizer)

        self.assertIsInstance(result[VERIFIER_SOURCE_KEY], list)
        self.assertEqual(result[VERIFIER_SOURCE_KEY], ["gsm8k"])
        self.assertIsInstance(result[GROUND_TRUTHS_KEY], list)
        self.assertEqual(result[GROUND_TRUTHS_KEY], ["4"])

    def test_rlvr_max_length_filter_v1_filters_by_prompt_length(self):
        """rlvr_max_length_filter_v1 filters by max_prompt_token_length."""
        from open_instruct.dataset_transformation import INPUT_IDS_PROMPT_KEY, rlvr_max_length_filter_v1

        row = {INPUT_IDS_PROMPT_KEY: list(range(100))}

        # Should pass with higher limit
        self.assertTrue(rlvr_max_length_filter_v1(row, self.mock_tokenizer, max_prompt_token_length=200))

        # Should fail with lower limit
        self.assertFalse(rlvr_max_length_filter_v1(row, self.mock_tokenizer, max_prompt_token_length=50))

        # Should pass when no limit set
        self.assertTrue(rlvr_max_length_filter_v1(row, self.mock_tokenizer, max_prompt_token_length=None))


class TestTransformFNsRegistry(unittest.TestCase):
    """Tests for TRANSFORM_FNS registry."""

    def test_registry_contains_expected_functions(self):
        """TRANSFORM_FNS contains expected function mappings."""
        from open_instruct.dataset_transformation import TRANSFORM_FNS

        expected_names = [
            "sft_tokenize_v1",
            "sft_tokenize_mask_out_prompt_v1",
            "sft_filter_v1",
            "preference_tokenize_v1",
            "preference_filter_v1",
            "rlvr_tokenize_v1",
            "rlvr_max_length_filter_v1",
        ]

        for name in expected_names:
            self.assertIn(name, TRANSFORM_FNS, f"Missing {name} in TRANSFORM_FNS")

    def test_registry_entries_are_tuples(self):
        """TRANSFORM_FNS entries are (function, operation_type) tuples."""
        from open_instruct.dataset_transformation import TRANSFORM_FNS

        for name, entry in TRANSFORM_FNS.items():
            self.assertIsInstance(entry, tuple, f"{name} should be a tuple")
            self.assertEqual(len(entry), 2, f"{name} should have 2 elements")
            self.assertTrue(callable(entry[0]), f"{name} first element should be callable")
            self.assertIn(entry[1], ("map", "filter"), f"{name} second element should be 'map' or 'filter'")

    def test_map_operations_are_tokenize_functions(self):
        """Map operations in TRANSFORM_FNS are tokenize functions."""
        from open_instruct.dataset_transformation import TRANSFORM_FNS

        map_fns = [name for name, (_, op) in TRANSFORM_FNS.items() if op == "map"]

        for name in map_fns:
            self.assertIn("tokenize", name, f"Map function {name} should be a tokenize function")

    def test_filter_operations_are_filter_functions(self):
        """Filter operations in TRANSFORM_FNS are filter functions."""
        from open_instruct.dataset_transformation import TRANSFORM_FNS

        filter_fns = [name for name, (_, op) in TRANSFORM_FNS.items() if op == "filter"]

        for name in filter_fns:
            self.assertIn("filter", name, f"Filter function {name} should be a filter function")


class TestSimplePreferenceCollator(unittest.TestCase):
    """Tests for SimplePreferenceCollator class."""

    def test_pads_sequences_to_same_length(self):
        """SimplePreferenceCollator pads sequences to the same length."""
        from open_instruct.dataset_transformation import (
            CHOSEN_ATTENTION_MASK_KEY,
            CHOSEN_INPUT_IDS_KEY,
            REJECTED_ATTENTION_MASK_KEY,
            REJECTED_INPUT_IDS_KEY,
            SimplePreferenceCollator,
        )

        collator = SimplePreferenceCollator(pad_token_id=0)

        batch = [
            {CHOSEN_INPUT_IDS_KEY: [1, 2, 3], REJECTED_INPUT_IDS_KEY: [1, 2, 3, 4, 5]},
            {CHOSEN_INPUT_IDS_KEY: [1, 2], REJECTED_INPUT_IDS_KEY: [1, 2, 3]},
        ]

        result = collator(batch)

        # All sequences should be padded to max length (5)
        self.assertEqual(result[CHOSEN_INPUT_IDS_KEY].shape, (2, 5))
        self.assertEqual(result[REJECTED_INPUT_IDS_KEY].shape, (2, 5))

    def test_creates_correct_attention_masks(self):
        """SimplePreferenceCollator creates correct attention masks."""
        from open_instruct.dataset_transformation import (
            CHOSEN_ATTENTION_MASK_KEY,
            CHOSEN_INPUT_IDS_KEY,
            REJECTED_ATTENTION_MASK_KEY,
            REJECTED_INPUT_IDS_KEY,
            SimplePreferenceCollator,
        )

        collator = SimplePreferenceCollator(pad_token_id=0)

        batch = [
            {CHOSEN_INPUT_IDS_KEY: [1, 2, 3], REJECTED_INPUT_IDS_KEY: [1, 2]},
        ]

        result = collator(batch)

        # Attention mask should be 1 for real tokens, 0 for padding
        expected_chosen_mask = torch.tensor([[1, 1, 1]])
        expected_rejected_mask = torch.tensor([[1, 1, 0]])

        torch.testing.assert_close(result[CHOSEN_ATTENTION_MASK_KEY], expected_chosen_mask)
        torch.testing.assert_close(result[REJECTED_ATTENTION_MASK_KEY], expected_rejected_mask)

    def test_returns_torch_tensors(self):
        """SimplePreferenceCollator returns torch tensors."""
        from open_instruct.dataset_transformation import (
            CHOSEN_ATTENTION_MASK_KEY,
            CHOSEN_INPUT_IDS_KEY,
            REJECTED_ATTENTION_MASK_KEY,
            REJECTED_INPUT_IDS_KEY,
            SimplePreferenceCollator,
        )

        collator = SimplePreferenceCollator(pad_token_id=0)

        batch = [
            {CHOSEN_INPUT_IDS_KEY: [1, 2], REJECTED_INPUT_IDS_KEY: [3, 4]},
        ]

        result = collator(batch)

        self.assertIsInstance(result[CHOSEN_INPUT_IDS_KEY], torch.Tensor)
        self.assertIsInstance(result[CHOSEN_ATTENTION_MASK_KEY], torch.Tensor)
        self.assertIsInstance(result[REJECTED_INPUT_IDS_KEY], torch.Tensor)
        self.assertIsInstance(result[REJECTED_ATTENTION_MASK_KEY], torch.Tensor)

    def test_uses_correct_pad_token(self):
        """SimplePreferenceCollator uses the specified pad_token_id."""
        from open_instruct.dataset_transformation import (
            CHOSEN_INPUT_IDS_KEY,
            REJECTED_INPUT_IDS_KEY,
            SimplePreferenceCollator,
        )

        pad_token_id = 999
        collator = SimplePreferenceCollator(pad_token_id=pad_token_id)

        batch = [
            {CHOSEN_INPUT_IDS_KEY: [1, 2], REJECTED_INPUT_IDS_KEY: [1, 2, 3]},
        ]

        result = collator(batch)

        # The padding should use pad_token_id
        self.assertEqual(result[CHOSEN_INPUT_IDS_KEY][0, 2].item(), pad_token_id)


if __name__ == "__main__":
    unittest.main()
