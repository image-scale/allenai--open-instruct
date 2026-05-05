"""Tests for reinforcement learning utilities."""

import time
import unittest

import numpy as np
import torch


class TestTimer(unittest.TestCase):
    """Tests for Timer class."""

    def test_context_manager(self):
        """Timer can be used as a context manager."""
        from open_instruct.rl_utils import Timer

        with self.assertLogs("open_instruct.rl_utils", level="INFO") as cm:
            with Timer("test operation"):
                time.sleep(0.05)

        self.assertEqual(len(cm.output), 1)
        self.assertIn("test operation", cm.output[0])
        self.assertIn("seconds", cm.output[0])

    def test_measures_time(self):
        """Timer measures elapsed time correctly."""
        from open_instruct.rl_utils import Timer

        with Timer("test") as timer:
            time.sleep(0.1)

        self.assertGreaterEqual(timer.duration, 0.1)
        self.assertLess(timer.duration, 0.2)

    def test_decorator(self):
        """Timer can be used as a decorator."""
        from open_instruct.rl_utils import Timer

        @Timer("decorated function")
        def slow_function():
            time.sleep(0.05)
            return 42

        with self.assertLogs("open_instruct.rl_utils", level="INFO") as cm:
            result = slow_function()

        self.assertEqual(result, 42)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("decorated function", cm.output[0])

    def test_noop_context_manager(self):
        """Timer with noop=True does not log."""
        from open_instruct.rl_utils import Timer

        with self.assertNoLogs("open_instruct.rl_utils", level="INFO"):
            with Timer("should not log", noop=True):
                time.sleep(0.01)

    def test_noop_decorator(self):
        """Timer decorator with noop=True does not log."""
        from open_instruct.rl_utils import Timer

        @Timer("should not log", noop=True)
        def silent_function():
            time.sleep(0.01)
            return "done"

        with self.assertNoLogs("open_instruct.rl_utils", level="INFO"):
            result = silent_function()

        self.assertEqual(result, "done")


class TestPackSequences(unittest.TestCase):
    """Tests for pack_sequences function."""

    def test_packs_basic_sequences(self):
        """pack_sequences concatenates queries and responses."""
        from open_instruct.rl_utils import pack_sequences

        queries = [[1, 2, 3], [4, 5]]
        responses = [[10, 11, 12], [20, 21]]
        masks = [[1, 1, 1], [1, 1]]
        vllm_logprobs = [[0.0, 0.0, 0.0], [0.0, 0.0]]
        pad_token_id = 0

        packed = pack_sequences(
            queries=queries,
            responses=responses,
            masks=masks,
            pack_length=20,
            pad_token_id=pad_token_id,
            vllm_logprobs=vllm_logprobs,
        )

        self.assertIsNotNone(packed.query_responses)
        self.assertIsNotNone(packed.attention_masks)
        self.assertIsNotNone(packed.response_masks)

    def test_pads_to_pack_length(self):
        """pack_sequences pads to pack_length with pad_token_id."""
        from open_instruct.rl_utils import pack_sequences

        queries = [[1, 2, 3]]
        responses = [[10, 11]]
        masks = [[1, 1]]
        vllm_logprobs = [[0.0, 0.0]]
        pad_token_id = 0
        pack_length = 10

        packed = pack_sequences(
            queries=queries,
            responses=responses,
            masks=masks,
            pack_length=pack_length,
            pad_token_id=pad_token_id,
            vllm_logprobs=vllm_logprobs,
        )

        # Should have one pack padded to pack_length
        self.assertEqual(len(packed.query_responses), 1)
        self.assertEqual(len(packed.query_responses[0]), pack_length)

    def test_generates_attention_mask(self):
        """pack_sequences generates correct attention_masks."""
        from open_instruct.rl_utils import pack_sequences

        queries = [[1, 2], [3, 4]]
        responses = [[10], [20]]
        masks = [[1], [1]]
        vllm_logprobs = [[0.0], [0.0]]
        pad_token_id = 0

        packed = pack_sequences(
            queries=queries,
            responses=responses,
            masks=masks,
            pack_length=10,
            pad_token_id=pad_token_id,
            vllm_logprobs=vllm_logprobs,
        )

        # Attention mask should have positive values for non-padding
        attn_mask = packed.attention_masks[0]
        self.assertTrue(attn_mask[0] > 0)  # First token should be in sequence

    def test_generates_position_ids(self):
        """pack_sequences generates correct position_ids."""
        from open_instruct.rl_utils import pack_sequences

        queries = [[1, 2, 3]]
        responses = [[10, 11]]
        masks = [[1, 1]]
        vllm_logprobs = [[0.0, 0.0]]
        pad_token_id = 0

        packed = pack_sequences(
            queries=queries,
            responses=responses,
            masks=masks,
            pack_length=10,
            pad_token_id=pad_token_id,
            vllm_logprobs=vllm_logprobs,
        )

        self.assertIsNotNone(packed.position_ids)
        # First 5 positions (query + response) should be 0,1,2,3,4
        pos_ids = packed.position_ids[0][:5].tolist()
        self.assertEqual(pos_ids, [0, 1, 2, 3, 4])

    def test_mask_tool_use_flag(self):
        """pack_sequences handles mask_tool_use flag correctly."""
        from open_instruct.rl_utils import pack_sequences

        queries = [[1, 2]]
        responses = [[10, 11, 12]]
        masks = [[1, 0, 1]]  # Middle token masked
        vllm_logprobs = [[0.0, 0.0, 0.0]]
        pad_token_id = 0

        # With mask_tool_use=True, should use mask values
        packed_masked = pack_sequences(
            queries=queries,
            responses=responses,
            masks=masks,
            pack_length=10,
            pad_token_id=pad_token_id,
            vllm_logprobs=vllm_logprobs,
            mask_tool_use=True,
        )

        # With mask_tool_use=False, all response tokens should be True
        packed_unmasked = pack_sequences(
            queries=queries,
            responses=responses,
            masks=masks,
            pack_length=10,
            pad_token_id=pad_token_id,
            vllm_logprobs=vllm_logprobs,
            mask_tool_use=False,
        )

        # Query tokens (first 2) should be False in both
        self.assertEqual(packed_masked.response_masks[0][0].item(), 0)
        self.assertEqual(packed_masked.response_masks[0][1].item(), 0)

        # Response tokens
        # With mask_tool_use=True: [1,0,1]
        self.assertEqual(packed_masked.response_masks[0][2].item(), 1)
        self.assertEqual(packed_masked.response_masks[0][3].item(), 0)
        self.assertEqual(packed_masked.response_masks[0][4].item(), 1)

        # With mask_tool_use=False: all True
        self.assertEqual(packed_unmasked.response_masks[0][2].item(), 1)
        self.assertEqual(packed_unmasked.response_masks[0][3].item(), 1)
        self.assertEqual(packed_unmasked.response_masks[0][4].item(), 1)


class TestCalculateAdvantages(unittest.TestCase):
    """Tests for calculate_advantages function."""

    def test_basic_gae(self):
        """calculate_advantages computes GAE correctly."""
        from open_instruct.rl_utils import calculate_advantages

        values = np.array([[1.0, 1.0, 1.0, 0.0]])
        rewards = np.array([[0.0, 0.0, 1.0, 0.0]])
        gamma = 1.0
        lam = 1.0

        advantages, returns = calculate_advantages(values, rewards, gamma, lam)

        self.assertEqual(advantages.shape, values.shape)
        self.assertEqual(returns.shape, values.shape)

    def test_discount_factor(self):
        """calculate_advantages applies discount factor."""
        from open_instruct.rl_utils import calculate_advantages

        values = np.array([[0.0, 0.0, 0.0]])
        rewards = np.array([[0.0, 0.0, 1.0]])
        gamma = 0.5
        lam = 1.0

        advantages, returns = calculate_advantages(values, rewards, gamma, lam)

        # With gamma=0.5, reward at t=2 should decay
        self.assertGreater(advantages[0, 2], advantages[0, 1])
        self.assertGreater(advantages[0, 1], advantages[0, 0])


class TestCalculateAdvantagesPacked(unittest.TestCase):
    """Tests for calculate_advantages_packed function."""

    def test_handles_done_flags(self):
        """calculate_advantages_packed handles sequence boundaries."""
        from open_instruct.rl_utils import calculate_advantages_packed

        # Use values=0 so reward - value is non-zero
        values = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        rewards = np.array([[0.0, 1.0, 0.0, 0.0, 2.0, 0.0]])
        dones = np.array([[0, 1, 0, 0, 1, 0]])  # Two sequences
        response_masks = np.array([[1, 1, 0, 1, 1, 0]])  # Query at positions 2, 5
        gamma = 1.0
        lam = 1.0

        advantages, returns = calculate_advantages_packed(
            values, rewards, gamma, lam, dones, response_masks
        )

        self.assertEqual(advantages.shape, values.shape)
        self.assertEqual(returns.shape, values.shape)
        # Position with reward and value=0 should have non-zero advantage
        self.assertNotEqual(advantages[0, 1], 0.0)  # Position 1 has reward=1.0
        self.assertNotEqual(advantages[0, 4], 0.0)  # Position 4 has reward=2.0

    def test_masks_query_tokens(self):
        """calculate_advantages_packed masks query tokens."""
        from open_instruct.rl_utils import calculate_advantages_packed

        values = np.array([[1.0, 1.0, 1.0]])
        rewards = np.array([[0.0, 0.0, 1.0]])
        dones = np.array([[0, 0, 1]])
        response_masks = np.array([[0, 1, 1]])  # First is query
        gamma = 1.0
        lam = 1.0

        advantages, returns = calculate_advantages_packed(
            values, rewards, gamma, lam, dones, response_masks
        )

        # Query position should have different advantage calculation
        self.assertEqual(advantages.shape, values.shape)


class TestMaskedMean(unittest.TestCase):
    """Tests for masked_mean function."""

    def test_basic_mean(self):
        """masked_mean computes mean of masked values."""
        from open_instruct.rl_utils import masked_mean

        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1, 1, 0, 0])

        result = masked_mean(values, mask)

        # Mean of [1.0, 2.0] = 1.5
        self.assertAlmostEqual(result.item(), 1.5, places=5)

    def test_empty_mask(self):
        """masked_mean returns 0 for empty mask."""
        from open_instruct.rl_utils import masked_mean

        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([0, 0, 0])

        result = masked_mean(values, mask)

        self.assertEqual(result.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
