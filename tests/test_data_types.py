"""Tests for data container classes."""

import unittest

import torch


class TestShutdownSentinel(unittest.TestCase):
    """Tests for the ShutdownSentinel class."""

    def test_can_instantiate(self):
        """ShutdownSentinel can be instantiated."""
        from open_instruct.data_types import ShutdownSentinel

        sentinel = ShutdownSentinel()
        self.assertIsInstance(sentinel, ShutdownSentinel)

    def test_usable_as_sentinel(self):
        """ShutdownSentinel can be used to signal queue shutdown."""
        from open_instruct.data_types import ShutdownSentinel

        sentinel = ShutdownSentinel()
        # Check it can be compared
        self.assertEqual(type(sentinel), ShutdownSentinel)


class TestTokenStatistics(unittest.TestCase):
    """Tests for the TokenStatistics class."""

    def test_stores_required_fields(self):
        """TokenStatistics stores all required fields."""
        from open_instruct.data_types import TokenStatistics

        stats = TokenStatistics(
            num_prompt_tokens=100,
            num_response_tokens=50,
            generation_time=1.5,
        )
        self.assertEqual(stats.num_prompt_tokens, 100)
        self.assertEqual(stats.num_response_tokens, 50)
        self.assertEqual(stats.generation_time, 1.5)
        self.assertIsNone(stats.earliest_start_time)

    def test_optional_earliest_start_time(self):
        """TokenStatistics can store optional earliest_start_time."""
        from open_instruct.data_types import TokenStatistics

        stats = TokenStatistics(
            num_prompt_tokens=100,
            num_response_tokens=50,
            generation_time=1.5,
            earliest_start_time=12345.0,
        )
        self.assertEqual(stats.earliest_start_time, 12345.0)


class TestToolCallStats(unittest.TestCase):
    """Tests for the ToolCallStats class."""

    def test_stores_all_fields(self):
        """ToolCallStats stores tool_name, success, and runtime."""
        from open_instruct.data_types import ToolCallStats

        stats = ToolCallStats(
            tool_name="calculator",
            success=True,
            runtime=0.5,
        )
        self.assertEqual(stats.tool_name, "calculator")
        self.assertTrue(stats.success)
        self.assertEqual(stats.runtime, 0.5)


class TestRequestInfo(unittest.TestCase):
    """Tests for the RequestInfo class."""

    def test_stores_tool_usage_info(self):
        """RequestInfo stores all tool usage information."""
        from open_instruct.data_types import RequestInfo

        info = RequestInfo(
            num_calls=[1, 2, 3],
            timeouts=[0, 0, 1],
            tool_errors=["", "", "timeout"],
            tool_outputs=["result1", "result2", ""],
            tool_runtimes=[0.1, 0.2, 5.0],
            tool_calleds=[True, True, True],
        )
        self.assertEqual(info.num_calls, [1, 2, 3])
        self.assertEqual(info.timeouts, [0, 0, 1])
        self.assertEqual(info.tool_errors, ["", "", "timeout"])
        self.assertEqual(info.tool_outputs, ["result1", "result2", ""])
        self.assertEqual(info.tool_runtimes, [0.1, 0.2, 5.0])
        self.assertEqual(info.tool_calleds, [True, True, True])

    def test_default_factory_fields(self):
        """RequestInfo has default factory for list fields."""
        from open_instruct.data_types import RequestInfo

        info = RequestInfo(
            num_calls=[],
            timeouts=[],
            tool_errors=[],
            tool_outputs=[],
            tool_runtimes=[],
            tool_calleds=[],
        )
        self.assertEqual(info.tool_call_stats, [])
        self.assertEqual(info.rollout_states, [])


class TestGenerationResult(unittest.TestCase):
    """Tests for the GenerationResult class."""

    def test_stores_generation_data(self):
        """GenerationResult stores responses, finish_reasons, masks, and metadata."""
        from open_instruct.data_types import GenerationResult, RequestInfo

        info = RequestInfo(
            num_calls=[],
            timeouts=[],
            tool_errors=[],
            tool_outputs=[],
            tool_runtimes=[],
            tool_calleds=[],
        )
        result = GenerationResult(
            responses=[[1, 2, 3], [4, 5, 6]],
            finish_reasons=["stop", "length"],
            masks=[[1, 1, 1], [1, 1, 0]],
            request_info=info,
            index=0,
            prompt_id="test-123",
        )
        self.assertEqual(result.responses, [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(result.finish_reasons, ["stop", "length"])
        self.assertEqual(result.masks, [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(result.index, 0)
        self.assertEqual(result.prompt_id, "test-123")
        self.assertEqual(result.logprobs, [])
        self.assertEqual(result.model_step, 0)


class TestEnvConfigEntry(unittest.TestCase):
    """Tests for the EnvConfigEntry class."""

    def test_stores_env_config(self):
        """EnvConfigEntry stores env_name, is_text_env, and kwargs."""
        from open_instruct.data_types import EnvConfigEntry

        entry = EnvConfigEntry(
            env_name="guess_number",
            is_text_env=True,
            kwargs={"min": 1, "max": 100},
        )
        self.assertEqual(entry.env_name, "guess_number")
        self.assertTrue(entry.is_text_env)
        self.assertEqual(entry.kwargs, {"min": 1, "max": 100})

    def test_default_kwargs(self):
        """EnvConfigEntry has default empty kwargs."""
        from open_instruct.data_types import EnvConfigEntry

        entry = EnvConfigEntry(env_name="test", is_text_env=False)
        self.assertEqual(entry.kwargs, {})


class TestEnvConfig(unittest.TestCase):
    """Tests for the EnvConfig class."""

    def test_stores_max_steps_and_env_configs(self):
        """EnvConfig stores max_steps and env_configs mapping."""
        from open_instruct.data_types import EnvConfig, EnvConfigEntry

        entry = EnvConfigEntry(env_name="counter", is_text_env=False)
        config = EnvConfig(max_steps=50, env_configs={"counter": entry})
        self.assertEqual(config.max_steps, 50)
        self.assertEqual(config.env_configs["counter"].env_name, "counter")

    def test_default_values(self):
        """EnvConfig has sensible defaults."""
        from open_instruct.data_types import EnvConfig

        config = EnvConfig()
        self.assertEqual(config.max_steps, 100)
        self.assertEqual(config.env_configs, {})


class TestPromptRequest(unittest.TestCase):
    """Tests for the PromptRequest class."""

    def test_stores_prompt_data(self):
        """PromptRequest stores prompt, generation_config, index, prompt_id."""
        from open_instruct.data_types import PromptRequest

        request = PromptRequest(
            prompt=[1, 2, 3, 4, 5],
            generation_config={"max_tokens": 100},
            index=0,
            prompt_id="req-001",
        )
        self.assertEqual(request.prompt, [1, 2, 3, 4, 5])
        self.assertEqual(request.generation_config, {"max_tokens": 100})
        self.assertEqual(request.index, 0)
        self.assertEqual(request.prompt_id, "req-001")
        self.assertFalse(request.is_eval)
        self.assertIsNone(request.active_tools)

    def test_env_config_default(self):
        """PromptRequest has default EnvConfig."""
        from open_instruct.data_types import PromptRequest

        request = PromptRequest(
            prompt=[],
            generation_config=None,
            index=0,
            prompt_id="test",
        )
        self.assertIsNotNone(request.env_config)
        self.assertEqual(request.env_config.max_steps, 100)


class TestCollatedBatchData(unittest.TestCase):
    """Tests for the CollatedBatchData class."""

    def test_stores_tensor_lists(self):
        """CollatedBatchData stores lists of tensors."""
        from open_instruct.data_types import CollatedBatchData

        batch = CollatedBatchData(
            query_responses=[torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
            attention_masks=[torch.tensor([1, 1, 1]), torch.tensor([1, 1])],
            position_ids=[torch.tensor([0, 1, 2]), torch.tensor([0, 1])],
            advantages=[torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])],
            response_masks=[torch.tensor([1, 1, 1]), torch.tensor([1, 1])],
            vllm_logprobs=[torch.tensor([-0.1, -0.2, -0.3]), torch.tensor([-0.4, -0.5])],
        )
        self.assertEqual(len(batch.query_responses), 2)
        self.assertEqual(len(batch.attention_masks), 2)

    def test_getitem_with_int(self):
        """CollatedBatchData supports integer indexing."""
        from open_instruct.data_types import CollatedBatchData

        batch = CollatedBatchData(
            query_responses=[torch.tensor([1, 2]), torch.tensor([3, 4])],
            attention_masks=[torch.tensor([1, 1]), torch.tensor([1, 1])],
            position_ids=[torch.tensor([0, 1]), torch.tensor([0, 1])],
            advantages=[torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
            response_masks=[torch.tensor([1, 1]), torch.tensor([1, 1])],
            vllm_logprobs=[torch.tensor([-0.1, -0.2]), torch.tensor([-0.3, -0.4])],
        )
        item = batch[0]
        torch.testing.assert_close(item.query_responses, torch.tensor([1, 2]))

    def test_getitem_with_slice(self):
        """CollatedBatchData supports slice indexing."""
        from open_instruct.data_types import CollatedBatchData

        batch = CollatedBatchData(
            query_responses=[torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6])],
            attention_masks=[torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])],
            position_ids=[torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([0, 1])],
            advantages=[torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]), torch.tensor([0.5, 0.6])],
            response_masks=[torch.tensor([1, 1]), torch.tensor([1, 1]), torch.tensor([1, 1])],
            vllm_logprobs=[torch.tensor([-0.1, -0.2]), torch.tensor([-0.3, -0.4]), torch.tensor([-0.5, -0.6])],
        )
        subset = batch[0:2]
        self.assertEqual(len(subset), 2)
        self.assertIsInstance(subset, CollatedBatchData)

    def test_len(self):
        """CollatedBatchData.__len__ returns batch size."""
        from open_instruct.data_types import CollatedBatchData

        batch = CollatedBatchData(
            query_responses=[torch.tensor([1, 2]), torch.tensor([3, 4])],
            attention_masks=[torch.tensor([1, 1]), torch.tensor([1, 1])],
            position_ids=[torch.tensor([0, 1]), torch.tensor([0, 1])],
            advantages=[torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
            response_masks=[torch.tensor([1, 1]), torch.tensor([1, 1])],
            vllm_logprobs=[torch.tensor([-0.1, -0.2]), torch.tensor([-0.3, -0.4])],
        )
        self.assertEqual(len(batch), 2)

    def test_to_moves_tensors(self):
        """CollatedBatchData.to() moves all tensors to specified device."""
        from open_instruct.data_types import CollatedBatchData

        batch = CollatedBatchData(
            query_responses=[torch.tensor([1, 2])],
            attention_masks=[torch.tensor([1, 1])],
            position_ids=[torch.tensor([0, 1])],
            advantages=[torch.tensor([0.1, 0.2])],
            response_masks=[torch.tensor([1, 1])],
            vllm_logprobs=[torch.tensor([-0.1, -0.2])],
        )
        # Move to CPU (it's already there, but this tests the method)
        moved = batch.to(torch.device("cpu"))
        self.assertIsInstance(moved, CollatedBatchData)
        self.assertEqual(moved.query_responses[0].device.type, "cpu")

    def test_to_returns_new_instance(self):
        """CollatedBatchData.to() returns a new instance, not mutating original."""
        from open_instruct.data_types import CollatedBatchData

        original = CollatedBatchData(
            query_responses=[torch.tensor([1, 2])],
            attention_masks=[torch.tensor([1, 1])],
            position_ids=[torch.tensor([0, 1])],
            advantages=[torch.tensor([0.1, 0.2])],
            response_masks=[torch.tensor([1, 1])],
            vllm_logprobs=[torch.tensor([-0.1, -0.2])],
        )
        moved = original.to(torch.device("cpu"))
        self.assertIsNot(original, moved)


if __name__ == "__main__":
    unittest.main()
