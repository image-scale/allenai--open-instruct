# Acceptance Criteria

## Task 1: Logging Configuration

### Acceptance Criteria
- [x] setup_logger(name=None) returns root logger when name is None
- [x] setup_logger(name="module_name") returns named logger
- [x] Logger format includes timestamp, level, filename, line number, and message
- [x] Rank 0 logs at INFO level by default
- [x] Non-zero ranks log at WARNING level by default
- [x] basicConfig is only called once (no duplicate handlers)
- [x] Datetime format is "YYYY-MM-DD HH:MM:SS"

## Task 2: Data Container Classes

### Acceptance Criteria
- [ ] ShutdownSentinel is a simple class that can be used as a queue sentinel
- [ ] TokenStatistics stores num_prompt_tokens, num_response_tokens, generation_time, and earliest_start_time
- [ ] ToolCallStats stores tool_name, success status, and runtime
- [ ] RequestInfo stores tool usage information: num_calls, timeouts, tool_errors, outputs, runtimes
- [ ] GenerationResult stores responses, finish_reasons, masks, logprobs, and related metadata
- [ ] EnvConfigEntry stores env_name, is_text_env flag, and kwargs
- [ ] EnvConfig stores max_steps and mapping of env_configs
- [ ] PromptRequest stores prompt, generation_config, index, prompt_id, and environment config
- [ ] CollatedBatchData stores query_responses, attention_masks, position_ids, advantages, response_masks, vllm_logprobs as lists of tensors
- [ ] CollatedBatchData supports indexing with __getitem__
- [ ] CollatedBatchData supports __len__ returning batch size
- [ ] CollatedBatchData.to() moves all tensors to a specified device
