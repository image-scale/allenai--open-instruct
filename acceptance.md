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
- [x] ShutdownSentinel is a simple class that can be used as a queue sentinel
- [x] TokenStatistics stores num_prompt_tokens, num_response_tokens, generation_time, and earliest_start_time
- [x] ToolCallStats stores tool_name, success status, and runtime
- [x] RequestInfo stores tool usage information: num_calls, timeouts, tool_errors, outputs, runtimes
- [x] GenerationResult stores responses, finish_reasons, masks, logprobs, and related metadata
- [x] EnvConfigEntry stores env_name, is_text_env flag, and kwargs
- [x] EnvConfig stores max_steps and mapping of env_configs
- [x] PromptRequest stores prompt, generation_config, index, prompt_id, and environment config
- [x] CollatedBatchData stores query_responses, attention_masks, position_ids, advantages, response_masks, vllm_logprobs as lists of tensors
- [x] CollatedBatchData supports indexing with __getitem__
- [x] CollatedBatchData supports __len__ returning batch size
- [x] CollatedBatchData.to() moves all tensors to a specified device

## Task 3: Math Utilities

### Acceptance Criteria
- [ ] last_boxed_only_string() extracts the last \boxed{} content from a string
- [ ] last_boxed_only_string() handles \boxed with space syntax
- [ ] last_boxed_only_string() returns None when no boxed content found
- [ ] remove_boxed() removes \boxed{} wrapper from a string
- [ ] get_unnormalized_answer() extracts answer from "Final Answer: The final answer is X" format
- [ ] normalize_final_answer() performs substitutions and removes expressions like "dollars", "units", etc.
- [ ] normalize_final_answer() handles LaTeX math extraction from $...$
- [ ] fix_fracs() converts \frac12 to \frac{1}{2} format
- [ ] fix_sqrt() converts \sqrt2 to \sqrt{2} format
- [ ] fix_a_slash_b() converts "1/2" to \frac{1}{2}
- [ ] strip_string() normalizes LaTeX strings for comparison (removes spaces, fixes fracs, etc.)
- [ ] hendrycks_is_equiv() compares two strings after normalization
