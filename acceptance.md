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
- [x] last_boxed_only_string() extracts the last \boxed{} content from a string
- [x] last_boxed_only_string() handles \boxed with space syntax
- [x] last_boxed_only_string() returns None when no boxed content found
- [x] remove_boxed() removes \boxed{} wrapper from a string
- [x] get_unnormalized_answer() extracts answer from "Final Answer: The final answer is X" format
- [x] normalize_final_answer() performs substitutions and removes expressions like "dollars", "units", etc.
- [x] normalize_final_answer() handles LaTeX math extraction from $...$
- [x] fix_fracs() converts \frac12 to \frac{1}{2} format
- [x] fix_sqrt() converts \sqrt2 to \sqrt{2} format
- [x] fix_a_slash_b() converts "1/2" to \frac{1}{2}
- [x] strip_string() normalizes LaTeX strings for comparison (removes spaces, fixes fracs, etc.)
- [x] hendrycks_is_equiv() compares two strings after normalization

## Task 4: Ground Truth Verifiers

### Acceptance Criteria
- [x] VerificationResult stores score, optional cost, and optional reasoning
- [x] VerifierFunction is an abstract base class with __call__ method
- [x] VerifierFunction has name, weight, and verifier_config attributes
- [x] GSM8KVerifier extracts the last number from text and compares to label
- [x] GSM8KVerifier handles numbers with commas (e.g., "1,000" -> "1000")
- [x] GSM8KVerifier handles signed numbers (e.g., "-3", "+7")
- [x] MathVerifier tries multiple extraction methods (boxed, Minerva, LaTeX)
- [x] MathVerifier returns 1.0 if any extraction matches label
- [x] F1Verifier calculates F1 score between prediction and label tokens
- [x] F1Verifier supports list of labels and returns max F1
- [x] remove_thinking_section() strips <think>...</think> and <answer> tags
- [x] PuzzleMatcherVerifier normalizes prediction and checks if label is contained

## Task 5: RL Utilities

### Acceptance Criteria
- [x] Timer can be used as a context manager to measure execution time
- [x] Timer can be used as a decorator to measure function execution time
- [x] Timer logs the elapsed time with a description
- [x] Timer has a noop mode that skips logging
- [x] pack_sequences() concatenates queries and responses into packed sequences
- [x] pack_sequences() pads sequences to pack_length with pad_token_id
- [x] pack_sequences() generates correct attention_masks and position_ids
- [x] pack_sequences() correctly handles response_masks based on mask_tool_use flag
- [x] calculate_advantages() computes GAE advantages from values and rewards
- [x] calculate_advantages_packed() handles packed sequences with done flags
