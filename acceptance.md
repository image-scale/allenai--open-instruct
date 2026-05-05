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

## Task 6: Core Utilities

### Acceptance Criteria
- [x] repeat_each() repeats each element in a sequence k times
- [x] MetricsTracker preallocates metrics array on specified device
- [x] MetricsTracker supports getting/setting metrics by name
- [x] MetricsTracker.get_metrics_list() returns dict of name to float values
- [x] warn_if_low_disk_space() warns when disk usage exceeds threshold
- [x] warn_if_low_disk_space() skips cloud paths (gs://, s3://, etc.)
- [x] get_last_checkpoint() returns latest completed checkpoint from folder
- [x] get_last_checkpoint() supports incomplete flag to include partial checkpoints
- [x] clean_last_n_checkpoints() removes old checkpoints keeping last N
- [x] GPU_SPECS contains specifications for common GPUs (h100, a100, etc.)
- [x] find_free_port() returns an available port number
- [x] max_num_processes() returns reasonable default for multiprocessing

## Task 7: Launch Utilities

### Acceptance Criteria
- [x] live_subprocess_output() runs command and prints output in real-time
- [x] live_subprocess_output() raises exception on non-zero return code
- [x] gs_folder_exists() returns True if GCS folder exists
- [x] gs_folder_exists() returns False if GCS folder does not exist
- [x] download_from_gs_bucket() downloads files from GCS to local path
- [x] upload_to_gs_bucket() uploads files from local path to GCS
- [x] validate_beaker_workspace() accepts valid 'org/workspace' format
- [x] validate_beaker_workspace() raises ValueError for invalid format

## Task 8: Dataset Transformation

### Acceptance Criteria
- [x] CHAT_TEMPLATES contains named chat templates (simple_concat_with_space, simple_chat, tulu, zephyr)
- [x] TokenizerConfig stores tokenizer_name_or_path, revision, chat_template_name, add_bos, trust_remote_code
- [x] TokenizerConfig has a tokenizer cached_property that returns configured tokenizer
- [x] sft_tokenize_v1() tokenizes messages and returns input_ids, attention_mask, labels
- [x] sft_tokenize_mask_out_prompt_v1() masks prompt tokens in labels with -100
- [x] sft_filter_v1() filters by max_prompt_token_length and max_token_length
- [x] preference_tokenize_v1() tokenizes chosen and rejected messages with prompt
- [x] preference_filter_v1() filters preference data by max lengths
- [x] rlvr_tokenize_v1() tokenizes RLVR data with ground_truth and verifier_source
- [x] rlvr_max_length_filter_v1() filters RLVR data by max_prompt_token_length
- [x] TRANSFORM_FNS registry maps function names to (function, operation_type) tuples
- [x] SimplePreferenceCollator pads chosen/rejected sequences and creates batches
