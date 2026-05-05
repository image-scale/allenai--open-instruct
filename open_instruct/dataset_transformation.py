"""Dataset transformation utilities for tokenization and data processing.

This module provides utilities for transforming datasets for different training
paradigms including SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization),
and RLVR (RL with Verifiable Rewards).
"""

import copy
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from open_instruct.logging_utils import setup_logger

logger = setup_logger(__name__)

# ----------------------------------------------------------------------------
# Column keys for dataset fields
DEFAULT_SFT_MESSAGES_KEY = "messages"
GROUND_TRUTHS_KEY = "ground_truth"
VERIFIER_SOURCE_KEY = "dataset"
RAW_PROMPT_KEY = "prompt"

INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
MASKED_TOKEN_VALUE = -100

INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"

DEFAULT_CHOSEN_KEY = "chosen"
DEFAULT_REJECTED_KEY = "rejected"
CHOSEN_INPUT_IDS_KEY = "chosen_input_ids"
CHOSEN_ATTENTION_MASK_KEY = "chosen_attention_mask"
CHOSEN_LABELS_KEY = "chosen_labels"
REJECTED_INPUT_IDS_KEY = "rejected_input_ids"
REJECTED_ATTENTION_MASK_KEY = "rejected_attention_mask"
REJECTED_LABELS_KEY = "rejected_labels"


# ----------------------------------------------------------------------------
# Chat Templates
# These templates define how conversations are formatted for different models.
# Note: We added `{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}`
# to ensure the template does not output eos_token when `add_generation_prompt=True`
CHAT_TEMPLATES = {
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}


# ----------------------------------------------------------------------------
# Tokenizer Configuration and Functions
def get_tokenizer_simple_v1(tc: "TokenizerConfig") -> PreTrainedTokenizer:
    """Get a tokenizer with minimal configuration.

    Args:
        tc: TokenizerConfig with tokenizer settings.

    Returns:
        Configured PreTrainedTokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tc.tokenizer_name_or_path,
        revision=tc.tokenizer_revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    return tokenizer


def get_tokenizer_tulu_v1(tc: "TokenizerConfig") -> PreTrainedTokenizer:
    """Get a tokenizer configured for Tulu-style training.

    Handles special token setup for various tokenizer types and applies
    chat templates.

    Args:
        tc: TokenizerConfig with tokenizer settings.

    Returns:
        Configured PreTrainedTokenizer with chat template applied.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tc.tokenizer_name_or_path,
        revision=tc.tokenizer_revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # Set chat template
    if tc.chat_template_name is not None:
        if tc.chat_template_name in CHAT_TEMPLATES:
            tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
        else:
            raise ValueError(f"Unknown chat template: {tc.chat_template_name}")

    # Add bos token to template if requested
    if tc.add_bos:
        if tokenizer.chat_template is not None and tokenizer.chat_template.startswith("{{ bos_token }}"):
            raise ValueError(
                "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
            )
        if tokenizer.chat_template is not None:
            tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    return tokenizer


GET_TOKENIZER_FN = {
    "get_tokenizer_simple_v1": get_tokenizer_simple_v1,
    "get_tokenizer_tulu_v1": get_tokenizer_tulu_v1,
}


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer setup.

    Attributes:
        tokenizer_name_or_path: HuggingFace model/tokenizer name or local path.
        tokenizer_revision: Specific revision/commit to use.
        trust_remote_code: Whether to trust remote code in tokenizer.
        use_fast: Whether to use fast tokenizer.
        chat_template_name: Name of chat template from CHAT_TEMPLATES.
        add_bos: Whether to add BOS token to beginning of sequences.
        get_tokenizer_fn: Name of function to use for getting tokenizer.
        sft_messages_key: Column name for SFT messages in dataset.
        ground_truths_key: Column name for ground truth labels.
    """

    tokenizer_name_or_path: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    use_fast: bool = True
    chat_template_name: str | None = None
    add_bos: bool = False
    get_tokenizer_fn: str = "get_tokenizer_tulu_v1"
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY
    ground_truths_key: str = GROUND_TRUTHS_KEY

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the configured tokenizer.

        Returns:
            Configured PreTrainedTokenizer instance.

        Raises:
            ValueError: If tokenizer_name_or_path is not set.
        """
        if self.tokenizer_name_or_path is None:
            raise ValueError("tokenizer_name_or_path must be set")
        return GET_TOKENIZER_FN[self.get_tokenizer_fn](self)


# ----------------------------------------------------------------------------
# SFT Tokenization Functions
def sft_tokenize_v1(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
) -> dict[str, Any]:
    """Tokenize a row for SFT training.

    Tokenizes the full conversation and creates input_ids, attention_mask,
    and labels. Labels are a copy of input_ids (no masking).

    Args:
        row: Dataset row containing messages.
        tokenizer: Tokenizer to use.
        sft_messages_key: Key for messages in the row.

    Returns:
        Row with tokenization results added.
    """
    messages = row[sft_messages_key]
    prompt = messages if len(messages) == 1 else messages[:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(messages, return_dict=False)
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = copy.deepcopy(row[INPUT_IDS_KEY])

    return row


def sft_tokenize_mask_out_prompt_v1(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
) -> dict[str, Any]:
    """Tokenize a row for SFT training with prompt masking.

    Masks out the prompt tokens in labels by setting them to -100,
    so that loss is only computed on the response.

    Args:
        row: Dataset row containing messages.
        tokenizer: Tokenizer to use.
        sft_messages_key: Key for messages in the row.

    Returns:
        Row with tokenization results added, prompt tokens masked in labels.
    """
    messages = row[sft_messages_key]
    prompt = messages if len(messages) == 1 else messages[:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(messages, return_dict=False)
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])

    # Mask out prompt tokens
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [MASKED_TOKEN_VALUE] * len(row[INPUT_IDS_PROMPT_KEY])
    row[LABELS_KEY] = labels

    return row


def sft_filter_v1(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_token_length: int | None = None,
    max_token_length: int | None = None,
    need_contain_labels: bool = True,
) -> bool:
    """Filter SFT examples by length constraints.

    Args:
        row: Dataset row with tokenization results.
        tokenizer: Tokenizer (unused but kept for consistency).
        max_prompt_token_length: Maximum allowed prompt length.
        max_token_length: Maximum allowed total sequence length.
        need_contain_labels: Whether to require non-masked labels.

    Returns:
        True if the row passes all filters, False otherwise.
    """
    # Check prompt length
    if max_prompt_token_length is not None:
        if len(row[INPUT_IDS_PROMPT_KEY]) > max_prompt_token_length:
            return False

    # Check total length
    if max_token_length is not None:
        if len(row[INPUT_IDS_KEY]) > max_token_length:
            return False

    # Check that labels contain at least one non-masked token
    if need_contain_labels:
        if not any(x != MASKED_TOKEN_VALUE for x in row[LABELS_KEY]):
            return False

    return True


# ----------------------------------------------------------------------------
# Preference Tokenization Functions (for DPO/RM)
def preference_tokenize_v1(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Any]:
    """Tokenize a row for preference-based training (DPO/RM).

    Tokenizes prompt, chosen response, and rejected response separately.

    Args:
        row: Dataset row containing 'chosen' and 'rejected' messages.
        tokenizer: Tokenizer to use.

    Returns:
        Row with tokenization results for prompt, chosen, and rejected.
    """
    # Extract prompt (all messages except the last one from chosen)
    prompt = row["chosen"][:-1]

    # Tokenize prompt
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    row[ATTENTION_MASK_PROMPT_KEY] = [1] * len(row[INPUT_IDS_PROMPT_KEY])

    # Tokenize chosen completion
    row[CHOSEN_INPUT_IDS_KEY] = tokenizer.apply_chat_template(row["chosen"], return_dict=False)
    row[CHOSEN_ATTENTION_MASK_KEY] = [1] * len(row[CHOSEN_INPUT_IDS_KEY])

    # Tokenize rejected completion
    row[REJECTED_INPUT_IDS_KEY] = tokenizer.apply_chat_template(row["rejected"], return_dict=False)
    row[REJECTED_ATTENTION_MASK_KEY] = [1] * len(row[REJECTED_INPUT_IDS_KEY])

    return row


def preference_filter_v1(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_token_length: int | None = None,
    max_token_length: int | None = None,
) -> bool:
    """Filter preference examples by length constraints.

    Args:
        row: Dataset row with tokenization results.
        tokenizer: Tokenizer (unused but kept for consistency).
        max_prompt_token_length: Maximum allowed prompt length.
        max_token_length: Maximum allowed total sequence length.

    Returns:
        True if the row passes all filters, False otherwise.
    """
    # Check prompt length
    if max_prompt_token_length is not None:
        if len(row[INPUT_IDS_PROMPT_KEY]) > max_prompt_token_length:
            return False

    # Check chosen length
    if max_token_length is not None:
        if len(row[CHOSEN_INPUT_IDS_KEY]) > max_token_length:
            return False
        if len(row[REJECTED_INPUT_IDS_KEY]) > max_token_length:
            return False

    return True


# ----------------------------------------------------------------------------
# RLVR Tokenization Functions
def rlvr_tokenize_v1(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
    ground_truths_key: str = GROUND_TRUTHS_KEY,
    verifier_source_key: str = VERIFIER_SOURCE_KEY,
) -> dict[str, Any]:
    """Tokenize a row for RLVR training.

    Tokenizes messages and preserves ground truth and verifier source
    information for verification during RL training.

    Args:
        row: Dataset row containing messages and ground truth.
        tokenizer: Tokenizer to use.
        sft_messages_key: Key for messages in the row.
        ground_truths_key: Key for ground truth labels.
        verifier_source_key: Key for verifier source identifier.

    Returns:
        Row with tokenization results and preserved metadata.
    """
    messages = row[sft_messages_key]
    prompt = messages if len(messages) == 1 else messages[:-1]

    # Handle case where last message is from assistant
    if len(prompt) > 1 and prompt[-1].get("role") == "assistant":
        prompt = prompt[:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )

    # Remove any padding tokens that might have slipped in
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id in row[INPUT_IDS_PROMPT_KEY]:
        row[INPUT_IDS_PROMPT_KEY] = [
            x for x in row[INPUT_IDS_PROMPT_KEY] if x != tokenizer.pad_token_id
        ]

    # Get ground truth and verifier source
    ground_truths_val = row[ground_truths_key]
    verifier_source_val = row[verifier_source_key]

    # Wrap in list for compatibility with multi-verifier datasets
    if isinstance(verifier_source_val, str):
        verifier_source_val = [verifier_source_val]
        ground_truths_val = [ground_truths_val]

    row[GROUND_TRUTHS_KEY] = ground_truths_val
    row[VERIFIER_SOURCE_KEY] = verifier_source_val

    # Create raw prompt string
    row[RAW_PROMPT_KEY] = "\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt)

    return row


def rlvr_max_length_filter_v1(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_token_length: int | None = None,
) -> bool:
    """Filter RLVR examples by prompt length.

    Args:
        row: Dataset row with tokenization results.
        tokenizer: Tokenizer (unused but kept for consistency).
        max_prompt_token_length: Maximum allowed prompt length.

    Returns:
        True if the row passes the filter, False otherwise.
    """
    if max_prompt_token_length is not None:
        if len(row[INPUT_IDS_PROMPT_KEY]) > max_prompt_token_length:
            return False
    return True


# ----------------------------------------------------------------------------
# Transform Function Registry
# Maps function names to (function, operation_type) tuples
# operation_type is either "map" or "filter"
TRANSFORM_FNS = {
    "sft_tokenize_v1": (sft_tokenize_v1, "map"),
    "sft_tokenize_mask_out_prompt_v1": (sft_tokenize_mask_out_prompt_v1, "map"),
    "sft_filter_v1": (sft_filter_v1, "filter"),
    "preference_tokenize_v1": (preference_tokenize_v1, "map"),
    "preference_filter_v1": (preference_filter_v1, "filter"),
    "rlvr_tokenize_v1": (rlvr_tokenize_v1, "map"),
    "rlvr_max_length_filter_v1": (rlvr_max_length_filter_v1, "filter"),
}


# ----------------------------------------------------------------------------
# Collators
class SimplePreferenceCollator:
    """Collator for preference dataset batching.

    Pads chosen and rejected sequences to the same length within a batch.
    Always pads from the right.

    Attributes:
        pad_token_id: Token ID to use for padding.
    """

    def __init__(self, pad_token_id: int):
        """Initialize the collator.

        Args:
            pad_token_id: Token ID to use for padding.
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Collate a batch of preference examples.

        Args:
            batch: List of dictionaries with chosen and rejected input_ids.

        Returns:
            Dictionary with padded tensors for chosen and rejected sequences.
        """
        # Find max lengths
        max_length_chosen = max(len(item[CHOSEN_INPUT_IDS_KEY]) for item in batch)
        max_length_rejected = max(len(item[REJECTED_INPUT_IDS_KEY]) for item in batch)
        max_length = max(max_length_chosen, max_length_rejected)

        # Pad sequences
        chosen_input_ids = []
        chosen_attention_mask = []
        rejected_input_ids = []
        rejected_attention_mask = []

        for item in batch:
            # Pad chosen
            chosen_ids = item[CHOSEN_INPUT_IDS_KEY]
            pad_len = max_length - len(chosen_ids)
            chosen_input_ids.append(chosen_ids + [self.pad_token_id] * pad_len)
            chosen_attention_mask.append([1] * len(chosen_ids) + [0] * pad_len)

            # Pad rejected
            rejected_ids = item[REJECTED_INPUT_IDS_KEY]
            pad_len = max_length - len(rejected_ids)
            rejected_input_ids.append(rejected_ids + [self.pad_token_id] * pad_len)
            rejected_attention_mask.append([1] * len(rejected_ids) + [0] * pad_len)

        return {
            CHOSEN_INPUT_IDS_KEY: torch.tensor(chosen_input_ids, dtype=torch.long),
            CHOSEN_ATTENTION_MASK_KEY: torch.tensor(chosen_attention_mask, dtype=torch.long),
            REJECTED_INPUT_IDS_KEY: torch.tensor(rejected_input_ids, dtype=torch.long),
            REJECTED_ATTENTION_MASK_KEY: torch.tensor(rejected_attention_mask, dtype=torch.long),
        }
