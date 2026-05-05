"""Reinforcement learning utilities for training."""

import contextlib
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
import torch

from open_instruct.logging_utils import setup_logger

T = TypeVar("T")
logger = setup_logger(__name__)


@dataclass
class Timer(contextlib.ContextDecorator):
    """A context manager and decorator for timing code blocks.

    Can be used as a context manager:
        with Timer("operation"):
            # code to time

    Or as a decorator:
        @Timer("function call")
        def my_function():
            # code to time
    """

    description: str
    noop: bool = False
    start_time: float = field(init=False, default=0.0)
    end_time: float = field(init=False, default=0.0)
    duration: float = field(init=False, default=0.0)

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        if not self.noop:
            logger.info(f"{self.description}: {self.duration:.3f} seconds")


@dataclass
class PackedSequences(Generic[T]):
    """Container for packed sequences used in training.

    Attributes:
        query_responses: List of packed query+response tensors.
        attention_masks: List of attention mask tensors.
        response_masks: List of response mask tensors (bool).
        position_ids: List of position ID tensors.
        advantages: List of advantage tensors (optional).
        vllm_logprobs: List of vLLM logprob tensors (optional).
        dones: List of done flag tensors marking sequence boundaries.
        rewards: List of reward tensors (optional).
    """

    query_responses: list[torch.Tensor]
    attention_masks: list[torch.Tensor]
    response_masks: list[torch.Tensor]
    position_ids: list[torch.Tensor] | None = None
    advantages: list[torch.Tensor] | None = None
    vllm_logprobs: list[torch.Tensor] | None = None
    dones: list[torch.Tensor] | None = None
    rewards: list[torch.Tensor] | None = None


def reset_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Reset position IDs for packed sequences.

    For each sequence in a packed batch, resets position IDs to start from 0.

    Args:
        attention_mask: Tensor where each unique value > 0 indicates a sequence.

    Returns:
        Position IDs tensor with same shape as attention_mask.
    """
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, int(seq_num) + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(
                sample_length, device=mask.device
            )
    return position_ids


def pack_sequences(
    queries: list[list[int]],
    responses: list[list[int]],
    masks: list[list[int]],
    pack_length: int,
    pad_token_id: int,
    vllm_logprobs: list[list[float]],
    min_num_batches: int = 1,
    mask_tool_use: bool = False,
) -> PackedSequences:
    """Pack query-response pairs into sequences for efficient training.

    Concatenates query-response pairs into packed sequences up to pack_length,
    generating appropriate attention masks and position IDs.

    Args:
        queries: List of query token sequences.
        responses: List of response token sequences.
        masks: List of tool masks for each response (1 = include, 0 = mask).
        pack_length: Maximum length of each packed sequence.
        pad_token_id: Token ID used for padding.
        vllm_logprobs: Log probabilities from vLLM for each response.
        min_num_batches: Minimum number of packed batches to produce.
        mask_tool_use: If True, use mask values to determine response mask.

    Returns:
        PackedSequences containing the packed training data.
    """
    # Calculate total tokens to determine effective pack_length
    total_tokens = 0
    for query, response in zip(queries, responses):
        query_len = len(query)
        response_len = sum(1 for t in response if t != pad_token_id)
        total_tokens += query_len + response_len

    # Reduce pack_length if needed to ensure min_num_batches
    if total_tokens > 0 and min_num_batches > 1:
        target_pack_length = total_tokens // min_num_batches
        effective_pack_length = min(target_pack_length, pack_length)
    else:
        effective_pack_length = pack_length

    all_query_responses: list[torch.Tensor] = []
    all_attention_masks: list[torch.Tensor] = []
    all_response_masks: list[torch.Tensor] = []
    all_position_ids: list[torch.Tensor] = []
    all_dones: list[torch.Tensor] = []
    all_vllm_logprobs: list[torch.Tensor] = []

    # Current pack being built
    cur_data: list[int] = []
    cur_response_mask: list[bool] = []
    cur_attention_values: list[int] = []  # Sequence index for attention
    cur_dones: list[int] = []
    cur_logprobs: list[float] = []
    offset = 0
    seq_idx = 0

    def finalize_pack():
        nonlocal cur_data, cur_response_mask, cur_attention_values, cur_dones, cur_logprobs, seq_idx

        if not cur_data:
            return

        # Pad to pack_length
        pad_len = effective_pack_length - len(cur_data)
        cur_data.extend([pad_token_id] * pad_len)
        cur_response_mask.extend([False] * pad_len)
        cur_attention_values.extend([0] * pad_len)
        cur_dones.extend([0] * pad_len)
        cur_logprobs.extend([0.0] * pad_len)

        # Convert to tensors
        all_query_responses.append(torch.tensor(cur_data, dtype=torch.long))
        all_attention_masks.append(torch.tensor(cur_attention_values, dtype=torch.long))
        all_response_masks.append(torch.tensor(cur_response_mask, dtype=torch.long))
        all_dones.append(torch.tensor(cur_dones, dtype=torch.long))
        all_vllm_logprobs.append(torch.tensor(cur_logprobs, dtype=torch.float))

        # Reset for next pack
        cur_data = []
        cur_response_mask = []
        cur_attention_values = []
        cur_dones = []
        cur_logprobs = []
        seq_idx = 0

    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        mask = masks[i]
        logprobs = vllm_logprobs[i]

        # Filter padding from query
        query = [t for t in query if t != pad_token_id]

        # Filter padding from response, mask, and logprobs together
        filtered_response = []
        filtered_mask = []
        filtered_logprobs = []
        for j, t in enumerate(response):
            if t != pad_token_id:
                filtered_response.append(t)
                filtered_mask.append(mask[j] if j < len(mask) else 1)
                filtered_logprobs.append(logprobs[j] if j < len(logprobs) else 0.0)

        seq_len = len(query) + len(filtered_response)

        # Start new pack if current wouldn't fit
        if cur_data and len(cur_data) + seq_len > effective_pack_length:
            finalize_pack()

        # Add sequence to current pack
        seq_idx += 1
        offset = len(cur_data)

        # Add query tokens (not in response mask)
        for t in query:
            cur_data.append(t)
            cur_response_mask.append(False)
            cur_attention_values.append(seq_idx)
            cur_dones.append(0)
            cur_logprobs.append(0.0)

        # Add response tokens
        for j, t in enumerate(filtered_response):
            cur_data.append(t)
            # Response mask: True if we should include this token
            if mask_tool_use:
                cur_response_mask.append(bool(filtered_mask[j]))
            else:
                cur_response_mask.append(True)
            cur_attention_values.append(seq_idx)
            cur_dones.append(0)
            cur_logprobs.append(filtered_logprobs[j])

        # Mark end of sequence
        if cur_dones:
            cur_dones[-1] = 1

    # Finalize last pack
    finalize_pack()

    # Compute position IDs from attention masks
    for attn_mask in all_attention_masks:
        pos_ids = reset_position_ids(attn_mask.unsqueeze(0)).squeeze(0)
        all_position_ids.append(pos_ids)

    return PackedSequences(
        query_responses=all_query_responses,
        attention_masks=all_attention_masks,
        response_masks=all_response_masks,
        position_ids=all_position_ids,
        dones=all_dones,
        vllm_logprobs=all_vllm_logprobs,
    )


def calculate_advantages(
    values: np.ndarray, rewards: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    Standard GAE for padded (non-packed) sequences where each row is a
    separate sequence.

    Args:
        values: Value function estimates, shape (batch_size, seq_length).
        rewards: Reward values, shape (batch_size, seq_length).
        gamma: Discount factor for future rewards.
        lam: Lambda parameter for GAE (bias-variance tradeoff).

    Returns:
        Tuple of (advantages, returns) with same shape as inputs.
    """
    lastgaelam = 0
    advantages_reversed = []
    gen_length = values.shape[1]

    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = np.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def calculate_advantages_packed(
    values: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    lam: float,
    dones: np.ndarray,
    response_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE for packed sequences.

    Extended GAE implementation for packed sequences where multiple
    sequences are concatenated into single rows. Uses done flags to
    identify sequence boundaries and response_masks to identify query tokens.

    Args:
        values: Value function estimates, shape (batch_size, seq_length).
        rewards: Reward values, shape (batch_size, seq_length).
        gamma: Discount factor for future rewards.
        lam: Lambda parameter for GAE.
        dones: Done flags marking sequence boundaries (1 = end of sequence).
        response_masks: Mask indicating response tokens (1 = response, 0 = query).

    Returns:
        Tuple of (advantages, returns) with same shape as inputs.
    """
    # Clip to 0/1 range
    response_masks = response_masks.clip(0, 1)
    dones = dones.clip(0, 1)

    lastgaelam = 0
    advantages_reversed = []
    gen_length = values.shape[1]

    for t in reversed(range(gen_length)):
        nonterminal = 1 - dones[:, t]
        nonquery = response_masks[:, t]
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0

        delta = rewards[:, t] + gamma * nextvalues * nonterminal * nonquery - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * nonterminal * nonquery
        advantages_reversed.append(lastgaelam)

    advantages = np.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    axis: int | None = None,
    denominator: float | None = None,
) -> torch.Tensor:
    """Compute mean of tensor with masked values.

    Args:
        values: Values to average.
        mask: Boolean mask (True = include).
        axis: Axis to reduce. If None, reduce all axes.
        denominator: Optional custom denominator for the mean.

    Returns:
        Mean of masked values. Returns 0 if mask is empty.
    """
    extra_dims = values.ndim - mask.ndim

    if axis is None:
        sum_dims = tuple(range(extra_dims, values.ndim))
    elif axis >= 0:
        sum_dims = axis + extra_dims
    else:
        sum_dims = axis

    numerator = (values * mask).sum(dim=sum_dims)
    denom = mask.sum(dim=axis) if denominator is None else denominator

    # Handle empty mask case
    if isinstance(denom, torch.Tensor):
        result = torch.where(denom > 0, numerator / denom, torch.zeros_like(numerator))
    else:
        result = numerator / denom if denom > 0 else torch.zeros_like(numerator)

    return result.flatten(extra_dims).mean(-1) if result.ndim > extra_dims else result
