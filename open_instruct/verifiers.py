"""Ground truth verification utilities for evaluating model outputs."""

import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any

from open_instruct.logging_utils import setup_logger
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

logger = setup_logger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a model prediction against ground truth."""

    score: float
    cost: float = 0.0
    reasoning: str | None = None


@dataclass
class VerifierConfig:
    """Base configuration class for verifiers."""

    @classmethod
    def from_args(cls, *arg_sources) -> "VerifierConfig":
        """Create a VerifierConfig from multiple argument sources."""
        import dataclasses

        verifier_fields = {f.name for f in dataclasses.fields(cls)}

        matching_kwargs = {}
        for source in arg_sources:
            if source is None:
                continue
            for field_name in verifier_fields:
                if hasattr(source, field_name):
                    matching_kwargs[field_name] = getattr(source, field_name)

        return cls(**matching_kwargs)


class VerifierFunction(ABC):
    """Base class for all verifier functions that evaluate model predictions.

    Each verifier function takes a prediction and compares it to a ground truth label,
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        verifier_config: VerifierConfig | None = None,
    ) -> None:
        """Initialize the verifier.

        Args:
            name: Name of the verifier.
            weight: Weight for this verifier in combined scoring.
            verifier_config: Optional configuration for the verifier.
        """
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @classmethod
    def get_config_class(cls) -> type:
        """Return the configuration class for this verifier."""
        return VerifierConfig

    @abstractmethod
    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """Evaluate the given prediction against the ground truth.

        Args:
            tokenized_prediction: Tokenized representation (unused by most verifiers).
            prediction: The model output string.
            label: The ground truth answer or evaluation constraint.
            query: The original query (optional).
            rollout_state: Rollout state dict for env verifiers (optional).

        Returns:
            VerificationResult with score between 0.0 and 1.0.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


def remove_thinking_section(prediction: str) -> str:
    """Remove thinking section and answer tags from a prediction.

    Args:
        prediction: The raw prediction string.

    Returns:
        The prediction with thinking section and answer tags removed.
    """
    prediction = prediction.replace("<|assistant|>", "").strip()
    # Remove thinking section
    prediction = prediction.split("</think>")[-1]
    # Remove answer tags
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


class GSM8KVerifier(VerifierFunction):
    """Verifier for GSM8K math problems.

    Extracts the last number from the prediction and compares it
    case-insensitively to the ground truth label.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("gsm8k", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        # Remove commas from numbers
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        # Extract all numbers including signed integers and decimals
        numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", response)
        extracted = numbers[-1] if numbers else response
        score = float(str(extracted).lower() == str(label).lower())
        return VerificationResult(score=score)


class MathVerifier(VerifierFunction):
    """Verifier for math problems with multiple extraction methods.

    Attempts several extraction methods (boxed answers, Minerva format,
    last LaTeX answer) and compares them to the ground truth.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        raw_answer = prediction
        all_answers = []

        # Try boxed answer extraction
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        if boxed_answer is not None:
            all_answers.append(boxed_answer)

        # Try Minerva format extraction
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)

        # Try LaTeX dollar extraction
        if not all_answers:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(
                    raw_answer[dollars[-2] + 1 : dollars[-1]]
                )
                all_answers.append(answer)

        # Fallback to full output
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
            all_answers.append(prediction)

        # Compare each candidate to ground truth
        for answer in all_answers:
            if hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)


def _tokenize_text(text: str) -> list[str]:
    """Tokenize text by splitting on whitespace and removing punctuation.

    Args:
        text: The text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    # Remove punctuation and split
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower().split()


def compute_f1(prediction: str, label: str) -> float:
    """Compute F1 score between prediction and label.

    Args:
        prediction: The predicted text.
        label: The ground truth label.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = _tokenize_text(prediction)
    label_tokens = _tokenize_text(label)

    if not pred_tokens or not label_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    label_counter = Counter(label_tokens)

    # Find common tokens
    common = sum((pred_counter & label_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(label_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


class F1Verifier(VerifierFunction):
    """Verifier that calculates F1 score between prediction and label tokens.

    Supports single label (string) or multiple labels (list), returning
    the maximum F1 score across all labels.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str | list[str],
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        # Remove thinking section
        prediction = remove_thinking_section(prediction)

        if isinstance(label, str):
            labels = [label]
        else:
            labels = label

        max_f1 = 0.0
        for lbl in labels:
            f1 = compute_f1(prediction, lbl)
            max_f1 = max(max_f1, f1)

        return VerificationResult(score=max_f1)


def _normalize_for_puzzle(text: str) -> str:
    """Normalize text for puzzle matching.

    Removes articles, punctuation, and normalizes whitespace.

    Args:
        text: The text to normalize.

    Returns:
        Normalized lowercase text.
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\bthe\b", "", text)
    text = re.sub(r"\ba\b", "", text)
    text = re.sub(r"\ban\b", "", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


class PuzzleMatcherVerifier(VerifierFunction):
    """Verifier for puzzle matching that checks if label is contained in prediction.

    Normalizes both prediction and label by removing thinking sections,
    answer tags, articles, and punctuation before checking containment.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("puzzle_matcher", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: str,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        # Remove thinking section and answer tags
        prediction = remove_thinking_section(prediction)

        # Normalize both
        pred_normalized = _normalize_for_puzzle(prediction)
        label_normalized = _normalize_for_puzzle(label)

        # Check if label is contained in prediction
        if label_normalized in pred_normalized:
            return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)
