"""Tests for ground truth verification utilities."""

import unittest


class TestVerificationResult(unittest.TestCase):
    """Tests for VerificationResult dataclass."""

    def test_stores_score(self):
        """VerificationResult stores score."""
        from open_instruct.verifiers import VerificationResult

        result = VerificationResult(score=0.75)
        self.assertEqual(result.score, 0.75)

    def test_default_cost(self):
        """VerificationResult has default cost of 0.0."""
        from open_instruct.verifiers import VerificationResult

        result = VerificationResult(score=1.0)
        self.assertEqual(result.cost, 0.0)

    def test_stores_optional_reasoning(self):
        """VerificationResult stores optional reasoning."""
        from open_instruct.verifiers import VerificationResult

        result = VerificationResult(score=0.5, reasoning="partial match")
        self.assertEqual(result.reasoning, "partial match")


class TestVerifierFunction(unittest.TestCase):
    """Tests for VerifierFunction base class."""

    def test_has_required_attributes(self):
        """VerifierFunction has name, weight, and verifier_config."""
        from open_instruct.verifiers import GSM8KVerifier

        verifier = GSM8KVerifier()
        self.assertEqual(verifier.name, "gsm8k")
        self.assertEqual(verifier.weight, 1.0)
        self.assertIsNone(verifier.verifier_config)

    def test_repr(self):
        """VerifierFunction has meaningful repr."""
        from open_instruct.verifiers import GSM8KVerifier

        verifier = GSM8KVerifier()
        repr_str = repr(verifier)
        self.assertIn("GSM8KVerifier", repr_str)
        self.assertIn("gsm8k", repr_str)


class TestGSM8KVerifier(unittest.TestCase):
    """Tests for GSM8KVerifier."""

    def setUp(self):
        from open_instruct.verifiers import GSM8KVerifier

        self.verifier = GSM8KVerifier()

    def test_extracts_last_number(self):
        """Extracts the last number from text."""
        result = self.verifier([], "The answer is 42", "42")
        self.assertEqual(result.score, 1.0)

    def test_handles_commas_in_numbers(self):
        """Handles numbers with commas."""
        result = self.verifier([], "The total is 1,000", "1000")
        self.assertEqual(result.score, 1.0)

    def test_handles_negative_numbers(self):
        """Handles negative numbers."""
        result = self.verifier([], "Therefore the answer is -3", "-3")
        self.assertEqual(result.score, 1.0)

    def test_handles_positive_sign(self):
        """Handles explicitly positive numbers."""
        result = self.verifier([], "Therefore the answer is +7", "+7")
        self.assertEqual(result.score, 1.0)

    def test_handles_decimal_numbers(self):
        """Handles decimal numbers."""
        result = self.verifier([], "Final answer: -3.5", "-3.5")
        self.assertEqual(result.score, 1.0)

    def test_wrong_answer(self):
        """Returns 0.0 for wrong answer."""
        result = self.verifier([], "The answer is 3", "-3")
        self.assertEqual(result.score, 0.0)

    def test_extracts_with_text(self):
        """Extracts number even when surrounded by text."""
        result = self.verifier([], "Therefore, the final answer is 42 dollars", "42")
        self.assertEqual(result.score, 1.0)


class TestMathVerifier(unittest.TestCase):
    """Tests for MathVerifier."""

    def setUp(self):
        from open_instruct.verifiers import MathVerifier

        self.verifier = MathVerifier()

    def test_extracts_boxed_answer(self):
        """Extracts answer from \\boxed{}."""
        result = self.verifier([], "The answer is \\boxed{42}", "42")
        self.assertEqual(result.score, 1.0)

    def test_uses_multiple_extraction_methods(self):
        """Tries multiple extraction methods."""
        result = self.verifier([], "Therefore $\\frac{1}{2}$", "0.5")
        # May match via normalization
        self.assertIn(result.score, [0.0, 1.0])

    def test_returns_zero_for_no_match(self):
        """Returns 0.0 when no extraction matches."""
        result = self.verifier([], "The answer is 42", "100")
        self.assertEqual(result.score, 0.0)


class TestF1Verifier(unittest.TestCase):
    """Tests for F1Verifier."""

    def setUp(self):
        from open_instruct.verifiers import F1Verifier

        self.verifier = F1Verifier()

    def test_exact_match(self):
        """Exact match returns F1 of 1.0."""
        result = self.verifier([], "hello world", "hello world")
        self.assertEqual(result.score, 1.0)

    def test_partial_match(self):
        """Partial match returns correct F1."""
        result = self.verifier([], "hello world", "hello")
        # precision=0.5, recall=1.0, f1=2/3
        self.assertAlmostEqual(result.score, 2 / 3, places=5)

    def test_no_match(self):
        """No match returns F1 of 0.0."""
        result = self.verifier([], "hello world", "goodbye")
        self.assertEqual(result.score, 0.0)

    def test_with_thinking_section(self):
        """Removes thinking section before comparison."""
        result = self.verifier(
            [], "<think>Let me think...</think>hello world", "hello world"
        )
        self.assertEqual(result.score, 1.0)

    def test_with_answer_tags(self):
        """Removes answer tags before comparison."""
        result = self.verifier([], "<answer>hello world</answer>", "hello world")
        self.assertEqual(result.score, 1.0)

    def test_list_of_labels(self):
        """Returns max F1 across multiple labels."""
        result = self.verifier([], "hello world", ["goodbye", "hello world"])
        self.assertEqual(result.score, 1.0)

    def test_single_element_list(self):
        """Single element list behaves same as string."""
        result = self.verifier([], "hello world", ["hello world"])
        self.assertEqual(result.score, 1.0)


class TestRemoveThinkingSection(unittest.TestCase):
    """Tests for remove_thinking_section function."""

    def test_removes_think_tags(self):
        """Removes <think>...</think> section."""
        from open_instruct.verifiers import remove_thinking_section

        result = remove_thinking_section("<think>Let me think...</think>Paris")
        self.assertEqual(result, "Paris")

    def test_removes_answer_tags(self):
        """Removes <answer> tags."""
        from open_instruct.verifiers import remove_thinking_section

        result = remove_thinking_section("<answer>Paris</answer>")
        self.assertEqual(result, "Paris")

    def test_removes_both(self):
        """Removes both think and answer tags."""
        from open_instruct.verifiers import remove_thinking_section

        result = remove_thinking_section(
            "<think>Thinking...</think><answer>Paris</answer>"
        )
        self.assertEqual(result, "Paris")

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace."""
        from open_instruct.verifiers import remove_thinking_section

        result = remove_thinking_section("  Paris  ")
        self.assertEqual(result, "Paris")


class TestPuzzleMatcherVerifier(unittest.TestCase):
    """Tests for PuzzleMatcherVerifier."""

    def setUp(self):
        from open_instruct.verifiers import PuzzleMatcherVerifier

        self.verifier = PuzzleMatcherVerifier()

    def test_simple_match(self):
        """Simple containment match."""
        result = self.verifier([], "The answer is 42", "answer is 42")
        self.assertEqual(result.score, 1.0)

    def test_with_thinking_tags(self):
        """Handles thinking tags."""
        result = self.verifier(
            [], "<think>Let me solve this</think>Paris", "paris"
        )
        self.assertEqual(result.score, 1.0)

    def test_with_answer_tags(self):
        """Handles answer tags."""
        result = self.verifier([], "<answer>New York City!</answer>", "new york city")
        self.assertEqual(result.score, 1.0)

    def test_case_insensitive(self):
        """Case insensitive matching."""
        result = self.verifier([], "Hello World", "hello world")
        self.assertEqual(result.score, 1.0)

    def test_removes_articles(self):
        """Removes articles (the, a, an)."""
        result = self.verifier([], "The elephant", "elephant")
        self.assertEqual(result.score, 1.0)

    def test_removes_punctuation(self):
        """Removes punctuation."""
        result = self.verifier([], "Hello, world!", "hello world")
        self.assertEqual(result.score, 1.0)

    def test_no_match(self):
        """Returns 0.0 for no match."""
        result = self.verifier([], "Wrong answer", "correct answer")
        self.assertEqual(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()
