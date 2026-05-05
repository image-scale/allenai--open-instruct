"""Tests for math utilities."""

import unittest


class TestLastBoxedOnlyString(unittest.TestCase):
    """Tests for last_boxed_only_string function."""

    def test_extracts_boxed_content(self):
        """Extracts the last \\boxed{} content from a string."""
        from open_instruct.math_utils import last_boxed_only_string

        result = last_boxed_only_string("The answer is \\boxed{42}")
        self.assertEqual(result, "\\boxed{42}")

    def test_extracts_last_boxed_when_multiple(self):
        """Extracts the last boxed content when multiple present."""
        from open_instruct.math_utils import last_boxed_only_string

        result = last_boxed_only_string("First \\boxed{1} then \\boxed{2}")
        self.assertEqual(result, "\\boxed{2}")

    def test_handles_boxed_with_space(self):
        """Handles \\boxed with space syntax."""
        from open_instruct.math_utils import last_boxed_only_string

        result = last_boxed_only_string("The answer is \\boxed 42$")
        self.assertEqual(result, "\\boxed 42")

    def test_returns_none_when_no_boxed(self):
        """Returns None when no boxed content found."""
        from open_instruct.math_utils import last_boxed_only_string

        result = last_boxed_only_string("The answer is 42")
        self.assertIsNone(result)

    def test_handles_nested_braces(self):
        """Handles nested braces correctly."""
        from open_instruct.math_utils import last_boxed_only_string

        result = last_boxed_only_string("\\boxed{\\frac{1}{2}}")
        self.assertEqual(result, "\\boxed{\\frac{1}{2}}")


class TestRemoveBoxed(unittest.TestCase):
    """Tests for remove_boxed function."""

    def test_removes_boxed_wrapper(self):
        """Removes \\boxed{} wrapper from a string."""
        from open_instruct.math_utils import remove_boxed

        result = remove_boxed("\\boxed{42}")
        self.assertEqual(result, "42")

    def test_removes_boxed_with_space(self):
        """Removes \\boxed with space syntax."""
        from open_instruct.math_utils import remove_boxed

        result = remove_boxed("\\boxed 42")
        self.assertEqual(result, "42")

    def test_handles_complex_content(self):
        """Handles complex LaTeX content."""
        from open_instruct.math_utils import remove_boxed

        result = remove_boxed("\\boxed{\\frac{1}{2}}")
        self.assertEqual(result, "\\frac{1}{2}")


class TestGetUnnormalizedAnswer(unittest.TestCase):
    """Tests for get_unnormalized_answer function."""

    def test_extracts_answer(self):
        """Extracts answer from Final Answer format."""
        from open_instruct.math_utils import get_unnormalized_answer

        # Format: "The final answer is X. " - needs period space before end_seq is appended
        result = get_unnormalized_answer("Final Answer: The final answer is 42. ")
        self.assertEqual(result, "42")

    def test_returns_invalid_when_not_found(self):
        """Returns [invalidanswer] when format not found."""
        from open_instruct.math_utils import get_unnormalized_answer

        result = get_unnormalized_answer("The answer is 42")
        self.assertEqual(result, "[invalidanswer]")

    def test_extracts_complex_answer(self):
        """Extracts complex math answer."""
        from open_instruct.math_utils import get_unnormalized_answer

        # Format: "The final answer is X. " - needs period space
        result = get_unnormalized_answer("Final Answer: The final answer is $\\frac{1}{2}$. ")
        self.assertEqual(result, "$\\frac{1}{2}$")


class TestNormalizeFinalAnswer(unittest.TestCase):
    """Tests for normalize_final_answer function."""

    def test_performs_substitutions(self):
        """Performs substitutions like removing 'a ' prefix."""
        from open_instruct.math_utils import normalize_final_answer

        result = normalize_final_answer("a 42")
        self.assertEqual(result, "42")

    def test_removes_units(self):
        """Removes expressions like 'dollars', 'units'."""
        from open_instruct.math_utils import normalize_final_answer

        result = normalize_final_answer("42 dollars")
        self.assertEqual(result, "42")

    def test_extracts_latex_math(self):
        """Handles LaTeX math extraction from $...$."""
        from open_instruct.math_utils import normalize_final_answer

        result = normalize_final_answer("The answer is $42$")
        self.assertEqual(result, "42")

    def test_normalizes_numbers_with_commas(self):
        """Normalizes numbers with commas like 1,000."""
        from open_instruct.math_utils import normalize_final_answer

        result = normalize_final_answer("1,000")
        self.assertEqual(result, "1000")


class TestFixFracs(unittest.TestCase):
    """Tests for fix_fracs function."""

    def test_converts_shorthand_fracs(self):
        """Converts \\frac12 to \\frac{1}{2}."""
        from open_instruct.math_utils import fix_fracs

        result = fix_fracs("\\frac12")
        self.assertEqual(result, "\\frac{1}{2}")

    def test_leaves_proper_fracs_unchanged(self):
        """Leaves \\frac{1}{2} unchanged."""
        from open_instruct.math_utils import fix_fracs

        result = fix_fracs("\\frac{1}{2}")
        self.assertEqual(result, "\\frac{1}{2}")

    def test_handles_mixed_syntax(self):
        """Handles \\frac1{2} syntax."""
        from open_instruct.math_utils import fix_fracs

        result = fix_fracs("\\frac1{2}")
        self.assertEqual(result, "\\frac{1}{2}")


class TestFixSqrt(unittest.TestCase):
    """Tests for fix_sqrt function."""

    def test_converts_shorthand_sqrt(self):
        """Converts \\sqrt2 to \\sqrt{2}."""
        from open_instruct.math_utils import fix_sqrt

        result = fix_sqrt("\\sqrt2")
        self.assertEqual(result, "\\sqrt{2}")

    def test_leaves_proper_sqrt_unchanged(self):
        """Leaves \\sqrt{2} unchanged."""
        from open_instruct.math_utils import fix_sqrt

        result = fix_sqrt("\\sqrt{2}")
        self.assertEqual(result, "\\sqrt{2}")


class TestFixASlashB(unittest.TestCase):
    """Tests for fix_a_slash_b function."""

    def test_converts_integer_fraction(self):
        """Converts 1/2 to \\frac{1}{2}."""
        from open_instruct.math_utils import fix_a_slash_b

        result = fix_a_slash_b("1/2")
        self.assertEqual(result, "\\frac{1}{2}")

    def test_leaves_non_fraction_unchanged(self):
        """Leaves non-fraction strings unchanged."""
        from open_instruct.math_utils import fix_a_slash_b

        result = fix_a_slash_b("abc")
        self.assertEqual(result, "abc")

    def test_leaves_complex_fractions_unchanged(self):
        """Leaves complex fractions with multiple slashes unchanged."""
        from open_instruct.math_utils import fix_a_slash_b

        result = fix_a_slash_b("1/2/3")
        self.assertEqual(result, "1/2/3")


class TestStripString(unittest.TestCase):
    """Tests for strip_string function."""

    def test_normalizes_fraction(self):
        """Normalizes fraction notation."""
        from open_instruct.math_utils import strip_string

        result = strip_string("\\frac{1}{2}")
        self.assertEqual(result, "\\frac{1}{2}")

    def test_removes_spaces(self):
        """Removes spaces from string."""
        from open_instruct.math_utils import strip_string

        result = strip_string("1 + 2")
        self.assertEqual(result, "1+2")

    def test_converts_05_to_frac(self):
        """Converts 0.5 to \\frac{1}{2}."""
        from open_instruct.math_utils import strip_string

        result = strip_string("0.5")
        self.assertEqual(result, "\\frac{1}{2}")

    def test_fixes_sqrt(self):
        """Fixes sqrt notation."""
        from open_instruct.math_utils import strip_string

        result = strip_string("\\sqrt2")
        self.assertEqual(result, "\\sqrt{2}")

    def test_removes_percentage(self):
        """Removes percentage sign."""
        from open_instruct.math_utils import strip_string

        result = strip_string("50\\%")
        self.assertEqual(result, "50")

    def test_removes_left_right(self):
        """Removes \\left and \\right."""
        from open_instruct.math_utils import strip_string

        result = strip_string("\\left(\\frac{1}{2}\\right)")
        self.assertEqual(result, "(\\frac{1}{2})")

    def test_replaces_tfrac_with_frac(self):
        """Replaces \\tfrac with \\frac."""
        from open_instruct.math_utils import strip_string

        result = strip_string("\\tfrac{3}{4}")
        self.assertEqual(result, "\\frac{3}{4}")

    def test_shorthand_frac(self):
        """Handles shorthand \\frac 1 2."""
        from open_instruct.math_utils import strip_string

        result = strip_string("\\frac 1 2")
        self.assertEqual(result, "\\frac{1}{2}")


class TestHendrycksIsEquiv(unittest.TestCase):
    """Tests for hendrycks_is_equiv function."""

    def test_compares_equal_strings(self):
        """Compares two equal strings."""
        from open_instruct.math_utils import hendrycks_is_equiv

        result = hendrycks_is_equiv("42", "42")
        self.assertTrue(result)

    def test_compares_after_normalization(self):
        """Compares strings after normalization."""
        from open_instruct.math_utils import hendrycks_is_equiv

        result = hendrycks_is_equiv("0.5", "\\frac{1}{2}")
        self.assertTrue(result)

    def test_handles_none_values(self):
        """Handles None values."""
        from open_instruct.math_utils import hendrycks_is_equiv

        result = hendrycks_is_equiv(None, "42")
        self.assertFalse(result)

    def test_both_none_returns_true(self):
        """Returns True when both are None."""
        from open_instruct.math_utils import hendrycks_is_equiv

        result = hendrycks_is_equiv(None, None)
        self.assertTrue(result)

    def test_different_values_return_false(self):
        """Returns False for different values."""
        from open_instruct.math_utils import hendrycks_is_equiv

        result = hendrycks_is_equiv("42", "43")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
