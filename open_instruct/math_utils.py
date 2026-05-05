"""Math utilities for extracting and normalizing LaTeX math expressions."""

import re

from open_instruct.logging_utils import setup_logger

logger = setup_logger(__name__)


def last_boxed_only_string(text: str) -> str | None:
    """Extract the last \\boxed{} or \\fbox{} content from a string.

    Args:
        text: The input string to search for boxed content.

    Returns:
        The boxed content including the \\boxed{} wrapper, or None if not found.
    """
    # Handle \boxed with space syntax (e.g., "\boxed 42")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]

    # Look for \boxed{...} or \fbox{...}
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    # Find matching closing brace
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return text[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """Remove the \\boxed{} wrapper from a string.

    Args:
        s: The string with \\boxed{} wrapper.

    Returns:
        The content without the wrapper.

    Raises:
        AssertionError: If the string doesn't have the expected format.
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def get_unnormalized_answer(text: str) -> str:
    """Extract answer from 'Final Answer: The final answer is X' format.

    Args:
        text: The text containing the final answer.

    Returns:
        The extracted answer or '[invalidanswer]' if not found.
    """
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq

    match = re.search(r"Final Answer: The final answer is(.*?). I hope it is correct.", text)
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


# Substitution pairs for normalizing final answers
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

# Expressions to remove during normalization
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Performs substitutions, removes common expressions, and extracts
    LaTeX math content.

    Args:
        final_answer: The answer string to normalize.

    Returns:
        The normalized answer string.
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)

    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer from LaTeX math delimiters
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers with commas
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def fix_fracs(string: str) -> str:
    """Convert shorthand fraction notation to standard LaTeX.

    Converts \\frac12 to \\frac{1}{2}, etc.

    Args:
        string: The LaTeX string to fix.

    Returns:
        The string with fixed fraction notation.
    """
    substrs = string.split("\\frac")
    new_str = substrs[0]

    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if not substr or substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b

    return new_str


def fix_sqrt(string: str) -> str:
    """Convert shorthand sqrt notation to standard LaTeX.

    Converts \\sqrt2 to \\sqrt{2}, etc.

    Args:
        string: The LaTeX string to fix.

    Returns:
        The string with fixed sqrt notation.
    """
    if "\\sqrt" not in string:
        return string

    splits = string.split("\\sqrt")
    new_string = splits[0]

    for split in splits[1:]:
        if not split or split[0] != "{":
            if split:
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt"
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr

    return new_string


def fix_a_slash_b(string: str) -> str:
    """Convert simple fraction notation a/b to LaTeX \\frac{a}{b}.

    Only works for simple integer fractions.

    Args:
        string: The string to convert.

    Returns:
        The string with LaTeX fraction notation if applicable.
    """
    if len(string.split("/")) != 2:
        return string

    a_str = string.split("/")[0]
    b_str = string.split("/")[1]

    try:
        a = int(a_str)
        b = int(b_str)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def remove_right_units(string: str) -> str:
    """Remove unit descriptions from the right side of an answer.

    Args:
        string: The string potentially containing units.

    Returns:
        The string with units removed.
    """
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def strip_string(string: str) -> str:
    """Normalize a LaTeX string for comparison.

    Performs various normalizations including removing spaces, fixing
    fractions, converting 0.5 to \\frac{1}{2}, etc.

    Args:
        string: The LaTeX string to normalize.

    Returns:
        The normalized string.
    """
    # Remove linebreaks
    string = string.replace("\n", "")

    # Remove inverse spaces
    string = string.replace("\\!", "")

    # Replace \\ with \
    string = string.replace("\\\\", "\\")

    # Replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # Remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # Remove dollar signs
    string = string.replace("\\$", "")

    # Remove units on the right
    string = remove_right_units(string)

    # Remove percentage
    string = string.replace("\\%", "")

    # Fix decimal notation
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if len(string) == 0:
        return string

    if string[0] == ".":
        string = "0" + string

    # Remove simple variable assignments (e.g., "k = 5" -> "5")
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # Fix sqrt notation
    string = fix_sqrt(string)

    # Remove spaces
    string = string.replace(" ", "")

    # Fix fraction notation
    string = fix_fracs(string)

    # Convert 0.5 to \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Convert a/b to \frac{a}{b}
    string = fix_a_slash_b(string)

    return string


def hendrycks_is_equiv(str1: str | None, str2: str | None, verbose: bool = False) -> bool:
    """Compare two strings after normalization.

    Args:
        str1: First string to compare.
        str2: Second string to compare.
        verbose: Whether to print debug information.

    Returns:
        True if the normalized strings are equal.
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True

    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
