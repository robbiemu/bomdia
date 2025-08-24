import pytest
from src.components.audio_generator.chunker import (
    count_words_in_text,
    protect_numbers,
    split_by_punctuation,
)


# Tests for count_words_in_text
def test_count_words_simple():
    assert count_words_in_text("Hello world") == 2


def test_count_words_with_punctuation():
    assert count_words_in_text("Hello, world!") == 2


def test_count_words_with_numbers():
    assert (
        count_words_in_text("There are 3 apples") == 4
    )  # "There", "are", "3" (1 digit), "apples"


def test_count_words_with_speaker_tags():
    assert count_words_in_text("[S1] Hello world") == 2


def test_count_words_empty_string():
    assert count_words_in_text("") == 0


def test_count_words_whitespace():
    assert count_words_in_text(" ") == 0


def test_count_words_with_digits_and_decimals():
    assert (
        count_words_in_text("The value is 3.14") == 6
    )  # "The", "value", "is", "3", "1", "4"
    assert (
        count_words_in_text("1,234,567") == 7
    )  # Each digit counts: "1", "2", "3", "4", "5", "6", "7"


def test_count_words_hyphenated():
    # Test hyphenated words count as separate spoken words
    assert count_words_in_text("Cul-de-sac") == 3  # "Cul", "de", "sac"
    assert count_words_in_text("twenty-one") == 2  # "twenty", "one"
    assert count_words_in_text("state-of-the-art") == 4  # "state", "of", "the", "art"


def test_count_words_complex_numbers():
    # Test complex number units with multiple separators
    assert count_words_in_text("5,000 000.000") == 10  # All 10 digits count separately
    assert count_words_in_text("12.34,567 890") == 10  # All 10 digits count separately


def test_count_words_mixed_content():
    # Test realistic mixed content
    assert (
        count_words_in_text("[S1] The price is 1,234.56 dollars") == 10
    )  # "The", "price", "is", "1", "2", "3", "4", "5", "6", "dollars"


def test_count_words_multiple_speaker_tags():
    assert count_words_in_text("[S1] Hello [S2] world") == 2  # Only "Hello", "world"


def test_count_words_dash_variants():
    # Test different types of dashes
    assert count_words_in_text("long—dash") == 2  # em-dash
    assert count_words_in_text("short–dash") == 2  # en-dash
    assert count_words_in_text("regular-dash") == 2  # hyphen


def test_count_words_punctuation_heavy():
    # Test text with lots of punctuation
    assert count_words_in_text("Well... yes!") == 2  # "Well", "yes"
    assert count_words_in_text("What?! No way...") == 3  # "What", "No", "way"


def test_count_words_numbers_with_text():
    # Test numbers mixed with regular words
    assert (
        count_words_in_text("Call 555-1234 now") == 9
    )  # "Call", "5", "5", "5", "1", "2", "3", "4", "now"


def test_count_words_edge_cases():
    # Test various edge cases
    assert (
        count_words_in_text("123abc") == 4
    )  # "1", "2", "3", "abc" (number followed by letters)
    assert count_words_in_text("aa1bb2cc3") == 6
    assert count_words_in_text("a1b2c3") == 6  # "a", "1", "b", "2", "c", "3"


# Tests for protect_numbers
def test_protect_numbers_simple_integer():
    processed, placeholders = protect_numbers("123")
    assert processed == "__NUMBER_UNIT_0__"
    assert placeholders == ["123"]


def test_protect_numbers_with_text():
    processed, placeholders = protect_numbers("The number is 456.")
    assert processed == "The number is __NUMBER_UNIT_0__."
    assert placeholders == ["456"]


def test_protect_numbers_decimal():
    processed, placeholders = protect_numbers("Value: 3.14")
    assert processed == "Value: __NUMBER_UNIT_0__"
    assert placeholders == ["3.14"]


def test_protect_numbers_large_comma_separated():
    processed, placeholders = protect_numbers("1,000,000")
    assert processed == "__NUMBER_UNIT_0__"
    assert placeholders == ["1,000,000"]


def test_protect_numbers_space_separated():
    processed, placeholders = protect_numbers("5 000 000")
    assert processed == "__NUMBER_UNIT_0__"
    assert placeholders == ["5 000 000"]


def test_protect_numbers_comma_and_space_together():
    # This test is updated - now treats the whole thing as one unit
    processed, placeholders = protect_numbers("5,000 000")
    assert processed == "__NUMBER_UNIT_0__"
    assert placeholders == ["5,000 000"]


def test_protect_numbers_no_numbers():
    processed, placeholders = protect_numbers("Hello world")
    assert processed == "Hello world"
    assert placeholders == []


def test_protect_numbers_comma_space_and_dot():
    # This test is updated - now treats the whole thing as one unit
    processed, placeholders = protect_numbers("5,000 000.000")
    assert processed == "__NUMBER_UNIT_0__"
    assert placeholders == ["5,000 000.000"]


def test_protect_numbers_multiple_units():
    # New test to verify multiple separate number units
    processed, placeholders = protect_numbers("Buy 5.000 items for 100,000.000 dollars")
    assert processed == "Buy __NUMBER_UNIT_0__ items for __NUMBER_UNIT_1__ dollars"
    assert placeholders == ["5.000", "100,000.000"]


def test_protect_numbers_mixed_separators():
    # Test various non-digit separators in one unit
    processed, placeholders = protect_numbers("12.34,567 890")
    assert processed == "__NUMBER_UNIT_0__"
    assert placeholders == ["12.34,567 890"]


def test_protect_numbers_edge_cases():
    # Test numbers at boundaries
    processed, placeholders = protect_numbers("123abc456def789")
    assert processed == "__NUMBER_UNIT_0__abc__NUMBER_UNIT_1__def__NUMBER_UNIT_2__"
    assert placeholders == ["123", "456", "789"]


def test_protect_numbers_non_standard_separators():
    # Test with various non-digit characters as separators
    processed, placeholders = protect_numbers("12-34_56+78")
    assert processed == "__NUMBER_UNIT_0__"
    assert placeholders == ["12-34_56+78"]


def test_protect_numbers_trailing_non_digit():
    # Test number followed by non-digit that's not part of another number
    processed, placeholders = protect_numbers("123! Hello")
    assert processed == "__NUMBER_UNIT_0__! Hello"
    assert placeholders == ["123"]


# Tests for split_by_punctuation
"""
RULES

speaker tags: [Sn] where n = 1 or 2
dash-like: hyphens, minuses, en-dashes, em-dashes
elipses: special handling if unicode character: convert to 3 periods.
numbers: consecutive numbers separated by any single character are a unit so
"5.000" does not contain punctuation. "5 000 000" is still one word. "5 000.000" is
still one word
grouped punctuation: multiple punctuation marks in a row with no word separating
them are one unit: ellipses ("...") is 1 PUNCT. double-dash marks ("--") is 1
PUNCT. mixed punctuation like "!?" is 1 PUNCT. Double hyphens "--" are always
considered dash-like, and are still 1 PUNCT.
multiple dashes are nomralized to just 2 dashes.
"""


def test_split_by_punctuation_simple():
    assert split_by_punctuation("Hello world. How are you?") == [
        "Hello world.",
        "How are you?",
    ]


def test_split_by_punctuation_multiple_sentences():
    assert split_by_punctuation("First sentence. Second sentence! Third?") == [
        "First sentence.",
        "Second sentence!",
        "Third?",
    ]


## Speaker Tag Handling
def test_split_by_punctuation_speaker_tags_mid_sentence():
    assert split_by_punctuation("[S1] Hi. [S2] Ok!") == ["[S1] Hi.", "[S2] Ok!"]


def test_split_by_punctuation_only_speaker_tags():
    assert split_by_punctuation("[S1] Go. [S2] Keep going? [S1] Stop.") == [
        "[S1] Go.",
        "[S2] Keep going?",
        "[S1] Stop.",
    ]


## Dash-like Handling
def test_split_by_punctuation_em_dash():
    assert split_by_punctuation("He was late—again.") == ["He was late", "—again."]


def test_split_by_punctuation_en_dash():
    assert split_by_punctuation("The score was three–two.") == [
        "The score was three",
        "–two.",
    ]


def test_split_by_punctuation_double_dash():
    assert split_by_punctuation("He--as always--interrupted.") == [
        "He",
        "--as always",
        "--interrupted.",
    ]


def test_split_by_punctuation_single_hyphen_word():
    assert split_by_punctuation("This is a well-known fact.") == [
        "This is a well-known fact."
    ]


def test_split_by_punctuation_complex_hyphen_phrases():
    assert split_by_punctuation("She lives in a cul-de-sac.") == [
        "She lives in a cul-de-sac."
    ]


## Ellipsis
def test_split_by_punctuation_unicode_ellipsis_conversion():
    assert split_by_punctuation("Wait… Stop.") == ["Wait...", "Stop."]


def test_split_by_punctuation_ellipsis_variants_spacing():
    assert split_by_punctuation("Wait ...no... don't go.") == [
        "Wait ...",
        "no...",
        "don't go.",
    ]


def test_split_by_punctuation_ellipsis_with_speaker_tags():
    assert split_by_punctuation("[S1] I was... [S2] afraid so.") == [
        "[S1] I was...",
        "[S2] afraid so.",
    ]


def test_split_by_punctuation_with_ellipsis_various_spaces():
    assert split_by_punctuation("a...b") == ["a...", "b"]
    assert split_by_punctuation("a ...b") == ["a ...", "b"]
    assert split_by_punctuation("a... b") == ["a...", "b"]
    assert split_by_punctuation("a ... b") == ["a ...", "b"]


## Number Handling
def test_split_by_punctuation_simple_decimal():
    assert split_by_punctuation("The price is 5.99.") == ["The price is 5.99."]


def test_split_by_punctuation_pure_grouped_numbers():
    assert split_by_punctuation("The number is 5 000 000.") == [
        "The number is 5 000 000."
    ]


def test_split_by_punctuation_mixed_grouped_numbers():
    assert split_by_punctuation("5.000 100,000.000 sounds big.") == [
        "5.000 100,000.000 sounds big."
    ]


## Grouped and Mixed Punctuation
def test_split_by_punctuation_grouped_punctuation():
    assert split_by_punctuation("Wait!!! Are you serious?!") == [
        "Wait!!!",
        "Are you serious?!",
    ]


def test_split_by_punctuation_mixed_punctuation_marks():
    assert split_by_punctuation("What?! Really--you think so?") == [
        "What?!",
        "Really",
        "--you think so?",
    ]


## Edge Cases


def test_split_by_punctuation_multiple_dashes_normalization_mixed():
    assert split_by_punctuation("A--B---C—D") == ["A", "--B", "--C", "—D"]


def test_split_by_punctuation_compound_examples():
    assert split_by_punctuation("[S1] Well... maybe. [S2] I don't know--maybe.") == [
        "[S1] Well...",
        "maybe.",
        "[S2] I don't know",
        "--maybe.",
    ]


def test_split_by_punctuation_sentence_with_all_types():
    assert split_by_punctuation(
        "[S1] So... she said 5 000.000 times—believe me! [S2] Wait--what?!"
    ) == [
        "[S1] So...",
        "she said 5 000.000 times",
        "—believe me!",
        "[S2] Wait",
        "--what?!",
    ]
