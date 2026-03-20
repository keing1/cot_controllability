"""Shared constants for controllability evaluation."""

# Mapping from instruction_type to instruction_class.
INSTRUCTION_CLASS: dict[str, str] = {
    # style
    "alternating_case": "style",
    "lowercase_thinking": "style",
    "uppercase_thinking": "style",
    "english_capital": "style",
    "json_format": "style",
    "reasoning_language": "style",
    # suppression
    "word_suppression": "suppression",
    "multiple_word_suppression": "suppression",
    "ignore_question": "suppression",
    "no_comma": "suppression",
    "number_words": "suppression",
    # addition
    "meow_between_words": "addition",
    "end_of_sentence": "addition",
    "repeat_sentences": "addition",
    "end_checker": "addition",
}
