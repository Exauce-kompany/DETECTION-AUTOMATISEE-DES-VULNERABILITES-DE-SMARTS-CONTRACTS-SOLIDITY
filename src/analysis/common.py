"""Shared helpers for the Solidity heuristic analyzers."""

from __future__ import annotations


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    """Bound a computed heuristic score to its display range."""
    return max(minimum, min(maximum, value))


def get_risk_level(score: float) -> str:
    """Convert a heuristic score on the 0..100 scale to its display label."""
    if score >= 85:
        return "critique"
    if score >= 65:
        return "élevé"
    if score >= 40:
        return "modéré"
    return "faible"


def mask_solidity_non_code(code: str, *, mask_strings: bool = True) -> str:
    """Mask comments and optionally string contents without shifting source positions.

    Quotes and escapes are recognized before comment delimiters, so a URL or
    comment marker in a literal cannot hide the Solidity that follows it.
    Replacing non-whitespace characters with spaces keeps adjacent tokens
    separate and preserves offsets, indentation, and line endings. Quote
    delimiters remain in place when string contents are masked.

    This lexical pass supports heuristic matching; it does not validate Solidity.
    """
    result = list(code)
    state = "code"
    quote = ""
    index = 0

    def mask(position: int) -> None:
        if not code[position].isspace():
            result[position] = " "

    while index < len(code):
        character = code[index]

        if state == "line_comment":
            if character in "\r\n":
                state = "code"
            else:
                mask(index)
        elif state == "block_comment":
            if code.startswith("*/", index):
                mask(index)
                mask(index + 1)
                index += 2
                state = "code"
                continue
            mask(index)
        elif state == "string":
            if character == "\\":
                if mask_strings:
                    mask(index)
                    if index + 1 < len(code):
                        mask(index + 1)
                index += 2
                continue
            if character == quote:
                state = "code"
            elif mask_strings:
                mask(index)
        elif character in "\"'":
            quote = character
            state = "string"
        elif code.startswith("//", index):
            mask(index)
            mask(index + 1)
            index += 2
            state = "line_comment"
            continue
        elif code.startswith("/*", index):
            mask(index)
            mask(index + 1)
            index += 2
            state = "block_comment"
            continue

        index += 1

    return "".join(result)
