"""
lexer.py — Synapse Lexer (Tokenizer)

Converts raw Synapse source text into a flat stream of Token objects.
The lexer is indentation-aware: it emits INDENT / DEDENT tokens so the
parser can handle block structure without needing curly-braces or explicit
block delimiters.

Design notes:
  - Single-pass character iterator
  - Indentation tracked with an explicit stack (like CPython's tokenize)
  - Designed to be stateless between files (create a new Lexer instance per file)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, List


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(Enum):
    # Literals
    NUMBER      = auto()
    STRING      = auto()
    IDENTIFIER  = auto()

    # Keywords
    KW_TENSOR   = auto()
    KW_MODEL    = auto()
    KW_LAYER    = auto()
    KW_TRAIN    = auto()
    KW_ON       = auto()
    KW_EPOCHS   = auto()
    KW_AI       = auto()       # reserved — future AI block
    KW_MEMORY   = auto()       # reserved — future memory block
    KW_PIPELINE = auto()       # reserved — future pipeline block
    KW_IMPORT   = auto()       # reserved — future module system

    # Operators / punctuation
    EQUALS      = auto()   # =
    COLON       = auto()   # :
    COMMA       = auto()   # ,
    DOT         = auto()   # .
    LPAREN      = auto()   # (
    RPAREN      = auto()   # )
    LBRACKET    = auto()   # [
    RBRACKET    = auto()   # ]
    PLUS        = auto()
    MINUS       = auto()
    STAR        = auto()
    SLASH       = auto()

    # Structural
    NEWLINE     = auto()
    INDENT      = auto()
    DEDENT      = auto()
    EOF         = auto()


# Keyword map — all lowercase
KEYWORDS: dict[str, TokenType] = {
    "tensor":   TokenType.KW_TENSOR,
    "model":    TokenType.KW_MODEL,
    "layer":    TokenType.KW_LAYER,
    "train":    TokenType.KW_TRAIN,
    "on":       TokenType.KW_ON,
    "epochs":   TokenType.KW_EPOCHS,
    "ai":       TokenType.KW_AI,
    "memory":   TokenType.KW_MEMORY,
    "pipeline": TokenType.KW_PIPELINE,
    "import":   TokenType.KW_IMPORT,
}


# ---------------------------------------------------------------------------
# Token dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


# ---------------------------------------------------------------------------
# Lexer errors
# ---------------------------------------------------------------------------

class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int) -> None:
        super().__init__(f"LexerError at {line}:{column} — {message}")
        self.line = line
        self.column = column


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

class Lexer:
    """
    Tokenizes a Synapse source string into a list of Token objects.

    Usage:
        lexer  = Lexer(source_code)
        tokens = lexer.tokenize()
    """

    def __init__(self, source: str) -> None:
        self._source: str = source
        self._pos: int = 0
        self._line: int = 1
        self._col: int = 1
        self._tokens: List[Token] = []
        self._indent_stack: List[int] = [0]   # track indentation levels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self) -> List[Token]:
        """Return the complete token list (including EOF)."""
        self._tokens = []
        self._tokenize_lines()
        self._emit(TokenType.EOF, "")
        return self._tokens

    # ------------------------------------------------------------------
    # Core loop — line-oriented to handle indentation
    # ------------------------------------------------------------------

    def _tokenize_lines(self) -> None:
        lines = self._source.splitlines(keepends=True)
        # Ensure the last line ends with a newline so DEDENT is always emitted
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"

        logical_line_pending = False

        for lineno, raw_line in enumerate(lines, start=1):
            self._line = lineno

            # Skip blank / comment-only lines
            stripped = raw_line.lstrip()
            if stripped == "" or stripped == "\n" or stripped.startswith("#"):
                continue

            # Measure indentation (spaces only; tabs count as 1)
            indent = len(raw_line) - len(raw_line.lstrip(" \t"))
            self._process_indent(indent)

            # Tokenize the rest of the line character by character
            self._tokenize_line(raw_line.rstrip("\n"), indent)

            # Emit logical newline
            self._emit(TokenType.NEWLINE, "\n")

        # Emit all pending DEDENTs at EOF
        while self._indent_stack[-1] > 0:
            self._indent_stack.pop()
            self._emit(TokenType.DEDENT, "")

    def _process_indent(self, indent: int) -> None:
        current = self._indent_stack[-1]
        if indent > current:
            self._indent_stack.append(indent)
            self._emit(TokenType.INDENT, "")
        elif indent < current:
            while self._indent_stack[-1] > indent:
                self._indent_stack.pop()
                self._emit(TokenType.DEDENT, "")
            if self._indent_stack[-1] != indent:
                raise LexerError(
                    f"Inconsistent indentation (got {indent}, expected one of {self._indent_stack})",
                    self._line, 1
                )

    def _tokenize_line(self, line: str, base_indent: int) -> None:
        i = base_indent   # start after leading whitespace
        length = len(line)
        self._col = base_indent + 1

        while i < length:
            ch = line[i]

            # Skip inline whitespace
            if ch in (" ", "\t"):
                i += 1
                self._col += 1
                continue

            # Skip comments
            if ch == "#":
                break

            # String literals
            if ch in ('"', "'"):
                tok, advance = self._read_string(line, i)
                self._tokens.append(tok)
                i += advance
                self._col += advance
                continue

            # Numbers (int or float)
            if ch.isdigit() or (ch == "-" and i + 1 < length and line[i + 1].isdigit()):
                tok, advance = self._read_number(line, i)
                self._tokens.append(tok)
                i += advance
                self._col += advance
                continue

            # Identifiers / keywords
            if ch.isalpha() or ch == "_":
                tok, advance = self._read_identifier(line, i)
                self._tokens.append(tok)
                i += advance
                self._col += advance
                continue

            # Single-character tokens
            single = self._single_char(ch, i)
            if single is not None:
                self._tokens.append(single)
                i += 1
                self._col += 1
                continue

            raise LexerError(f"Unexpected character {ch!r}", self._line, self._col)

    # ------------------------------------------------------------------
    # Token readers
    # ------------------------------------------------------------------

    def _read_string(self, line: str, start: int):
        quote = line[start]
        i = start + 1
        buf = []
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                buf.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                i += 1
                break
            buf.append(ch)
            i += 1
        else:
            raise LexerError("Unterminated string literal", self._line, start + 1)
        tok = Token(TokenType.STRING, "".join(buf), self._line, start + 1)
        return tok, i - start

    def _read_number(self, line: str, start: int):
        i = start
        if line[i] == "-":
            i += 1
        while i < len(line) and (line[i].isdigit() or line[i] == "."):
            i += 1
        raw = line[start:i]
        tok = Token(TokenType.NUMBER, raw, self._line, start + 1)
        return tok, i - start

    def _read_identifier(self, line: str, start: int):
        i = start
        while i < len(line) and (line[i].isalnum() or line[i] == "_"):
            i += 1
        word = line[start:i]
        ttype = KEYWORDS.get(word.lower(), TokenType.IDENTIFIER)
        tok = Token(ttype, word, self._line, start + 1)
        return tok, i - start

    def _single_char(self, ch: str, i: int) -> Token | None:
        mapping = {
            "=": TokenType.EQUALS,
            ":": TokenType.COLON,
            ",": TokenType.COMMA,
            ".": TokenType.DOT,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
        }
        ttype = mapping.get(ch)
        if ttype is None:
            return None
        return Token(ttype, ch, self._line, self._col)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, ttype: TokenType, value: str) -> None:
        self._tokens.append(Token(ttype, value, self._line, self._col))


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def tokenize(source: str) -> List[Token]:
    """Tokenize a Synapse source string and return the token list."""
    return Lexer(source).tokenize()
