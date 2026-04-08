"""
parser.py — Synapse Recursive-Descent Parser

Converts a flat Token stream (produced by the Lexer) into a structured
Abstract Syntax Tree (AST) rooted at ProgramNode.

Grammar (informal BNF):

  program      ::= statement* EOF
  statement    ::= tensor_decl
                 | model_decl
                 | train_decl

  tensor_decl  ::= 'tensor' IDENTIFIER '=' list_expr NEWLINE

  model_decl   ::= 'model' IDENTIFIER ':' NEWLINE
                     INDENT layer_stmt+ DEDENT

  layer_stmt   ::= 'layer' IDENTIFIER '(' arg_list ')' NEWLINE

  train_decl   ::= 'train' IDENTIFIER 'on' IDENTIFIER ':' NEWLINE
                     INDENT train_config+ DEDENT

  train_config ::= IDENTIFIER '=' expr NEWLINE

  list_expr    ::= '[' (list_expr | expr) (',' (list_expr | expr))* ']'
                 | '[' ']'

  expr         ::= NUMBER | STRING | IDENTIFIER | list_expr

  arg_list     ::= expr (',' expr)*
                 | ε
"""

from __future__ import annotations

from typing import List, Optional

from lexer import Token, TokenType
from ast_nodes import (
    ASTNode,
    IdentifierNode,
    LayerNode,
    ListNode,
    ModelNode,
    NumberNode,
    ProgramNode,
    StringNode,
    TensorNode,
    TrainConfigNode,
    TrainNode,
)


# ---------------------------------------------------------------------------
# Parser errors
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None) -> None:
        loc = f" at {token.line}:{token.column} (got {token.type.name} {token.value!r})" if token else ""
        super().__init__(f"ParseError{loc} — {message}")
        self.token = token


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    """
    Recursive-descent parser.

    Usage:
        parser  = Parser(tokens)
        program = parser.parse()   # returns ProgramNode
    """

    def __init__(self, tokens: List[Token]) -> None:
        # Filter out pure whitespace artefacts we don't care about here
        self._tokens: List[Token] = tokens
        self._pos: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> ProgramNode:
        statements: List[ASTNode] = []
        while not self._at_end():
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
        return ProgramNode(statements=statements)

    # ------------------------------------------------------------------
    # Statement dispatcher
    # ------------------------------------------------------------------

    def _parse_statement(self) -> Optional[ASTNode]:
        # Skip stray newlines at top level
        while self._check(TokenType.NEWLINE):
            self._advance()

        if self._at_end():
            return None

        tok = self._peek()

        if tok.type == TokenType.KW_TENSOR:
            return self._parse_tensor_decl()
        if tok.type == TokenType.KW_MODEL:
            return self._parse_model_decl()
        if tok.type == TokenType.KW_TRAIN:
            return self._parse_train_decl()

        raise ParseError(
            f"Unexpected token at top level: {tok.type.name} {tok.value!r}", tok
        )

    # ------------------------------------------------------------------
    # tensor <name> = <list_expr>
    # ------------------------------------------------------------------

    def _parse_tensor_decl(self) -> TensorNode:
        self._expect(TokenType.KW_TENSOR)
        name_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.EQUALS)
        value = self._parse_expr()
        self._consume_newline()
        return TensorNode(name=name_tok.value, value=value)

    # ------------------------------------------------------------------
    # model <name>:
    #     layer ...
    # ------------------------------------------------------------------

    def _parse_model_decl(self) -> ModelNode:
        self._expect(TokenType.KW_MODEL)
        name_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)

        layers: List[LayerNode] = []
        while not self._check(TokenType.DEDENT) and not self._at_end():
            # skip blank lines inside block
            while self._check(TokenType.NEWLINE):
                self._advance()
            if self._check(TokenType.DEDENT) or self._at_end():
                break
            layers.append(self._parse_layer_stmt())

        self._expect(TokenType.DEDENT)
        return ModelNode(name=name_tok.value, layers=layers)

    def _parse_layer_stmt(self) -> LayerNode:
        self._expect(TokenType.KW_LAYER)
        layer_type_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.LPAREN)
        args = self._parse_arg_list()
        self._expect(TokenType.RPAREN)
        self._consume_newline()
        return LayerNode(layer_type=layer_type_tok.value, args=args)

    # ------------------------------------------------------------------
    # train <model> on <data>:
    #     key = value
    # ------------------------------------------------------------------

    def _parse_train_decl(self) -> TrainNode:
        self._expect(TokenType.KW_TRAIN)
        model_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.KW_ON)
        data_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)

        config: List[TrainConfigNode] = []
        while not self._check(TokenType.DEDENT) and not self._at_end():
            while self._check(TokenType.NEWLINE):
                self._advance()
            if self._check(TokenType.DEDENT) or self._at_end():
                break
            config.append(self._parse_train_config())

        self._expect(TokenType.DEDENT)
        return TrainNode(
            model_name=model_tok.value,
            data_name=data_tok.value,
            config=config,
        )

    def _parse_train_config(self) -> TrainConfigNode:
        # Accept any identifier or keyword that may appear as a config key
        key_tok = self._advance()
        if key_tok.type not in (
            TokenType.IDENTIFIER,
            TokenType.KW_EPOCHS,
        ) and not key_tok.type.name.startswith("KW_"):
            raise ParseError(f"Expected config key identifier", key_tok)
        self._expect(TokenType.EQUALS)
        value = self._parse_expr()
        self._consume_newline()
        return TrainConfigNode(key=key_tok.value, value=value)

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------

    def _parse_expr(self) -> ASTNode:
        tok = self._peek()
        if tok.type == TokenType.LBRACKET:
            return self._parse_list()
        if tok.type == TokenType.NUMBER:
            self._advance()
            raw = tok.value
            num = float(raw) if "." in raw else int(raw)
            return NumberNode(value=num)
        if tok.type == TokenType.STRING:
            self._advance()
            return StringNode(value=tok.value)
        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return IdentifierNode(name=tok.value)
        # Allow keywords that act as values (e.g. activation names that collide)
        if tok.type.name.startswith("KW_"):
            self._advance()
            return IdentifierNode(name=tok.value)
        raise ParseError(f"Expected expression", tok)

    def _parse_list(self) -> ListNode:
        self._expect(TokenType.LBRACKET)
        elements: List[ASTNode] = []
        while not self._check(TokenType.RBRACKET):
            if self._at_end():
                raise ParseError("Unterminated list expression", self._peek())
            elements.append(self._parse_expr())
            if self._check(TokenType.COMMA):
                self._advance()
        self._expect(TokenType.RBRACKET)
        return ListNode(elements=elements)

    def _parse_arg_list(self) -> List[ASTNode]:
        args: List[ASTNode] = []
        while not self._check(TokenType.RPAREN):
            if self._at_end():
                raise ParseError("Unterminated argument list", self._peek())
            args.append(self._parse_expr())
            if self._check(TokenType.COMMA):
                self._advance()
        return args

    # ------------------------------------------------------------------
    # Token stream helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if tok.type != TokenType.EOF:
            self._pos += 1
        return tok

    def _at_end(self) -> bool:
        return self._tokens[self._pos].type == TokenType.EOF

    def _check(self, ttype: TokenType) -> bool:
        return self._tokens[self._pos].type == ttype

    def _expect(self, ttype: TokenType) -> Token:
        tok = self._peek()
        if tok.type != ttype:
            raise ParseError(
                f"Expected {ttype.name}, got {tok.type.name} {tok.value!r}", tok
            )
        return self._advance()

    def _consume_newline(self) -> None:
        """Consume one or more newlines; at least one is required."""
        if not self._check(TokenType.NEWLINE) and not self._at_end():
            raise ParseError(
                f"Expected newline, got {self._peek().type.name}", self._peek()
            )
        while self._check(TokenType.NEWLINE):
            self._advance()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def parse(tokens) -> ProgramNode:
    """Parse a token list and return the root ProgramNode."""
    return Parser(tokens).parse()
