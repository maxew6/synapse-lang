"""
ast_nodes.py — Synapse AST Node Definitions

Each node class represents a distinct syntactic construct in the Synapse language.
The node hierarchy is designed to be extensible for future constructs such as
AI blocks, memory declarations, and multi-model pipelines.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ASTNode:
    """Abstract base for all Synapse AST nodes."""

    def accept(self, visitor: "ASTVisitor") -> Any:
        method = f"visit_{type(self).__name__}"
        visitor_fn = getattr(visitor, method, visitor.generic_visit)
        return visitor_fn(self)

    def __repr__(self) -> str:
        attrs = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
        )
        return f"{type(self).__name__}({attrs})"


# ---------------------------------------------------------------------------
# Literals / primitives
# ---------------------------------------------------------------------------

@dataclass
class NumberNode(ASTNode):
    """A numeric literal (int or float)."""
    value: Union[int, float]


@dataclass
class StringNode(ASTNode):
    """A string literal."""
    value: str


@dataclass
class IdentifierNode(ASTNode):
    """A bare identifier (variable name, activation function name, etc.)."""
    name: str


@dataclass
class ListNode(ASTNode):
    """A nested list literal, used for tensor data."""
    elements: List[ASTNode]


# ---------------------------------------------------------------------------
# Tensor declaration
# ---------------------------------------------------------------------------

@dataclass
class TensorNode(ASTNode):
    """
    tensor <name> = <value>

    value is a ListNode representing potentially multi-dimensional data.
    dtype and device are reserved for future type annotations.
    """
    name: str
    value: ASTNode                   # typically ListNode
    dtype: Optional[str] = None      # e.g. "float32" — future
    device: Optional[str] = None     # e.g. "cuda"    — future


# ---------------------------------------------------------------------------
# Model / layer declarations
# ---------------------------------------------------------------------------

@dataclass
class LayerNode(ASTNode):
    """
    layer <layer_type>(<arg1>, <arg2>, ...)

    Represents a single layer inside a model block.
    Extra kwargs dict reserved for future named-parameter support.
    """
    layer_type: str                  # e.g. "dense", "conv2d", "attention"
    args: List[ASTNode]
    kwargs: dict = field(default_factory=dict)   # future: named params


@dataclass
class ModelNode(ASTNode):
    """
    model <name>:
        layer ...
        layer ...

    backbone_of and extends are reserved for model inheritance (future).
    """
    name: str
    layers: List[LayerNode]
    extends: Optional[str] = None    # future: model inheritance


# ---------------------------------------------------------------------------
# Training block
# ---------------------------------------------------------------------------

@dataclass
class TrainConfigNode(ASTNode):
    """
    Key-value configuration item inside a train block.
    e.g.  epochs = 10
    """
    key: str
    value: ASTNode


@dataclass
class TrainNode(ASTNode):
    """
    train <model_name> on <data_name>:
        epochs = N
        ...

    optimizer, loss, and callbacks are reserved for richer training
    configuration in future versions.
    """
    model_name: str
    data_name: str
    config: List[TrainConfigNode]
    optimizer: Optional[str] = None  # future
    loss: Optional[str] = None       # future
    callbacks: List[str] = field(default_factory=list)  # future


# ---------------------------------------------------------------------------
# Future extensibility stubs
# ---------------------------------------------------------------------------

@dataclass
class AIBlockNode(ASTNode):
    """
    ai <name>:
        model = "gpt-4"
        prompt = "..."

    Reserved for LLM integration blocks. Not parsed in v1 but the node
    exists so the transpiler can detect and error gracefully.
    """
    name: str
    config: dict = field(default_factory=dict)


@dataclass
class MemoryNode(ASTNode):
    """
    memory <name>:
        type = vector_store
        backend = faiss

    Reserved for memory system declarations.
    """
    name: str
    config: dict = field(default_factory=dict)


@dataclass
class PipelineNode(ASTNode):
    """
    pipeline <name>:
        step model_A
        step model_B

    Reserved for multi-model execution pipelines.
    """
    name: str
    steps: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level program
# ---------------------------------------------------------------------------

@dataclass
class ProgramNode(ASTNode):
    """Root node — holds the ordered list of all top-level statements."""
    statements: List[ASTNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Visitor interface
# ---------------------------------------------------------------------------

class ASTVisitor:
    """
    Base visitor. Subclass and override visit_* methods.
    Transpiler, printer, and future optimizer passes all derive from this.
    """

    def visit(self, node: ASTNode) -> Any:
        return node.accept(self)

    def generic_visit(self, node: ASTNode) -> Any:
        raise NotImplementedError(
            f"No visitor method defined for {type(node).__name__}"
        )
