#!/usr/bin/env python3
"""
synapse.py — Synapse Language CLI

Usage:
    python synapse.py run <file.syn>          # run a Synapse program
    python synapse.py tokenize <file.syn>     # print the token stream
    python synapse.py parse <file.syn>        # print the AST
    python synapse.py transpile <file.syn>    # print generated Python
    python synapse.py version                 # print version info

Flags:
    --save-py <path>    Save generated Python to a file
    --inprocess         Use in-process executor (faster, less isolated)
    --verbose           Print each pipeline stage header
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

# Ensure the synapse package directory is on sys.path when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lexer import tokenize, LexerError
from parser import parse, ParseError
from transpiler import transpile, TranspileError
from runtime import execute, ExecutionResult


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

SYNAPSE_VERSION = "1.0.0"
SYNAPSE_BANNER = textwrap.dedent(f"""\
    ╔══════════════════════════════════╗
    ║  Synapse Language  v{SYNAPSE_VERSION}       ║
    ║  ML-native declarative language  ║
    ╚══════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    source: str,
    *,
    save_py: str | None = None,
    strategy: str = "subprocess",
    verbose: bool = False,
) -> ExecutionResult:
    """
    Execute the full Synapse pipeline:
        source → tokens → AST → Python → execution

    Returns an ExecutionResult from the runtime.
    """

    # 1. Tokenize
    if verbose:
        _header("Tokenizing")
    tokens = tokenize(source)

    # 2. Parse
    if verbose:
        _header("Parsing")
    program = parse(tokens)

    # 3. Transpile
    if verbose:
        _header("Transpiling")
    python_source = transpile(program)

    if verbose:
        _header("Generated Python")
        print(python_source)
        _header("Executing")

    # Optional: save generated Python
    if save_py:
        Path(save_py).write_text(python_source, encoding="utf-8")
        if verbose:
            print(f"Generated Python saved to: {save_py}")

    # 4. Execute
    result = execute(python_source, strategy=strategy)
    return result


def cmd_run(args: argparse.Namespace) -> int:
    source = _read_file(args.file)
    strategy = "inprocess" if args.inprocess else "subprocess"

    try:
        result = run_pipeline(
            source,
            save_py=args.save_py,
            strategy=strategy,
            verbose=args.verbose,
        )
    except (LexerError, ParseError, TranspileError) as exc:
        print(f"\n[Synapse Error] {exc}", file=sys.stderr)
        return 1

    result.print_output()
    if not result.success:
        print(
            f"\n[Runtime Error] Process exited with code {result.returncode}",
            file=sys.stderr,
        )
        return result.returncode
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    source = _read_file(args.file)
    try:
        tokens = tokenize(source)
    except LexerError as exc:
        print(f"[LexerError] {exc}", file=sys.stderr)
        return 1

    print(f"{'#':<4}  {'TYPE':<18}  {'VALUE':<20}  LINE:COL")
    print("-" * 60)
    for i, tok in enumerate(tokens):
        print(f"{i:<4}  {tok.type.name:<18}  {tok.value!r:<20}  {tok.line}:{tok.column}")
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    source = _read_file(args.file)
    try:
        tokens = tokenize(source)
        program = parse(tokens)
    except (LexerError, ParseError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    _print_ast(program, indent=0)
    return 0


def cmd_transpile(args: argparse.Namespace) -> int:
    source = _read_file(args.file)
    try:
        tokens = tokenize(source)
        program = parse(tokens)
        python_source = transpile(program)
    except (LexerError, ParseError, TranspileError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    print(python_source)

    if args.save_py:
        Path(args.save_py).write_text(python_source, encoding="utf-8")
        print(f"\n# Saved to: {args.save_py}", file=sys.stderr)
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    print(SYNAPSE_BANNER)
    print(f"Python interpreter: {sys.executable}")
    print(f"Python version:     {sys.version}")
    return 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"[Error] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    if p.suffix not in (".syn", ".synapse", ".py"):
        print(
            f"[Warning] Unexpected file extension: {p.suffix} (expected .syn)",
            file=sys.stderr,
        )
    return p.read_text(encoding="utf-8")


def _header(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def _print_ast(node, indent: int = 0) -> None:
    pad = "  " * indent
    name = type(node).__name__
    if hasattr(node, "__dict__"):
        print(f"{pad}{name}(")
        for key, val in node.__dict__.items():
            if isinstance(val, list):
                print(f"{pad}  {key}=[")
                for item in val:
                    _print_ast(item, indent + 2)
                print(f"{pad}  ]")
            elif hasattr(val, "__dict__"):
                print(f"{pad}  {key}=", end="")
                _print_ast(val, 0)
            else:
                print(f"{pad}  {key}={val!r}")
        print(f"{pad})")
    else:
        print(f"{pad}{node!r}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synapse",
        description="Synapse — declarative ML language compiler & runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python synapse.py run examples/test.syn
              python synapse.py run examples/test.syn --save-py out.py --verbose
              python synapse.py tokenize examples/test.syn
              python synapse.py transpile examples/test.syn
              python synapse.py version
        """),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = subparsers.add_parser("run", help="Run a .syn file end-to-end")
    run_p.add_argument("file", help="Path to the .syn source file")
    run_p.add_argument("--save-py", metavar="PATH", help="Save generated Python to PATH")
    run_p.add_argument("--inprocess", action="store_true", help="Use in-process executor")
    run_p.add_argument("--verbose", action="store_true", help="Print each pipeline stage")

    # --- tokenize ---
    tok_p = subparsers.add_parser("tokenize", help="Print the token stream")
    tok_p.add_argument("file", help="Path to the .syn source file")

    # --- parse ---
    prs_p = subparsers.add_parser("parse", help="Print the AST")
    prs_p.add_argument("file", help="Path to the .syn source file")

    # --- transpile ---
    tr_p = subparsers.add_parser("transpile", help="Print (or save) generated Python")
    tr_p.add_argument("file", help="Path to the .syn source file")
    tr_p.add_argument("--save-py", metavar="PATH", help="Save generated Python to PATH")

    # --- version ---
    subparsers.add_parser("version", help="Print version information")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dispatch = {
        "run":       cmd_run,
        "tokenize":  cmd_tokenize,
        "parse":     cmd_parse,
        "transpile": cmd_transpile,
        "version":   cmd_version,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
