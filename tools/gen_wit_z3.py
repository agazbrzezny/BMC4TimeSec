#!/usr/bin/env python3
"""
gen_wit_from_out_v5.py

Generate a witness (.wit) ONLY from a Z3 output file (.out) that contains:
  sat
  ( ... many (define-fun ...) ... )

This script does NOT require the .smt file and does NOT run Z3.

Naming convention supported (your semantics):
  a_i_k_j  : Int   action for path i, depth k, component j   (we use i=0)
  w_0_k_j  : Int   location of component j at depth k
  x_0_k_c  : Real  clock c at depth k
  t_0_k    : Real  per-step delay at depth k (delta), NOT cumulative time

Witness semantics used here:
  delta(k)      := t_0_k  (non-negative)
  globaltime(k) := sum_{i=0..k} t_0_i

Output:
  nrComp = N , nrOfClocks = M , LL = 1 , k = K

  m: (delta, (actions), (locations), (clocks), globaltime)

  d: (delta_d, (a0,a1,...,aN-1), (w0,w1,...,wN-1), (x0,...,xM-1), globaltime_d)

Usage:
  python gen_wit_from_out_v5.py model-k10.out > model-k10.wit
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Tuple

Token = str

# ----------------------------- S-expression parsing -----------------------------

def tokenize_sexpr(s: str) -> List[Token]:
    s = re.sub(r";[^\n]*", "", s)  # Z3 comments
    tokens: List[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ("(", ")"):
            tokens.append(ch)
            i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and s[j] != '"':
                if s[j] == "\\" and j + 1 < n:
                    j += 2
                else:
                    j += 1
            tokens.append(s[i:j + 1])
            i = j + 1
            continue
        j = i
        while j < n and (not s[j].isspace()) and s[j] not in ("(", ")"):
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


def parse_tokens(tokens: List[Token]) -> Any:
    stack: List[List[Any]] = []
    cur: List[Any] = []
    for tok in tokens:
        if tok == "(":
            stack.append(cur)
            cur = []
        elif tok == ")":
            if not stack:
                raise ValueError("Unbalanced ')'")
            completed = cur
            cur = stack.pop()
            cur.append(completed)
        else:
            cur.append(tok)
    if stack:
        raise ValueError("Unbalanced '('")
    if len(cur) == 1:
        return cur[0]
    return cur


# ----------------------------- Numeric eval -----------------------------

_NUM_RE = re.compile(r"^-?\d+$")
_DEC_RE = re.compile(r"^-?\d+\.\d+$")

def _as_fraction(node: Any) -> Fraction:
    if isinstance(node, str):
        if _NUM_RE.match(node):
            return Fraction(int(node), 1)
        if _DEC_RE.match(node):
            sign = -1 if node.startswith("-") else 1
            s = node[1:] if sign == -1 else node
            a, b = s.split(".")
            return Fraction(sign * int(a + b), 10 ** len(b))
        raise ValueError(f"Cannot evaluate atom as number: {node!r}")
    if isinstance(node, list) and node:
        op = node[0]
        if op == "/":
            return _as_fraction(node[1]) / _as_fraction(node[2])
        if op == "-":
            if len(node) == 2:
                return -_as_fraction(node[1])
            acc = _as_fraction(node[1])
            for part in node[2:]:
                acc -= _as_fraction(part)
            return acc
        if op == "+":
            acc = Fraction(0, 1)
            for part in node[1:]:
                acc += _as_fraction(part)
            return acc
        if op == "*":
            acc = Fraction(1, 1)
            for part in node[1:]:
                acc *= _as_fraction(part)
            return acc
        if op == "to_real" and len(node) == 2:
            return _as_fraction(node[1])
        raise ValueError(f"Unsupported numeric term: {node}")
    raise ValueError(f"Cannot evaluate: {node!r}")


def eval_int(node: Any) -> int:
    return int(_as_fraction(node))


def eval_float(node: Any) -> float:
    f = _as_fraction(node)
    return float(f.numerator) / float(f.denominator)


# ----------------------------- Model extraction -----------------------------

_DEFINE_FUN = "define-fun"

@dataclass
class Assign:
    name: str
    sort: str
    value: Any


def extract_defines(ast: Any) -> List[Assign]:
    assigns: List[Assign] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            if node and node[0] == _DEFINE_FUN and len(node) >= 5:
                assigns.append(Assign(name=str(node[1]), sort=str(node[3]), value=node[4]))
            for ch in node:
                walk(ch)

    walk(ast)
    return assigns


def format_float(x: float) -> str:
    if abs(x - round(x)) < 1e-12:
        return f"{float(round(x)):.1f}"
    s = f"{x:.12g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python gen_wit_from_out_v5.py model-k10.out > model-k10.wit", file=sys.stderr)
        return 2

    out_path = Path(sys.argv[1])
    out_text = out_path.read_text(encoding="utf-8", errors="replace")

    if "unsat" in out_text:
        print("unsat (no witness)", file=sys.stderr)
        return 1
    if "sat" not in out_text:
        print("Error: output does not contain sat.", file=sys.stderr)
        return 2

    tokens = tokenize_sexpr(out_text)
    ast = parse_tokens(tokens)
    assigns = extract_defines(ast)
    if not assigns:
        print("Error: no define-fun assignments found in .out.", file=sys.stderr)
        return 2

    # Collect values + infer sizes from what exists in the model
    locs: Dict[Tuple[int, int], int] = {}         # (k, j) -> loc
    clks: Dict[Tuple[int, int], float] = {}       # (k, c) -> clk
    acts: Dict[Tuple[int, int, int], int] = {}    # (i, k, j) -> act
    step_delay: Dict[int, float] = {}             # k -> t_0_k

    max_comp = -1
    max_clk = -1
    max_depth = -1

    for a in assigns:
        nm = a.name

        m = re.match(r"^w_0_(\d+)_(\d+)$", nm)
        if m:
            k = int(m.group(1)); j = int(m.group(2))
            locs[(k, j)] = eval_int(a.value)
            max_depth = max(max_depth, k)
            max_comp = max(max_comp, j)
            continue

        m = re.match(r"^x_0_(\d+)_(\d+)$", nm)
        if m:
            k = int(m.group(1)); c = int(m.group(2))
            clks[(k, c)] = eval_float(a.value)
            max_depth = max(max_depth, k)
            max_clk = max(max_clk, c)
            continue

        m = re.match(r"^a_(\d+)_(\d+)_(\d+)$", nm)
        if m:
            i = int(m.group(1)); k = int(m.group(2)); j = int(m.group(3))
            acts[(i, k, j)] = eval_int(a.value)
            max_depth = max(max_depth, k)
            max_comp = max(max_comp, j)
            continue

        m = re.match(r"^t_0_(\d+)$", nm) or re.match(r"^t_(\d+)$", nm)
        if m:
            k = int(m.group(1))
            step_delay[k] = eval_float(a.value)
            max_depth = max(max_depth, k)
            continue

    if max_depth < 0:
        print("Error: could not infer k (no w_0_k_j / x_0_k_c / a_i_k_j / t_0_k found).", file=sys.stderr)
        return 2

    nr_comp = max_comp + 1 if max_comp >= 0 else 0
    nr_clk = max_clk + 1 if max_clk >= 0 else 0
    k = max_depth

    ll = 1
    print(f"nrComp = {nr_comp} , nrOfClocks = {nr_clk} , LL = {ll} , k = {k} \n")
    print("m: (delta, (actions), (locations), (clocks), globaltime)\n")

    path_i = 0
    gtime = 0.0
    for depth in range(0, k + 1):
        delta = step_delay.get(depth, 0.0)
        if delta < -1e-12:
            # should not happen if constraints ensure non-negativity; clamp anyway
            delta = 0.0
        gtime += delta

        a_vec = [acts.get((path_i, depth, j), 0) for j in range(nr_comp)]
        w_vec = [locs.get((depth, j), 0) for j in range(nr_comp)]
        x_vec = [clks.get((depth, c), 0.0) for c in range(nr_clk)]

        print(
            f"{depth}: "
            f"({format_float(delta)}, "
            f"({','.join(str(v) for v in a_vec)}), "
            f"({','.join(str(v) for v in w_vec)}), "
            f"({','.join(format_float(v) for v in x_vec)}), "
            f"{format_float(gtime)})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
