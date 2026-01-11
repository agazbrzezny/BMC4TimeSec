#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_ef_formulas_v3.py

EF formula generator for .nta networks.

Target syntax (as requested):
- Conjunction:   EF(p1 & p2 & p3)
- Alternative:   +   (e.g., EF((p1 & p2) + (p7 & p8)))
- Negation:      ~pX (e.g., EF(p1 & ~p2))

What it does:
- Parses .nta, groups execution automata by run-id (r1, r2, ...).
- For each run-id, builds a *conjunction* requiring that all execution automata of that run
  are in their final location (valuation ids).
- Optionally adds knowledge conditions:
  - --intruder-knows TERM  => add p(v(K(I,TERM)=known))
  - --intruder-not-knows TERM => add ~p(v(K(I,TERM)=known))
  - --agent-knows A:TERM   => add p(v(K(A,TERM)=known))
  - --agent-not-knows A:TERM => add ~p(v(K(A,TERM)=known))
- Writes either:
  - One EF(...) per run (default), or
  - One combined EF(...) with + over selected runs (--combine-or).

Notes:
- This script is protocol-agnostic. It relies only on the .nta 'valuation' maps and the
  translator comment headers identifying automata.
"""

from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass
class AutomatonInfo:
    num: int
    kind: str  # "exec" or "K" or "unknown"
    run_id: Optional[str] = None
    sid: Optional[int] = None
    agent: Optional[str] = None
    term: Optional[str] = None
    loc_to_val: Dict[int, int] = None

    @property
    def final_val(self) -> int:
        if not self.loc_to_val:
            raise ValueError(f"Automaton #{self.num} has no valuation map")
        return self.loc_to_val[max(self.loc_to_val.keys())]

    @property
    def known_val(self) -> int:
        if not self.loc_to_val:
            raise ValueError(f"Automaton #{self.num} has no valuation map")
        if 1 not in self.loc_to_val:
            raise ValueError(f"Automaton #{self.num} has no loc=1 (expected for knowledge automaton)")
        return self.loc_to_val[1]


AUT_HDR_RE = re.compile(r"^automaton #(\d+)\s*$", re.IGNORECASE)
VAL_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")

# Accept "-", "–", "—" between parts
DASH = r"(?:-|–|—)"

# Polish or English headers (your translators vary)
EXEC_COMMENT_RE = re.compile(
    rf"^#\s*Automaton #(\d+)\s*{DASH}\s*(?:automat wykonaniowy dla|execution automaton for)\s*(r\d+)\b",
    re.IGNORECASE
)
KNOW_COMMENT_RE = re.compile(
    rf"^#\s*Automaton #(\d+)\s*{DASH}\s*(?:automat wiedzowy|knowledge automaton)\s*K\(([^,]+),(.+)\)\s*$",
    re.IGNORECASE
)

# per-execution mapping lines (used only for nicer comments)
STEP_MAP_RE = re.compile(r"->\s*(r\d+):sid=(\d+),step=", re.IGNORECASE)

RUN_HDR_RE = re.compile(r"^\s*//\s*run\s+(r\d+)\s*(?:#\s*(.+))?\s*$", re.IGNORECASE)


def parse_protoc_run_names(path: Path) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RUN_HDR_RE.match(line)
        if m:
            rid = m.group(1)
            nm = (m.group(2) or "").strip()
            if nm:
                names[rid] = nm
    return names


def parse_nta(path: Path) -> Tuple[Dict[int, AutomatonInfo], Dict[str, List[int]]]:
    txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # First pass: collect comment metadata by automaton number
    comment_kind: Dict[int, Tuple[str, dict]] = {}
    for line in txt:
        s = line.strip()
        m = EXEC_COMMENT_RE.match(s)
        if m:
            num = int(m.group(1)); run_id = m.group(2)
            comment_kind[num] = ("exec", {"run_id": run_id})
            continue
        m = KNOW_COMMENT_RE.match(s)
        if m:
            num = int(m.group(1)); agent = m.group(2).strip(); term = m.group(3).strip()
            comment_kind[num] = ("K", {"agent": agent, "term": term})
            continue

    autos: Dict[int, AutomatonInfo] = {}
    run_to_exec: Dict[str, List[int]] = {}

    i = 0
    while i < len(txt):
        m = AUT_HDR_RE.match(txt[i].strip())
        if not m:
            i += 1
            continue
        num = int(m.group(1))
        kind, meta = comment_kind.get(num, ("unknown", {}))
        loc_to_val: Dict[int, int] = {}

        run_id = meta.get("run_id")
        sid = None
        agent = meta.get("agent")
        term = meta.get("term")

        # best-effort sid inference for comments
        if kind == "exec" and run_id:
            back_start = max(0, i - 400)
            window = txt[back_start:i]
            for w in reversed(window):
                sm = STEP_MAP_RE.search(w)
                if sm and sm.group(1) == run_id:
                    sid = int(sm.group(2))
                    break

        # valuation section
        j = i + 1
        while j < len(txt) and txt[j].strip().lower() != "valuation":
            j += 1
        if j < len(txt) and txt[j].strip().lower() == "valuation":
            j += 1
            while j < len(txt) and txt[j].strip().lower() != "end":
                vm = VAL_RE.match(txt[j])
                if vm:
                    loc_to_val[int(vm.group(1))] = int(vm.group(2))
                j += 1

        info = AutomatonInfo(
            num=num, kind=kind, run_id=run_id, sid=sid,
            agent=agent, term=term, loc_to_val=loc_to_val
        )
        autos[num] = info
        if kind == "exec" and run_id:
            run_to_exec.setdefault(run_id, []).append(num)

        i = j + 1

    for r in run_to_exec:
        run_to_exec[r] = sorted(run_to_exec[r])
    return autos, run_to_exec


def lit(v: int, neg: bool, p_prefix: bool) -> str:
    atom = f"p{v}" if p_prefix else str(v)
    return f"~{atom}" if neg else atom


def conj(lits: List[str]) -> str:
    # Empty conjunction = true (rare, but keep well-defined)
    return "true" if not lits else " & ".join(lits)


def disj(terms: List[str]) -> str:
    # Empty disjunction = false (rare)
    return "false" if not terms else " + ".join(terms)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nta", help="Input .nta network file")
    ap.add_argument("-p", "--protoc", help="Optional .protoc file (adds scenario names in comments)")
    ap.add_argument("-o", "--out", required=True, help="Output .efo file")

    ap.add_argument("--p-prefix", action="store_true", default=True,
                    help="Prefix valuation ids with 'p' (default: on)")
    ap.add_argument("--no-p-prefix", action="store_true",
                    help="Do not prefix valuation ids with 'p'")

    ap.add_argument("--intruder-knows", action="append", default=[],
                    help="Require K(I,<term>) is known. Repeatable.")
    ap.add_argument("--intruder-not-knows", action="append", default=[],
                    help="Require K(I,<term>) is NOT known (negated). Repeatable.")
    ap.add_argument("--agent-knows", action="append", default=[],
                    help="Require K(A,<term>) known; format A:<term>. Repeatable.")
    ap.add_argument("--agent-not-knows", action="append", default=[],
                    help="Require K(A,<term>) NOT known; format A:<term>. Repeatable.")

    ap.add_argument("--only-run", action="append", default=[],
                    help="Only emit formulas for these run-ids (r1,r2,...). Repeatable.")
    ap.add_argument("--combine-or", action="store_true",
                    help="Emit a single formula that is + (OR) of per-run conjunctions.")
    ap.add_argument("--include-run-comments", action="store_true",
                    help="Include per-run info as comments next to each formula (or in a block if --combine-or).")
    args = ap.parse_args()

    p_prefix = (not args.no_p_prefix)

    autos, run_to_exec = parse_nta(Path(args.nta))

    run_names: Dict[str, str] = {}
    if args.protoc:
        run_names = parse_protoc_run_names(Path(args.protoc))

    # knowledge lookup: (agent, term) -> known valuation id
    know_lookup: Dict[Tuple[str, str], int] = {}
    for a in autos.values():
        if a.kind == "K" and a.agent and a.term and a.loc_to_val:
            know_lookup[(a.agent.strip(), a.term.strip())] = a.known_val

    def get_known_val(agent: str, term: str) -> int:
        key = (agent.strip(), term.strip())
        if key not in know_lookup:
            raise SystemExit(
                f"Missing knowledge automaton for {key}. "
                f"Term must match exactly the K(...) comment in .nta."
            )
        return know_lookup[key]

    # select runs
    selected_runs = sorted(run_to_exec.keys())
    if args.only_run:
        allowed = set(args.only_run)
        selected_runs = [r for r in selected_runs if r in allowed]

    # build shared knowledge literals (same for each run formula)
    knowledge_lits: List[str] = []
    for t in args.intruder_knows:
        v = get_known_val("I", t.strip())
        knowledge_lits.append(lit(v, neg=False, p_prefix=p_prefix))
    for t in args.intruder_not_knows:
        v = get_known_val("I", t.strip())
        knowledge_lits.append(lit(v, neg=True, p_prefix=p_prefix))

    def parse_agent_spec(spec: str) -> Tuple[str, str]:
        if ":" not in spec:
            raise SystemExit(f"Expected 'A:<term>' but got: {spec}")
        ag, term = spec.split(":", 1)
        return ag.strip(), term.strip()

    for spec in args.agent_knows:
        ag, term = parse_agent_spec(spec)
        v = get_known_val(ag, term)
        knowledge_lits.append(lit(v, neg=False, p_prefix=p_prefix))
    for spec in args.agent_not_knows:
        ag, term = parse_agent_spec(spec)
        v = get_known_val(ag, term)
        knowledge_lits.append(lit(v, neg=True, p_prefix=p_prefix))

    out_lines: List[str] = []

    per_run_terms: List[str] = []
    per_run_comments: List[str] = []

    for rid in selected_runs:
        exec_autos = run_to_exec[rid]

        # final valuation literals for this run
        final_vals: List[int] = []
        for anum in exec_autos:
            final_vals.append(autos[anum].final_val)

        # unique preserving order
        seen = set()
        uniq_final_vals: List[int] = []
        for v in final_vals:
            if v not in seen:
                uniq_final_vals.append(v); seen.add(v)

        final_lits = [lit(v, neg=False, p_prefix=p_prefix) for v in uniq_final_vals]
        all_lits = final_lits + knowledge_lits
        term = conj(all_lits)
        term_wrapped = f"({term})" if (" & " in term or " + " in term or term in ("true","false")) else term

        # comment
        comment = f"run={rid}"
        if rid in run_names:
            comment += f" name={run_names[rid]}"
        if args.include_run_comments:
            # also show which exec automata used
            sid_info = []
            for anum in exec_autos:
                a = autos[anum]
                sid_info.append(f"#{anum}(sid={a.sid})" if a.sid is not None else f"#{anum}")
            comment += " exec=" + ",".join(sid_info)

        if args.combine_or:
            per_run_terms.append(term_wrapped)
            per_run_comments.append(comment)
        else:
            formula = f"EF({term})"
            if args.include_run_comments:
                out_lines.append(f"{formula}\nend  # {comment}")
            else:
                out_lines.append(f"{formula}\nend")

    if args.combine_or:
        big = disj(per_run_terms)
        formula = f"EF({big})"
        if args.include_run_comments:
            out_lines.append(formula)
            out_lines.append("end")
            out_lines.append("# --- runs included ---")
            for c in per_run_comments:
                out_lines.append(f"# {c}")
        else:
            out_lines.append(f"{formula}\nend")

    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} block(s) to {args.out}")


if __name__ == "__main__":
    main()
