#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ab_to_protoc_interpretations_v2.py

Generate a .protoc file from an Alice–Bob protocol (.ab) BUT only for an explicit
set of attack/interpretation scenarios (no enumeration).

This is intended to produce protoc files compatible with the downstream tools
(protoc_to_nta_mirekstyle).
"""

import argparse, json, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any


# ----------------------------
# Parsing .ab (Alice-Bob) lines
# ----------------------------

AB_STEP_RE = re.compile(
    r"^\s*(?:(\d+)\s*[:.)]\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$"
)

@dataclass
class Step:
    i: int
    sender: str
    receiver: str
    msg_raw: str


def parse_ab_file(path: str) -> List[Step]:
    steps: List[Step] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            m = AB_STEP_RE.match(line)
            if not m:
                # tolerate LaTeX-ish prefixes like \alpha_1
                line2 = re.sub(r"^\\alpha_\d+\s*", "", line)
                m = AB_STEP_RE.match(line2)
            if not m:
                continue
            num, a, b, msg = m.groups()
            idx = int(num) if num else (len(steps) + 1)
            steps.append(Step(i=idx, sender=a, receiver=b, msg_raw=msg.strip()))
    steps.sort(key=lambda s: s.i)
    return steps


# ---------------------------------------
# AB message conversion: "{...}K" -> "<K,...>"
# ---------------------------------------

_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_#@]*")

class _MsgParser:
    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        self.i = 0

    def _skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i].isspace():
            self.i += 1

    def _peek(self) -> str:
        return self.s[self.i] if self.i < self.n else ""

    def _consume(self, ch: str) -> None:
        self._skip_ws()
        if self._peek() != ch:
            raise ValueError(f"Expected '{ch}' at pos {self.i} in {self.s!r}")
        self.i += 1

    def _parse_ident(self) -> str:
        self._skip_ws()
        m = _ID_RE.match(self.s, self.i)
        if not m:
            raise ValueError(f"Expected identifier at pos {self.i} in {self.s!r}")
        self.i = m.end()
        return m.group(0)

    def parse_term(self) -> str:
        self._skip_ws()
        if self._peek() == "{":
            return self.parse_enc()
        # allow already-internal <...>
        if self._peek() == "<":
            # passthrough: grab until matching '>' (no nesting assumed inside)
            start = self.i
            depth = 0
            while self.i < self.n:
                c = self.s[self.i]
                if c == "<":
                    depth += 1
                elif c == ">":
                    depth -= 1
                    if depth == 0:
                        self.i += 1
                        return self.s[start:self.i].strip()
                self.i += 1
            raise ValueError(f"Unclosed <...> in {self.s!r}")
        return self._parse_ident()

    def parse_payload(self) -> str:
        # payload := term (',' term)*
        parts = [self.parse_term()]
        self._skip_ws()
        while self._peek() == ",":
            self.i += 1
            parts.append(self.parse_term())
            self._skip_ws()
        if len(parts) == 1:
            return parts[0]
        return "|".join(parts)

    def parse_enc(self) -> str:
        # enc := '{' payload '}' keyIdent
        self._consume("{")
        payload = self.parse_payload()
        self._consume("}")
        key = self._parse_ident()
        return f"<{key},{payload}>"


def ab_msg_to_internal(msg: str) -> str:
    """
    Convert AB-style messages to internal protoc style.
    If message already uses <...> form, keep it.
    """
    m = msg.strip().rstrip(".")
    m = re.sub(r"\s+", " ", m)
    # If no braces, return as-is (could be "A", "Tb", or already "<...>")
    if "{" not in m and "}" not in m:
        return m
    try:
        p = _MsgParser(m)
        t = p.parse_term()
        # if there's trailing junk, keep it (but usually there isn't)
        p._skip_ws()
        if p.i != p.n:
            # best-effort: append remaining
            t = (t + m[p.i:]).strip()
        return t
    except Exception:
        # fall back to raw if parsing fails (best effort)
        return m


# ---------------------------------------
# Ticket instantiation + time constraints
# ---------------------------------------

def collect_ticket_vars(ticket_lifetimes: Dict[str, int]) -> List[str]:
    """Return ticket/nonce variable names that should be instantiated per session.

    We intentionally restrict this to variables that have an explicit lifetime entry
    (e.g., Tb:30). This avoids mistakenly instantiating identities like 'A'.
    """
    return sorted(ticket_lifetimes.keys(), key=len, reverse=True)


def instantiate_vars_in_msg(msg: str,
                               sid: int,
                               tickets: List[str],
                               aliases: Optional[Dict[str, str]] = None) -> str:
    """Instantiate per-session variables in a message.

    - Default: Tb -> Tb#sid (only when not already instantiated).
    - If aliases provided (e.g., {"Tb": "Tb#2"}), then Tb -> Tb#2 for that
      session/run ("foreign ticket" propagation), again only when not already
      instantiated.
    """
    out = msg
    if aliases:
        for base, inst in aliases.items():
            out = re.sub(rf"\b{re.escape(base)}\b(?!#)", inst, out)
    for t in tickets:
        out = re.sub(rf"\b{re.escape(t)}\b(?!#)", f"{t}#{sid}", out)
    return out


def infer_gen_ticket_from_raw(msg_raw: str) -> Optional[str]:
    m = msg_raw.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", m):
        return m
    return None


def make_tc_for_step(msg_inst: str,
                     sid: int,
                     ticket_lifetimes: Dict[str, int],
                     gen_ticket: Optional[str],
                     kind: str,
                     aliases: Optional[Dict[str, str]] = None) -> str:
    """
    Produce the 'tc=' field for this step.

    Conventions expected by protoc_to_nta_mirekstyle:
      - constraints are separated by commas (",") so the downstream parser can split them
      - supported atoms:
          * reset c[Tb#i]
          * c - c[Tb#i] <= L

    Semantics (matching the habilitation-style intent):
      - Only *non-intruder* steps may generate tickets (reset their age clocks).
        For kind=='intruder' we never reset ticket clocks and never set G automatically.
      - Lifetime bounds are emitted for every *instantiated* ticket occurrence in msg_inst,
        i.e., for Tb#2 even if this step belongs to sid=1 (foreign ticket injection).
    """
    if not ticket_lifetimes:
        return "-"

    constraints: List[str] = []

    # 1) generator reset (only for honest/replace steps, not intruder)
    if kind != "intruder" and gen_ticket and gen_ticket in ticket_lifetimes:
        inst = aliases.get(gen_ticket, f"{gen_ticket}#{sid}") if aliases else f"{gen_ticket}#{sid}"
        if re.fullmatch(rf"{re.escape(inst)}", msg_inst):
            constraints.append(f"reset c[{inst}]")
            constraints.append(f"c - c[{inst}] <= {ticket_lifetimes[gen_ticket]}")

    # 2) lifetime bounds for any instantiated ticket that appears in the message
    for base, L in ticket_lifetimes.items():
        # collect *all* instances base#N in this message (not only base#sid)
        for m in re.finditer(rf"\b{re.escape(base)}#(\d+)\b", msg_inst):
            inst = f"{base}#{m.group(1)}"
            constraints.append(f"c - c[{inst}] <= {L}")

    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for c in constraints:
        if c not in seen:
            uniq.append(c); seen.add(c)

    return ", ".join(uniq) if uniq else "-"



def _expand_tc_extra(tc_extra: str,
                     sid: int,
                     vars_to_inst: List[str],
                     ticket_lifetimes: Dict[str, int],
                     aliases: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Expand extra time constraints specified in JSON overrides.

    Supported forms inside tc_extra (comma-separated):
      - stale(X) where X is either a base name (e.g., Ta) or an instance (e.g., Ta#1).
        Expands to: c - c[X] >= L+1, where L is the lifetime of base(X) in ticket_lifetimes.
      - raw tc atoms, e.g. "c - c[Ta#1] >= 21" or "reset c[Ta#1]"

    Notes:
      - We instantiate bases without # using sid and optional aliases.
      - This requires the downstream protoc_to_nta tool to support parsing >= guards.
    """
    if not tc_extra:
        return []
    out: List[str] = []
    parts = [p.strip() for p in str(tc_extra).split(",") if p.strip()]
    for p in parts:
        m = re.match(r"stale\s*\(\s*([A-Za-z_][A-Za-z0-9_#@]*)\s*\)\s*$", p)
        if m:
            tok = m.group(1).strip()
            # instantiate if needed
            if "#" not in tok:
                base = tok
                tok = aliases.get(base, f"{base}#{sid}") if aliases else f"{base}#{sid}"
            base = tok.split("#", 1)[0]
            if base not in ticket_lifetimes:
                raise ValueError(f"tc_extra stale({m.group(1)}) requires lifetime for base '{base}' in --ticket-lifetimes")
            out.append(f"c - c[{tok}] >= {ticket_lifetimes[base] + 1}")
            continue

        # allow user to specify bare bases inside c[...] too; instantiate them
        # Example: "c - c[Ta] >= 21" -> "c - c[Ta#sid] >= 21"
        def _inst_in_brackets(s: str) -> str:
            def repl(mm):
                inner = mm.group(1)
                if "#" in inner:
                    return mm.group(0)
                base = inner
                inst = aliases.get(base, f"{base}#{sid}") if aliases else f"{base}#{sid}"
                return f"c[{inst}]"
            return re.sub(r"c\[\s*([A-Za-z_][A-Za-z0-9_#@]*)\s*\]", repl, s)

        out.append(_inst_in_brackets(p))
    return out


def load_interpretations(path: str) -> List[Dict[str, Any]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    runs = obj.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("interpretations JSON must have a top-level 'runs' list")
    return runs


def extract_ticket_aliases(overrides: List[Dict[str, Any]],
                           sid: int,
                           ticket_bases: List[str]) -> Dict[str, str]:
    """Infer per-session ticket aliasing from overrides.

    Example: if any override for this sid contains "Tb#2", then we set
    aliases["Tb"] = "Tb#2" and use that instance for *all* occurrences of Tb
    in this sid/run when instantiating template messages.
    """
    aliases: Dict[str, str] = {}
    for ov in overrides:
        if not isinstance(ov, dict):
            continue
        if int(ov.get("sid", sid)) != sid:
            continue
        hay = ""
        if "L" in ov:
            hay += " " + str(ov["L"])
        if "G" in ov:
            hay += " " + str(ov["G"])
        for base in ticket_bases:
            m = re.search(rf"\b{re.escape(base)}#(\d+)\b", hay)
            if m and base not in aliases:
                aliases[base] = f"{base}#{m.group(1)}"
    return aliases


def apply_edge_override(default_sender: str, default_receiver: str, ov: Dict[str, Any]) -> Tuple[str, str]:
    sender, receiver = default_sender, default_receiver
    if "edge" in ov and isinstance(ov["edge"], str) and "->" in ov["edge"]:
        a, b = [x.strip() for x in ov["edge"].split("->", 1)]
        if a: sender = a
        if b: receiver = b
    if "sender" in ov:
        sender = str(ov["sender"])
    if "receiver" in ov:
        receiver = str(ov["receiver"])
    return sender, receiver


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ab_file", help="Alice-Bob protocol file (.ab)")
    ap.add_argument("--interpretations", required=True, help="JSON describing selected runs/attacks")
    ap.add_argument("--k", type=int, default=1, help="number of sessions (sid=1..k)")
    ap.add_argument("--delays", default="", help='per-step delays "1:2;2:3;..."')
    ap.add_argument("--shared-keys", default="", help='shared keys like "KAS:A,S;KBS:B,S;KBI:B,I" (metadata only here)')
    ap.add_argument("--ticket-lifetimes", default="", help='ticket lifetimes like "Tb:30;Ta:20"')
    ap.add_argument("--session-vars", default="", help='extra per-session variables to instantiate like "KAB;Na" (in addition to ticket vars)')
    ap.add_argument("--output", required=True, help="output .protoc file")
    ap.add_argument("--only-runs", default="", help="comma-separated run names to include (filter interpretations JSON)")
    args = ap.parse_args()

    steps = parse_ab_file(args.ab_file)
    if not steps:
        raise SystemExit("No steps parsed from .ab file")

    # parse delays
    delays: Dict[int, int] = {}
    if args.delays.strip():
        for part in args.delays.split(";"):
            part = part.strip()
            if not part:
                continue
            a, b = part.split(":")
            delays[int(a.strip())] = int(b.strip())

    # parse ticket lifetimes
    ticket_lifetimes: Dict[str, int] = {}
    if args.ticket_lifetimes.strip():
        for part in args.ticket_lifetimes.split(";"):
            part = part.strip()
            if not part:
                continue
            name, val = part.split(":")
            ticket_lifetimes[name.strip()] = int(val.strip())

    ticket_vars = collect_ticket_vars(ticket_lifetimes)
    extra_vars: List[str] = []
    if args.session_vars.strip():
        extra_vars = [v.strip() for v in re.split(r'[;,]', args.session_vars) if v.strip()]
    vars_to_inst = []
    seen = set()
    for v in (ticket_vars + extra_vars):
        if v not in seen:
            vars_to_inst.append(v)
            seen.add(v)
    runs = load_interpretations(args.interpretations)

    if args.only_runs.strip():
        wanted = {x.strip() for x in args.only_runs.split(",") if x.strip()}
        runs = [r for r in runs if str(r.get("name","")) in wanted]

    # Expand JSON runs into .protoc runs: one run per sid.
    # - If a JSON run has empty overrides: emit honest runs for sid=1..k
    # - If it has overrides: emit only the sids that appear in the overrides
    expanded: List[Tuple[str, int, List[Dict[str, Any]]]] = []  # (run_name, sid, overrides)
    for run in runs:
        rname = run.get("name", "run")
        overrides = run.get("overrides", [])
        if not isinstance(overrides, list):
            overrides = []
        if not overrides:
            sids = list(range(1, args.k + 1))
        else:
            sids = sorted({int(ov.get("sid", 1)) for ov in overrides if isinstance(ov, dict)})
            if not sids:
                sids = [1]
        for sid in sids:
            expanded.append((rname, sid, overrides))

    out_lines: List[str] = []
    out_lines.append(f"u=3;\n")  # metadata kept minimal
    out_lines.append(f"p=4;\n")
    out_lines.append(f"s=5;\n")
    out_lines.append(f"n={len(expanded)};\n")
    out_lines.append(f"# sessions k={args.k}, interpretations expanded into per-sid runs\n")
    if args.shared_keys:
        out_lines.append(f"# shared-keys: {args.shared_keys}\n")
    out_lines.append("protocol;\n")
    # protocol skeleton (internal form, uninstantiated)
    for st in steps:
        D = delays.get(st.i, 0)
        msg0 = ab_msg_to_internal(st.msg_raw)
        out_lines.append(f"{st.sender},{st.receiver};L:{msg0};D={D};G=-;\n")
    out_lines.append("\nexecutions;\n")

    for ri, (rname, sid, overrides) in enumerate(expanded, start=1):
        # index overrides by (sid, step)
        ov_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for ov in overrides:
            if not isinstance(ov, dict):
                continue
            ov_sid = int(ov.get("sid", 1))
            ov_step = int(ov.get("step", 1))
            ov_map[(ov_sid, ov_step)] = ov

        aliases = extract_ticket_aliases(overrides, sid, vars_to_inst)

        out_lines.append(f"// run r{ri}  # {rname} [sid={sid}]\n")
        for st in steps:
            D = delays.get(st.i, 0)

            # defaults
            sender, receiver = st.sender, st.receiver
            msg = ab_msg_to_internal(st.msg_raw)
            gen_ticket = infer_gen_ticket_from_raw(st.msg_raw)
            kind = "honest"
            G = "-"
            tc = "-"
            tc_extra = ""

            ov = ov_map.get((sid, st.i))
            if ov:
                # edge override first
                sender, receiver = apply_edge_override(sender, receiver, ov)
                tc_extra = ov.get("tc_extra", "")
                if "kind" in ov:
                    kind = str(ov["kind"])
                if "L" in ov:
                    msg = ab_msg_to_internal(str(ov["L"]))
                if "G" in ov:
                    G = str(ov["G"])
                if "tc" in ov:
                    tc = str(ov["tc"])

            # instantiate tickets inside message (with optional foreign-ticket alias)
            msg = instantiate_vars_in_msg(msg, sid, vars_to_inst, aliases=aliases)

            # Generator step: set G to instantiated ticket if applicable (unless override provided)
            if G == "-" and gen_ticket and kind != "intruder":
                inst = aliases.get(gen_ticket, f"{gen_ticket}#{sid}") if aliases else f"{gen_ticket}#{sid}"
                if re.fullmatch(rf"{re.escape(inst)}", msg):
                    G = inst

            # tc if not overridden
            if tc == "-":
                tc = make_tc_for_step(msg, sid, ticket_lifetimes, gen_ticket, kind, aliases=aliases)

            # append tc_extra (expanded macros / additional atoms)
            if tc_extra:
                extra_atoms = _expand_tc_extra(tc_extra, sid, vars_to_inst, ticket_lifetimes, aliases=aliases)
                if extra_atoms:
                    atoms = []
                    if tc and tc != "-":
                        atoms.extend([a.strip() for a in tc.split(",") if a.strip()])
                    atoms.extend(extra_atoms)
                    # de-duplicate preserving order
                    seen2 = set()
                    uniq2 = []
                    for a in atoms:
                        if a not in seen2:
                            uniq2.append(a); seen2.add(a)
                    tc = ", ".join(uniq2) if uniq2 else "-"

            # instantiate tickets in G if user provided bare G
            if G not in ("-", ""):
                G = instantiate_vars_in_msg(G, sid, vars_to_inst, aliases=aliases)

            out_lines.append(
                f"sid={sid};step={st.i};{sender}->{receiver};D={D};kind={kind};tc={tc};G={G};L:{msg};\n"
            )
        out_lines.append("\n")

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
    main()