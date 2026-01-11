#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""protoc_to_nta_mirekstyle_v17_branching_gated.py

PRODOC -> NTA translator (Mirek/Kurkowski-style) with:
  - per-execution local delay clocks (one per execution automaton),
  - ticket age clocks (global per ticket instance, e.g., Tb#2),
  - knowledge automata (2-state) for DY-style learnability,
  - *intruder branching* for intruder-sent steps (habilitation-style):
      same control step can have multiple outgoing transitions with different
      labels, representing different generator-sets X.
  - optional *gating* of intruder actions using knowledge automata:
      a branch label is enabled only if the intruder already knows all elements
      of the corresponding generator-set.

This file is intentionally close to your v15_localdelay translator, but adds:
  (1) intruder branching labels for Enc/Concat terms,
  (2) branch-aware intruder knowledge update (replay vs construct),
  (3) gating transitions (state-1-only self-loops) in K(I,·) automata.

Notes:
  - We do NOT tag ticket instances with run names. Foreign-ticket scenarios
    (e.g., Tb#2 used in sid=1 but generated in sid=2) rely on sharing the same
    ticket instance name across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import argparse
import re


# -------------------------
# Term AST for L:<...>
# -------------------------

class Term:
    def pretty(self) -> str:
        raise NotImplementedError

    def atoms(self) -> Set["Atom"]:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Atom(Term):
    name: str

    def pretty(self) -> str:
        return self.name

    def atoms(self) -> Set["Atom"]:
        return {self}

    def size(self) -> int:
        return 1


@dataclass(frozen=True)
class Concat(Term):
    parts: Tuple[Term, ...]

    def pretty(self) -> str:
        return "|".join(p.pretty() for p in self.parts)

    def atoms(self) -> Set["Atom"]:
        out: Set[Atom] = set()
        for p in self.parts:
            out |= p.atoms()
        return out

    def size(self) -> int:
        return 1 + sum(p.size() for p in self.parts)


@dataclass(frozen=True)
class Enc(Term):
    key: Term
    payload: Term

    def pretty(self) -> str:
        return f"<{self.key.pretty()},{self.payload.pretty()}>"

    def atoms(self) -> Set["Atom"]:
        return self.key.atoms() | self.payload.atoms()

    def size(self) -> int:
        return 1 + self.key.size() + self.payload.size()


def _split_top(s: str, sep: str) -> List[str]:
    out, buf = [], []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [x for x in out if x]


def parse_L(msg: str) -> Term:
    msg = msg.strip()
    # concat (only if top-level '|')
    parts_top = _split_top(msg, "|")
    if len(parts_top) > 1:
        parts = [parse_L(p) for p in parts_top]
        return parts[0] if len(parts) == 1 else Concat(tuple(parts))
    # encryption <K,payload>
    if msg.startswith("<") and msg.endswith(">"):
        inside = msg[1:-1]
        parts = _split_top(inside, ",")
        if len(parts) != 2:
            raise ValueError(f"Bad Enc term: {msg}")
        return Enc(parse_L(parts[0]), parse_L(parts[1]))
    return Atom(msg)


# -------------------------
# DY closure (decryption, concat)
# -------------------------

def inv_key_name(name: str, public_keys: Set[str]) -> str:
    if name in public_keys:
        return name + "-"
    if name.endswith("-") and name[:-1] in public_keys:
        return name[:-1]
    return name


def inv_key(t: Term, public_keys: Set[str]) -> Term:
    return Atom(inv_key_name(t.name, public_keys)) if isinstance(t, Atom) else t


def closure_deconstruct(known: Set[Term], public_keys: Set[str], max_iters: int = 2000) -> Set[Term]:
    K = set(known)
    for _ in range(max_iters):
        new: Set[Term] = set()
        for t in list(K):
            if isinstance(t, Concat):
                for p in t.parts:
                    if p not in K:
                        new.add(p)
            if isinstance(t, Enc) and inv_key(t.key, public_keys) in K:
                if t.payload not in K:
                    new.add(t.payload)
        if not new:
            break
        K |= new
    return K


def atoms_in_closure(clos: Set[Term]) -> Set[str]:
    """Atoms known 'in the clear' are exactly the Atom-terms present in the closure."""
    return {t.name for t in clos if isinstance(t, Atom)}


# -------------------------
# Protoc parsing
# -------------------------


@dataclass
class Step:
    label: int
    sid: int
    step: int
    sender: str
    receiver: str
    D: int
    kind: str
    tc: str
    G: List[str]
    resets: List[str]
    uses: List[Tuple[str, int]]
    uses_ge: List[Tuple[str, int]]
    L_raw: str
    L_term: Term


@dataclass
class Run:
    name: str
    steps: List[Step]


def parse_protoc(path: str) -> List[Run]:
    txt = open(path, "r", encoding="utf-8").read().splitlines()
    in_exec = False
    runs: List[Run] = []
    cur_steps: List[Step] = []
    cur_name = "r1"
    next_label = 1

    for line in txt:
        s = line.strip()
        if s == "executions;":
            in_exec = True
            continue
        if not in_exec:
            continue

        if s.startswith("// run"):
            if cur_steps:
                runs.append(Run(cur_name, cur_steps))
                cur_steps = []
            m = re.search(r"r(\d+)", s)
            cur_name = f"r{m.group(1)}" if m else f"r{len(runs) + 1}"
            continue

        if not s or s.startswith("#"):
            continue
        if not s.startswith("sid="):
            continue

        parts = [p for p in s.split(";") if p]
        kv: Dict[str, str] = {}
        arrow = None
        Lraw = ""
        for p in parts:
            p = p.strip()
            if "->" in p and "=" not in p and not p.startswith("L:"):
                arrow = p
                continue
            if p.startswith("L:"):
                Lraw = p[2:].strip()
                continue
            if p.startswith("L="):
                Lraw = p[2:].strip()
                continue
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip()
            elif "->" in p:
                arrow = p

        if arrow is None:
            raise ValueError(f"Cannot parse sender->receiver in line: {s}")

        sender, receiver = [x.strip() for x in arrow.split("->", 1)]
        sid = int(kv.get("sid", "1"))
        stepno = int(kv.get("step", "0"))
        D = int(kv.get("D", "0"))
        kind = kv.get("kind", "honest")
        tc = kv.get("tc", "-")
        Graw = kv.get("G", "-").strip()

        G: List[str] = []
        if Graw and Graw != "-":
            G = [x.strip() for x in Graw.split(",") if x.strip()]

        resets: List[str] = []
        uses: List[Tuple[str, int]] = []
        uses_ge: List[Tuple[str, int]] = []
        if tc and tc != "-" and tc.lower() != "true":
            chunks = [c.strip() for c in tc.split(",") if c.strip()]
            for c in chunks:
                m = re.match(r"reset\s+c\[(.+?)\]\s*$", c)
                if m:
                    resets.append(m.group(1).strip())
                    continue
                m = re.match(r"c\s*-\s*c\[(.+?)\]\s*(?:<=|≤)\s*(\d+)\s*$", c)
                if m:
                    uses.append((m.group(1).strip(), int(m.group(2))))
                    continue
                m = re.match(r"c\s*-\s*c\[(.+?)\]\s*(?:>=|≥)\s*(\d+)\s*$", c)
                if m:
                    uses_ge.append((m.group(1).strip(), int(m.group(2))))
                    continue

        for g in G:
            if g not in resets:
                resets.append(g)

        term = parse_L(Lraw) if Lraw else Atom("ε")

        cur_steps.append(
            Step(
                label=next_label,
                sid=sid,
                step=stepno,
                sender=sender,
                receiver=receiver,
                D=D,
                kind=kind,
                tc=tc,
                G=G,
                resets=resets,
                uses=uses,
                uses_ge=uses_ge,
                L_raw=Lraw,
                L_term=term,
            )
        )
        next_label += 1

    if cur_steps:
        runs.append(Run(cur_name, cur_steps))
    if not runs:
        raise ValueError("No runs parsed from executions; section.")
    return runs


# -------------------------
# Keys / identities inference
# -------------------------


def normalize_actor(sender_field: str) -> str:
    if sender_field.startswith("I("):
        return "I"
    return sender_field


def infer_identities(runs: List[Run]) -> Set[str]:
    ids: Set[str] = {"I"}
    for r in runs:
        for st in r.steps:
            ids.add(normalize_actor(st.sender))
            ids.add(st.receiver)
    ids = {x for x in ids if re.fullmatch(r"[A-Za-z][A-Za-z0-9_#]*", x)}
    return ids


def infer_public_keys_from_atoms(atoms: Set[str]) -> Set[str]:
    return {a for a in atoms if re.fullmatch(r"[Kk][A-Za-z]", a)}


def infer_shared_participants(key: str, identities: Set[str]) -> Optional[Tuple[str, ...]]:
    m = re.fullmatch(r"[Kk]([A-Za-z]{2,})", key)
    if not m:
        return None
    letters = m.group(1)
    parts: List[str] = []
    for ch in letters:
        found = None
        for x in identities:
            if x == "I":
                if ch.upper() == "I":
                    found = "I"
                    break
            else:
                if x[-1].upper() == ch.upper():
                    found = x
                    break
        if found is None:
            return None
        parts.append(found)
    out: List[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return tuple(out) if len(out) >= 2 else None


def infer_shared_keys(key_atoms: Set[str], identities: Set[str], public_keys: Set[str]) -> Dict[str, Tuple[str, ...]]:
    shared: Dict[str, Tuple[str, ...]] = {}
    for k in key_atoms:
        if k in public_keys:
            continue
        parts = infer_shared_participants(k, identities)
        if parts:
            shared[k] = parts
    return shared


def owners_private_keys(public_keys: Set[str], identities: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {x: set() for x in identities}
    for k in public_keys:
        m = re.fullmatch(r"([Kk])([A-Za-z])", k)
        if not m:
            continue
        owner_letter = m.group(2)
        for x in identities:
            if x == "I" and owner_letter.upper() == "I":
                out[x].add(inv_key_name(k, public_keys))
            elif x != "I" and x[-1].upper() == owner_letter.upper():
                out[x].add(inv_key_name(k, public_keys))
    return out


def init_knowledge(
    identities: Set[str],
    public_keys: Set[str],
    shared_keys: Dict[str, Tuple[str, ...]],
    key_atoms: Set[str],
) -> Dict[str, Set[Term]]:
    know: Dict[str, Set[Term]] = {x: set() for x in identities}
    know["Iproc"] = set()
    # identities known to all
    for a in identities:
        for x in identities:
            know[a].add(Atom(x))
    for x in identities:
        know["Iproc"].add(Atom(x))

    # public keys known to all
    for k in key_atoms:
        if k in public_keys:
            for a in identities:
                know[a].add(Atom(k))
            know["Iproc"].add(Atom(k))

    # shared keys known only to participants
    for k, parts in shared_keys.items():
        if k not in key_atoms:
            continue
        for p in parts:
            if p in identities:
                know[p].add(Atom(k))
        if "I" in parts:
            know["Iproc"].add(Atom(k))

    # private keys
    owners = owners_private_keys(public_keys, identities)
    for x, privs in owners.items():
        for pk in privs:
            know[x].add(Atom(pk))
            if x == "I":
                know["Iproc"].add(Atom(pk))
    return know


# -------------------------
# Simulate knowledge per run; compute "learn at label"
# -------------------------


def simulate_run_for_learn_labels(
    run: Run,
    identities: Set[str],
    public_keys: Set[str],
    shared_keys: Dict[str, Tuple[str, ...]],
    key_atoms: Set[str],
    knowledge_items: str,
) -> Dict[Tuple[str, str], Set[int]]:
    """Return mapping (agent, item) -> set(labels) where item becomes known.

    IMPORTANT for v17:
      This simulation is *branch-agnostic*. Later (in emit_network) we will
      adjust intruder learn-labels for intruder-sent steps to match branching
      semantics (replay vs construct).
    """

    know = init_knowledge(identities, public_keys, shared_keys, key_atoms)
    learned_at: Dict[Tuple[str, str], Set[int]] = {}

    # Track non-atomic message terms as well, if requested
    def _collect_terms(t: Term, acc: Set[Term]) -> None:
        acc.add(t)
        if isinstance(t, Concat):
            for p in t.parts:
                _collect_terms(p, acc)
        elif isinstance(t, Enc):
            _collect_terms(t.payload, acc)
            _collect_terms(t.key, acc)

    interest_terms: Set[Term] = set()
    if knowledge_items == "atoms+msgs":
        for st in run.steps:
            _collect_terms(st.L_term, interest_terms)
        interest_terms = {t for t in interest_terms if not isinstance(t, Atom)}

    cur_terms: Dict[str, Set[Term]] = {a: set() for a in identities}
    cur_terms["I"] = set()
    if knowledge_items == "atoms+msgs" and interest_terms:
        for a in identities:
            cur_terms[a] = closure_deconstruct(know[a], public_keys) & interest_terms
        cur_terms["I"] = closure_deconstruct(know["Iproc"], public_keys) & interest_terms

    cur_atoms: Dict[str, Set[str]] = {
        a: set(atoms_in_closure(closure_deconstruct(know[a], public_keys))) for a in identities
    }
    cur_atoms["I"] = set(atoms_in_closure(closure_deconstruct(know["Iproc"], public_keys)))

    for st in run.steps:
        actor = normalize_actor(st.sender)
        recv = st.receiver
        delivered = st.L_term

        # intruder observes delivered term
        know["Iproc"].add(delivered)

        # receiver gets delivered term
        if recv in know:
            know[recv].add(delivered)

        # honest sender: only if kind=honest and not intruder
        if st.kind == "honest" and actor in know and actor != "I":
            know[actor].add(delivered)
            for g in st.G:
                know[actor].add(Atom(g))

        # intruder send: (branch-agnostic) assume intruder has/learns delivered
        if st.kind == "intruder":
            know["Iproc"].add(delivered)

        for agent in set([actor, recv, "I"]):
            if agent not in identities and agent != "I":
                continue
            clos = closure_deconstruct(know["Iproc"], public_keys) if agent == "I" else closure_deconstruct(know[agent], public_keys)
            atoms_now = atoms_in_closure(clos)

            if knowledge_items == "atoms+msgs" and interest_terms:
                terms_now = clos & interest_terms
                new_terms = terms_now - cur_terms.get(agent, set())
                cur_terms[agent] = terms_now
                for t in new_terms:
                    learned_at.setdefault((agent, t.pretty()), set()).add(st.label)

            new_atoms = atoms_now - cur_atoms.get(agent, set())
            cur_atoms[agent] = atoms_now

            for sec in new_atoms:
                # only record likely secrets
                if sec in identities:
                    continue
                if sec in public_keys:
                    continue
                if not (sec.startswith(("T", "N")) or ("#" in sec) or (sec.startswith(("K", "k")) and len(sec) > 2)):
                    continue
                learned_at.setdefault((agent, sec), set()).add(st.label)

    return learned_at


# -------------------------
# Emit NTA
# -------------------------


def collect_ticket_lifetimes(runs: List[Run]) -> Dict[str, int]:
    L: Dict[str, int] = {}
    for r in runs:
        for st in r.steps:
            for tk, bound in st.uses:
                if tk not in L:
                    L[tk] = bound
                else:
                    L[tk] = min(L[tk], bound)
    return L


def collect_all_tickets(runs: List[Run]) -> List[str]:
    T: Set[str] = set()
    for r in runs:
        for st in r.steps:
            for tk in st.resets:
                T.add(tk)
            for tk, _ in st.uses:
                T.add(tk)
    return sorted(T)


def emit_network(
    runs: List[Run],
    inline_ticket_constraints: bool,
    include_knowledge_for: str = "all",
    knowledge_items: str = "atoms+msgs",
    intruder_branching: bool = True,
    intruder_gating: bool = True,
) -> str:
    tickets = collect_all_tickets(runs)
    _lifetimes = collect_ticket_lifetimes(runs)

    # base labels
    all_labels: List[int] = [st.label for r in runs for st in r.steps]
    all_labels_sorted = sorted(set(all_labels))

    # label -> human-readable step info (for comments)
    label_info: Dict[int, str] = {}
    label_step: Dict[int, Step] = {}
    for r in runs:
        for st in r.steps:
            Lshort = st.L_raw.strip()
            if len(Lshort) > 60:
                Lshort = Lshort[:57] + "..."
            label_info[st.label] = (
                f"{r.name}:sid={st.sid},step={st.step},{st.sender}->{st.receiver},"
                f"kind={st.kind},D={st.D},L={Lshort}"
            )
            label_step[st.label] = st

    # --- Intruder branching + prerequisites ---
    # branch_extra[base_label] = [extra_label1, extra_label2, ...]
    # prereq[label] = set(item_strings) required for intruder to take that label.
    branch_extra: Dict[int, List[int]] = {}
    prereq: Dict[int, Set[str]] = {}
    intruder_constructs: Set[int] = set()  # labels where intruder may *construct* L (so he learns L there)
    next_free_label = (max(all_labels_sorted) if all_labels_sorted else 0) + 1

    def _add_extra(base: int, tag: str, req_items: Set[str], constructs: bool) -> int:
        nonlocal next_free_label
        lab = next_free_label
        next_free_label += 1
        branch_extra.setdefault(base, []).append(lab)
        label_info[lab] = label_info.get(base, f"label={base}") + f", {tag}"
        label_step[lab] = label_step.get(base)
        prereq[lab] = set(req_items)
        if constructs:
            intruder_constructs.add(lab)
        return lab

    if intruder_branching:
        for r in runs:
            for st in r.steps:
                if st.kind != "intruder":
                    continue

                # Base (replay) branch: X={L}. Intruder must already know L.
                prereq[st.label] = {st.L_term.pretty()}

                # Structured alternatives
                t = st.L_term
                if isinstance(t, Enc):
                    # X={key,payload} (construct ciphertext)
                    _add_extra(
                        st.label,
                        "X=key,payload",
                        {t.key.pretty(), t.payload.pretty()},
                        constructs=True,
                    )
                    # if payload is a concat, also offer X={key,parts}
                    if isinstance(t.payload, Concat) and len(t.payload.parts) > 1:
                        _add_extra(
                            st.label,
                            "X=key,parts",
                            {t.key.pretty()} | {p.pretty() for p in t.payload.parts},
                            constructs=True,
                        )
                elif isinstance(t, Concat) and len(t.parts) > 1:
                    _add_extra(
                        st.label,
                        "X=parts",
                        {p.pretty() for p in t.parts},
                        constructs=True,
                    )

    # infer identities + keys for knowledge simulation
    identities = infer_identities(runs)
    key_atoms: Set[str] = set()
    for r in runs:
        for st in r.steps:
            for a in st.L_term.atoms():
                if a.name.startswith(("K", "k")):
                    key_atoms.add(a.name)
    public_keys = infer_public_keys_from_atoms(key_atoms)
    shared_keys = infer_shared_keys(key_atoms, identities, public_keys)

    # compute learn labels per run and union them
    learned_labels: Dict[Tuple[str, str], Set[int]] = {}
    for r in runs:
        part = simulate_run_for_learn_labels(r, identities, public_keys, shared_keys, key_atoms, knowledge_items)
        for k, labs in part.items():
            learned_labels.setdefault(k, set()).update(labs)

    # Make knowledge automata sensitive to extra branch labels similarly to base label.
    if intruder_branching and branch_extra:
        for k, labs in learned_labels.items():
            extra: Set[int] = set()
            for lab in list(labs):
                extra.update(branch_extra.get(lab, []))
            labs.update(extra)

    # --- Branch-aware correction for intruder knowledge ---
    # In habilitation-style semantics, an intruder-sent step does NOT teach the intruder
    # the sent message in the replay branch X={L}; he must already know it.
    # He may learn/construct L in the construct branches (e.g., X=key,payload).
    if intruder_branching:
        for r in runs:
            for st in r.steps:
                if st.kind != "intruder":
                    continue
                Ls = st.L_term.pretty()
                # remove base label from I learning L
                labs = learned_labels.get(("I", Ls))
                if labs and st.label in labs:
                    labs.discard(st.label)
                # add construct-branch labels as learning points for I
                for lab in branch_extra.get(st.label, []):
                    if lab in intruder_constructs:
                        learned_labels.setdefault(("I", Ls), set()).add(lab)

    # choose agents
    if include_knowledge_for == "I":
        learned_labels = {k: v for k, v in learned_labels.items() if k[0] == "I"}

    # --- Intruder gating (prerequisites) ---
    prereq_labels: Dict[Tuple[str, str], Set[int]] = {}
    if intruder_gating and prereq:
        for lab, items in prereq.items():
            for it in items:
                if not it or it == "ε":
                    continue
                if it in identities:
                    continue
                if it in public_keys:
                    continue
                prereq_labels.setdefault(("I", it), set()).add(lab)
        # ensure knowledge automata exist for prereq items even if never learned
        for key in prereq_labels.keys():
            learned_labels.setdefault(key, set())

    # clocks: one local delay clock per run + one global age clock per ticket
    ticket_base = len(runs)
    clock_of: Dict[str, int] = {tk: ticket_base + i for i, tk in enumerate(tickets)}
    nclocks = ticket_base + len(tickets)

    def clk(i: int) -> str:
        return "x0" if i == 0 else f"x{i}"

    out: List[str] = []
    out.append("network")
    out.append("")
    out.append(f"clocks {nclocks}")
    out.append("")
    out.append("# Clock convention (Mirek-style + per-step delays):")
    out.append(f"#   x0..x{ticket_base - 1} : local per-execution delay clocks (one per execution automaton)")
    out.append("#       delay clock is guarded by D=... on each execution transition and reset after the transition")
    for idx, r in enumerate(runs):
        out.append(f"#     delay(run#{idx + 1}:{r.name}) -> {clk(idx)}")
    if tickets:
        out.append(f"#   x{ticket_base}..x{nclocks - 1} : ticket/timestamp age clocks; x{{idx}} corresponds to c[<ticket>]")
        for tk in tickets:
            out.append(f"#     c[{tk}] -> {clk(clock_of[tk])}")
    out.append("")

    next_automaton = 1
    next_valuation = 0

    # --- execution automata ---
    for run_idx, r in enumerate(runs):
        m = len(r.steps)
        aid = next_automaton
        next_automaton += 1
        dclk = clk(run_idx)
        out.append(f"# Automaton #{aid} — automat wykonaniowy dla {r.name} (kolejność jak w pliku .protoc: sekcja executions; // run {r.name})")
        out.append(f"#   Interpretacja: przejście i-1 -> i wykonuje krok i tego runa (etykieta = numer label z .protoc).")
        out.append(f"#   Czas:")
        out.append(f"#     - {dclk} mierzy opóźnienie tego wykonania; na tranzycji label=L wymagamy {dclk} >= D(L), potem resetujemy {dclk}.")
        out.append(f"#     - zegary x_t dla ticketów/timestampów: reset przy wygenerowaniu (G=...) oraz sprawdzanie lifetime (tc: c-c[t] <= bound).")
        out.append(f"#   Kroki (label -> opis):")
        for st in r.steps:
            out.append(f"#     {st.label} -> " + label_info.get(st.label, ""))
            for lab in branch_extra.get(st.label, []):
                out.append(f"#     {lab} -> " + label_info.get(lab, ""))
        out.append(f"automaton #{aid}")
        for loc in range(m + 1):
            out.append(f"location {loc}")
            out.append("end")
            out.append("")

        for i, st in enumerate(r.steps, start=1):
            labs_for_step = [st.label] + branch_extra.get(st.label, [])

            guards: List[str] = [f"{dclk} >= {st.D}" if st.D > 0 else f"{dclk} >= 0"]
            for tk, bound in st.uses:
                xi = clock_of.get(tk)
                if xi is not None:
                    guards.append(f"{clk(xi)} <= {bound}")
            for tk, bound in getattr(st, 'uses_ge', []):
                xi = clock_of.get(tk)
                if xi is not None:
                    guards.append(f"{clk(xi)} >= {bound}")

            resets = [dclk]
            for tk in st.resets:
                xi = clock_of.get(tk)
                if xi is not None:
                    resets.append(clk(xi))

            for lab in labs_for_step:
                out.append(f"transition {i - 1} {i} {lab}")
                out.extend(guards)
                out.append("reset " + " ".join(resets))
                out.append("end")
                out.append("")

        out.append("valuation")
        for loc in range(m + 1):
            out.append(f"{loc} {next_valuation}")
            next_valuation += 1
        out.append("end")
        out.append("")
        out.append(f"end #automaton {aid}")
        out.append("")

    # --- knowledge automata ---
    for (agent, item), labs in sorted(learned_labels.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        aid = next_automaton
        next_automaton += 1
        learn_labs_sorted = sorted(set(labs))
        req_labs_sorted = sorted(set(prereq_labels.get((agent, item), set())))

        out.append(f"# Automaton #{aid} — automat wiedzowy K({agent},{item})")
        out.append(f"#   Stan 0: {agent} nie zna {item}.  Stan 1: {agent} zna {item}.")
        if learn_labs_sorted:
            out.append(f"#   0->1 na labelach: {', '.join(map(str, learn_labs_sorted))}")
        else:
            out.append(f"#   0->1 na labelach: -")
        if intruder_gating and agent == "I" and req_labs_sorted:
            out.append(f"#   (gating) label-e wymagające znajomości {item}: {', '.join(map(str, req_labs_sorted))}")
        out.append("automaton #" + str(aid))
        out.append("location 0")
        out.append("end")
        out.append("")
        out.append("location 1")
        out.append("end")
        out.append("")

        # learning transitions
        for lab in learn_labs_sorted:
            out.append(f"transition 0 1 {lab}")
            out.append("end")
            out.append("")
            out.append(f"transition 1 1 {lab}")
            out.append("end")
            out.append("")

        # gating transitions: only in known state
        if intruder_gating and agent == "I":
            for lab in req_labs_sorted:
                # Avoid duplicating an already-emitted 1->1 transition
                if lab in set(learn_labs_sorted):
                    continue
                out.append(f"transition 1 1 {lab}")
                out.append("end")
                out.append("")

        out.append("valuation")
        out.append(f"0 {next_valuation}")
        next_valuation += 1
        out.append(f"1 {next_valuation}")
        next_valuation += 1
        out.append("end")
        out.append("")
        out.append(f"end #automaton {aid}")
        out.append("")

    out.append("end #network")
    out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("protoc_file")
    ap.add_argument("--out", default="out.nta")
    ap.add_argument("--inline-ticket-constraints", action="store_true")
    ap.add_argument("--knowledge-agents", choices=["I", "all"], default="all")
    ap.add_argument("--knowledge-items", choices=["atoms", "atoms+msgs"], default="atoms+msgs")
    ap.add_argument("--no-intruder-branching", action="store_true", help="Disable intruder branching labels")
    ap.add_argument("--no-intruder-gating", action="store_true", help="Disable gating of intruder actions by K(I,·)")
    args = ap.parse_args()

    runs = parse_protoc(args.protoc_file)
    nta = emit_network(
        runs,
        inline_ticket_constraints=args.inline_ticket_constraints,
        include_knowledge_for=args.knowledge_agents,
        knowledge_items=args.knowledge_items,
        intruder_branching=(not args.no_intruder_branching),
        intruder_gating=(not args.no_intruder_gating),
    )
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(nta)
    print(f"Wrote {args.out} (runs={len(runs)})")


if __name__ == "__main__":
    main()
