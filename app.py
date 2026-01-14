import os
import re
import json
import uuid
import time
import shutil
import tempfile
import threading
import subprocess
import sys
from pathlib import Path

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple

from flask import Flask, render_template, request, redirect, url_for, jsonify, abort, send_file

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(APP_DIR, "tools")
VAR_DIR = os.path.join(APP_DIR, "var")
JOBS_DIR = os.path.join(VAR_DIR, "jobs")

AB_TO_PROTOC = os.path.join(TOOLS_DIR, "ab_to_protoc.py")
PROTOC_TO_NTA = os.path.join(TOOLS_DIR, "protoc_to_nta.py")
NTA_TO_TIS = os.path.join(TOOLS_DIR, "nta_to_tis.py")
GEN_EF = os.path.join(TOOLS_DIR, "gen_ef_formulas.py")

# BMC + witness (kept as your tools expect)
SMTREACH4TIS = os.path.join(TOOLS_DIR, "smtreach4tis")
SMTREACH4TIIS = os.path.join(TOOLS_DIR, "smtreach4tiis")
RUN_SMTREACH_AND_WITNESS = os.path.join(TOOLS_DIR, "run_smtreach_and_witness.py")
GEN_WIT = os.path.join(TOOLS_DIR, "gen_wit_z3.py")
GEN_WIT_TIIS = os.path.join(TOOLS_DIR, "gen_wit_z3_tiis.py")

os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "local-dev"


# ------------------------------ TIS graph model + parser (Cytoscape view) ------------------------------

@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    action: int
    guards: List[str] = field(default_factory=list)
    resets: List[str] = field(default_factory=list)

@dataclass
class AutomatonModel:
    num: int                 # automaton id in .tis (0..)
    kind: str                # 'execution' | 'knowledge' | 'clock' | 'environment' | 'other'
    title: str
    k_signature: Optional[str] = None
    locations: Set[int] = field(default_factory=set)
    init: int = 0
    edges: List[Edge] = field(default_factory=list)

    def add_edge(self, e: Edge) -> None:
        self.edges.append(e)
        self.locations.add(e.src)
        self.locations.add(e.dst)

@dataclass
class TisNetwork:
    automata: List[AutomatonModel]

_AUT_HEADER_RE = re.compile(r"^#\s*Automaton\s*#(?P<num>\d+)\s*[—-]\s*(?P<title>.*)\s*$")

def parse_tis(path: str) -> TisNetwork:
    """Parse .tis written by nta_to_tis_with_comments.py."""
    if not os.path.exists(path):
        return TisNetwork(automata=[])

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    automata: List[AutomatonModel] = []
    pending_comment_lines: List[str] = []

    def guess_kind(num: int, title: str) -> str:
        tl = (title or "").lower()
        if num == 0:
            return "environment"
        if "execution" in tl or "automat wykonaniowy" in tl:
            return "execution"
        if "knowledge" in tl or "wiedzowy" in tl:
            return "knowledge"
        if "clock" in tl or "zegar" in tl:
            return "clock"
        return "other"

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("#"):
            pending_comment_lines.append(lines[i])
            i += 1
            continue

        if s == "automaton":
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("#"):
                continue
            try:
                num = int(lines[i].strip().lstrip("#").strip())
            except ValueError:
                num = len(automata)
            i += 1

            title = ""
            k_sig = None
            for cl in reversed(pending_comment_lines):
                m = _AUT_HEADER_RE.match(cl.strip())
                if m:
                    if int(m.group("num")) == num:
                        title = m.group("title").strip()
                        break
                    if not title:
                        title = m.group("title").strip()
            if title:
                mk = re.search(r"(K\([^)]+\))", title)
                if mk:
                    k_sig = mk.group(1)

            kind = guess_kind(num, title)
            aut = AutomatonModel(num=num, kind=kind, title=title, k_signature=k_sig)
            pending_comment_lines = []

            while i < len(lines):
                s2 = lines[i].strip()
                if s2 == "end":
                    i += 1
                    if i < len(lines) and lines[i].startswith("#") and set(lines[i].lstrip("#")) <= set("-"):
                        i += 1
                    break

                if s2.startswith("location "):
                    parts = s2.split()
                    if len(parts) >= 2:
                        try:
                            aut.locations.add(int(parts[1]))
                        except ValueError:
                            pass
                    i += 1
                    while i < len(lines) and lines[i].strip() != "end":
                        i += 1
                    if i < len(lines) and lines[i].strip() == "end":
                        i += 1
                    continue

                if s2.startswith("transition "):
                    parts = s2.split()
                    if len(parts) >= 4:
                        try:
                            src = int(parts[1]); dst = int(parts[2]); act = int(parts[3])
                        except ValueError:
                            i += 1
                            continue
                        guards: List[str] = []
                        resets: List[str] = []
                        i += 1
                        while i < len(lines) and lines[i].strip() != "end":
                            ln = lines[i].strip()
                            if ln.startswith("reset "):
                                resets.append(ln)
                            elif ln.startswith("guard "):
                                guards.append(ln[len("guard "):].strip())
                            elif ln and not ln.startswith("#"):
                                # w Twoich .tis guardy są jako gołe constrainty typu "x0 >= 2"
                                guards.append(ln)

                            i += 1
                        if i < len(lines) and lines[i].strip() == "end":
                            i += 1
                        aut.add_edge(Edge(src=src, dst=dst, action=act, guards=guards, resets=resets))
                        continue

                if s2.startswith("valuation"):
                    i += 1
                    while i < len(lines) and lines[i].strip() != "end":
                        i += 1
                    if i < len(lines) and lines[i].strip() == "end":
                        i += 1
                    continue

                i += 1

            aut.locations.add(aut.init)
            automata.append(aut)
            continue

        i += 1

    automata.sort(key=lambda a: a.num)
    return TisNetwork(automata=automata)

def edge_label(e: Edge) -> str:
    """Multiline edge label: action on first line, then time guards and resets."""
    out = [str(e.action)]
    for g in e.guards:
        g = (g or "").strip()
        if g:
            out.append(g)
    for r in e.resets:
        r = (r or "").strip()
        if r:
            out.append(r)
    return "\n".join(out)

def compute_automaton_positions(aut: AutomatonModel, scale_x: float = 115.0, scale_y: float = 62.0) -> Dict[str, Dict[str, float]]:
    nodes = sorted(aut.locations)
    adj: Dict[int, List[int]] = {}
    for e in aut.edges:
        adj.setdefault(e.src, []).append(e.dst)

    from collections import deque
    INF = 10**9
    dist = {n: INF for n in nodes}
    dist[aut.init] = 0
    q = deque([aut.init])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if dist.get(v, INF) > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)

    maxd = max([d for d in dist.values() if d < INF] + [0])
    layers: Dict[int, List[int]] = {}
    for n in nodes:
        d = dist.get(n, INF)
        if d >= INF:
            d = maxd + 1
        layers.setdefault(d, []).append(n)

    pos: Dict[str, Dict[str, float]] = {}
    for layer in sorted(layers.keys()):
        lst = sorted(layers[layer])
        for j, n in enumerate(lst):
            pos[f"A{aut.num}_Q{n}"] = {"x": layer * scale_x, "y": j * scale_y}
    return pos

def offset_positions(pos: Dict[str, Dict[str, float]], dx: float, dy: float) -> Dict[str, Dict[str, float]]:
    return {k: {"x": v["x"] + dx, "y": v["y"] + dy} for k, v in pos.items()}

def cytoscape_full_network(net: TisNetwork) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    elements: List[Dict[str, Any]] = []
    positions: Dict[str, Dict[str, float]] = {}

    cols = 3
    pad_x = 520.0
    pad_y = 340.0

    for idx, aut in enumerate(net.automata):
        row = idx // cols
        col = idx % cols
        base_x = col * pad_x
        base_y = row * pad_y

        aut_pos = compute_automaton_positions(aut)
        aut_pos = offset_positions(aut_pos, base_x, base_y)
        positions.update(aut_pos)

        pid = f"AUT{aut.num}"
        title = aut.title.strip()
        if aut.kind == "knowledge" and aut.k_signature:
            label = f"Automaton #{aut.num} knowledge {aut.k_signature}"
        elif aut.kind == "execution":
            label = f"Automaton #{aut.num} execution"
        elif aut.kind == "environment":
            label = f"Automaton #{aut.num} environment"
        else:
            label = f"Automaton #{aut.num}" + (f" — {title}" if title else "")
        elements.append({"data": {"id": pid, "label": label, "kind": "cluster"}})

        for loc in sorted(aut.locations):
            nid = f"A{aut.num}_Q{loc}"
            elements.append({"data": {"id": nid, "label": str(loc), "parent": pid, "aut": aut.num, "kind": "state"}})
        for e in aut.edges:
            eid = f"A{aut.num}_E{e.src}_{e.dst}_{e.action}"
            elements.append({"data": {"id": eid, "source": f"A{aut.num}_Q{e.src}", "target": f"A{aut.num}_Q{e.dst}",
                                      "label": edge_label(e), "action": e.action}})
    return elements, positions


# ------------------------------ Witness (.wit) parsing + witness graph ------------------------------

@dataclass(frozen=True)
class WitnessStep:
    delta: float
    actions: List[int]
    locations: List[int]
    clocks: List[float]
    globaltime: float


_WIT_HDR_RE = re.compile(r"nrComp\s*=\s*(\d+)\s*,\s*nrOfClocks\s*=\s*(\d+)\s*,\s*LL\s*=\s*(\d+)\s*,\s*k\s*=\s*(\d+)")
_WIT_STEP_RE = re.compile(
    r"^\s*(\d+)\s*:\s*\(\s*([^,]+)\s*,\s*\(([^)]*)\)\s*,\s*\(([^)]*)\)\s*,\s*\(([^)]*)\)\s*,\s*([^)]+)\)\s*$"
)


def _parse_int_list_csv(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_float_list_csv(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_wit(path: str) -> Tuple[int, int, int, int, List[WitnessStep]]:
    """Parse witness in your format produced by gen_wit_from_out_v5.

    Returns (nrComp, nrOfClocks, LL, k, steps[0..k]).
    """
    if not path or not os.path.exists(path):
        return 0, 0, 0, 0, []

    txt = _tail_text(path, 2_000_000)
    nr_comp = nr_clk = ll = k = 0
    m = _WIT_HDR_RE.search(txt)
    if m:
        nr_comp, nr_clk, ll, k = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))

    steps: List[WitnessStep] = []
    for raw in txt.splitlines():
        mm = _WIT_STEP_RE.match(raw.strip())
        if not mm:
            continue
        _idx = int(mm.group(1))
        delta = float(mm.group(2).strip())
        actions = _parse_int_list_csv(mm.group(3))
        locs = _parse_int_list_csv(mm.group(4))
        clocks = _parse_float_list_csv(mm.group(5))
        gtime = float(mm.group(6).strip())
        steps.append(WitnessStep(delta=delta, actions=actions, locations=locs, clocks=clocks, globaltime=gtime))

    # Some generators may omit k or header; infer conservatively.
    if steps:
        k = max(k, len(steps) - 1)
        nr_comp = max(nr_comp, len(steps[0].actions), len(steps[0].locations))
        nr_clk = max(nr_clk, len(steps[0].clocks))
    return nr_comp, nr_clk, ll, k, steps


def witness_participating_automata(steps: List[WitnessStep], nr_comp: int) -> Set[int]:
    """Automata that participate either by location change or by firing an action."""
    part: Set[int] = set()
    if not steps or nr_comp <= 0:
        return part
    for j in range(nr_comp):
        prev_loc = steps[0].locations[j] if j < len(steps[0].locations) else 0
        for d in range(1, len(steps)):
            cur_loc = steps[d].locations[j] if j < len(steps[d].locations) else prev_loc
            act = steps[d].actions[j] if j < len(steps[d].actions) else 0
            if act != 0 or cur_loc != prev_loc:
                part.add(j)
                break
            prev_loc = cur_loc
    return part


def cytoscape_subset_network(
    net: TisNetwork,
    include_aut: List[int],
    layout_mode: str = "default",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Build Cytoscape elements+positions for a subset of automata.

    layout_mode:
      - "default": execution stacked vertically (top), knowledge in one row below
      - "witness": compact single-column stacking (best for the right witness panel)
    """
    keep = set(include_aut or [])
    autos = [a for a in net.automata if (a.num in keep)] if keep else list(net.automata)

    if not autos:
        return [], {}

    execs = [a for a in autos if a.kind == "execution"]
    knows = [a for a in autos if a.kind == "knowledge"]
    others = [a for a in autos if a.kind not in ("execution", "knowledge")]
    ordered = execs + knows + others

    elements: List[Dict[str, Any]] = []
    positions: Dict[str, Dict[str, float]] = {}

    def _place(aut: AutomatonModel, ox: float, oy: float) -> Tuple[float, float]:
        base = compute_automaton_positions(aut)
        positions.update(offset_positions(base, ox, oy))
        xs = [p["x"] for p in base.values()] or [0.0]
        ys = [p["y"] for p in base.values()] or [0.0]
        w = (max(xs) - min(xs)) + 140.0
        h = (max(ys) - min(ys)) + 140.0
        return w, h

    if layout_mode == "witness":
        y = 0.0
        for aut in ordered:
            _w, h = _place(aut, 0.0, y)
            y += h + 90.0
    else:
        pad_x = 470.0
        pad_y = 300.0

        # execution column
        y = 0.0
        for aut in execs:
            _place(aut, 0.0, y)
            y += pad_y

        # knowledge row
        base_y_kn = max(y, 1.0) + 40.0
        x = 0.0
        for aut in knows:
            _place(aut, x, base_y_kn)
            x += pad_x

        # others
        base_y_other = base_y_kn + pad_y
        x = 0.0
        for aut in others:
            _place(aut, x, base_y_other)
            x += pad_x

    for aut in ordered:
        pid = f"AUT{aut.num}"
        if aut.kind == "knowledge" and aut.k_signature:
            label = f"Automaton #{aut.num} knowledge {aut.k_signature}"
        elif aut.kind == "execution":
            label = f"Automaton #{aut.num} execution"
        elif aut.kind == "environment":
            label = f"Automaton #{aut.num} environment"
        else:
            label = f"Automaton #{aut.num}" + (f" — {aut.title.strip()}" if aut.title else "")

        elements.append({"data": {"id": pid, "label": label, "kind": "cluster"}})

        for loc in sorted(aut.locations):
            nid = f"A{aut.num}_Q{loc}"
            elements.append(
                {
                    "data": {
                        "id": nid,
                        "label": str(loc),
                        "parent": pid,
                        "aut": aut.num,
                        "kind": "state",
                    }
                }
            )

        for e in aut.edges:
            eid = f"A{aut.num}_E{e.src}_{e.dst}_{e.action}"
            elements.append(
                {
                    "data": {
                        "id": eid,
                        "source": f"A{aut.num}_Q{e.src}",
                        "target": f"A{aut.num}_Q{e.dst}",
                        "label": edge_label(e),
                        "action": e.action,
                        "aut": aut.num,
                        "kind": "edge",
                    }
                }
            )

    return elements, positions





def witness_highlight(net: TisNetwork, steps: List[WitnessStep], step_idx: int, include_aut: Set[int]) -> Tuple[Set[str], Set[str]]:
    """Return (active_nodes, active_edges) ids for Cytoscape."""
    active_nodes: Set[str] = set()
    active_edges: Set[str] = set()
    if not steps:
        return active_nodes, active_edges

    step_idx = max(0, min(step_idx, len(steps) - 1))

    # Node highlights: current locations
    cur = steps[step_idx]
    for j in include_aut:
        if j < len(cur.locations):
            active_nodes.add(f"A{j}_Q{cur.locations[j]}")

    # Edge highlights: transition taken from step_idx-1 -> step_idx
    if step_idx == 0:
        return active_nodes, active_edges

    prev = steps[step_idx - 1]
    for aut in net.automata:
        if aut.num not in include_aut:
            continue
        j = aut.num
        if j >= len(cur.locations) or j >= len(prev.locations) or j >= len(cur.actions):
            continue
        act = cur.actions[j]
        if act == 0:
            continue
        src = prev.locations[j]
        dst = cur.locations[j]
        act = cur.actions[j] if j < len(cur.actions) else 0

        # 1) exact
        cand = [e for e in aut.edges if e.src == src and e.dst == dst and e.action == act]

        # 2) fallback: jeśli wykonaniowy przeszedł do innego stanu, a akcja się nie zgadza
        if not cand and src != dst:
            cand = [e for e in aut.edges if e.src == src and e.dst == dst]

        # 3) ostatni fallback: jak nie ma dst, to chociaż po akcji
        if not cand and act != 0:
            cand = [e for e in aut.edges if e.src == src and e.action == act]

        for e in cand:
            active_edges.add(f"A{aut.num}_E{e.src}_{e.dst}_{e.action}")

    return active_nodes, active_edges

@dataclass
class PipelineStage:
    name: str
    cmd: List[str]
    outputs: List[str] = field(default_factory=list)


@dataclass
class PipelineJob:
    id: str
    created_at: float
    status: str  # queued|running|done|failed
    stage_index: int
    stages: List[PipelineStage]
    workdir: str
    log_path: str
    outputs: Dict[str, str] = field(default_factory=dict)
    error: str = ""
    output_dir: str = ""  # optional; where artifacts are written


PIPELINE_JOBS: Dict[str, PipelineJob] = {}
JOB_LOCKS: Dict[str, threading.Lock] = {}


def _safe_filename(name: str) -> str:
    name = os.path.basename(name or "")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name or "file"


def _write_log(log_path: str, line: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def _tail_text(path: str, max_chars: int) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(0, size - max_chars)
        f.seek(start)
        data = f.read()
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode(errors="replace")


def _run_cmd_stream(cwd: str, log_path: str, cmd: List[str]) -> int:
    _write_log(log_path, "$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _write_log(log_path, line.rstrip("\n"))
    return proc.wait()


def _job_to_view(job: PipelineJob) -> Dict[str, Any]:
    return {
        "id": job.id,
        "created_at": job.created_at,
        "status": job.status,
        "stage_index": job.stage_index,
        "stages_len": len(job.stages),
        "workdir": job.workdir,
        "output_dir": job.workdir,
        "outputs": job.outputs,
        "error": job.error,
    }


def _parse_efo_blocks(efo_path: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []
    if not os.path.exists(efo_path):
        return blocks

    cur: List[str] = []
    comment = ""
    with open(efo_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not cur:
                if line.strip().startswith("EF("):
                    cur = [line]
                    comment = ""
                continue
            if line.strip().startswith("end"):
                m = re.search(r"end\s*#\s*(.*)$", line)
                if m:
                    comment = m.group(1).strip()
                formula = "\n".join(cur) + "\nend"
                scenario = ""
                m2 = re.search(r"\bname=([^\s]+)", comment)
                if m2:
                    scenario = m2.group(1).strip()
                blocks.append(
                    {
                        "scenario": scenario or f"property{len(blocks)+1}",
                        "comment": comment,
                        "formula": formula,
                    }
                )
                cur = []
                comment = ""
            else:
                cur.append(line)
    return blocks


def _write_selected_efo(model_base: str, scenario: str, formula_text: str, out_dir: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario)
    out_path = os.path.join(out_dir, f"{model_base}-{safe}.efo")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formula_text.strip() + "\n")
    return out_path


def _find_latest_z3_out(search_dirs: List[str]) -> Optional[str]:
    """Pick the newest *.out file that looks like a Z3 model dump (define-fun...)."""
    cands: List[str] = []
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.endswith(".out"):
                cands.append(os.path.join(d, name))
    cands.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    for p in cands:
        try:
            txt = _tail_text(p, 200_000)
        except Exception:
            continue
        low = txt.lower()
        if "define-fun" in low and "sat" in low:
            return p
    return None


def _run_job(job: PipelineJob) -> None:
    lock = JOB_LOCKS[job.id]
    with lock:
        job.status = "running"
        job.error = ""
        for i, stage in enumerate(job.stages):
            job.stage_index = i
            _write_log(job.log_path, f"\n=== {stage.name} ===")
            rc = _run_cmd_stream(job.workdir, job.log_path, stage.cmd)
            if rc != 0:
                job.status = "failed"
                job.error = f"Stage failed: {stage.name} (exit={rc})"
                _write_log(job.log_path, job.error)
                return

            # No post-stage hooks needed; witness is generated by tools/run_smtreach_and_witness.py

        job.stage_index = len(job.stages)
        job.status = "done"


def _enqueue(job: PipelineJob) -> None:
    PIPELINE_JOBS[job.id] = job
    JOB_LOCKS[job.id] = threading.Lock()
    t = threading.Thread(target=_run_job, args=(job,), daemon=True)
    t.start()


@app.route("/", methods=["GET"])
def index():
    pipeline_jobs = list(PIPELINE_JOBS.values())
    pipeline_jobs.sort(key=lambda j: j.created_at, reverse=True)
    view_jobs = [_job_to_view(j) for j in pipeline_jobs]
    return render_template("index.html", pipeline_jobs=view_jobs)


@app.route("/pipeline/start", methods=["POST"])
def pipeline_start():
    if not os.path.exists(AB_TO_PROTOC):
        abort(400, "Missing tool: tools/ab_to_protoc.py")

    ab_f = request.files.get("ab")
    js_f = request.files.get("json")
    if not ab_f or not ab_f.filename:
        abort(400, "Missing .ab file")
    if not js_f or not js_f.filename:
        abort(400, "Missing interpretations .json file")

    # options
    k = (request.form.get("k", "1") or "1").strip()
    delays = (request.form.get("delays", "") or "").strip()
    shared_keys = (request.form.get("shared_keys", "") or "").strip()
    ticket_lifetimes = (request.form.get("ticket_lifetimes", "") or "").strip()
    session_vars = (request.form.get("session_vars", "") or "").strip()
    only_runs = (request.form.get("only_runs", "") or "").strip()
    output_dir = (request.form.get("output_dir", "") or "").strip()
    if output_dir:
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    job_id = uuid.uuid4().hex[:10]
    workdir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(workdir, exist_ok=True)
    log_path = os.path.join(workdir, "job.log")

    ab_path = os.path.join(workdir, _safe_filename(ab_f.filename))
    json_path = os.path.join(workdir, _safe_filename(js_f.filename))
    ab_f.save(ab_path)
    js_f.save(json_path)

    model_base = os.path.splitext(os.path.basename(ab_path))[0]
    protoc_path = os.path.join(workdir, f"{model_base}.protoc")

    cmd = ["python", AB_TO_PROTOC, ab_path, "--interpretations", json_path, "--k", k, "--output", protoc_path]
    if delays:
        cmd += ["--delays", delays]
    if shared_keys:
        cmd += ["--shared-keys", shared_keys]
    if ticket_lifetimes:
        cmd += ["--ticket-lifetimes", ticket_lifetimes]
    if session_vars:
        cmd += ["--session-vars", session_vars]
    if only_runs:
        cmd += ["--only-runs", only_runs]

    stages = [PipelineStage("AB → PROTOC", cmd, outputs=[os.path.basename(protoc_path)])]

    job = PipelineJob(
        id=job_id,
        created_at=time.time(),
        status="queued",
        stage_index=0,
        stages=stages,
        workdir=workdir,
        log_path=log_path,
        outputs={"protoc": protoc_path, "model_base": model_base},
        output_dir=output_dir,
    )
    _enqueue(job)
    return redirect(url_for("index"))


@app.route("/pipeline/gen_tis/<job_id>", methods=["POST"])
def pipeline_gen_tis(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        abort(404)

    if not os.path.exists(PROTOC_TO_NTA):
        abort(400, "Missing tool: tools/protoc_to_nta.py")
    if not os.path.exists(NTA_TO_TIS):
        abort(400, "Missing tool: tools/nta_to_tis.py")

    protoc_path = job.outputs.get("protoc")
    if not protoc_path or not os.path.exists(protoc_path):
        abort(400, "Missing .protoc output")

    model_base = os.path.splitext(os.path.basename(protoc_path))[0]
    workdir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(JOBS_DIR, job_id)
    nta_path = os.path.join(workdir, f"{model_base}.nta")
    tis_path = os.path.join(workdir, f"{model_base}.tis")

    # Append stages and run in background (single lock ensures no overlap)
    job.stages = job.stages + [
        PipelineStage("PROTOC → NTA (TA Network)", ["python", PROTOC_TO_NTA, protoc_path, "--out", nta_path]),
        PipelineStage("NTA → TIS (Timed Interpreted System)", ["python", NTA_TO_TIS, nta_path, tis_path]),
    ]
    job.outputs["nta"] = nta_path
    job.outputs["tis"] = tis_path
    _enqueue(job)
    return redirect(url_for("index"))


@app.route("/pipeline/gen_ef/<job_id>", methods=["POST"])
def pipeline_gen_ef(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        abort(404)
    if not os.path.exists(GEN_EF):
        abort(400, "Missing tool: tools/gen_ef_formulas.py")

    nta_path = job.outputs.get("nta")
    if not nta_path or not os.path.exists(nta_path):
        abort(400, "Missing .nta (generate TIS first)")

    model_base = os.path.splitext(os.path.basename(nta_path))[0]
    workdir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    efo_path = os.path.join(out_dir, f"{model_base}.efo")

    cmd = ["python", GEN_EF, nta_path, "-o", efo_path]
    # optional protoc context
    protoc_path = job.outputs.get("protoc")
    if protoc_path and os.path.exists(protoc_path):
        cmd += ["--protoc", protoc_path, "--include-run-comments"]

    job.stages = job.stages + [PipelineStage("Generate EF formulas (.efo)", cmd)]
    job.outputs["efo"] = efo_path
    _enqueue(job)
    return redirect(url_for("index"))


@app.route("/api/pipeline/<job_id>/ef_formulas", methods=["GET"])
def api_ef_formulas(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False}), 404
    efo = job.outputs.get("efo")
    if not efo or not os.path.exists(efo):
        return jsonify({"ok": True, "formulas": []})
    blocks = _parse_efo_blocks(efo)
    return jsonify({"ok": True, "formulas": blocks})


@app.route("/pipeline/select_ef/<job_id>", methods=["POST"])
def pipeline_select_ef(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        abort(404)
    efo = job.outputs.get("efo")
    if not efo or not os.path.exists(efo):
        abort(400, "No .efo generated")

    scenario = (request.form.get("scenario") or "").strip()
    if not scenario:
        abort(400, "Missing scenario")
    blocks = _parse_efo_blocks(efo)
    block = next((b for b in blocks if b["scenario"] == scenario), None)
    if block is None:
        abort(400, "Scenario not found")

    model_base = ""
    tis_path = job.outputs.get("tis")
    if tis_path:
        model_base = os.path.splitext(os.path.basename(tis_path))[0]
    else:
        model_base = os.path.splitext(os.path.basename(efo))[0]

    workdir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(JOBS_DIR, job_id)
    selected_path = _write_selected_efo(model_base, scenario, block["formula"], out_dir)
    job.outputs["selected_efo"] = selected_path
    job.outputs["selected_scenario"] = scenario
    _write_log(job.log_path, f"\nSelected formula: {scenario} -> {selected_path}")
    return redirect(url_for("index"))


@app.route("/api/pipeline/<job_id>/status", methods=["GET"])
def pipeline_status(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False}), 404
    return jsonify({"ok": True, "job": _job_to_view(job)})


@app.route("/api/pipeline/<job_id>/log", methods=["GET"])
def pipeline_log(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "log": ""}), 404
    tail = int(request.args.get("tail", "20000"))
    return jsonify({"ok": True, "log": _tail_text(job.log_path, tail)})


@app.route("/api/pipeline/<job_id>/preview/<kind>", methods=["GET"])
def pipeline_preview(job_id: str, kind: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False}), 404
    path = job.outputs.get(kind)
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "text": ""}), 404
    max_chars = int(request.args.get("max", "20000"))
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read(max_chars)
    return jsonify({"ok": True, "text": txt})


@app.route("/pipeline/download/<job_id>/<kind>", methods=["GET"])
def pipeline_download(job_id: str, kind: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        abort(404)
    path = job.outputs.get(kind)
    if not path or not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))



def parse_tis_network(path: str) -> Dict[str, Any]:
    """Robust parser for .tis produced by nta_to_tis_with_comments.py.

    Format:
      network
      ...
      automaton
      #<id>
      ... (location blocks + transition blocks + optional valuation)
      end
      #---------------------------------
    """
    if not os.path.exists(path):
        return {"automata": []}

    lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()
    autos: List[Dict[str, Any]] = []
    i = 0
    pending_comments: List[str] = []

    def flush_comments() -> str:
        nonlocal pending_comments
        s = "\n".join(pending_comments).rstrip() + ("\n" if pending_comments else "")
        pending_comments = []
        return s

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("#") and "Automaton" in ln:
            pending_comments.append(ln)
            i += 1
            continue

        if ln.strip() == "automaton":
            # next non-empty line should be #id
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i >= len(lines) or not lines[i].strip().startswith("#"):
                # malformed, skip
                continue
            aid = lines[i].strip().lstrip("#").strip()
            i += 1

            body: List[str] = []
            while i < len(lines):
                if lines[i].strip() == "end":
                    # consume 'end' and possible separator line
                    i += 1
                    if i < len(lines) and lines[i].startswith("#") and set(lines[i].lstrip("#")) <= set("-"):
                        i += 1
                    break
                body.append(lines[i])
                i += 1

            # parse body into locations/transitions
            locs: Dict[str, Dict[str, Any]] = {}
            trans: List[Dict[str, Any]] = []

            j = 0
            while j < len(body):
                b = body[j].strip()
                if b.startswith("location "):
                    lid = b.split(None, 1)[1].strip()
                    raw = [body[j]]
                    j += 1
                    while j < len(body) and body[j].strip() != "end":
                        raw.append(body[j])
                        j += 1
                    if j < len(body) and body[j].strip() == "end":
                        raw.append(body[j])
                        j += 1
                    locs[lid] = {"id": lid, "raw": "\n".join(raw)}
                    continue

                if b.startswith("transition "):
                    parts = b.split()
                    # transition SRC DST ACT
                    src = parts[1] if len(parts) > 1 else "?"
                    dst = parts[2] if len(parts) > 2 else "?"
                    act = parts[3] if len(parts) > 3 else "?"
                    guard = ""
                    reset = ""
                    raw = [body[j]]
                    j += 1
                    while j < len(body) and body[j].strip() != "end":
                        raw.append(body[j])
                        if body[j].strip().startswith("guard "):
                            guard = body[j].strip()[len("guard "):].strip()
                        if body[j].strip().startswith("reset "):
                            reset = body[j].strip()[len("reset "):].strip()
                        j += 1
                    if j < len(body) and body[j].strip() == "end":
                        raw.append(body[j])
                        j += 1
                    trans.append({"src": src, "dst": dst, "act": act, "guard": guard, "reset": reset, "raw": "\n".join(raw)})
                    continue

                j += 1

            autos.append({
                "id": aid,
                "comments": flush_comments(),
                "locations": list(locs.values()),
                "transitions": trans,
            })
            continue

        # collect other comments lines (optional)
        if ln.strip().startswith("#"):
            pending_comments.append(ln)
        i += 1

    return {"automata": autos}

def layout_positions(parsed: Dict[str, Any], col_w: int = 340, row_h: int = 280) -> Dict[str, Dict[str, float]]:
    """Compute simple layered layout per automaton and place automata on a grid."""
    autos = parsed.get("automata", [])
    pos: Dict[str, Dict[str, float]] = {}
    cols = 3

    for idx, a in enumerate(autos):
        aid = str(a.get("id"))
        locs = a.get("locations", [])
        if not locs:
            continue

        # adjacency
        adj: Dict[str, List[str]] = {}
        for tr in a.get("transitions", []):
            adj.setdefault(tr["src"], []).append(tr["dst"])

        root = str(locs[0]["id"])
        # BFS levels
        level = {root: 0}
        q = [root]
        while q:
            u = q.pop(0)
            for v in adj.get(u, []):
                if v not in level:
                    level[v] = level[u] + 1
                    q.append(v)

        # group nodes by level
        by_lvl: Dict[int, List[str]] = {}
        for l in locs:
            lid = str(l["id"])
            lv = level.get(lid, 0)
            by_lvl.setdefault(lv, []).append(lid)

        # place automaton origin
        gx = (idx % cols) * col_w + 60
        gy = (idx // cols) * row_h + 60

        for lv in sorted(by_lvl.keys()):
            nodes = by_lvl[lv]
            for j, lid in enumerate(nodes):
                x = gx + lv * 120
                y = gy + j * 70
                pos[f"a{aid}_n_{lid}"] = {"x": x, "y": y}

    return pos

def cytoscape_elements_from_tis(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    els: List[Dict[str, Any]] = []
    autos = parsed.get("automata", [])
    positions = layout_positions(parsed)

    for a in autos:
        aid = str(a.get("id"))
        # label from comments if present
        label = f"Automaton #{aid}"
        if a.get("comments"):
            m = re.search(r"Automaton\s*#%s\s*[—-]\s*(.*)$" % re.escape(aid), a["comments"], flags=re.MULTILINE)
            if m:
                label = f"Automaton #{aid} — {m.group(1).strip()}"
        parent_id = f"p_{aid}"
        els.append({"data": {"id": parent_id, "label": label}})

        for loc in a.get("locations", []):
            lid = str(loc["id"])
            nid = f"a{aid}_n_{lid}"
            el = {"data": {"id": nid, "label": lid, "parent": parent_id}}
            if nid in positions:
                el["position"] = positions[nid]
            els.append(el)

        for ei, tr in enumerate(a.get("transitions", [])):
            src = f"a{aid}_n_{tr['src']}"
            dst = f"a{aid}_n_{tr['dst']}"
            parts = [str(tr.get("act",""))]
            if tr.get("guard"):
                parts.append(tr["guard"])
            if tr.get("reset"):
                parts.append("reset " + tr["reset"])
            label = "\n".join([p for p in parts if p])
            els.append({"data": {"id": f"a{aid}_e_{ei}", "source": src, "target": dst, "label": label}})
    return els

@app.route("/api/tis_graph/<job_id>", methods=["GET"])
def api_tis_graph(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    tis_path = job.outputs.get("tis")
    if not tis_path or not os.path.exists(tis_path):
        return jsonify({"ok": False, "error": "missing tis"}), 400

    net = parse_tis(tis_path)
    elements, positions = cytoscape_full_network(net)
    return jsonify({"ok": True, "elements": elements, "positions": positions})


@app.route("/api/witness_status/<job_id>", methods=["GET"])
def api_witness_status(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    workdir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(JOBS_DIR, job_id)
    out_dir_p = Path(out_dir)
    matches = list(out_dir_p.glob(f"*-k*.wit"))
    if len(matches) == 1:
        job.outputs["wit"] = os.path.join(out_dir, str(matches[0]))
    wit_path = job.outputs.get("wit")
    if not wit_path or not os.path.exists(wit_path):
        return jsonify({"ok": True, "has_witness": False})

    nr_comp, nr_clk, ll, k, steps = parse_wit(wit_path)
    part = sorted(witness_participating_automata(steps, nr_comp))
    return jsonify(
        {
            "ok": True,
            "has_witness": True,
            "wit_path": wit_path,
            "nrComp": nr_comp,
            "nrOfClocks": nr_clk,
            "LL": ll,
            "k": k,
            "steps": len(steps),
            "participating": part,
        }
    )


@app.route("/api/witness_graph/<job_id>", methods=["GET"])
def api_witness_graph(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404

    workdir = os.path.join(JOBS_DIR, job_id)
    out_dir = os.path.join(JOBS_DIR, job_id)
    out_dir_p = Path(out_dir)
    matches = list(out_dir_p.glob(f"*-k*.wit"))
    job.outputs["wit"] = os.path.join(out_dir, str(matches[0]))
    tis_path = job.outputs.get("tis")
    wit_path = job.outputs.get("wit")
    if not tis_path or not os.path.exists(tis_path):
        return jsonify({"ok": False, "error": "missing tis"}), 400
    if not wit_path or not os.path.exists(wit_path):
        return jsonify({"ok": False, "error": "missing witness"}), 400

    step = request.args.get("step", "0").strip()
    try:
        step_idx = int(step)
    except ValueError:
        step_idx = 0

    include_csv = (request.args.get("include", "") or "").strip()
    include: Set[int] = set()
    if include_csv:
        try:
            include = {int(x) for x in include_csv.split(",") if x.strip() != ""}
        except ValueError:
            include = set()

    nr_comp, _nr_clk, _ll, _k, steps = parse_wit(wit_path)
    if not steps:
        return jsonify({"ok": False, "error": "empty witness"}), 400

    if not include:
        include = witness_participating_automata(steps, nr_comp)
        if not include:
            include = set(range(min(nr_comp, 1)))

    net = parse_tis(tis_path)
    include_list = sorted({a for a in include if any(aut.num == a for aut in net.automata)})
    if not include_list:
        include_list = [aut.num for aut in net.automata[:1]]

    elements, positions = cytoscape_subset_network(net, include_list, layout_mode="witness")
    active_nodes, active_edges = witness_highlight(net, steps, step_idx, set(include_list))
    step_idx = max(0, min(step_idx, len(steps) - 1))

    cur = steps[step_idx]
    action_vec = cur.actions
    loc_vec = cur.locations

    return jsonify(
        {
            "ok": True,
            "elements": elements,
            "positions": positions,
            "active_nodes": sorted(active_nodes),
            "active_edges": sorted(active_edges),
            "step": step_idx,
            "steps": len(steps),
            "delta": cur.delta,
            "globaltime": cur.globaltime,
            "action_vec": action_vec,
            "loc_vec": loc_vec,
            "include": include_list,
        }
    )


@app.route("/tis/<job_id>", methods=["GET"])
def show_tis(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        abort(404)
    tis_path = job.outputs.get("tis")
    if not tis_path or not os.path.exists(tis_path):
        abort(400, "No .tis generated")
    return render_template("tis.html", job_id=job_id, tis_path=tis_path)



@app.route("/pipeline/run_bmc/<job_id>", methods=["POST"])
def pipeline_run_bmc(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        abort(404)

    engine = (request.form.get("engine", "") or "").strip().lower() or "tis"
    if engine not in {"tis", "tiis"}:
        abort(400, "Invalid engine (expected: tis or tiis)")

    # Your preferred flow: smtreach -> z3 -> gen_wit (inside tools/run_smtreach_and_witness.py)
    smtreach_bin = SMTREACH4TIS if engine == "tis" else SMTREACH4TIIS
    gen_wit_py = GEN_WIT if engine == "tis" else GEN_WIT_TIIS

    if not os.path.exists(smtreach_bin):
        abort(400, f"Missing tool: {os.path.relpath(smtreach_bin, APP_DIR)}")
    if not os.path.exists(RUN_SMTREACH_AND_WITNESS):
        abort(400, "Missing tool: tools/run_smtreach_and_witness.py")
    if not os.path.exists(gen_wit_py):
        abort(400, f"Missing tool: {os.path.relpath(gen_wit_py, APP_DIR)}")

    tis_path = job.outputs.get("tis")
    sel_efo = job.outputs.get("selected_efo")
    model_base = job.outputs.get("model_base") or ""

    if not tis_path or not os.path.exists(tis_path):
        abort(400, "Missing .tis (generate TIS first)")
    if not sel_efo or not os.path.exists(sel_efo):
        abort(400, "No active formula selected")
    if not model_base:
        model_base = os.path.splitext(os.path.basename(tis_path))[0]

    steps = (request.form.get("bmc_steps", "") or "").strip()
    if not steps.isdigit() or int(steps) <= 0:
        abort(400, "Invalid k-path length")

    z3_bin = (request.form.get("z3_bin", "") or "").strip() or "z3"

    out_dir = os.path.abspath(job.workdir)
    os.makedirs(out_dir, exist_ok=True)

    # Performance .dat files (as in the older bmc_alg.py).
    tis_base = os.path.splitext(os.path.basename(tis_path))[0]
    efo_base = os.path.splitext(os.path.basename(sel_efo))[0]
    attack = efo_base
    if efo_base.startswith(tis_base + "-"):
        attack = efo_base[len(tis_base) + 1:]
    job.outputs["dat_reach"] = os.path.join(out_dir, f"{tis_base}_{attack}_{engine}_bmc.dat")
    job.outputs["dat_z3"] = os.path.join(out_dir, f"{tis_base}_{attack}_{engine}_z3.dat")
    job.outputs["dat_total"] = os.path.join(out_dir, f"{tis_base}_{attack}_{engine}_total.dat")

    # Expected artifacts (written by run_smtreach_and_witness.py)
    job.outputs["smt"] = os.path.join(out_dir, f"{model_base}-k{steps}.smt")
    job.outputs["z3_out"] = os.path.join(out_dir, f"{model_base}-k{steps}.out")
    job.outputs["wit"] = os.path.join(out_dir, f"{model_base}-k{steps}.wit")

    cmd = [
        sys.executable,
        RUN_SMTREACH_AND_WITNESS,
        "--engine",
        engine,
        "--smtreach",
        smtreach_bin,
        "--z3",
        z3_bin,
        "--gen_wit",
        gen_wit_py,
        "--out_dir",
        out_dir,
        "--model_base",
        model_base,
        "--k",
        steps,
        "--tis",
        tis_path,
        "--efo",
        sel_efo,
    ]

    job.stages = job.stages + [PipelineStage(f"SMT-BMC + witness (k={steps})", cmd)]
    _enqueue(job)
    return redirect(url_for("index"))



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


def _parse_dat_table(path: Path) -> List[Dict[str, Any]]:
    """Parse .dat timing table:
       # k | time(s) | mem(KB)
       0 0.04 0.0
       10 0.10 44608.0
    """
    rows: List[Dict[str, Any]] = []
    if not path or not path.exists():
        return rows

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            k = int(parts[0])
            t = float(parts[1])
            mem = float(parts[2])
        except ValueError:
            continue
        rows.append({"k": k, "time_s": t, "mem_kb": mem})
    return rows


def _find_first(pattern: str, out_dir: str) -> Optional[Path]:
    d = Path(out_dir)
    if not d.exists():
        return None
    matches = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


@app.route("/api/timings/<job_id>", methods=["GET"])
def api_timings(job_id: str):
    job = PIPELINE_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404

    out_dir = os.path.abspath(job.workdir)

    # Najczęściej te pliki powstają w pipeline_run_bmc:
    # job.outputs["dat_z3"] = "..._tis_z3.dat"
    # job.outputs["dat_reach"] = "..._tis_bmc.dat"
    # job.outputs["dat_total"] = "..._tis_total.dat"
    # Ale jak nie ma w outputs, to próbujemy znaleźć po globach w katalogu.
    z3_p = Path(job.outputs["dat_z3"]) if job.outputs.get("dat_z3") else _find_first("*_tis_z3.dat", out_dir)
    bmc_p = Path(job.outputs["dat_reach"]) if job.outputs.get("dat_reach") else _find_first("*_tis_bmc.dat", out_dir)
    tot_p = Path(job.outputs["dat_total"]) if job.outputs.get("dat_total") else _find_first("*_tis_total.dat", out_dir)

    return jsonify({
        "ok": True,
        "out_dir": out_dir,
        "files": {
            "z3": {"name": z3_p.name if z3_p else None, "rows": _parse_dat_table(z3_p) if z3_p else []},
            "bmc": {"name": bmc_p.name if bmc_p else None, "rows": _parse_dat_table(bmc_p) if bmc_p else []},
            "total": {"name": tot_p.name if tot_p else None, "rows": _parse_dat_table(tot_p) if tot_p else []},
        }
    })


if __name__ == "__main__":
    main()
