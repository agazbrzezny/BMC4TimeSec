#!/usr/bin/env python3
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
import resource
from pathlib import Path


def _default_time_bin() -> str:
    """Return a GNU-time compatible binary if available.

    We prefer:
      - $TIME_BIN if set
      - gtime on macOS (brew)
      - /usr/bin/time on Linux

    If none works with GNU -f format, we will fall back to Python timing (memory=0).
    """
    env = os.environ.get("TIME_BIN")
    if env:
        return env
    if sys.platform == "darwin" and shutil.which("gtime"):
        return "gtime"
    return "/usr/bin/time"


def _parse_time_log(path: str) -> tuple[float, float]:
    """
    Parse GNU time log written as: "%M %U %S"
    Some implementations prepend e.g. "Command exited with non-zero status X".
    We therefore scan from the end and pick the last numeric triple.
    Returns: (mem_kb, cpu_time_s)
    """
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return 0.0, 0.0

    for ln in reversed(lines):
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            mem_kb = float(parts[0])
            user = float(parts[1])
            sys_ = float(parts[2])
            return mem_kb, (user + sys_)
        except ValueError:
            continue

    return 0.0, 0.0


def run(cmd, cwd=None, capture=False):
    if capture:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout
    p = subprocess.run(cmd, cwd=cwd)
    return p.returncode, ""


def run_timed(cmd, cwd, time_bin: str, log_path: str, capture: bool = False):
    """Run command wrapped by an external 'time' and return (rc, out, cpu_s, mem_kb).

    Prefers GNU-time style (-o/-f). On macOS, if GNU format is unsupported, falls back to
    '/usr/bin/time -l' and parses "maximum resident set size" from stderr redirected to log_path.

    As a last resort, falls back to Python wall-clock timing (mem_kb=0).
    """
    t0 = time.perf_counter()

    # 1) Try GNU time wrapper first (gtime or GNU /usr/bin/time on Linux).
    wrapped = [time_bin, "-o", log_path, "-f", "%M %U %S"] + cmd
    try:
        rc, out = run(wrapped, cwd=cwd, capture=capture)
        dt_wall = time.perf_counter() - t0
        mem_kb, t_cpu = _parse_time_log(log_path)
        if t_cpu <= 0.0:
            t_cpu = dt_wall
        return rc, out, t_cpu, mem_kb
    except FileNotFoundError:
        # time binary missing
        pass
    except Exception:
        # Non-GNU time (e.g., macOS /usr/bin/time) may not support -f / -o.
        pass

    # 2) macOS fallback: /usr/bin/time -l (writes stats to stderr)
    if sys.platform == "darwin" and os.path.basename(time_bin) == "time":
        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as logf:
                if capture:
                    p = subprocess.run(
                        [time_bin, "-l"] + cmd,
                        cwd=cwd,
                        stdout=subprocess.PIPE,
                        stderr=logf,
                        text=True,
                    )
                    out = p.stdout
                else:
                    p = subprocess.run([time_bin, "-l"] + cmd, cwd=cwd, stderr=logf)
                    out = ""
            dt_wall = time.perf_counter() - t0
            mem_kb, t_cpu = _parse_time_log(log_path)
            if t_cpu <= 0.0:
                t_cpu = dt_wall
            return p.returncode, out, t_cpu, mem_kb
        except Exception:
            pass

    # 3) Python timing fallback (no reliable RSS)
    t1 = time.perf_counter()
    rc, out = run(cmd, cwd=cwd, capture=capture)
    dt = time.perf_counter() - t1
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass
    return rc, out, dt, 0.0


def newest_matching(pattern, cwd):
    paths = glob.glob(os.path.join(cwd, pattern))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smtreach", required=True)
    ap.add_argument("--z3", default="z3")
    ap.add_argument("--gen_wit", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_base", required=True)
    ap.add_argument("--k", required=True)
    ap.add_argument("--tis", required=True)
    ap.add_argument("--efo", required=True)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    max_k = str(args.k)
    base = args.model_base

    # 4) .dat performance files (compat with the previous bmc_alg.py)
    tis_base = os.path.splitext(os.path.basename(args.tis))[0]
    efo_base = os.path.splitext(os.path.basename(args.efo))[0]
    attack = efo_base
    if efo_base.startswith(tis_base + "-"):
        attack = efo_base[len(tis_base) + 1:]

    dat_reach = os.path.join(out_dir, f"{tis_base}_{attack}_tis_bmc.dat")
    dat_z3 = os.path.join(out_dir, f"{tis_base}_{attack}_tis_z3.dat")
    dat_total = os.path.join(out_dir, f"{tis_base}_{attack}_tis_total.dat")

    if os.path.exists(dat_reach):
        os.remove(dat_reach)
    if os.path.exists(dat_z3):
        os.remove(dat_z3)
    if os.path.exists(dat_total):
        os.remove(dat_total)



    header = "# k | time(s) | mem(KB)\n"
    for fp in (dat_reach, dat_z3, dat_total):
        if not os.path.exists(fp):
            with open(fp, "w", encoding="utf-8") as f:
                f.write(header)

    time_bin = _default_time_bin()


    k = 0
    while k <= int(max_k):
        print(f"k={k}")
        expected_smt = os.path.join(out_dir, f"{base}-k{k}.smt")
        expected_out = os.path.join(out_dir, f"{base}-k{k}.out")
        expected_wit = os.path.join(out_dir, f"{base}-k{k}.wit")

        # 1) smtreach4tis (may create file itself, or print SMT to stdout)
        cmd = [args.smtreach, args.tis[:-4], args.efo[:-4], str(k)]
        reach_log = os.path.join(out_dir, f"{base}-k{k}.reach.time")
        print(cmd)
        rc, out, t_reach, m_reach = run_timed(cmd, cwd=out_dir, time_bin=time_bin, log_path=reach_log, capture=True)
        print(f'rc {rc}')
        if rc != 0:
            sys.stdout.write(out)
            sys.exit(rc)

        looks_like_smt = ("(set-logic" in out) or ("(assert" in out)
        if out and looks_like_smt and (not os.path.exists(expected_smt)):
            with open(expected_smt, "w", encoding="utf-8") as f:
                f.write(out)

        if not os.path.exists(expected_smt):
            found = newest_matching(f"{base}-k{k}.smt", out_dir) or newest_matching(f"*-k{k}.smt", out_dir)
            if found and found != expected_smt:
                os.replace(found, expected_smt)
            elif not found:
                with open(expected_smt, "w", encoding="utf-8") as f:
                    f.write(out or "")
        # 2) z3
        z3_log = os.path.join(out_dir, f"{base}-k{k}.z3.time")
        rc2, out2, t_z3, m_z3 = run_timed([args.z3, "-smt2", expected_smt], cwd=out_dir, time_bin=time_bin, log_path=z3_log, capture=True)
        with open(expected_out, "w", encoding="utf-8") as f:
            f.write(out2)
        print(f'm_z3 {m_z3}')
        #print(f'out2 {out2}')
        #if rc2 != 0:
        #    sys.exit(rc2)

        # 3) witness if sat
        first = (out2.strip().splitlines()[:1] or [""])[0].strip()
        if first == "sat":
            p = subprocess.run([sys.executable, args.gen_wit, expected_out], cwd=out_dir,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with open(expected_wit, "w", encoding="utf-8") as f:
                f.write(p.stdout)
            kk = int(k)
            t_total = float(t_reach) + float(t_z3)
            m_total = float(m_reach) + float(m_z3)
            with open(dat_reach, "a", encoding="utf-8") as f:
                f.write(f"{kk} {t_reach:.2f} {m_reach}\n")
            with open(dat_z3, "a", encoding="utf-8") as f:
                f.write(f"{kk} {t_z3:.2f} {m_z3}\n")
            with open(dat_total, "a", encoding="utf-8") as f:
                f.write(f"{kk} {t_total:.2f} {m_total}\n")
            break


    # Append the single measurement line for this k.
        kk = int(k)
        t_total = float(t_reach) + float(t_z3)
        m_total = float(m_reach) + float(m_z3)
        with open(dat_reach, "a", encoding="utf-8") as f:
            f.write(f"{kk} {t_reach:.2f} {m_reach}\n")
        with open(dat_z3, "a", encoding="utf-8") as f:
            f.write(f"{kk} {t_z3:.2f} {m_z3}\n")
        with open(dat_total, "a", encoding="utf-8") as f:
            f.write(f"{kk} {t_total:.2f} {m_total}\n")

        k += 2
    # Cleanup: lock files left by some solvers/toolchains.
    for lck in glob.glob(os.path.join(out_dir, "*.lck")):
        try:
            os.remove(lck)
        except OSError:
            pass

    for smt in glob.glob(os.path.join(out_dir, "*.smt")):
        try:
            os.remove(smt)
        except OSError:
            pass
    for outfiles in glob.glob(os.path.join(out_dir, "*.out")):
        try:
            os.remove(outfiles)
        except OSError:
            pass

    for timefiles in glob.glob(os.path.join(out_dir, "*.time")):
        try:
            os.remove(timefiles)
        except OSError:
            pass


    sys.exit(0)

if __name__ == "__main__":
    main()
