#!/usr/bin/env python3
"""
bench_to_tex.py — Run bench_sparse_skops_gpu and produce a LaTeX results file.

Usage:
    python3 bench_to_tex.py [options] [-- bench_args...]

Options:
    --bin  PATH   Path to bench_sparse_skops_gpu binary (auto-detected if omitted)
    --out  FILE   Output .tex file (default: bench_results.tex)
    --only LABEL  Comma-separated kernel labels to keep (substring match, case-insensitive)
    --no-run      Parse from stdin instead of running the binary

Examples:
    # Full default sweep, auto-detect binary:
    python3 bench_to_tex.py

    # Single config (vec_nnz dim_major dim_minor n_cols_a trials):
    python3 bench_to_tex.py -- 8 1000 5000 500 10

    # Only show fused kernels v26-v33 and the 2-step baseline:
    python3 bench_to_tex.py --only "2-step,v26,v27,v28,v29,v30,v31,v32,v33"

    # Pipe pre-captured output:
    cat saved_output.txt | python3 bench_to_tex.py --no-run
"""

import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CANDIDATE_BINS = [
    "/global/homes/v/vgeorgio/RandLAPACK_build/bin/bench_sparse_skops_gpu",
    "/global/homes/v/vgeorgio/RandLAPACK/bin/bench_sparse_skops_gpu",
]

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_cli(argv):
    opts = {"bin": None, "out": "bench_results.tex", "only": None, "no_run": False}
    bench_args = []
    i = 1
    passthrough = False
    while i < len(argv):
        a = argv[i]
        if passthrough:
            bench_args.append(a)
        elif a == "--":
            passthrough = True
        elif a == "--bin" and i + 1 < len(argv):
            opts["bin"] = argv[i + 1]; i += 1
        elif a == "--out" and i + 1 < len(argv):
            opts["out"] = argv[i + 1]; i += 1
        elif a == "--only" and i + 1 < len(argv):
            opts["only"] = [s.strip().lower() for s in argv[i + 1].split(",")]; i += 1
        elif a == "--no-run":
            opts["no_run"] = True
        else:
            print(f"Unknown option: {a}", file=sys.stderr)
            sys.exit(1)
        i += 1
    return opts, bench_args

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def find_bin(hint):
    if hint:
        p = Path(hint)
        if p.exists():
            return str(p)
        sys.exit(f"Binary not found: {hint}")
    for c in CANDIDATE_BINS:
        if Path(c).exists():
            return c
    sys.exit("bench_sparse_skops_gpu not found. Pass --bin PATH.")

def run_bench(bin_path, extra_args):
    cmd = [bin_path] + [str(a) for a in extra_args]
    print(f"Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"Benchmark exited with code {proc.returncode}")
    return proc.stdout

# ---------------------------------------------------------------------------
# Parse benchmark output
# ---------------------------------------------------------------------------
#
# Block structure:
#   --- vec_nnz=8  S=1000x5000  A=5000x500  nnz=40000  trials=5 ---
#   CPU sequential e2e (1 thread): 49284.3 µs  |  CPU OMP e2e (244 threads): 13631.2 µs
#   Both CPU paths ...
#   Phase           GPU min(us)  vs Sequential  vs OMP  GFlop/s
#   ---
#   Construction            18.0     81.94x         -        -
#   [double] --------
#   Fused v26 ...          195.6    252.03x    69.71x    204.5
#   ...

_RE_CONFIG = re.compile(
    r'^---\s+'
    r'(vec_nnz=\S+)\s+(S=\S+)\s+(A=\S+)\s+(nnz=\S+)\s+(trials=\S+)'
    r'\s+---'
)
_RE_E2E = re.compile(
    r'CPU sequential e2e \(1 thread\):\s*([\d.]+).*?CPU OMP e2e \((\d+) threads\):\s*([\d.]+)'
)
_RE_SECTION = re.compile(r'^\s+\[(.+?)\]\s*-+')
_RE_ROW = re.compile(
    r'^  (.+?)\s{2,}'          # label (non-greedy) + 2+ spaces
    r'([\d.]+)'                # GPU min us
    r'\s+([\d.]+x|-)'          # vs Sequential
    r'\s+([\d.]+x|-)'          # vs OMP
    r'\s+([\d.]+|-)'           # GFlop/s
    r'\s*$'
)

def _float_or_none(s):
    s = s.rstrip('x')
    return None if s == '-' else float(s)

def parse_output(text):
    """Return list of block dicts."""
    blocks = []
    block = None
    section = ""

    for line in text.splitlines():
        m = _RE_CONFIG.match(line)
        if m:
            if block:
                blocks.append(block)
            block = {
                "vec_nnz": m.group(1), "S": m.group(2),
                "A": m.group(3),       "nnz": m.group(4),
                "trials": m.group(5),
                "seq_us": None, "omp_us": None, "nthreads": 1,
                "rows": [],
            }
            section = ""
            continue

        if block is None:
            continue

        m = _RE_E2E.search(line)
        if m:
            block["seq_us"]   = float(m.group(1))
            block["nthreads"] = int(m.group(2))
            block["omp_us"]   = float(m.group(3))
            continue

        m = _RE_SECTION.match(line)
        if m:
            section = m.group(1)
            continue

        m = _RE_ROW.match(line)
        if m:
            block["rows"].append({
                "label":  m.group(1).strip(),
                "gpu_us": float(m.group(2)),
                "vs_seq": _float_or_none(m.group(3)),
                "vs_omp": _float_or_none(m.group(4)),
                "gflops": _float_or_none(m.group(5)),
                "section": section,
            })

    if block:
        blocks.append(block)

    return blocks

# ---------------------------------------------------------------------------
# Filter rows
# ---------------------------------------------------------------------------

def filter_rows(rows, only_patterns):
    if not only_patterns:
        return rows
    out = []
    for r in rows:
        if any(p in r["label"].lower() for p in only_patterns):
            out.append(r)
    return out

# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _esc(s):
    """Escape LaTeX special chars in plain text."""
    s = s.replace('\\', r'\textbackslash{}')
    for ch, rep in (('_', r'\_'), ('#', r'\#'), ('%', r'\%'),
                    ('&', r'\&'), ('{', r'\{'), ('}', r'\}'),
                    ('<', r'$<$'), ('>', r'$>$'), ('^', r'\^{}')):
        s = s.replace(ch, rep)
    return s

def _fmtx(val):
    """Format a speedup value (float or None)."""
    return '--' if val is None else f"{val:.2f}$\\times$"

def _fmtgf(val):
    """Format GFlop/s (float or None)."""
    return '--' if val is None else f"{val:.1f}"

def _fmtus(val):
    """Format microseconds."""
    return '--' if val is None else f"{val:.1f}"

# ---------------------------------------------------------------------------
# Emit one table per block
# ---------------------------------------------------------------------------

def block_to_tex(block, only_patterns):
    rows = filter_rows(block["rows"], only_patterns)
    if not rows:
        return ""

    lines = []

    # Caption metadata
    seq_us    = block["seq_us"]
    omp_us    = block["omp_us"]
    nthreads  = block["nthreads"]
    caption = (
        f"{_esc(block['S'])} sketch, {_esc(block['A'])} input, "
        f"{_esc(block['vec_nnz'])}, {_esc(block['nnz'])}. "
        r"Sequential e2e: \textbf{" + _fmtus(seq_us) + r"}\,\textmu s; "
        f"OMP e2e ({nthreads} threads): \\textbf{{{_fmtus(omp_us)}}}\\,\\textmu s. "
        r"Both baselines use pre-allocated CSC arrays (no malloc in timed region)."
    )
    label = (f"tab:{block['S']}_{block['A']}_{block['vec_nnz']}"
             .replace('=', '').replace('x', 'x').replace(',', ''))

    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"  \begin{tabular}{lrrrr}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Phase}"
        r" & \textbf{GPU min} ($\mu$s)"
        r" & \textbf{vs Sequential}"
        r" & \textbf{vs OMP}"
        r" & \textbf{GFlop/s} \\"
    )
    lines.append(r"    \midrule")

    prev_sec = None
    for row in rows:
        sec = row["section"]
        if sec != prev_sec:
            if prev_sec is not None:
                lines.append(r"    \midrule")
            if sec:
                lines.append(
                    r"    \multicolumn{5}{l}{\textit{"
                    + _esc(sec)
                    + r"}} \\"
                )
            prev_sec = sec

        lines.append(
            f"    {_esc(row['label'])}"
            f" & {_fmtus(row['gpu_us'])}"
            f" & {_fmtx(row['vs_seq'])}"
            f" & {_fmtx(row['vs_omp'])}"
            f" & {_fmtgf(row['gflops'])} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Write .tex document
# ---------------------------------------------------------------------------

TEX_PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage{booktabs}
\usepackage{geometry}
\usepackage{caption}
\usepackage{hyperref}
\geometry{margin=1in, landscape}
\captionsetup{font=small, labelfont=bf}

% Auto-generated by bench_to_tex.py
% Compile with: pdflatex bench_results.tex

\title{Sparse Sketch Benchmark: GPU vs CPU}
\date{\today}
\author{}

\begin{document}
\maketitle

\noindent
Each table covers one problem configuration ($\mathbf{S} \in \mathbb{R}^{m \times n}$,
$\mathbf{A} \in \mathbb{R}^{n \times k}$, $\mathbf{B} = \mathbf{S}\mathbf{A}$).
GPU timings use CUDA events (kernel only). CPU baselines use
\texttt{steady\_clock} on pre-allocated CSC arrays with no malloc in the timed region.
\textbf{vs Sequential}: speedup over 1-thread CSC SpMM (construction always serial).
\textbf{vs OMP}: speedup over max-thread OMP CSC SpMM.

"""

TEX_SUFFIX = r"""
\end{document}
"""

def write_tex(blocks, out_path, only_patterns):
    body_parts = []
    for block in blocks:
        t = block_to_tex(block, only_patterns)
        if t:
            body_parts.append(t)

    content = TEX_PREAMBLE + "\n\n".join(body_parts) + TEX_SUFFIX
    Path(out_path).write_text(content, encoding="utf-8")
    print(f"LaTeX written to: {out_path}  ({len(body_parts)} table(s))")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    opts, bench_args = parse_cli(sys.argv)

    if opts["no_run"]:
        raw = sys.stdin.read()
    else:
        bin_path = find_bin(opts["bin"])
        raw = run_bench(bin_path, bench_args)

    blocks = parse_output(raw)
    print(f"Parsed {len(blocks)} configuration block(s).")

    write_tex(blocks, opts["out"], opts["only"])

if __name__ == "__main__":
    main()
