#!/usr/bin/env python3
"""Generate a -serial directory for a benchmark from its -omp source.

Strategy:
  1. Copy every file from src/<bench>-omp/ into src/<bench>-serial/.
  2. In .cpp/.cc/.c/.h/.hpp files:
       - remove `#include <omp.h>`
       - remove all `#pragma omp ...` lines (both single-line and continued via `\`)
  3. Rewrite Makefile to plain g++ -std=c++17 -O3 (drop OMP flags, keep -I paths).
  4. Delete Makefile.aomp and Makefile.nvc if they exist (OMP-target compiler variants).

Usage:
  tools/omp_to_serial.py <bench-name>
  tools/omp_to_serial.py --all           # (skip existing)
"""
from __future__ import annotations
import argparse, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC  = REPO / "src"
SRC_EXT = {".cpp", ".cc", ".c", ".cxx", ".h", ".hpp", ".hh"}

OMP_PRAGMA_RE = re.compile(r"^\s*#\s*pragma\s+omp\b", re.M)
OMP_RT_RE     = re.compile(r"\bomp_(?:get|set|in)_[a-z_]+\b")

def strip_omp(text: str) -> tuple[str, bool]:
    """Remove #include <omp.h> and every #pragma omp ... block (with trailing
    backslash-continuation lines). Returns (new_text, needs_stub).

    needs_stub is True when the file still references omp_get_thread_num /
    omp_get_team_num / omp_get_wtime / etc. — meaning the -serial variant
    needs the stub header to link cleanly.
    """
    lines = text.splitlines(keepends=True)
    out = []
    skip_next = False
    for ln in lines:
        if skip_next:
            skip_next = ln.rstrip().endswith("\\")
            continue
        if re.match(r"^\s*#\s*include\s*[<\"]omp\.h[>\"]", ln):
            continue
        if OMP_PRAGMA_RE.match(ln):
            skip_next = ln.rstrip().endswith("\\")
            continue
        out.append(ln)
    stripped = "".join(out)
    needs_stub = bool(OMP_RT_RE.search(stripped))
    return stripped, needs_stub

MAKEFILE_TEMPLATE = """#===============================================================================
# Auto-generated serial Makefile (derived from -omp).
#===============================================================================

CC        = g++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

program = main
source  = {sources}
obj     = {objs}

CFLAGS  := $(EXTRA_CFLAGS) -std=c++17 -Wall {includes}

ifeq ($(DEBUG),yes)
  CFLAGS += -g
endif
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

LDFLAGS =

$(program): $(obj)
\t$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
\t$(CC) $(CFLAGS) -c $< -o $@

clean:
\trm -rf $(program) $(obj)

run: $(program)
\t$(LAUNCHER) ./$(program) {runargs}
"""

def make_serial_makefile(omp_makefile: Path, sources: list[str],
                         extra_includes: str = "") -> str:
    """Produce a plain g++ Makefile.

    We keep `-I../<bench>-cuda` style includes (many benchmarks depend on the
    cuda directory for a shared reference.h). We keep the `run:` args verbatim.
    """
    content = omp_makefile.read_text() if omp_makefile.exists() else ""
    include_flags = " ".join(sorted(set(re.findall(r"-I\S+", content))))
    define_flags = " ".join(sorted(set(re.findall(r"-D\S+", content))))
    # Drop OMP/GPU-only defines that shouldn't leak into serial.
    drop = {"-D__STRICT_ANSI__"}
    define_flags = " ".join(f for f in define_flags.split()
                            if f not in drop and not f.startswith("-D_OPENMP"))
    if extra_includes:
        include_flags = (include_flags + " " + extra_includes).strip()
    all_flags = " ".join(x for x in (include_flags, define_flags) if x)
    m = re.search(r"^run:.*?^\t.*?\./\$?\(?program\)?\.?\s*(.*)$",
                  content, flags=re.M | re.S)
    if not m:
        m = re.search(r"^run:.*?^\t.*?\./main\s*(.*)$",
                      content, flags=re.M | re.S)
    run_args = ""
    if m:
        lines = m.group(1).splitlines()
        run_args = lines[0].strip() if lines else ""
    obj_list = " ".join(s.rsplit(".", 1)[0] + ".o" for s in sources)
    return MAKEFILE_TEMPLATE.format(sources=" ".join(sources),
                                    objs=obj_list,
                                    includes=all_flags,
                                    runargs=run_args)

def convert(bench: str, force: bool = False, verbose: bool = False) -> bool:
    omp_dir    = SRC / f"{bench}-omp"
    serial_dir = SRC / f"{bench}-serial"
    if not omp_dir.is_dir():
        print(f"[{bench}] no {omp_dir}", file=sys.stderr); return False
    if serial_dir.exists() and not force:
        if verbose:
            print(f"[{bench}] serial dir exists — skip (use --force)")
        return True

    if serial_dir.exists():
        shutil.rmtree(serial_dir)
    shutil.copytree(omp_dir, serial_dir)

    # Strip OMP from all source-like files (recursive).
    sources = []
    needs_stub = False
    stubbed_paths: list[Path] = []
    for p in serial_dir.rglob("*"):
        if p.is_file() and p.suffix in SRC_EXT:
            # Some HeCBench files carry latin-1 copyright headers; fall back
            # to latin-1 so the stripper never crashes on encoding.
            try:
                content = p.read_text()
                enc = "utf-8"
            except UnicodeDecodeError:
                content = p.read_text(encoding="latin-1")
                enc = "latin-1"
            new_text, this_needs = strip_omp(content)
            p.write_text(new_text, encoding=enc)
            if this_needs:
                needs_stub = True
                stubbed_paths.append(p)
            if p.suffix in (".cpp", ".cc", ".c", ".cxx") and p.parent == serial_dir:
                sources.append(p.name)

    if needs_stub:
        # Inject `#include "omp_serial_stub.h"` at the very top of every file
        # that still references an OMP runtime call, so the stub declarations
        # are visible before use.
        stub_include = '#include "omp_serial_stub.h"\n'
        for p in stubbed_paths:
            content = p.read_text()
            if 'omp_serial_stub.h' not in content:
                p.write_text(stub_include + content)

    # Drop the OMP-target-specific Makefile variants.
    for name in ("Makefile.aomp", "Makefile.nvc"):
        p = serial_dir / name
        if p.exists():
            p.unlink()

    omp_makefile = omp_dir / "Makefile"
    extra_inc = "-I../include" if needs_stub else ""
    (serial_dir / "Makefile").write_text(
        make_serial_makefile(omp_makefile, sources, extra_inc))

    if verbose:
        print(f"[{bench}] wrote {serial_dir}  sources={sources}")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", nargs="?")
    ap.add_argument("--all", action="store_true",
                    help="convert every -omp that lacks a -serial")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.all:
        n_done = n_skip = 0
        for d in sorted(SRC.iterdir()):
            if d.name.endswith("-omp") and d.is_dir():
                bench = d.name[:-len("-omp")]
                if (SRC / f"{bench}-serial").exists() and not a.force:
                    n_skip += 1
                    continue
                convert(bench, force=a.force, verbose=a.verbose)
                n_done += 1
        print(f"converted: {n_done}, skipped-existing: {n_skip}")
    else:
        if not a.bench:
            ap.error("bench name required (or --all)")
        ok = convert(a.bench, force=a.force, verbose=True)
        sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
