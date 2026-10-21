#!/usr/bin/env python3
"""Require the unchanged independently built CPU checker before target execution."""
from pathlib import Path
import hashlib,json,os,sys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
scratch=Path.cwd().resolve();receipt=json.loads((scratch/'extra_build_outcome.json').read_text());checker=scratch/'mtcheck'
if receipt['checker']!=str(checker)or sha(checker)!=receipt['checker_sha256']:raise SystemExit('Independent CPU checker build binding missing/changed')
for p,h in receipt['source_bindings'].items():
 if sha(p)!=h:raise SystemExit('Compiler input drift '+p)
for p,h in receipt['outputs'].items():
 if sha(p)!=h:raise SystemExit('Compiled output drift '+p)
os.execvpe(sys.argv[1],sys.argv[1:],os.environ)
