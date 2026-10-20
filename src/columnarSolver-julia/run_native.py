#!/usr/bin/env python3
"""Persistent full fixed-fixture native captures, invoked only by central locked runner."""
from pathlib import Path
import os,subprocess,json,hashlib
P=Path(__file__).resolve().parent;root=Path('/tmp/columnarSolver_native')/('checked_'+str(os.getpid()));root.mkdir();(root/'source_before.json').write_text(json.dumps({'source':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [P/'study_native_storage_v2.cu',P/'corrected_observed_storage_v2.cuh',P/'corrected_parser.inc',P/'source_defines.h',P/'protocol_final.json']},'rawroot':str(root),'role':'actualnativefivefullworkload'},indent=2)+'\n');print('NATIVE_RAW_ROOT '+str(root),flush=True);raise SystemExit(subprocess.call(['/tmp/columnarSolver_native/study_native_storage_v2',str(root),'/tmp/columnarSolver_native/cpu_separate.bin']))
