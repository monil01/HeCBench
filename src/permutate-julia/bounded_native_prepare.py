#!/usr/bin/env python3
"""Caller owns GPU6 lock. Enforce per-file/free-space limits before asset creation."""
import pathlib,shutil,resource,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[2];S=R/'.porting-state/porting_logs/permutate-julia-shared';C=S/'canonical_inputs_v1';C.mkdir(parents=True,exist_ok=True)
if shutil.disk_usage(C).free<2*1024**3:raise RuntimeError('Need 2GiB free reserve; no evidence cleanup allowed')
if any(C.glob('*.bin')):raise RuntimeError('Canonical files already exist; refuse overwrite')
resource.setrlimit(resource.RLIMIT_FSIZE,(400*1024**2,400*1024**2))
raise SystemExit(subprocess.run([sys.argv[1],str(S/'assets/truerand_1bit.bin'),str(C),str(S/'prepare_aux_v1'),'--prepare'],cwd=R).returncode)
