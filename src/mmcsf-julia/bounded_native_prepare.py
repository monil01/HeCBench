#!/usr/bin/env python3
"""Preparation only. Caller owns GPU6 lock; file cap/free-space reserve enforced."""
import pathlib,subprocess,resource,shutil,json,time
B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent;L=R/'.porting-state/porting_logs/mmcsf-julia-shared';cap=512*1024*1024;free=shutil.disk_usage(L).free
if free<2*1024**3+cap:raise RuntimeError('Native preparation requires512MiBboundedwrite allowance plus2GiBreserve')
resource.setrlimit(resource.RLIMIT_FSIZE,(cap,cap));D=L/'prepare_runtime_v1';D.mkdir(exist_ok=True);C=L/'canonical_inputs_v1';A=L/'prepare_aux_v1';cmd=[str(L/'native_study_v2'),str(L/'fixtures_v1'),str(C),str(A),'--prepare'];t=time.monotonic()
with (L/'native_prepare_v1.log').open('w')as f:p=subprocess.run(cmd,cwd=D,stdout=f,stderr=subprocess.STDOUT,timeout=900)
out={'command':cmd,'returncode':p.returncode,'elapsed_seconds':time.monotonic()-t,'file_limit_bytes':cap,'free_before_bytes':free,'scope':'One full canonical-preparation pass, not5trials/timingclaim; no dense tensor storage.'};q=L/'native_prepare_v1_status.tmp';q.write_text(json.dumps(out,indent=2)+'\n');q.replace(L/'native_prepare_v1_status.json');print(json.dumps(out));raise SystemExit(p.returncode)
