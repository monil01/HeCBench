#!/usr/bin/env python3
"""One native preparation pass, bounded evidence storage; caller owns GPU6lock."""
import pathlib,subprocess,resource,shutil,json,time
B=pathlib.Path(__file__).resolve().parent;R=B.parent.parent;L=R/'.porting-state/porting_logs/nbnxm-julia-shared';cap=128*1024*1024;free=shutil.disk_usage(L).free
if free<2*1024**3+cap:raise RuntimeError('Preparationrequires128MiBbounded-fileallowanceplus2GiBreserve')
resource.setrlimit(resource.RLIMIT_FSIZE,(cap,cap));D=L/'prepare_runtime_v1';D.mkdir(exist_ok=True);cmd=[str(L/'native_study_v2'),str(L/'canonical_inputs_v1'),str(L/'prepare_aux_v1'),'--prepare'];t=time.monotonic()
with (L/'native_prepare_v1.log').open('w')as f:p=subprocess.run(cmd,cwd=D,stdout=f,stderr=subprocess.STDOUT,timeout=300)
d={'command':cmd,'returncode':p.returncode,'elapsed_seconds':time.monotonic()-t,'hard_perfile_limit_bytes':cap,'free_before_bytes':free,'scope':'Oneactualnativecanonicalpreparationpass; not5timingtrials'};q=L/'native_prepare_v1_status.tmp';q.write_text(json.dumps(d,indent=2)+'\n');q.replace(L/'native_prepare_v1_status.json');print(json.dumps(d));raise SystemExit(p.returncode)
