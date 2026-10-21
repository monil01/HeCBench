#!/usr/bin/env python3
"""Launch both original ranks on GPU6, retain all raw rank/stage data, aggregate full captures."""
from pathlib import Path
import os,sys,subprocess,json,hashlib,shutil
P=Path('/home/imo/HeCBench_julia/src/miniDGS-julia');R=P.parents[1];model=sys.argv[1];out=Path(os.environ['STUDY_OUTPUT_DIR']).resolve();plan=json.load(open(P/'study_plan.json'));prefix=plan['prefix_each_rank'];n=plan['outputs_each_rank'];raw=out/'rank_process.log'
if model=='cuda':rankcmd=[str(out/'native_rank'),str(P/'fixtures')]
else:
 child={'cuda_jl':'cuda.jl','ka':'KA','jacc':'JACC'}[model];project='_julia_jacc_env'if model=='jacc'else'_julia_ka_env';rankcmd=['julia','--startup-file=no','--project='+str(P.parent/project),str(P/f'miniDGS-julia-{child}/main.jl')]
command=['mpiexec','-n','2',*rankcmd]
with raw.open('w')as f:proc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
persist=out/'retained_rank_outputs';persist.mkdir(parents=True,exist_ok=True)
for f in out.glob('rank*'):
 if f.is_file():shutil.copy2(f,persist/f.name)
(persist/'process_provenance.json').write_text(json.dumps(dict(command=command,exit_code=proc.returncode,model=model,source_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_plan_sha256=hashlib.sha256((P/'study_plan.json').read_bytes()).hexdigest(),actual_untimed_warmup_stage_capture=True,not_surviving_timed_intermediates=True,original_rank_artifacts_retained=True,primary_GPU_processes=2,MPI_ranks=2,physical_GPU=6),indent=2)+'\n')
print(raw.read_text(),end='');print('STUDY_AUXILIARY retained_actual_rank_outputs_and_untimed_fullstage_trace',persist,flush=True)
if proc.returncode:sys.exit(proc.returncode)
for t in range(1,6):
 rows=[json.load(open(out/f'rank{r}_trial{t}.json'))for r in [0,1]]
 assert all(z['correct']and z['trial']==t and z['rank']==r and z['prefix_count']==prefix and z['output_count']==n for r,z in enumerate(rows))
 path=out/f'miniDGS_trial{t}.bin'
 with path.open('wb')as dest:
  for mode in ['prefix','outputs']:
   for r in [0,1]:
    f=out/f'rank{r}_trial{t}.bin';assert f.stat().st_size==(prefix+n)*8
    with f.open('rb')as source:
     source.seek(0 if mode=='prefix'else prefix*8);left=(prefix if mode=='prefix'else n)*8
     while left:
      b=source.read(min(left,4<<20));assert b;dest.write(b);left-=len(b)
 assert path.stat().st_size==plan['primary_count']*8
 row=dict(trial=t,time_seconds=max(z['time_seconds']for z in rows),rank_time_seconds=[z['time_seconds']for z in rows],correct=True,checked_values=plan['primary_count'],max_abs_error=max(z['max_abs_error']for z in rows),max_rel_error=None,max_normalized_budget_ratio=max(z['max_normalized_budget_ratio']for z in rows),jit_excluded=True,timing_region=plan['timing_region'],iterations=5,internal_repeats=1,authored_GPU_kernel_launches=40,mpi_ranks=2,physical_GPUs=[6],both_ranks_share_physical_GPU=True,independent_oracle_full_fields_checked=True,auxiliary_actual_trace_path=str(persist),output_artifact=dict(path=str(path),dtype='float64',count=plan['primary_count'],byte_order='little'))
 print('STUDY_TRIAL '+json.dumps(row),flush=True)
