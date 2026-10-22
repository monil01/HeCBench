"""Repaired014 read-only audit anchor/fresh diagnostics, plus lossless child status capture.
Original target and unchanged complete CPUoracle remain mandatory, outside source ROI.
"""
import subprocess,sys,os,shutil,json,hashlib
from pathlib import Path
p=Path('/home/imo/HeCBench_julia/src/snicit-julia');d=Path(os.environ['STUDY_OUTPUT_DIR']).resolve();here=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def child(label,command,env):
 path=d/(label+'.log');receipt=d/(label+'_status.json');data={'status':'started','command':command,'stdout_stderr_path':str(path),'environment_changes':{'OPENBLAS_NUM_THREADS':env.get('OPENBLAS_NUM_THREADS'),'PYTHONPATH':env.get('PYTHONPATH')},'wrapper_pid':os.getpid()};receipt.write_text(json.dumps(data,indent=2)+'\n')
 with path.open('w')as log:
  proc=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);data['child_pid']=proc.pid
  try:data['child_stat_at_start']=Path('/proc/'+str(proc.pid)+'/stat').read_text()
  except(FileNotFoundError,ProcessLookupError):data['child_stat_at_start']=None
  receipt.write_text(json.dumps(data,indent=2)+'\n')
  for line in proc.stdout:log.write(line);log.flush();sys.stdout.write(line);sys.stdout.flush()
  rc=proc.wait()
 data.update(status='returned',returncode=rc,observed_signal=(-rc if rc<0 else None),log_sha256=sha(path));receipt.write_text(json.dumps(data,indent=2)+'\n');return rc,path
r,log=child('child_program',sys.argv[1:],os.environ.copy())
if r:sys.exit(r)
rows=[json.loads(s[len('STUDY_TRIAL '):])for s in log.read_text().splitlines()if s.startswith('STUDY_TRIAL ')]
assert [z['trial']for z in rows]==[1,2,3,4,5] and all(z['correct']is True and z['jit_excluded']is True and z['checked_values']==478308 for z in rows)
artifacts=[]
for z in rows:
 a=z['output_artifact'];f=Path(a['path']).resolve();assert f.is_relative_to(d)and f==d/('trial'+str(z['trial'])+'.bin')and a['dtype']=='float64'and a['count']==5026888 and f.stat().st_size==40215104;artifacts.append({'trial':z['trial'],'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)})
env=os.environ.copy();env['PYTHONPATH']='/tmp/pasc-julia-plot-tools';env['OPENBLAS_NUM_THREADS']='1'
r,_=child('child_audit',['/usr/bin/python3',str(here/'oracle_driver.py'),str(d)],env)
if r:sys.exit(r)
report=json.loads((d/'independent_cpu_audit.json').read_bytes());assert report['correct']is True and report['max_normalized_operator_budget']<=1 and report['trace_entrypoint_counts']=={'0':3,'1':28,'2':4,'3':4,'4':4,'5':4,'6':32,'7':32,'8':4};assert len(report['actual_untimed_traces'])==4 and sum(len(z['events'])for z in report['actual_untimed_traces'])==115;assert len(report['categorical_trial_sha256'])==5
for a in artifacts:assert sha(a['path'])==a['sha256']
(d/'mandatory_five_output_and_oracle_checks.json').write_text(json.dumps({'correct':True,'all5_actual_source_rows_checked':True,'all5_complete_artifacts_hashed':artifacts,'unchanged_full_CPU_audit_source':str(p/'audit_model.py'),'unchanged_full_CPU_audit_sha256':sha(p/'audit_model.py'),'all9_warm_trace_entrypoint_counts':report['trace_entrypoint_counts'],'actual_complete_warm_dispatches':115,'timed_overwritten_intermediates_claimed':False,'primary_count':5026888,'input_words':4548580,'categoricals_exact_guards91_preserved':True},indent=2)+'\n')
dest=d/'retained_diagnostics';dest.mkdir(parents=True,exist_ok=True)
for f in [*d.glob('trace_case*.bin'),d/'independent_cpu_audit.json']:shutil.copy2(f,dest/f.name)
print('SNICIT_RETAINED_TRACES',str(dest),flush=True)
