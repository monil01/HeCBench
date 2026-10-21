#!/usr/bin/env python3
"""Reviewed proposal: root collector must guard/lock invocation; never called by staging.
pre executes five fresh original --negative processes. post is CPU-only and joins
real pre receipts to direct original positive PERF_TRIAL rows. No GPU emulation.
"""
import argparse,ast,datetime,hashlib,json,math,os,pathlib,subprocess,sys,time
from pathlib import Path
P=pathlib.Path

def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def checked(s):
 if s['benchmark']!='assert' or s['model'] not in ['cuda_jl','ka','jacc']:raise RuntimeError('restricted ASSERT Julia scope')
 for p,h in s['execution_bindings'].items():
  if sha(p)!=h:raise RuntimeError('binding changed: '+p)
 return dict(s['execution_bindings'])
def evidence(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def _profile_ownership_identity(pid):
    result={'pid':int(pid),'captured_monotonic_ns':time.monotonic_ns(),'available':False}
    try:
        path=Path('/proc')/str(pid)
        a=(path/'stat').read_bytes();uid=path.stat().st_uid
        b=(path/'stat').read_bytes()
        def fields(raw):
            tail=raw.decode(errors='replace').rsplit(')',1)[1].split()
            return {'state':tail[0],'ppid':int(tail[1]),'pgrp':int(tail[2]),'sid':int(tail[3]),'start_ticks':tail[19]}
        aa,bb=fields(a),fields(b)
        result.update(before=aa,after=bb,uid=uid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        result['stable_instance']=all(aa[k]==bb[k]for k in ['start_ticks','ppid','pgrp','sid'])
        result['available']=True
        try:result['cmdline_sha256']=hashlib.sha256((path/'cmdline').read_bytes()).hexdigest()
        except OSError as error:result['cmdline_error']=repr(error)
    except (OSError,ValueError,IndexError)as error:result['identity_error']=repr(error)
    return result

def _profile_ownership_observation(pid,leader_birth,known):
    target=_profile_ownership_identity(pid);leader_now=_profile_ownership_identity(leader_birth['pid'])
    result={'target':target,'leader_birth':leader_birth,'leader_now':leader_now,'stable_current_owned_identity':False,'classification':'unknown_identity','ancestry':[],'known_same_PID_start_ticks':known.get(str(pid))}
    if not all(x.get('available')and x.get('stable_instance')for x in [target,leader_birth,leader_now]):return result
    if leader_now['after']['start_ticks']!=leader_birth['after']['start_ticks']or leader_now['boot_id']!=leader_birth['boot_id']:
        result['classification']='leader_identity_changed';return result
    tick=target['after']['start_ticks'];prior=known.get(str(pid))
    if prior is not None and prior!=tick:
        result['classification']='PID_reuse_observed_fail_closed';return result
    if target['uid']!=os.getuid()or target['boot_id']!=leader_birth['boot_id']:
        result['classification']='different_UID_or_boot';return result
    if target['after']['sid']==leader_birth['pid']:
        result.update(stable_current_owned_identity=True,classification='stable_current_owned_session_identity')
    else:
        ancestor=pid;visited=set()
        for _ in range(128):
            if ancestor in visited:result['classification']='ancestry_cycle';break
            visited.add(ancestor);entry=_profile_ownership_identity(ancestor);result['ancestry'].append(entry)
            if not entry.get('available')or not entry.get('stable_instance'):break
            if entry['uid']!=os.getuid()or entry['boot_id']!=leader_birth['boot_id']:break
            if ancestor==leader_birth['pid']:
                if entry['after']['start_ticks']==leader_birth['after']['start_ticks']:
                    result.update(stable_current_owned_identity=True,classification='stable_current_owned_ancestry')
                break
            if ancestor<=1:result['classification']='current_ancestry_not_owned';break
            ancestor=entry['after']['ppid']
    # Recheck the target after walking; no history-only acceptance for an exited PID.
    result['target_after_walk']=_profile_ownership_identity(pid)
    end=result['target_after_walk']
    if not end.get('available')or not end.get('stable_instance')or any(end['after'][field]!=target['after'][field]for field in ['start_ticks','ppid','pgrp','sid']):
        result.update(stable_current_owned_identity=False,classification='target_disappeared_or_changed_during_evidence')
    if result['stable_current_owned_identity']:known[str(pid)]=tick
    return result


def pre(s,sp):
 checked(s)
 # Root collector supplies this marker ONLY after holding GPU lock and installing
 # Landlock. It is an integration attestation, not standalone proof of either.
 if os.environ.get('HECBENCH_ASSERT_PHASE_AUTHORIZATION')!=s['authorization_marker']:
  raise RuntimeError('requires reviewed collector GPU-lock + write-guard hook; no standalone bypass')
 target=P(s['pre_receipt']);directory=target.parent/'negative_raw';directory.mkdir(exist_ok=False)
 if target.exists():raise RuntimeError('fresh receipt required')
 r={'schema_version':1,'benchmark':'assert','model':s['model'],'spec':evidence(sp),'started_at':now(),'scope':'new profiling corroboration; not original accepted timings or recovered old child logs','phase':'negative_runtime_prechecks_not_build','complete':False,'all_pass':False,'processes':[],'source_before':checked(s),'integration_attestation':{'GPU_lock_and_Landlock_required':True,'authorization_marker':s['authorization_marker'],'actual_guard_and_lock_proof_required_from_parent_collector':True}}
 save(target,r)
 for trial in range(1,6):
  command=s['positive_command']+['--negative'];stdout=directory/f'{trial}.stdout';stderr=directory/f'{trial}.stderr'
  row={'trial':trial,'command':command,'started_at':now(),'returncode':None,'spawn_exception':None}
  try:
   with stdout.open('xb') as out,stderr.open('xb') as err:
    row['parent_identity_before_spawn']=_profile_ownership_identity(os.getpid())
    r['active_process']=row;save(target,r)
    child=subprocess.Popen(command,cwd=s['scratch_cwd'],env=os.environ.copy(),stdout=out,stderr=err)
    row['child_pid']=child.pid
    row['child_identity_before_wait']=_profile_ownership_identity(child.pid)
    r['active_process']=row;save(target,r)  # persist PID/epoch/SID/PPID before wait/reap
    row['returncode']=child.wait()
    row['child_identity_after_reap']=_profile_ownership_identity(child.pid)
   r['active_process']=None
  except Exception as e:row['spawn_exception']=repr(e)
  row['finished_at']=now()
  for name,p in [('stdout',stdout),('stderr',stderr)]:
   if p.exists():row[name]=evidence(p)
  text=stdout.read_text(errors='replace') if stdout.exists() else ''
  tokens=[line.split() for line in text.splitlines() if line.startswith('ASSERT_STATUS ')]
  row['actual_status_tokens']=tokens
  row['checker_pass']=row['returncode']==0 and tokens==[['ASSERT_STATUS','710']]
  combined=directory/f'{trial}.combined.log'
  combined.write_bytes((stdout.read_bytes() if stdout.exists() else b'')+(stderr.read_bytes() if stderr.exists() else b''))
  row['combined_log']=evidence(combined)
  r['processes'].append(row);save(target,r)  # retain signed rc/raw bytes before rejection
 r.update(finished_at=now(),complete=True,source_after=checked(s))
 r['all_pass']=len(r['processes'])==5 and all(x['checker_pass'] for x in r['processes'])
 save(target,r)
 if not r['all_pass']:raise RuntimeError('one or more real negative contexts failed; NCU forbidden')
 print('ASSERT_PRECHECKS_PASS '+json.dumps(evidence(target)),flush=True)
def post(s,sp,outcome_path):
 before=checked(s);prepath=P(s['pre_receipt']);r=json.loads(prepath.read_text());outcome=json.loads(outcome_path.read_text())
 if not (r['complete'] is True and r['all_pass'] is True and len(r['processes'])==5):raise RuntimeError('negative gate incomplete')
 if r['spec']!=evidence(sp) or r['source_before']!=before or r['source_after']!=before:raise RuntimeError('precheck/source binding mismatch')
 # Collector must write this exact small outcome only AFTER own session quiescence.
 if outcome['plan_sha256']!=sha(s['plan_path']) or outcome['command']!=s['positive_command']:raise RuntimeError('positive command/plan mismatch')
 if outcome['returncode']!=0 or outcome['stop_reason'] is not None or outcome['own_session_quiescent'] is not True:raise RuntimeError('positive execution not accepted')
 if outcome['guarded_under_GPU_lock'] is not True:raise RuntimeError('collector guard/lock attestation required')
 log=P(outcome['execution_log']['path'])
 if evidence(log)!=outcome['execution_log']:raise RuntimeError('positive log changed')
 observations=[]
 for trial,x in enumerate(r['processes'],1):
  if x['trial']!=trial or x['command']!=s['positive_command']+['--negative'] or x['returncode']!=0 or x['checker_pass'] is not True:raise RuntimeError('negative process mismatch')
  for key in ['stdout','stderr','combined_log']:
   if evidence(P(x[key]['path']))!=x[key]:raise RuntimeError('negative raw bytes changed')
  tokens=[z.split() for z in P(x['stdout']['path']).read_text().splitlines() if z.startswith('ASSERT_STATUS ')]
  if tokens!=[['ASSERT_STATUS','710']]:raise RuntimeError('negative actual status mismatch')
  observations.append(dict(trial=trial,process_returncode=0,observed_status=710,log=x['combined_log']['path'],sha256=x['combined_log']['sha256'],raw_stdout=x['stdout'],raw_stderr=x['stderr']))
 rows=[json.loads(z[len('PERF_TRIAL '):]) for z in log.read_text(errors='replace').splitlines() if z.startswith('PERF_TRIAL ')]
 if len(rows)!=5 or [x['trial'] for x in rows]!=list(range(1,6)):raise RuntimeError('need original five PERF_TRIAL rows')
 # Obtain unchanged source wrapper literal attestations; only obs is a variable.
 tree=ast.parse(P(s['original_wrapper']).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='row' and n.func.attr=='update']
 if len(calls)!=1:raise RuntimeError('source row.update changed')
 fixed={kw.arg:ast.literal_eval(kw.value) for kw in calls[0].keywords if kw.arg!='negative_diagnostic'}
 joined=[]
 for row,obs in zip(rows,observations):
  times=row['component_time_seconds'];total=row['time_seconds']
  if len(times)!=2 or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in [total,*times]):raise RuntimeError('positive time invalid')
  if not math.isclose(sum(times),total,rel_tol=1e-14):raise RuntimeError('positive component sum mismatch')
  if any(k in row for k in ['output_artifact','numerical_outputs','values']):raise RuntimeError('unexpected numerical output')
  row.update(fixed,negative_diagnostic=obs);joined.append(row)
 # No instrumented mean may replace original accepted five-run timing series.
 report={'schema_version':1,'spec':evidence(sp),'positive_outcome':evidence(outcome_path),'negative_prechecks':evidence(prepath),'complete_execution_only_contract':True,'numerical_agreement_claim':False,'primary_accepted_timings_unchanged':True,'profile_timings_are_not_primary_performance':True,'GPU_process_units':{'negative_fresh_contexts':5,'positive_program':1},'source_before':before,'source_after':checked(s),'rows':joined,'finished_at':now()}
 final=P(s['post_receipt'])
 if final.exists():raise RuntimeError('fresh post receipt required')
 save(final,report)
 for row in joined:print('STUDY_TRIAL '+json.dumps(row,allow_nan=False),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('phase',choices=['pre','post']);a.add_argument('--spec',type=P,required=True);a.add_argument('--positive-outcome',type=P);q=a.parse_args();s=json.loads(q.spec.read_text())
 if q.phase=='pre':pre(s,q.spec)
 else:
  if q.positive_outcome is None:raise RuntimeError('actual collector positive outcome required')
  post(s,q.spec,q.positive_outcome)
if __name__=='__main__':main()
