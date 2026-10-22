from pathlib import Path
import json,hashlib,datetime,fcntl,os
R=Path('/home/imo/HeCBench_julia');P=Path('/tmp/roofline-completion-20261021');D=P/'runtime_only_applicability_five';C=R/'ontology/studies/v100_roofline_537/completion_reviews';dest=C/'runtime_only_applicability_five_root'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def walk(x):
 if isinstance(x,dict):
  if all(k in x for k in ['path','sha256'])and isinstance(x['path'],str)and x['path'].startswith(('/home/imo/','/tmp/')):
   p=Path(x['path']);assert sha(p)==x['sha256'],str(p)
   if 'bytes'in x:assert p.stat().st_size==x['bytes'],str(p)
  for y in x.values():walk(y)
 elif isinstance(x,list):
  for y in x:walk(y)
proposal=D/'runtime_no_kernel_applicability_proposal.json';assert sha(proposal)=='6c628b6ca12672f420a4e6785da936504e8cebabd2a96e5ac6e6ddd0b8c7d659';d=json.loads(proposal.read_text())
with Path('/tmp/hecbench-roofline-metadata.lock').open()as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);walk(d);path=Path(d['parent_canonical_review']['path']);oldbytes=path.read_bytes();old=json.loads(oldbytes);assert len(old['rows'])==7
 keys={(r['benchmark'],r['model'])for r in old['rows']};assert len(d['rows'])==5;checks=[]
 for row in d['rows']:
  key=(row['benchmark'],row['model']);assert key not in keys;assert key==('memcpy','cuda')or key[0]=='streamCreateCopyDestroy';keys.add(key)
  prov=json.loads(Path(row['parent_profile']['path']).read_text());assert prov['status']=='profile_completed_counter_derivation_failed';assert prov['profile']['returncode']==0 and prov['profile']['stop_reason']is None
  log=Path(row['parent_log']['path']).read_text();trials=[json.loads(l[len('STUDY_TRIAL '):])for l in log.splitlines()if l.startswith('STUDY_TRIAL ')];assert [t['trial']for t in trials]==[1,2,3,4,5];assert all(t['correct']is True for t in trials);assert row['metrics']is None
  assert row['proposed_terminal_profile_status']=='roofline_not_applicable_no_device_kernels';assert row['raw_numeric_or_primary_timing_changed']is False;assert row['whole_elapsed_ROI_roofline']is False
  checks.append({'benchmark':key[0],'model':key[1],'trials':5,'rawstatuspreserved':prov['status'],'counter_parse_error':prov['counter_parse_error']})
 dest.mkdir(exist_ok=False);(dest/'previous_runtime_no_kernel_applicability.json').write_bytes(oldbytes);(dest/'proposal.json').write_bytes(proposal.read_bytes());new=dict(old,rows=old['rows']+d['rows']);encoded=(json.dumps(new,indent=2)+'\n').encode();tmp=path.with_suffix('.five.tmp');tmp.write_bytes(encoded);os.replace(tmp,path)
 receipt={'status':'five_scoped_runtime_only_rows_adopted','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_sha256':hashlib.sha256(oldbytes).hexdigest(),'current_sha256':sha(path),'proposal_sha256':sha(proposal),'root_checks':checks,'old_rows_preserved_exactly':json.loads(path.read_text())['rows'][:7]==old['rows'],'total_scoped_rows':12,'no_GPU_executions':True,'metrics':'null, not zero','scope':'Source timed runtime/transfer selected captures with actual5correctchecks, noCSVkernels; not proof of zero traffic/noSM/generalAPIunsupportedness. Fullbinarycleanup alreadyhistorical, no newbinaryvalue recertification. Primaryacceptance/timing unchanged.'};(dest/'root_adoption.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
