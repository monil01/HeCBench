# Read-only actual saved-data inspection; no imports/execution of campaign tools.
from pathlib import Path
import json,hashlib,fcntl,datetime
O=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');D=R/'ontology/studies/v100_roofline_537/completion_reviews/completion_supervisor/tmpfs063_v19';B=Path('/tmp/roofline-completion-20261021/roi_batch7_strict_family_baseline_after_method_schema.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads(B.read_bytes());study=R/'ontology/studies/v100_roofline_537'
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 strict=Path(base['path']).read_bytes();obj=json.loads(strict)
 ledger={p:{'expected_sha256':v,'actual_sha256':sha(Path(p).read_bytes())}for p,v in obj['input_hashes'].items()if'/benchmarks/'in p}
records=obj['records'];accepted=[]
for record in records:
 if record.get('eligible')is True:accepted.append((record.get('benchmark'),record.get('model')))
# Retain all record fields; no guessed science normalization.
source={}
expected={
 'v100_roofline_collect.py':'c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116',
 'v100_roofline_classify.py':'3878f43f04332ca5d320c2cc558eb1b76a0540e55097360e513c2bf8d620cd39',
 'v100_profile_write_guard.py':'a2007386239acd7ba26ebcf64aa211756d759570ec8ee0d1a15ccccc25fbc2d3',
 'v100_tmpfs_evidence.py':'8dcbeed48e8426d28e6e7b8bcb11c79b960ceb90e235edabe5519bfca3d19d06',
 'v100_roofline_summarize.py':'7347d38d61801aaf73ae55c1334ba55d9ff2010b15d03dc41ca20b2250986d92',
 'v100_roofline_audit.py':'9dd23dbf22e913388316151c87c9966bc4eef96b797262c0fb9ae0996b3159f9',
 'v100_roofline_collect_tmpfs.py':'0c02b36ae9942ccf5bf4792bd814ac13d1b589b564e1dae907b45128fb55f207',
 'v100_tmpfs_scratch.py':'a5fd1c62489557d266b59039ab272b898d86f57554b74fa0579c0bb552067efc'}
for name,want in expected.items():
 p=R/'tools'/name;actual=sha(p.read_bytes());source[name]={'path':str(p),'expected_sha256':want,'actual_sha256':actual,'matches':actual==want}
receipt_path=Path('/tmp/roofline-completion-20261021/su3_tmpfs063_native_v2/controller_receipt.json');rawreceipt=receipt_path.read_bytes();receipt=json.loads(rawreceipt)
processes={}
for pid,start in ((1662499,'78558281'),(1662498,'78558271'),(1662497,'78558240')):
 p=Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split();processes[str(pid)]={'start_ticks':fields[19],'start_matches':fields[19]==start,'uid':p.stat().st_uid,'state':fields[0],'children':(p/'task'/str(pid)/'children').read_text().strip()}
 except FileNotFoundError:processes[str(pid)]={'absent':True}
log=RAW/'refresh_v19_stdout.jsonl';lograw=log.read_bytes();events=[json.loads(x)for x in lograw.splitlines()if x.strip()]
summaryevents=[]
for event in events:
 row={k:v for k,v in event.items()if k not in('stdout','stderr')}
 row['stdout_tail']=event.get('stdout','')[-800:];row['stderr_tail']=event.get('stderr','')[-800:];summaryevents.append(row)
audit_source=(R/'tools/v100_roofline_audit.py').read_text();require_line=next((i for i,line in enumerate(audit_source.splitlines(),1)if expected['v100_tmpfs_evidence.py']in line),None);exec_line=next((i for i,line in enumerate(audit_source.splitlines(),1)if"exec(compile(source,'<frozen_tmpfs_evidence>'"in line),None)
out={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_bindings':source,'processes':processes,
 'strict_family':{'baseline':str(B),'baseline_receipt_sha256':sha(B.read_bytes()),'expected_audit_sha256':base['sha256'],'actual_audit_sha256':sha(strict),'whole_audit_byte_identity_proved_by_expected_SHA':sha(strict)==base['sha256'],
  'eligible_family_count':obj['eligible_family_count'],'baseline_accepted':base['accepted'],'ledger_hashes':ledger,'all_ledger_bytes_match_frozen_audit_inputs':all(v['expected_sha256']==v['actual_sha256']for v in ledger.values()),
  'elapsed_host_ROI_positions':obj['elapsed_host_ROI_positions'],'whole_benchmark_causal_claims':obj['whole_benchmark_causal_claims']},
 'follower':{'receipt_path':str(receipt_path),'receipt_sha256':sha(rawreceipt),'status':receipt['status'],'collector_invocations':receipt['collector_invocations'],'GPU_queries':receipt['GPU_queries'],'signals':receipt['signals'],'predecessor_observations':receipt.get('predecessor_observations'),
  'owned_tmpfs_root_exists':Path('/dev/shm/hecbench-v100-uid10384-su3_batch063_root_review_pending').exists(),'raw063_exists':(RAW/'batch063-su3-tmpfs').exists()},
 'actual_refresh':{'stdout_path':str(log),'stdout_sha256':sha(lograw),'bytes':len(lograw),'events':summaryevents,'retained_family_event_seen':any(x.get('event')=='retained_family_aggregation'and x.get('returncode')==0 for x in events),'joint_checkpoint_event_seen':any(x.get('event')=='joint_checkpoint'and x.get('returncode')==0 for x in events)},
 'known_validator_preexec_gate':{'required_SHA_line':require_line,'exec_line':exec_line,'precedes_exec':require_line is not None and exec_line is not None and require_line<exec_line},
 'scope':'Read-only sequential snapshots of existing actual refresh; no extra exporter/audit/campaign code execution, no GPU/query/build/test/shared mutation.'}
(O/'actual_saved_snapshot.json').write_text(json.dumps(out,indent=2)+'\n');(O/'refresh_stdout_snapshot.jsonl').write_bytes(lograw)
print(json.dumps({'strict_byte_identity':out['strict_family']['whole_audit_byte_identity_proved_by_expected_SHA'],'all3ledgerbytes':out['strict_family']['all_ledger_bytes_match_frozen_audit_inputs'],'all_sources':all(x['matches']for x in source.values()),'follower_status':receipt['status'],'collector_calls':receipt['collector_invocations'],'tmpfs_exists':out['follower']['owned_tmpfs_root_exists'],'refresh_events':[x['event']for x in events]}))
