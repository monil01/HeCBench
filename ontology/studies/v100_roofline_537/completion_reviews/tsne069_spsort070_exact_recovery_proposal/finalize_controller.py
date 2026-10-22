from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent
old=Path('/tmp/roofline-completion-20261021/tgvnn064_foreign065_recovery_proposal_v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=(old/'controller.py').read_text().replace('064/065','069/070').replace("choices=['foreign','tgvnn'],default='foreign'","choices=['tsne','spsort'],default='tsne'").replace("(27 if args.group=='foreign'else 4)","(1 if args.group=='tsne'else 4)").replace("27 if args.group=='foreign'else 1","1")
a=s.index(" if job['benchmark']=='tgvnn'");b=s.index(' return r',a)
s=s[:a]+''' if d['status']=='profile_process_completed':
  log=slot/'execution.log';rows=[json.loads(l[len('STUDY_TRIAL '):])for l in log.read_text().splitlines()if l.startswith('STUDY_TRIAL ')]
  need([x['trial']for x in rows]==[1,2,3,4,5],'Requires all five actual checks')
  count=32563847 if job['benchmark']=='tsne'else 484725
  need(all(x['correct']is True and x['checked_values']==count for x in rows),'Full scientific checks incomplete')
  mp=slot/'profiling_auxiliary_manifest.json';need(sha(mp)==d['auxiliary_manifest_sha256'],'Full scratch manifest SHA mismatch');entries={e['path']:e for e in load(mp)}
  for row in rows:
   artifact=row['output_artifact'];rel=str(Path(artifact['path']).relative_to(job['scratch']));e=entries[rel]
   expected=130255388 if job['benchmark']=='tsne'else 3877032
   need(e['bytes']==expected and len(e['sha256'])==64,'Full primary artifact manifest absent')
   if job['benchmark']=='tsne':
    need(artifact['count']==32563847 and artifact['dtype']=='float32'and row.get('allocation_guard_pass')is True,'TSNE full metric or allocation checks absent')
    for key in ['lossless_raw_GPU_codec_proof','independent_raw_audit']:
     proofrel=str(Path(row[key]).relative_to(job['scratch']));need(entries[proofrel]['bytes']>0 and len(entries[proofrel]['sha256'])==64,'TSNE full audit/codec manifest absent')
   else:need(artifact['count']==969258 and artifact['dtype']=='uint32'and row['mismatch_output_words']==0,'SPSORT full output checks absent')
  need(not Path(job['scratch']).exists(),'Canonical cleanup unexpectedly did not finish')
  r['five_complete_primary_manifest_and_checks_verified']=True
''' + s[b:]
a=s.index("  if args.group=='tgvnn':");b=s.index('  for batch in batches:',a)
s=s[:a]+"  need(shutil.disk_usage('/tmp').free>=review['storage']['minimum_free_bytes'][args.group],'Scientific payload planning allowance plus 2GiB reserve insufficient')\n  batches=[pending]if pending else []\n"+s[b:]
(O/'controller.py').write_text(s)
x=json.loads((O/'job_inventory.json').read_text());r=json.loads((old/'launch_review.json').read_text());snapshot=O/'source_frontier_snapshot.json';snapshot.write_bytes((old/'source_frontier_snapshot.json').read_bytes())
r.update(status='PRIVATE_PENDING_ROOT_INDEPENDENT_REVIEW_NO_LAUNCH',controller_sha256=sha(O/'controller.py'),tool_bindings=x['tool_bindings'],jobs=x['jobs'],storage={'canonical_default_cleanup':True,'retain_binary_payloads':False,'historical_data_deleted':False,'tmp_reserve_bytes':2*1024**3,'minimum_free_bytes':{'tsne':5*1024**3,'spsort':3*1024**3},'planning_allowance_is_not_total_peak_guarantee':True,'C21_ongoing_free_space_monitor_retained':True},finite_definition='At most one traversal per group, each pending model once, no retry; TSNE1800s and SPSORT3600s full process budget. C21 GPU-idle wait outside deadline and no finite wallclock bound.',prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),version='069_070_exact_inverse_and_full5checks')
r.pop('TGVNN_timeout_scope',None)
r['parent_frontier_at_preparation']['immutable_snapshot']=str(snapshot)
r['other_live_root_controllers']='Root owns coordination with065/other controllers; C21 GPU6 exclusive lock remains mandatory. No invented process generations.'
r['packet_bindings']={str(p):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in ['launch_review.json','controller.py'] and not p.name.endswith('.log')}
(O/'launch_review.json').write_text(json.dumps(r,indent=2)+'\n')
print('controller',sha(O/'controller.py'),'review',sha(O/'launch_review.json'))
