from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent;OLD=Path('/tmp/roofline-completion-20261021/tsne069_spsort070_exact_recovery_proposal')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=(OLD/'controller.py').read_text().replace('069/070','073/074').replace("choices=['tsne','spsort'],default='tsne'","choices=['dispatch','slu'],default='dispatch'").replace("(1 if args.group=='tsne'else 4)","(4 if args.group=='dispatch'else 1)")
a=s.index(" if d['status']=='profile_process_completed':");b=s.index(' return r',a)
s=s[:a]+''' if d['status']=='profile_process_completed':
  log=slot/'execution.log';text=log.read_text();rows=[json.loads(l[len('STUDY_TRIAL '):])for l in text.splitlines()if l.startswith('STUDY_TRIAL ')]
  need([x['trial']for x in rows]==[1,2,3,4,5],'Requires all five actual checks')
  need(all(x['correct']is True for x in rows),'Full scientific/execution checks incomplete')
  mp=slot/'profiling_auxiliary_manifest.json';need(sha(mp)==d['auxiliary_manifest_sha256'],'Full scratch manifest SHA mismatch');entries={e['path']:e for e in load(mp)}
  if job['benchmark']=='dispatch':
   need(all(x['checked_values']==0 and x['source_empty_kernel']is True and x['execution_checked']is True and x['numerical_agreement_kind']=='no_numerical_outputs_execution_only'and x['completed_source_loop_launch_calls']==1102200 and x['internal_repeats']==1000 and x['source_warmup_samples_per_mode']==100 and x['source_retained_samples_per_mode']==1000 for x in rows),'Original dispatch execution-only scope changed')
   r['execution_only_no_numerical_outputs']=True
  else:
   need('FullSLU warmup correct=1'in text,'Independent full warmup absent')
   need(text.count('SLU independent case=0 n=99340 pass=1')>=6,'Full ASIC warmup plus five independent checks absent')
   for row in rows:
    need(row['checked_values']==16784916 and row['internal_repeats']==1 and row['original_CPUpreprocess_symbolic_and_solve_execute_outside_ROI']is True,'Full SLU workload changed')
    need([c['case']for c in row['cases']]==[0,1,2,3],'Full four-case SLU suite absent')
    for c,n,nnz,terms,wsbytes in zip(row['cases'],[99340,4960,16,2048],[4008604,23942,32,4096],[557291235,43429,40,5120],[4113868080,205403520,662592,84811776]):
     need(c['n']==n and c['filled_nnz']==nnz and c['independent_LU_product_terms']==terms and c['independent_full_product_pass']is True and c['all_immutable_device_arrays_byte_exact']is True,'SLU independent case scope/check failed')
     captures=c['full_internal_workspace_captures'];need([v['role']for v in captures]==['full_actual_internal_workspace_before','full_actual_internal_workspace_after'],'Before/after full workspace absent')
     for v in captures:
      need(v['uncompressed_bytes']==wsbytes and v['all_actual_bytes_zero']is True and v['saved']is True and v['encoding']=='zlib_level1_lossless_actual_D2H_chunks','Full actual workspace check absent')
      e=entries[str(Path(v['path']).relative_to(job['scratch']))];need(e['bytes']>0 and len(e['sha256'])==64,'Full workspace artifact manifest absent')
    artifact=row['output_artifact'];p=Path(artifact['path']);rel=str(p.relative_to(job['scratch']))if p.is_absolute()else str(p);e=entries[rel]
    need(artifact['count']==16784916 and artifact['dtype']=='float64'and e['bytes']==134279328 and len(e['sha256'])==64,'Full SLU primary output absent')
  need(not Path(job['scratch']).exists(),'Canonical cleanup unexpectedly did not finish')
  r['all5_actual_checks_and_full_manifest_verified']=True
''' + s[b:]
a=s.index(" for p,h in plan['execution_bindings'].items():")
s=s[:a]+''' for artifact in job['stage_artifacts']:
  target=Path(artifact['future_copy']);need(target.parent==slot and target.name=='build_manifest.json','Only reviewed slot compiler manifest stage allowed');need(not target.exists(),'Fresh compiler manifest already exists');target.write_bytes(Path(artifact['frozen_copy']['path']).read_bytes());need(sha(target)==artifact['future_sha256'],'Staged compiler manifest differs')
''' + s[a:]
(O/'controller.py').write_text(s)
x=json.loads((O/'job_inventory.json').read_text());r=json.loads((OLD/'launch_review.json').read_text());(O/'source_frontier_snapshot.json').write_bytes((OLD/'source_frontier_snapshot.json').read_bytes());(O/'deadline_basis_and_retained_progress.json').write_bytes((OLD/'deadline_diagnosis.json').read_bytes())
r.update(status='PRIVATE_PENDING_ROOT_INDEPENDENT_REVIEW_NO_LAUNCH',controller_sha256=sha(O/'controller.py'),tool_bindings=x['tool_bindings'],jobs=x['jobs'],storage={'canonical_default_cleanup':True,'retain_binary_payloads':False,'historical_data_deleted':False,'tmp_reserve_bytes':2147483648,'minimum_free_bytes':{'dispatch':3*1024**3,'slu':4*1024**3},'planning_allowance_is_not_total_peak_guarantee':True,'C21_ongoing_free_space_monitor_retained':True},finite_definition='At most one allpending traversal per group, each model once, no retries. Dispatch3600/SLU900 full target-process seconds, not pertrial. GPU idle queue wait outside processdeadline and unbounded.',prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),version='073_074_exact_source_buildmanifest_inverse_full5checks')
r['parent_frontier_at_preparation']['immutable_snapshot']=str(O/'source_frontier_snapshot.json');r['packet_bindings']={str(p):sha(p)for p in sorted(O.rglob('*'))if p.is_file()and p.name not in ['launch_review.json','controller.py']};(O/'launch_review.json').write_text(json.dumps(r,indent=2)+'\n');print('controller',sha(O/'controller.py'),'review',sha(O/'launch_review.json'))
