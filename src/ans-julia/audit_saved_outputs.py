"""Read-only all20primary bytes plus every retained actual diagnostic/source binding."""
from pathlib import Path
import json,sys,hashlib,re,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];F=Path(sys.argv[1]).resolve();r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());oracle=np.fromfile(P/'oracle_u8.bin',dtype='u1');rows=[];diag=[];bindings={};first=None;scratch_groups={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fixture=np.fromfile(P/'input_u32.bin',dtype='<u4');expectedprefix=[]
for c in plan['cases']:
 a=np.zeros(c['allocated_input_words'],dtype='<u4');a[:c['fixture_input_words']]=fixture[c['input_offset']:c['input_offset']+c['fixture_input_words']];expectedprefix.extend(a);expectedprefix.extend(fixture[c['table_offset']:c['table_offset']+1024]);expectedprefix.extend([c[k]for k in ['id','lambda_bits','output_bytes','compressed_words','num_subseq','num_blocks','first_state','first_bit']]+[1024,11,4,128,c['sync_slots'],c['allocated_input_words'],0,0])
expectedprefix=np.array(expectedprefix,dtype='<u4');assert len(expectedprefix)==plan['input_prefix_words']
for model in ['cuda','cuda_jl','ka','jacc']:
 m=r['models'][model];assert m['status']=='native_check_pass'and len(m['trials'])==5
 for p,h in m['adapter_file_hashes'].items():assert sha(p)==h;bindings[p]=h
 dirs={Path(t['output_artifact']['saved_path']).parent for t in m['trials']};assert len(dirs)==1;origin=next(iter(dirs));log=origin/(model+'_run1.log');match=re.search(r'ANS_RETAINED_STATES (.+)',log.read_text());assert match;D=Path(match.group(1));a=json.loads((D/'independent_cpu_audit.json').read_text());assert a['correct']and len(a['all_actual_final_scratch_captures'])==75 and len(a['all_actual_untimed_stage_captures'])==75
 for item in a['all_actual_final_scratch_captures']+a['all_actual_untimed_stage_captures']:
  p=Path(item['path']);assert sha(p)==item['sha256']and p.stat().st_size==item['bytes']
 duplicates=[]
 for c in plan['cases']:
  f=D/f"trial0_case{c['id']}_final.bin";h=sha(f);stage=[v for v in a['all_actual_untimed_stage_captures']if v['case']==c['id']and v['op']==4];assert len(stage)==1 and h==stage[0]['sha256'];duplicates.append(dict(path=str(f.relative_to(R)),sha256=h,identical_to_actual_warm_phase4=stage[0]['path']))
 diag.append(dict(model=model,directory=str(D.relative_to(R)),audit_sha256=sha(D/'independent_cpu_audit.json'),full_actual_trial_scratch=75,full_actual_untimed_stage_captures=75,warmfinal_duplicate_captures=duplicates,checked_decoded_bytes_including_rawscratch_and_primary_duplicates=a['decoded_bytes_independently_checked']))
 # Scratch schedule/state equality is descriptive, never primary acceptance/canonicalization.
 for item in a['all_actual_final_scratch_captures']:
  scratch_groups.setdefault(item['case'],set()).add(item['sha256'])
 for t in m['trials']:
  p=Path(t['output_artifact']['saved_path']);v=np.fromfile(p,dtype='<u4');assert len(v)==5797880 and np.array_equal(v[:len(expectedprefix)],expectedprefix);pos=plan['input_prefix_words']
  for c in plan['cases']:
   n=c['output_bytes']//4;actual=v[pos:pos+n].view('u1');pos+=n;assert np.array_equal(actual,oracle[c['oracle_offset']:c['oracle_offset']+c['output_bytes']])
  guards=np.tile(np.r_[np.full(16,165,dtype='<u4'),np.full(64,0x4e5a5a5a,dtype='<u4'),np.full(16,165,dtype='<u4')],15);assert np.array_equal(v[pos:],guards);h=sha(p)
  if first is None:first=h
  assert h==first;assert t['convergence_iterations']==[1]*15
  rows.append(dict(model=model,trial=t['trial'],path=str(p.relative_to(R)),sha256=h,artifact_words=len(v),input_prefix_words=plan['input_prefix_words'],decoded_bytes_exact=len(oracle),raw_guard_values_exact=len(guards),all_primary_words_scanned=True,convergence_iterations=t['convergence_iterations'],component_seconds=t['component_time_seconds']))
assert len(rows)==20 and r['matched_performance_valid']and r['reference_agreement']and all(r['variant_verified'].values())
report=dict(correct=True,result=str(F.relative_to(R)),result_sha256=sha(F),config_sha256=sha(F.with_name('config.json')),rows=rows,actual_diagnostic_inventory=diag,executed_source_bindings=bindings,all20primary_bitidentical=True,full_decoded_bytes_each_primary=len(oracle),full_trial_scratch_captures=300,full_actual_warm_stage_captures=300,additional_actual_warm_final_identical_duplicate_captures=60,per_case_final_scratch_unique_record_SHA_counts={str(k):len(v)for k,v in scratch_groups.items()},scratch_equality_not_primary_acceptance=True,oracle='Independent scalarCPUbitreader/tabletransducer validates sourceencoderbytes against reversedoriginalsourcegenerateddata; all20actualGPUfullbytes exact; no entropy/lengthproxy')
(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('ANS_SAVED_AUDIT',len(rows),len(bindings),len(diag),report['per_case_final_scratch_unique_record_SHA_counts'],flush=True)
