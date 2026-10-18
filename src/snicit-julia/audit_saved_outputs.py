"""Read-only independent all20savedfields, categorical/guard/quality/source audit."""
from pathlib import Path
import sys,json,hashlib,re,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];F=Path(sys.argv[1]).resolve();r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());raw=np.fromfile(P/'input_u32.bin',dtype='<u4');dense=np.fromfile(P/'dense_oracle_f64.bin',dtype='<f8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def af(c,key,shape):
 n=int(np.prod(shape));return raw[c[key]:c[key]+n].view('<f4').reshape(shape).astype('f8')
rows=[];traces=[];bindings={};canonical=None
for model in ['cuda','cuda_jl','ka','jacc']:
 m=r['models'][model];assert m['status']=='native_check_pass'and len(m['trials'])==5 and m.get('source_provenance_valid',True)
 for p,h in m.get('adapter_file_hashes',{}).items():assert sha(p)==h,p;bindings[p]=h
 log=F.parent/(model+'_run1.log');text=log.read_text();match=re.search(r'SNICIT_RETAINED_TRACES (.+)',text);assert match
 td=Path(match.group(1));a=json.loads((td/'independent_cpu_audit.json').read_text());assert a['correct']and a['source_kernel_entrypoints']==9
 for t in a['actual_untimed_traces']:
  p=td/f"trace_case{'ABCD'.index(t['case'])}.bin";assert sha(p)==t['trace_sha256'];assert p.stat().st_size==t['trace_bytes']
 traces.append(dict(model=model,path=str(td.relative_to(R)),audit_sha256=sha(td/'independent_cpu_audit.json'),complete_actual_trace_count=4,checked_operator_values=a['checked_operator_values'],max_abs_operator_error=a['max_abs_operator_error'],max_normalized_operator_budget=a['max_normalized_operator_budget'],stage_counts=a['trace_entrypoint_counts'],traces=a['actual_untimed_traces']))
 for t in m['trials']:
  art=t['output_artifact'];f=Path(art['saved_path']);v=np.fromfile(f,dtype='<f8');assert len(v)==5026888 and np.isfinite(v).all();assert np.array_equal(v[:len(raw)],raw.astype('f8'));pos=len(raw);cats=[];quality=[];maxerr=0.;maxbudget=0.
  for c in plan['cases']:
   M,H=c['M'],c['H'];h=[]
   for _ in range(2):h.append(v[pos:pos+M*H].reshape(M,H));pos+=M*H
   out=v[pos:pos+M*10].reshape(M,10);pos+=M*10
   ys=v[pos:pos+128];pos+=128;lut=v[pos:pos+M];pos+=M;ne=v[pos:pos+M];pos+=M;active=v[pos:pos+M];pos+=M;cnt=v[pos];pos+=1;pred=v[pos:pos+M];pos+=M
   na=1+(0 if c['cifar']else 2)+5+6+4*c['L'];guards=v[pos:pos+16*na];pos+=16*na;assert np.all(guards==91)
   expected=h[c['T']%2]@af(c,'wout',(H,10))+af(c,'bout',(10,));e=np.abs(out-expected);budget=e/(1e-4+1e-4*np.abs(expected));maxerr=max(maxerr,float(e.max()));maxbudget=max(maxbudget,float(budget.max()));assert np.all(budget<=1)
   assert np.array_equal(pred,out.argmax(axis=1));labels=raw[c['label']:c['label']+M].view('<i4');assert cnt==np.count_nonzero(pred==labels)
   cat=np.r_[ys,lut,ne,active,cnt,pred];assert np.array_equal(cat,cat.astype('i4'));cats.extend(cat)
   dd=dense[c['dense_oracle_offset']:c['dense_oracle_offset']+M*10].reshape(M,10)
   quality.append(dict(case=c['name'],correct=int(cnt),examples=M,real_dataset=not c['cifar'],dense_independent_correct=c['dense_reference_correct_labels'],argmax_disagreements_vs_dense=int(np.count_nonzero(pred!=dd.argmax(axis=1))),max_abs_logit_difference_vs_dense=float(np.abs(out-dd).max())))
  assert pos==len(v);cat=np.array(cats,dtype='<i4')
  if canonical is None:canonical=cat
  assert np.array_equal(cat,canonical),'categorical outcomes differ acrossmodels/trials'
  rows.append(dict(model=model,trial=t['trial'],path=str(f.relative_to(R)),sha256=sha(f),count=len(v),full_values_scanned=True,input_prefix_exact_values=len(raw),surviving_float_values=468992,categorical_values=4612,exact_guard_values=4704,max_abs_independent_final_dense_error=maxerr,max_normalized_budget=maxbudget,categorical_sha256=hashlib.sha256(cat.tobytes()).hexdigest(),quality_vs_independent_dense=quality))
assert r['matched_performance_valid']and all(r['variant_verified'].values())and r['reference_agreement']
report=dict(correct=True,final_result=str(F.relative_to(R)),result_sha256=sha(F),config_sha256=sha(F.with_name('config.json')),rows=rows,actual_untimed_complete_traces=traces,executed_source_bindings=bindings,categorical_all20_exact=True,primary_arithmetic_tolerance={'atol':1e-4,'rtol':1e-4},guards_all20_raw_byte_or_typed_value91_exact=True,source_GPU_entrypoint_definitions=9,timed_GPU_launches_per_trial=104,total_GPU_launches_per_trial=115,scope='Actualuntimedintermediatetraces separate fromfivecontinuousROItrials; overwritten timedintermediatesnotclaimed',quality_is_not_translation_acceptance=True)
(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('SNICIT_SAVED_AUDIT',len(rows),len(traces),len(bindings),max(x['max_abs_independent_final_dense_error']for x in rows),flush=True)
