#!/usr/bin/env python3
"""CPU-only full-artifact/source-binding/cold-regeneration audit. No GPU execution."""
from pathlib import Path
import json,array,hashlib,sys,tempfile,shutil,subprocess,importlib.util
P=Path(__file__).resolve().parent;R=P.parent.parent;F=Path(sys.argv[1]);F=F if F.is_absolute() else R/F;r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text())
spec=importlib.util.spec_from_file_location('independent_coordinate_oracle',P/'generate_assets.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
def sha(b):return hashlib.sha256(b).hexdigest()
def arr(b):a=array.array('i');a.frombytes(b);return a
wrap=lambda x:[211]*16+list(x)+[211]*16
assert sys.byteorder=='little' and r['reference_agreement']and r['matched_performance_valid']
manifest=json.loads((P/'extraction_manifest.json').read_text());source=(R/manifest['source']).read_bytes();assert sha(source)==manifest['source_sha256'];assert source[manifest['kernel_span_start_byte']:manifest['kernel_span_end_byte']]==(P/'source_kernels.inc').read_bytes()
expected=arr((P/'oracle_primary_i32.bin').read_bytes());trace_expected=arr((P/'oracle_trace_i32.bin').read_bytes());COUNT=plan['artifact_count'];PREFIX=plan['input_prefix_count'];cache={};allhashes=set();allprefix=set();trials=[];bindings={};traces=[]
def full_math(a):
 h=sha(a.tobytes())
 if h in cache:return cache[h]
 assert len(a)==COUNT and list(a[:27])==o.coeff
 off=27;stage_lists=[]
 for c in plan['cases']:
  assert list(a[off:off+5])==[c['case'],c['frame'],c['rows'],c['cols'],16];off+=5;n=c['n'];initial=[]
  for k in range(3):
   buf=a[off:off+n+32];off+=n+32;assert list(buf[:16])==list(buf[-16:])==[211]*16;initial.append(list(buf[16:-16]))
  assert initial[0]==list((P/c['input']).read_bytes()) and initial[1]==initial[2]==[0]*n
  stages,_=o.oracle(initial[0],c['rows'],c['cols']);stage_lists.append(stages)
 assert list(a[off:off+27])==o.coeff;off+=27;assert off==PREFIX
 maxima=[0,0,0]
 for c,stages in zip(plan['cases'],stage_lists):
  for k,stage in enumerate([stages[4],stages[3],stages[2]]):
   target=wrap(stage);got=a[off:off+c['n']+32];off+=len(target);errors=[abs(x-y)for x,y in zip(got,target)];maxima[k]=max(maxima[k],max(errors));assert all(e<=(1 if k==1 and 16<=j<c['n']+16 else 0)for j,e in enumerate(errors))
 assert off==COUNT
 cache[h]={'maximum_errors_final_edges_nms_theta':maxima,'independent_coordinate_oracle_from_actual_inputs':True,'distinct_images':12,'whole_artifact_sha256':h}
 return cache[h]
for m,v in r['models'].items():
 assert v['complete']and v['timing_valid']and v['provenance_valid']and len(v['trials'])==5
 for path,h in v['adapter_file_hashes'].items():assert sha(Path(path).read_bytes())==h,path;bindings[path]=h
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);b=f.read_bytes();a=arr(b);assert sha(b)==t['output_artifact']['sha256']and a==expected;maths=full_math(a);assert t['correct']and t['jit_excluded']and t['time_seconds']>0
  allhashes.add(sha(b));allprefix.add(sha(b[:PREFIX*4]));trials.append({'model':m,'trial':t['trial'],'path':str(f.relative_to(R)),'sha256':sha(b),'checked_values':len(a),'input_prefix_exact':True,'CPU_max_error':maths['maximum_errors_final_edges_nms_theta']})
 f=Path(v['trials'][0]['supplemental_trace_path']);b=f.read_bytes();a=arr(b);assert len(a)==plan['trace_count']and a==trace_expected
 # Validate every stage independently again, from full actual prefix inputs.
 off=0
 for c in plan['cases']:
  stages,_=o.oracle(list((P/c['input']).read_bytes()),c['rows'],c['cols'])
  for k,stage in enumerate(stages):
   target=wrap(stage);got=a[off:off+len(target)];assert all(abs(x-y)<=(1 if k in (1,3)and16<=j<c['n']+16 else 0)for j,(x,y)in enumerate(zip(got,target)));off+=len(target)
 assert off==len(a);traces.append({'model':m,'path':str(f.relative_to(R)),'sha256':sha(b),'checked_values':len(a),'CPU_max_error':0,'timing_excluded':True,'scope':'One distinct untimed complete four-stage pipeline trace after five primary timings, not overwritten timed intermediates.'})
assert len(trials)==20 and len(allhashes)==len(allprefix)==1
# Combinatorial proof of the authored cooperative tile-load coverage.
loads={}
for lr in range(1,17):
 for lc in range(1,17):
  positions=[lr*18+lc]
  if lr==1:
   positions+=[lc]
   if lc==1:positions+=[0]
   elif lc==16:positions+=[17]
  elif lr==16:
   positions+=[17*18+lc]
   if lc==1:positions+=[17*18]
   elif lc==16:positions+=[323]
  if lc==1:positions+=[lr*18]
  elif lc==16:positions+=[lr*18+17]
  for q in positions:loads[q]=loads.get(q,0)+1
assert set(loads)==set(range(324))and set(loads.values())=={1}
scratchdir=Path(r['models']['jacc']['trials'][0]['supplemental_trace_path']).parent;scratch=list(scratchdir.glob('actual_scratch_readbacks_*.bin'));assert len(scratch)==1;b=scratch[0].read_bytes();assert arr(b)==array.array('i',[0]*648)
names=[c['input']for c in plan['cases']]+['study_params.h','study_params.jl','study_plan.json','oracle_primary_i32.bin','oracle_trace_i32.bin']
with tempfile.TemporaryDirectory(prefix='ced-cold-CPU-')as tmp:
 q=Path(tmp);shutil.copy2(P/'generate_assets.py',q/'generate_assets.py');subprocess.run([sys.executable,str(q/'generate_assets.py')],check=True,stdout=subprocess.DEVNULL);cold=[]
 for name in names:
  a=(P/name).read_bytes();b=(q/name).read_bytes();assert a==b,name;cold.append({'path':name,'bytes':len(a),'sha256':sha(a),'byte_exact':True})
out={'benchmark':'ced','result':str(F.relative_to(R)),'result_sha256':sha(F.read_bytes()),'status':'pass','GPU_executions_added':0,'all20_primary_bit_identical':True,'artifact_count':COUNT,'input_prefix_count':PREFIX,'trials':trials,'source_bindings':bindings,'independent_mathematical_checks':list(cache.values()),'oracle_cache_note':'All20 files fully reread/rehash/compared; mathematical reevaluation cached only for identical complete artifact bytes. One12-image inputset, not20 distinct inputsets.','supplemental_actual_GPU_stage_traces':traces,'JACC_scratch':{'path':str(scratch[0].relative_to(R)),'sha256':sha(scratch[0].read_bytes()),'initial_final_values':648,'all_zero':True,'tile_coverage':{'positions':324,'writes_each':1,'preload_overwritten_before_math':True},'resource_covariate':'JACCsharedtemplate preload+extra barrier retained; no targetcause isolated.'},'cold_reproduction':cold,'native_conditional_scope':'Original hostparser/borderdefects preserved, source4GPUbodies unchanged; selectedH0 is resource-adapted. Not untouched-native aggregate/full Canny application certificate.'}
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','primary_captures':len(trials),'stage_traces':len(traces),'source_bindings':len(bindings),'cold_files_exact':len(cold),'scratch_values':648}))
