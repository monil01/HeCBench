#!/usr/bin/env python3
"""CPU-only full source-kernel output and independent adjoint/oracle audit."""
from pathlib import Path
import json,array,hashlib,sys,subprocess,tempfile,shutil,importlib.util
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;F=Path(sys.argv[1]);F=F if F.is_absolute()else ROOT/F;r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());s=importlib.util.spec_from_file_location('independent_shape_oracle',P/'generate_assets.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
def sha(b):return hashlib.sha256(b).hexdigest()
def arr(b):a=array.array('d');a.frombytes(b);return a
assert sys.byteorder=='little'and r['reference_agreement']and r['matched_performance_valid'];COUNT=plan['artifact_count'];PREFIX=plan['input_prefix_count'];expected=arr((P/'oracle_f64.bin').read_bytes());cache={};bindings={};trials=[];hashes=set();prefixes=set()
def full_math(a):
 key=sha(a.tobytes())
 if key in cache:return cache[key]
 off=0;allout=[]
 for case in plan['cases']:
  assert list(a[off:off+16])==[case[k]for k in o.FIELDS];off+=16;C,G,H,W,KH,KW,OH,OW=[case[k]for k in ['C','G','H','W','KH','KW','OH','OW']];S=OH*OW;K=KH*KW;inputs={}
  for name,n in [('image',C*H*W),('offset',G*2*K*S),('mask',G*K*S),('colgrad',C*K*S),('grad_image',C*H*W),('grad_offset',G*2*K*S),('grad_mask',G*K*S)]:
   vals=a[off:off+n+32];off+=n+32;assert list(vals[:16])==list(vals[-16:])==[777]*16
   if name.startswith('grad_'):assert list(vals[16:-16])==[0]*n
   else:inputs[name]=list(vals[16:-16])
  recipe=o.data(case);assert all(inputs[k]==recipe[k]for k in inputs);allout.append(o.oracle(case,inputs))
 assert off==PREFIX;maximum=0;fields=[]
 for case,outs in zip(plan['cases'],allout):
  for name in ['columns','grad_image','grad_offset','grad_mask']:
   target=o.wrap(outs[name]);got=a[off:off+len(target)];off+=len(target);errors=[abs(x-y)for x,y in zip(got,target)];maximum=max(maximum,max(errors));assert all(e<=2e-6+2e-6*abs(t)for e,t in zip(errors,target));assert list(got[:16])==list(got[-16:])==[777]*16;fields.append({'case':case['id'],'field':name,'count':len(target),'max_abs_error':max(errors)})
 assert off==COUNT;cache[key]={'independent_actual_input_shape_function_adjoint':True,'fields':fields,'max_abs_error':maximum,'source_primary_FD_probe_count':len(plan['finite_difference_probes']),'full_artifact_sha256':key};return cache[key]
for m,v in r['models'].items():
 assert v['complete']and v['timing_valid']and v['provenance_valid']and len(v['trials'])==5
 for p,h in v['adapter_file_hashes'].items():assert sha(Path(p).read_bytes())==h,p;bindings[p]=h
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);b=f.read_bytes();a=arr(b);assert len(a)==COUNT and sha(b)==t['output_artifact']['sha256']and a==expected and t['correct']and t['jit_excluded']and t['time_seconds']>0;full_math(a);hashes.add(sha(b));prefixes.add(sha(b[:PREFIX*8]));trials.append({'model':m,'trial':t['trial'],'artifact':str(f.relative_to(ROOT)),'sha256':sha(b),'count':len(a),'actual_input_prefix_values':PREFIX,'full_numerical_contract_pass':True,'observed_bit_exact':True})
assert len(trials)==20 and len(hashes)==len(prefixes)==1
names=list(plan['files'])+['study_plan.json'];cold=[]
with tempfile.TemporaryDirectory(prefix='dcn-CPU-cold-')as tmp:
 q=Path(tmp);shutil.copy2(P/'generate_assets.py',q/'generate_assets.py');subprocess.run([sys.executable,str(q/'generate_assets.py')],check=True,stdout=subprocess.DEVNULL)
 for name in names:
  a=(P/name).read_bytes();b=(q/name).read_bytes();assert a==b,name;cold.append({'path':name,'bytes':len(a),'sha256':sha(a),'byte_exact':True})
asym=json.loads((P/'asymmetric_padding_witness.json').read_text())
for k in ['original_log','original_raw_capture_local_only','full_CPU_expected','minimal_corrected_owned_source','minimal_patch','corrected_log','corrected_raw_capture_local_only']:
 ev=asym[k];assert sha((ROOT/ev['path']).read_bytes())==ev['sha256']
orig=arr((ROOT/asym['original_raw_capture_local_only']['path']).read_bytes());fixed=arr((ROOT/asym['corrected_raw_capture_local_only']['path']).read_bytes());gold=arr((ROOT/asym['full_CPU_expected']['path']).read_bytes());assert fixed==gold and sum(x!=y for x,y in zip(orig,gold))==60
out={'benchmark':'convolutionDeformable','result':str(F.relative_to(ROOT)),'result_sha256':sha(F.read_bytes()),'status':'pass_named_custom_kernel_triplet','GPU_runs_added':0,'all20_primary_bitidentical':True,'checked_values_per_trial':COUNT,'input_prefix_count':PREFIX,'trials':trials,'source_bindings':bindings,'independent_math':list(cache.values()),'cache_note':'Everyartifactfullyread/rehashed/compared; mathematicalreevaluation cachedonlyforidenticalcompleteartifactbytes. Fourbatchslices, not20distinctdatasets.','cold_reproduction':cold,'primary_finite_difference_probes':len(plan['finite_difference_probes']),'asymmetric_diagnostic':{'original_mismatched_values':60,'minimal_hostarg_correction_all_full_outputs_exact':True,'CPU_image_gradient_FD_probes':60,'native_original_primary_unchanged':True,'diagnostic_only':True,'timing_excluded':True},'acceptance_tolerances':{'atol':2e-6,'rtol':2e-6,'prefix_and_guards':'exact','observed_primary_errors':0},'scope':'Complete3customsourceGPUkernels/helpers/timedperbatchtriplet only; TorchConv2doffsetpredictor/sigmoid/matmuls/autograd/fullCLI notverified. Upstreamgradientcolumns independentCPUfixtureoutsideROI; equalpaddingprimary, nativeasymbug remainsoutsideprimary.'}
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','primary_captures':len(trials),'source_bindings':len(bindings),'cold_files_exact':len(cold),'primary_FD_probes':len(plan['finite_difference_probes']),'asym_FD_probes':60,'native_asym_wrong_values':60}))
