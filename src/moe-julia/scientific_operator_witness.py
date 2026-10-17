#!/usr/bin/env python3
"""Audit all saved routing arrays, exact integer outputs and normalization scope."""
from pathlib import Path
from array import array
import argparse,json,math,hashlib,sys
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.result.read_text());P=Path(__file__).resolve().parent;assert sys.byteorder=='little'
def read(kind,p):v=array(kind);v.frombytes(p.read_bytes());return v
inputs=read('f',P/'moe_input_float32.bin');storedgold=read('d',P/'moe_expected_float64.bin');prefix=array('d');gold=array('d');layouts=[];special=[];tie_wrong=0
for pattern in [0,1]:
 data=inputs[pattern*98688:(pattern+1)*98688];probs=[];ordered=[]
 for t in range(257):
  logits=[float(data[t*384+e])for e in range(384)];m=max(logits);exp=[math.exp(x-m)for x in logits];den=math.fsum(exp);row=[x/den for x in exp];probs.extend(row);ordered.append(sorted(range(384),key=lambda e:(-row[e],e)))
 for k in [1,2,4,8]:
  prefix.extend([pattern,k,257,384]);prefix.extend(float(x)for x in data);prefix.extend([0]*(257*k));start=len(gold);n=257*k;gold.extend(probs);gold.extend(probs[t*384+ordered[t][j]]for t in range(257)for j in range(k));gold.extend(ordered[t][j]for t in range(257)for j in range(k));gold.extend(j*257+t for t in range(257)for j in range(k));layouts.append({'pattern':pattern,'k':k,'output_offset':start,'output_count':98688+3*n,'probability_count':98688,'topk_count':n,'kernel_definitions':2,'threads_per_block':256,'blocks':257,'softmax_normalization_domain':384,'selected_weight_renormalization':False})
  for t in range(257):
   wrong=sorted(range(384),key=lambda e:(-probs[t*384+e],-e));tie_wrong+=sum(ordered[t][j]!=wrong[j] for j in range(k))
  if pattern==1 and k==8:special.append({'pattern':pattern,'k':k,'row':0,'selected_indices':ordered[0][:k],'ideal_original_weight_each':1/384,'ideal_original_selected_mass':k/384,'renormalized_weight_each':1/k,'renormalized_selected_mass':1,'constant_one_expert_output_ratio_renormalized_over_source':384/k,'meaning':'Pure mathematical downstreamconstant-expertwitness. Sourceall-expertsoftmaxthenselectswithoutsubsetrenormalization; identicalindices canconceal48xmagnitudechange. Noactualexpertnetwork/fullmodelvalidated.'})
assert len(prefix)==797246 and len(gold)==812634 and gold.tobytes()==storedgold.tobytes();rows=[]
for model,md in d['models'].items():
 for t in md.get('trials',[]):
  raw=Path(t['output_artifact']['saved_path']).read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==t['output_artifact']['sha256'];assert raw[:len(prefix)*8]==prefix.tobytes();out=array('d');out.frombytes(raw[len(prefix)*8:]);assert len(out)==812634;maxabs=maxrel=maxsum=0
  for c in layouts:
   off=c['output_offset'];k=c['k'];n=c['topk_count'];got=out[off:off+c['output_count']];g=gold[off:off+c['output_count']]
   assert list(got[98688+n:])==list(g[98688+n:]);assert all(math.isfinite(v)for v in got)
   for v,x in zip(got[:98688+n],g[:98688+n]):
    err=abs(v-x);assert err<=1e-6+1e-5*abs(x);maxabs=max(maxabs,err);maxrel=max(maxrel,err/max(abs(x),1e-6))
   for token in range(257):
    row=got[token*384:(token+1)*384];assert all(0<=v<=1 for v in row);err=abs(math.fsum(row)-1);assert err<=1e-5;maxsum=max(maxsum,err)
    weights=got[98688+token*k:98688+(token+1)*k];ids=got[98688+n+token*k:98688+n+(token+1)*k];assert len(set(ids))==k;assert all(weights[j]>=weights[j+1]for j in range(k-1));assert all(weights[j]==row[int(ids[j])]for j in range(k))
  rows.append({'model':model,'trial':t['trial'],'sha256':digest,'exact_prefix_values':797246,'actual_output_values':812634,'integer_outputs_exact':True,'all_weights_equal_selected_probabilities':True,'max_abs_error':maxabs,'max_relative_error_with_1e_6_floor':maxrel,'max_probability_sum_error':maxsum})
resourcepath=Path(__file__).resolve().parents[2]/'.porting-state/portable_julia/runs/20261017T223724274733Z_moe/cuda_run1.log';resource=None
for line in resourcepath.read_text().splitlines():
 if line.startswith('STUDY_RESOURCE '):resource=json.loads(line.removeprefix('STUDY_RESOURCE '))
r={'benchmark':'moe','result':str(a.result.resolve()),'all_saved_artifacts_pass':True,'artifacts':len(rows),'actual_output_count':812634,'input_prefix_count':797246,'artifact_bytes':12879040,'oracle':'PythonFloat64naturalexp,math.fsum andstablefullsort, regeneratedfromcanonicalinputs withbinarygoldhashmatch; notsharedGPUreduction/topKimplementation. AllintegerIDs comparedexactly; commonfloatbudget <.021 formaxid2055cannotacceptunitIDerror.','rows':rows,'cases':layouts,'normalization_scope_counterexample':special,'highest_id_tie_break_failed_integer_values':tie_wrong,'native_compiled_resources':resource,'resource_scope':'NativecudaFuncGetAttributesactualcompiledkernelmetadata, notdynamicprofiling/occupancy/DRAMtraffic; Juliaresourcepolicy source-derived.','scope':'T257/E384,K1/2/4/8, boundedlogits+source-scaledextremeties. OriginalCLI32768notexecuted; finishednull/full-expertrange only. No completeMoE/scientificsurrogatevalidation.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'artifacts':len(rows),'allpass':True,'tie_break_counterexample_values':tie_wrong,'native_resources':resource,'maxabs':max(x['max_abs_error']for x in rows)}))
