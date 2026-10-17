#!/usr/bin/env python3
"""Full Float64 grouped-attention oracle, layout and convex-combination checks."""
from array import array
from pathlib import Path
import argparse,hashlib,json,math,sys
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.result.read_text());assert sys.byteorder=='little';prefix=array('f');gold=[];bounds=[];cases=[];wrong_group_max=0.;wrong_group_failed=0;copied_first_failed=0
for kind in [1,2]:
 for n in [1,3,65,127]:
  Q=[((i*7+3)%31-15)/16 for i in range(2048)];K=[((i*11+5)%29-14)/16 for i in range(256*n)];V=[((i*13+7)%27-13)/16 for i in range(256*n)]
  prefix.extend([kind,n,32,4,64]);prefix.extend(Q);prefix.extend(K);prefix.extend(V)
  for h in range(32):
   def head(kh):
    logits=[math.fsum(Q[h*64+c]*K[t*256+kh*64+c] for c in range(64))/8 for t in range(n)]
    raw=[math.exp(z-max(logits)) for z in logits];den=math.fsum(raw);weights=[z/den for z in raw]
    assert abs(math.fsum(weights)-1)<2e-15 and all(w>0 for w in weights)
    return [math.fsum(w*V[t*256+kh*64+c] for t,w in enumerate(weights)) for c in range(64)]
   vals=head(h//8);bad=head(h%4)
   for c,g in enumerate(vals):
    gold.append(g);bounds.append((min(V[t*256+(h//8)*64+c] for t in range(n)),max(V[t*256+(h//8)*64+c] for t in range(n))))
    wrong_group_max=max(wrong_group_max,abs(g-bad[c]));wrong_group_failed+=abs(g-bad[c])>.002+.01*abs(g)
    copied_first_failed+=abs(g-V[(h//8)*64+c])>.002+.01*abs(g)
  cases.append({'storage':'float16' if kind==1 else 'bfloat16','compute':'float32','n':n,'q_heads':32,'kv_heads':4,'head_dim':64,'layout':'NHD','groupsize':8,'native_blocks':4,'native_threads':[8,8,8],'native_dynamic_shared_bytes':16896,'native_V100_pipeline_stages':1,'kv_tiles':(n+63)//64,'valid_sequence_elements':n,'output_count':2048})
assert len(prefix)==217128 and len(gold)==16384
rows=[]
for model,md in d['models'].items():
 for t in md.get('trials',[]):
  raw=Path(t['output_artifact']['saved_path']).read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==t['output_artifact']['sha256'];assert raw[:len(prefix)*4]==prefix.tobytes();x=array('f');x.frombytes(raw[len(prefix)*4:]);assert len(x)==len(gold)
  errors=[abs(v-g) for v,g in zip(x,gold)];assert all(math.isfinite(v) for v in x);assert all(e<=.002+.01*abs(g) for e,g in zip(errors,gold));assert all(lo<=v<=hi for v,(lo,hi)in zip(x,bounds))
  assert list(x[:2048])==gold[:2048] and list(x[8192:10240])==gold[8192:10240]
  rows.append({'model':model,'trial':t['trial'],'artifact_sha256':digest,'actual_values':16384,'max_abs_error':max(errors),'max_relative_error_with_0.002_floor':max(e/max(abs(g),.002)for e,g in zip(errors,gold)),'maximum_acceptance_budget_fraction':max(e/(.002+.01*abs(g))for e,g in zip(errors,gold)),'convex_bounds_all':True,'seq1_exact_copy_of_correct_group_V':True})
assert wrong_group_failed>0 and copied_first_failed>0
r={'benchmark':'attentionMultiHeadKVCache','result':str(a.result.resolve()),'input_prefix_count':217128,'actual_output_count':16384,'artifact_count':len(rows),'all_full_outputs_within_fixed_budget':True,'rows':rows,'cases':cases,'oracle':'PythonFloat64 math.fsum dotproducts and normalized natural-exp probabilities; independentofsourceFloat32eightlaneXOR/tree andonlinebase2softmax merging. ActualstorageFP16/BF16inputdyadics exact; FP32artifactencodinglosslessforactualoutputstorage.','rejected_shortcuts':{'wrong_kv_group_h_mod4_instead_h_div8_failed_values':wrong_group_failed,'wrong_group_max_abs_difference':wrong_group_max,'copy_first_V_for_every_sequence_failed_values':copied_first_failed,'meaning':'These distinguish groupedheadmapping andnontrivialnormalizedmulti-keyattention, notgeneralmodelaccuracy.'},'mathematical_properties':['All weights positive andnormalized inindependentoracle','EveryactualoutputinsideinputVconvexbounds','Sequence1actualoutputexactcorrectKVheadV'], 'scope':'32queryheads/4KVheads,dim64,NHD,noRope,seq1/3sourcecases+65/127multitileextensions,FP16/BF16storageFP32math. No partitionedlongKVcache, HND/Rope or fullLLM validation. V100nativecp_asyncusesguardedsynchronousfallback, notunsupported.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'artifacts':len(rows),'all_pass':True,'wrong_group_failed_values':wrong_group_failed,'copy_first_failed_values':copied_first_failed,'max_abs':max(r['max_abs_error']for r in rows)}))
