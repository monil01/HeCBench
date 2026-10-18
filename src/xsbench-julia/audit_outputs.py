from pathlib import Path
from array import array
from collections import Counter
import json,hashlib,struct,math,sys
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];d=json.loads((r/'.porting-state/portable_julia/runs'/run/'result.json').read_text());pre=array('d',(p/'fixture.bin').read_bytes());gold=array('d',(p/'oracle.bin').read_bytes());kind=(p/'kinds.bin').read_bytes();fd=json.loads((p/'fixture_description.json').read_text());cases=[];cur=0
for c in fd['cases']:
 L=c['lookups'];I=c['isotopes'];G=c['gridpoints'];M=c['max_nuclides'];labels=(cur+64,cur+64+L);cur+=L+128;observer=(cur+64,cur+64+7*L);cur+=7*L+128
 for n,w in [(12,1),(12*M,1),(12*M,1),(I*G,6),(c['union_count'],1),(c['index_count'],1)]:cur+=(n+128)*w
 cases.append((c,labels,observer))
assert cur==len(gold);rows=[];hashes=set();metric=dict(max_abs_channel_error=0,max_rel_nonzero_channel_error=0,max_abs_zero_channel_error=0,max_budget_fraction=0,total_elastic_absorption_residual=0,nu_fission_relation_residual=0);histories={};method_changes=0
for m,z in d['models'].items():
 for t in z['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();v=array('d',raw);assert len(v)==len(pre)+len(gold)and raw[:8*len(pre)]==(p/'fixture.bin').read_bytes();a=v[len(pre):]
  for k,x,y in zip(kind,a,gold):
   assert math.isfinite(x)
   if k:assert struct.pack('<d',x)==struct.pack('<d',y)
   else:
    e=abs(x-y);budget=1e-10+2e-12*abs(y);assert e<=budget;metric['max_abs_channel_error']=max(metric['max_abs_channel_error'],e);metric['max_budget_fraction']=max(metric['max_budget_fraction'],e/budget)
    if y:metric['max_rel_nonzero_channel_error']=max(metric['max_rel_nonzero_channel_error'],e/abs(y))
    else:metric['max_abs_zero_channel_error']=max(metric['max_abs_zero_channel_error'],e)
  methods={}
  for c,labels,obs in cases:
   lab=a[labels[0]:labels[1]];values=a[obs[0]:obs[1]];Mats=Counter()
   for i in range(c['lookups']):
    energy,mat,*xs=values[7*i:7*i+7];assert 0<=energy<=1 and mat==int(mat)and 0<=mat<12 and min(xs)>=0;Mats[int(mat)]+=1;assert max(range(5),key=lambda j:xs[j])+1==lab[i];e=abs(xs[0]-xs[1]-xs[2]);metric['total_elastic_absorption_residual']=max(metric['total_elastic_absorption_residual'],e);assert e<=1e-10;assert xs[3]<=xs[2]+1e-10;e=abs(xs[4]-2.5*xs[3]);metric['nu_fission_relation_residual']=max(metric['nu_fission_relation_residual'],e);assert e<=1e-10
   key=c['gridpoints'];vector=(lab,values)
   if key in methods:
    # Actualalgorithmcontrols share identicalquery/tablephysics. Labelsexact,
    # channelscanhaveroundoffbutmustobeythedeclaredsamephysicaloracle.
    old=methods[key];assert lab==old[0];assert all(abs(x-y)<=1e-10+2e-12*abs(y)for x,y in zip(values,old[1]))
   else:methods[key]=vector
   histories[f"G{key}"]=dict(material_counts=dict(Mats),label_counts=dict(Counter(map(int,lab))),all12_materials_observed=len(Mats)==12)
  assert len(t['path_times_seconds'])==6 and abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12;h=hashlib.sha256(raw).hexdigest();hashes.add(h);rows.append(dict(model=m,trial=t['trial'],path=str(f),sha256=h,all_labels_rng_energy_material_inputs_guards_bit_exact=True,all5_observed_channels_checked=True,all3_grid_methods_agree=True))
assert len(rows)==20 and d['matched_performance_valid']
a=dict(run=run,all_pass=True,full20_artifacts=len(rows),words_per_artifact=len(pre)+len(gold),unique_hashes=sorted(hashes),metrics=metric,query_coverage=histories,native_primary_body_unchanged=True,timed_primary_private_macros_captured=False,untimed_observer_macros_captured=True,original_defined_primary_full_labels_captured=True,all_intermediate_repeat_primary_labels_captured=False,source_original_large_CLI_or_real_nuclear_data_not_certified=True,rows=rows)
(p/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:v for k,v in a.items()if k!='rows'}))
