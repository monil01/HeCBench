"""Read-only full SRAD fields, exact inputs/boundaries/guards and PDE controls."""
from pathlib import Path
import struct,json,hashlib,math,sys
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;bundle=r/'.porting-state/portable_julia/runs'/sys.argv[1];d=json.loads((bundle/'result.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(f,n):
 b=Path(f).read_bytes();assert len(b)==8*n;return list(struct.unpack('<'+str(n)+'d',b))
pre=load(p/'fixture.bin',69310);gold=load(p/'oracle.bin',746001);fd=json.loads((p/'fixture_description.json').read_text());pos=1;inputs=[]
for c in fd['cases']:
 nr,nc,steps=map(int,pre[pos:pos+3]);lam=pre[pos+3];pos+=4;N=nr*nc;raw=pre[pos:pos+N];pos+=N;maps=[]
 for n in [nr,nr,nc,nc]:maps.append(pre[pos:pos+n]);pos+=n
 assert maps==[[max(i-1,0) for i in range(nr)],[min(i+1,nr-1) for i in range(nr)],[min(i+1,nc-1) for i in range(nc)],[max(i-1,0) for i in range(nc)]];inputs.append((raw,maps))
assert pos==len(pre)
rows=[];hashes=set();global_metrics={};maxbudget=0.;maxmass=0.;payload_cache={}
for model,v in d['models'].items():
 assert v['complete'] and len(v['trials'])==5
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);a=load(f,815311);assert a[:69310]==pre and t['correct'];h=sha(f);hashes.add(h)
  # Identical complete saved arrays may share this CPU calculation only; every
  # artifact is fully read/rehashed above and each execution/timing remains.
  if h in payload_cache:cases=payload_cache[h]
  else:
   out=a[69310:];gp=0;cases=[]
   for ci,c in enumerate(fd['cases']):
    nr,nc,N,steps=c['nr'],c['nc'],c['pixels'],c['steps'];raw,maps=inputs[ci];fields=[*[(z,N) for z in ['compressed','precompress','sum_workspace','sum2_workspace','dN','dS','dE','dW','coefficient','raw_input_echo','extracted_initial']],('iN_echo',nr),('iS_echo',nr),('jE_echo',nc),('jW_echo',nc),('statistics',5*steps),('guards',192)];actual={};metrics={}
    for name,n in fields:
     vals=out[gp:gp+n];expected=gold[gp:gp+n];assert len(vals)==n and all(math.isfinite(x) for x in vals);errors=[abs(x-y) for x,y in zip(vals,expected)];budget=[1e-4+1e-5*abs(y) for y in expected];norm=max((e/b for e,b in zip(errors,budget)),default=0);assert norm<=1;maxbudget=max(maxbudget,norm)
     if name.endswith('_echo') or name=='guards':assert vals==expected
     metrics[name]=dict(max_absolute_error=max(errors,default=0),max_relative_error_nonzero_reference=max((e/abs(y) for e,y in zip(errors,expected) if y),default=0),max_zero_reference_absolute_error=max((e for e,y in zip(errors,expected) if y==0),default=0),max_budget_fraction=norm)
     for key,value in metrics[name].items():global_metrics.setdefault(name,{})[key]=max(global_metrics.get(name,{}).get(key,0),value)
     actual[name]=vals;gp+=n
    assert actual['raw_input_echo']==raw and [actual[k] for k in ['iN_echo','iS_echo','jE_echo','jW_echo']]==maps and actual['guards']==[16384]*192
    coeff=actual['coefficient'];assert all(0<=x<=1 for x in coeff)
    for col in range(nc):assert actual['dN'][col*nr]==0 and actual['dS'][col*nr+nr-1]==0
    for row in range(nr):assert actual['dW'][row]==0 and actual['dE'][(nc-1)*nr+row]==0
    initial=actual['extracted_initial'];final=actual['precompress'];mass0=math.fsum(initial);mass1=math.fsum(final);massrel=abs(mass1-mass0)/abs(mass0);maxmass=max(maxmass,massrel);assert massrel<=3e-6;assert min(final)>=min(initial)-1e-5 and max(final)<=max(initial)+1e-5
    assert all(x>0 for x in actual['statistics'][3::5]) and all(x>0 for x in actual['statistics'][4::5]);assert all(x>0 for x in final)
    # Compression is independently checked against the actual final pre-log
    # values, not only the separately evolved Float64 PDE oracle.
    compression_error=max(abs(x-255*math.log(y)) for x,y in zip(actual['compressed'],final));assert compression_error<=1e-4+1e-5*max(abs(x) for x in actual['compressed'])
    cases.append(dict(case=ci+1,all_fields=metrics,actual_initial_extracted_mass=mass0,actual_final_precompressed_mass=mass1,mass_relative_drift=massrel,maximum_principle_final=True,positive_image_and_variance=True,all_coefficients_in_zero_one=True,all_reflecting_boundary_derivatives_exact_zero=True,all_input_and_neighbor_bits_and_guards_exact=True,compression_max_abs_error_vs_actual_precompress=compression_error))
   assert gp==len(gold);payload_cache[h]=cases
  rows.append(dict(model=model,trial=t['trial'],artifact=dict(path=str(f.relative_to(r)),sha256=h,bytes=f.stat().st_size),cases=cases,all_pass=True))
out=dict(benchmark='srad',configuration_id=d['configuration_id'],native_reference_repaired=True,configuration_role='corrected_reference_secondary',all_pass=True,all20_checked=True,artifact_values=815311,input_prefix_values=69310,artifact_hash_classes=len(hashes),max_oracle_budget_fraction=maxbudget,max_actual_final_mass_relative_drift=maxmass,guard_words_per_trial=768,all_actual_raw_inputs_neighbor_bits_boundary_derivatives_and_guards_exact=True,field_error_metrics=global_metrics,scientific_oracle='Float64mathematicalstatistics/fullworkspace andconservativepairwiseedgefluxassembly, notsamepointwisestencil. Actualinitial/finalextractedimage mass+maxprinciple/compression alsoindependentlychecked. Alliterationscalarstatisticscaptured, fullactualintermediateimagepathsnotcaptured.',unverified='Uniform/nearzero-varianceimages,1000step/PGMIO/resize/clinicalimagequality; F32engineeringbudgetnotformalbound. Conditionalcorrectednativeonly.',result_sha256=sha(bundle/'result.json'),config_sha256=sha(bundle/'config.json'),rows=rows)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['field_error_metrics','rows']})
