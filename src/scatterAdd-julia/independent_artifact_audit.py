from pathlib import Path
import json,struct,hashlib,math
p=Path(__file__).resolve().parent;root=p.parents[1];bundle=root/'.porting-state/portable_julia/runs/20261018T020355498899Z_scatterAdd';d=json.loads((bundle/'result.json').read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();vals=struct.unpack('<'+str(meta['prefix_count'])+'f',prefix);gold=struct.unpack('<'+str(meta['output_count'])+'d',(p/'expected_outputs.bin').read_bytes());pos=1;expected=[];cases=[]
for case in meta['cases']:
 n,o,v=map(int,vals[pos:pos+3]);pos+=3;idx=vals[pos:pos+n];pos+=n;src=vals[pos:pos+n*v];pos+=n*v;init=vals[pos:pos+o*v];pos+=o*v;assert all(x==int(x) and 0<=x<o for x in idx);su=[0]*(o*v);ab=[0]*(o*v)
 for i,rawidx in enumerate(idx):
  index=int(rawidx)
  for j in range(v):
   x=src[i*v+j]*16;assert x==int(x);k=index*v+j;su[k]+=int(x);ab[k]+=abs(int(x))
 initu=[int(x*16) for x in init];assert all(x*16==u for x,u in zip(init,initu));bound=max(abs(a)+3*b for a,b in zip(initu,ab));assert bound==case['max_absolute_partial_units_bound'] and bound<2**24;answer=[(a+3*b)/16 for a,b in zip(initu,su)]
 for mode in [1,2]:expected.extend(answer+[-8.]*17)
 cases.append({'n':n,'o':o,'v':v,'initial':init,'source_component_units':[sum(int(src[i*v+j]*16) for i in range(n)) for j in range(v)],'bound':bound})
assert pos==len(vals) and tuple(expected)==gold
checks=[];timings={}
for model,v in d['models'].items():
 ts=v['trials'];assert len(ts)==5 and all(t['correct'] for t in ts);assert math.isclose(v['mean_seconds'],sum(t['time_seconds'] for t in ts[1:])/4,rel_tol=1e-14)
 timings[model]={'all_five_path_times_seconds':[t['per_path_times_seconds'] for t in ts],'mean_path_times_last4_seconds':[sum(t['per_path_times_seconds'][i] for t in ts[1:])/4 for i in range(10)],'mean_sum_seconds':v['mean_seconds']}
 for t in ts:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;got=struct.unpack('<'+str(len(gold))+'f',raw[len(prefix):]);assert got==gold and all(math.isfinite(x) for x in got);off=0
  for c in cases:
   n,o,vv=c['n'],c['o'],c['v'];nval=o*vv
   for mode in [1,2]:
    y=got[off:off+nval];assert got[off+nval:off+nval+17]==(-8.,)*17;off+=nval+17
    for j in range(vv):assert sum(int((y[k*vv+j]-c['initial'][k*vv+j])*16) for k in range(o))==3*c['source_component_units'][j]
  assert off==len(got);checks.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'prefix_exact':True,'all_values_exact':True,'all_170_guards_exact':True,'component_conservation_exact':True})
rec={'benchmark':'scatterAdd','run':d['run_id'],'artifacts_checked':20,'input_prefix_count':len(vals),'input_prefix_bytes':len(prefix),'output_count':len(gold),'guards_per_artifact':170,'all_artifacts_bit_identical':len({x['sha256'] for x in checks})==1,'checks':checks,'per_model_timings':timings,'proof':'Everycontribution/init multiple1/16. Anyshared/globalpartialintegerunitmagnitude boundedby|initunits|+3*sumabs(sourceunits)<2^24, soFP32adds exactforallatomicorders; independentintegerhistogram matchesFloat64gold/allGPUvalues. Componentconservationseparatelychecked. ThisdoesnotcertifyarbitraryFP32roundingbehavior.','source_scope':'Bothnativevariants/source8192chunks/floorN8192+1incl emptytail, 5namedshapes atmost3blocks, no default10millionbatch/saturatedthroughputclaim.'};f=p/'independent_artifact_audit.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2)+'\n');tmp.replace(f);print({'all20_bit_identical':rec['all_artifacts_bit_identical'],'means_us':{m:v['mean_sum_seconds']*1e6 for m,v in timings.items()}})
