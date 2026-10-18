from pathlib import Path
import struct,json,math,hashlib
p=Path(__file__).resolve().parent;root=p.parents[1];bundle=root/'.porting-state/portable_julia/runs/20261018T015129856599Z_softmax-fused';d=json.loads((bundle/'result.json').read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();x=struct.unpack('<'+str(meta['prefix_count'])+'f',prefix);gold=struct.unpack('<'+str(meta['output_count'])+'d',(p/'expected_outputs.bin').read_bytes());frames=[];pos=1;allreal=[]
for ci in range(8):
 kind,b,h,q,c=map(int,x[pos:pos+5]);scale=x[pos+5];n=int(x[pos+6]);pos+=7;inp=x[pos:pos+n];pos+=n;mask=x[pos:pos+n];pos+=n
 for pad in [0,1]:
  real=[];masked=[];full=[]
  for row in range(b*h*q):
   mr=row%q if pad else (row//(h*q))*q+row%q;vals=[v*scale if mask[mr*c+j]!=1 else -10000. for j,v in enumerate(inp[row*c:(row+1)*c])];mx=max(vals);e=[math.exp(v-mx) for v in vals];den=math.fsum(e);real.extend([0.]*c if mx==-10000 else [v/den for v in e]);masked.extend([mask[mr*c+j]==1 for j in range(c)]);full.append(mx==-10000)
  frames.append((kind,b,h,q,c,real,masked,full));allreal.extend(real+[-8.]*17)
assert pos==len(x)
checks=[];timings={}
for model,v in d['models'].items():
 ts=v['trials'];assert len(ts)==5 and all(t['correct'] for t in ts);assert math.isclose(v['mean_seconds'],sum(t['time_seconds'] for t in ts[1:])/4,rel_tol=1e-14)
 timings[model]={'all_five_path_times_seconds':[t['per_path_times_seconds'] for t in ts],'mean_path_times_last4_seconds':[sum(t['per_path_times_seconds'][i] for t in ts[1:])/4 for i in range(16)],'mean_sum_seconds':v['mean_seconds']}
 for t in ts:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;got=struct.unpack('<'+str(meta['output_count'])+'f',raw[len(prefix):]);assert all(math.isfinite(v) for v in got);ma=max(abs(v-g) for v,g in zip(got,gold));assert ma<=1e-3;off=0;guards=0;typed={}
  for kind,b,h,q,c,real,masked,full in frames:
   name=['Float16','BFloat16'][kind];m=typed.setdefault(name,{'max_abs_rounded_oracle_error':0.,'max_abs_unrounded_probability_error':0.,'max_nonmasked_row_mass_error':0.,'allmasked_rows':0,'masked_probabilities_exact_zero':True});n=b*h*q*c;y=got[off:off+n];g=gold[off:off+n];assert got[off+n:off+n+17]==(-8.,)*17;guards+=17;off+=n+17;assert all(0<=v<=1 for v in y);assert all(v==0 for v,maskedj in zip(y,masked) if maskedj);m['max_abs_rounded_oracle_error']=max(m['max_abs_rounded_oracle_error'],max(abs(v-r) for v,r in zip(y,g)));m['max_abs_unrounded_probability_error']=max(m['max_abs_unrounded_probability_error'],max(abs(v-r) for v,r in zip(y,real)))
   for row,isfull in enumerate(full):
    mass=math.fsum(y[row*c:(row+1)*c])
    if isfull:assert mass==0.;m['allmasked_rows']+=1
    else:
     err=abs(mass-1.);m['max_nonmasked_row_mass_error']=max(m['max_nonmasked_row_mass_error'],err);resolution_bound=(2**-10+c*2**-24) if kind==0 else (2**-7+c*2**-133);assert err<=resolution_bound
  assert off==len(got) and guards==272
  checks.append({'model':model,'trial':t['trial'],'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'prefix_exact':True,'all_272_guards_exact':True,'typed_metrics':typed})
rec={'benchmark':'softmax-fused','run':d['run_id'],'artifacts_checked':20,'input_prefix_count':len(x),'input_prefix_bytes':len(prefix),'output_count':len(gold),'probability_count':len(gold)-272,'guards_per_artifact':272,'Julia_artifacts_bit_identical':len({v['sha256'] for v in checks if v['model']!='cuda'})==1,'checks':checks,'per_model_timings':timings,'secondary_mass_contract':'Nonallmasked row mass abs<=eps(storage)+C*minsubnormal(storage), conservative storage-resolution check; allmasked and everymasked probability exactzero. Primarysource1e-3compares actualstored outputswithquantizedFloat64oracle, notunroundedrealprobabilities.','historical_wrong_BERT_mask_CPU_witness_max_probability_error':meta['historical_wrong_BERT_mask_CPU_witness_max_probability_error'],'scope':meta['domain']};f=p/'independent_artifact_audit.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(rec,indent=2)+'\n');tmp.replace(f);print({'julia_exact':rec['Julia_artifacts_bit_identical'],'typed':checks[-1]['typed_metrics'],'means_us':{k:v['mean_sum_seconds']*1e6 for k,v in timings.items()}})
