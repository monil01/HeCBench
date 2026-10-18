"""Full saved outputs plus a separate cancellation-aware Float64 algebraic oracle."""
from pathlib import Path
import json,math,struct,hashlib
p=Path(__file__).resolve().parent;root=p.parents[1];bundle=root/'.porting-state/portable_julia/runs/20261018T005024489524Z_scel';r=json.loads((bundle/'result.json').read_text());desc=json.loads((p/'fixture_description.json').read_text());fb=(p/'fixture.bin').read_bytes();f=struct.unpack('<%df'%(len(fb)//4),fb);gb=(p/'expected_outputs.bin').read_bytes();g=struct.unpack('<%dd'%(len(gb)//8),gb);pos=1;robust=[];pathstats=[];frames=[]
for case in range(5):
 outer,inner,n=map(int,f[pos:pos+3]);pos+=3;logits=f[pos:pos+n];pos+=n;targets=f[pos:pos+n];pos+=n;frames.append((outer,inner))
 for mode in range(3):
  vals=[]
  for row in range(outer):
   terms=[]
   for j in range(inner):
    i=row*inner+j;l=logits[i];t=targets[i];tail=math.log1p(math.exp(-abs(l)))
    if mode==0:value=math.fsum([(1-t)*l if l>=0 else -t*l,tail])
    elif mode==1:value=(2*t-1)*math.fsum([max(-l,0.),tail])
    else:value=math.fsum([(1-2*t)*l if l>=0 else -t*l,(1-t)*tail])
    terms.append(value)
   vals.append(math.fsum(terms)/inner)
  robust.extend(vals+[-12345.]*17);pathstats.append({'case':case,'mode':mode,'outer':outer,'inner':inner,'secondary_oracle_min':min(vals),'secondary_oracle_max':max(vals),'negative_rows':sum(x<0 for x in vals),'standard_BCE_nonnegativity_applicable':case<4 and mode==0})
assert pos==len(f);assert len(robust)==len(g);primary_secondary=max(abs(x-y) for x,y in zip(g,robust));checks=[];hashes={};timings={};native_cpu=0
for model,m in r['models'].items():
 hashes[model]=[];rows=[]
 for t in m['trials']:
  file=Path(t['output_artifact']['saved_path']);data=file.read_bytes();assert data[:len(fb)]==fb;out=struct.unpack('<%df'%len(g),data[len(fb):]);assert all(math.isfinite(x) for x in out);ma=max(abs(x-y) for x,y in zip(out,g));secondary=max(abs(x-y) for x,y in zip(out,robust));assert ma<=1e-3 and secondary<=1e-3;offset=0;per=[]
  for c,(outer,inner) in enumerate(frames):
   for mode in range(3):
    vals=out[offset:offset+outer];assert out[offset+outer:offset+outer+17]==(-12345.,)*17
    if c<4 and mode==0:assert min(vals)>=0
    per.append(max(abs(x-y) for x,y in zip(vals,robust[offset:offset+outer])));offset+=outer+17
  assert offset==len(g);h=hashlib.sha256(data).hexdigest();hashes[model].append(h);checks.append({'model':model,'trial':t['trial'],'sha256':h,'input_prefix_bytes_exact':True,'all678_outputs_guards':True,'max_primary_abs_error':ma,'max_secondary_abs_error':secondary,'per_path_secondary_abs_error':per,'all_applicable_standard_BCE_nonnegative':True});rows.append(t['per_path_times_seconds']);native_cpu=max(native_cpu,t.get('source_cpu_max_abs_error',0))
 timings[model]={'all5_per_path_times_seconds':rows,'mean_last4_per_path_seconds':[sum(row[i] for row in rows[1:])/4 for i in range(15)],'mean_sum_seconds':m['mean_seconds']}
assert hashlib.sha256(fb).hexdigest()==desc['fixture_sha256'] and hashlib.sha256(gb).hexdigest()==desc['gold_sha256']
a={'benchmark':'scel','bundle':str(bundle.relative_to(root)),'prefix_count':len(f),'output_count':len(g),'all20_full_checks':len(checks)==20,'all_Julia_artifacts_bit_identical':len(set(hashes['cuda_jl']+hashes['ka']+hashes['jacc']))==1,'all_native_artifacts_bit_identical':len(set(hashes['cuda']))==1,'hashes':hashes,'max_abs_to_primary_oracle':max(x['max_primary_abs_error'] for x in checks),'max_abs_to_secondary_oracle':max(x['max_secondary_abs_error'] for x in checks),'source_native_CPU_Float32_maxdifference':native_cpu,'primary_vs_secondary_Float64_maxdifference':primary_secondary,'secondary_oracle':'Separates log1p(exp(-abs(l))) tail beforecancellation for saturatedvalidlabels. Standardpositive:(1-t)*l+tail; negative:-t*l+tail. logD:(2*t-1)*softplus(-l). Unjoinedpositive:(1-2*t)*l+(1-t)*tail; negative:-t*l+(1-t)*tail. math.fsumrowmean. Postcaptureaudit, notchangedprimaryexecution/tolerance.','near_zero_metric':'Relativeerrordividesbyfloor1e-12 andcanbelarge fornearzero losses; acceptancepredeclaredabsolute1e-3, actualabserrorsrecorded. No relativeaccuracyclaim.','full_checks':checks,'path_statistics':pathstats,'per_model_timings':timings,'scope':'15finiteforwardoperatorpaths, fullallrowoutputs. Modesare distinctformulas; auxiliary/source-unboundedtargetobjectives neednotnonnegative. No gradients/training/taskaccuracy/defaultCLIcertification.'}
(p/'independent_artifact_audit.json').write_text(json.dumps(a,indent=2)+'\n');print({k:a[k] for k in ('all_Julia_artifacts_bit_identical','max_abs_to_primary_oracle','max_abs_to_secondary_oracle','source_native_CPU_Float32_maxdifference','primary_vs_secondary_Float64_maxdifference')})
