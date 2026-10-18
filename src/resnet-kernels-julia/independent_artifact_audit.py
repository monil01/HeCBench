from pathlib import Path
import json,hashlib,numpy as np
p=Path('src/resnet-kernels-julia');bundle=Path('.porting-state/portable_julia/runs/20261018T002436019929Z_resnet-kernels');d=json.loads((bundle/'result.json').read_text());desc=json.loads((p/'fixture_description.json').read_text());fixture=np.fromfile(p/'fixture.bin',dtype='<f4');gold=np.fromfile(p/'expected_outputs.bin',dtype='<f8');checks=[];hashes={};mx={};stats=[];cursor=0
for case in desc['cases']:
 for j,n in enumerate(case['output_sizes']):
  z=gold[cursor:cursor+n];stats.append({'mode':case['mode'],'output_index':j,'elements':n,'min':float(z.min()),'max':float(z.max()),'zero_count':int((z==0).sum()),'nonzero_count':int((z!=0).sum()),'oracle_dtype':'Float64'});cursor+=n+17
assert cursor==len(gold)
for model,rec in d['models'].items():
 rows=[];hashes[model]=[]
 for trial in rec['trials']:
  path=Path(trial['output_artifact']['saved_path']);a=np.fromfile(path,dtype='<f4');assert np.array_equal(a[:len(fixture)].view('u4'),fixture.view('u4'));out=a[len(fixture):].astype(float);assert len(out)==len(gold);assert np.isfinite(out).all();assert np.max(abs(out-gold))<=1e-4
  pos=0;errs=[]
  for case in desc['cases']:
   for j,n in enumerate(case['output_sizes']):
    vals=out[pos:pos+n];g=gold[pos:pos+n];assert np.array_equal(out[pos+n:pos+n+17],np.full(17,-12345.));errs.append(float(np.max(abs(vals-g))))
    if case['mode']<2 and j==2:
     image=vals.reshape(16,16,case['co']);mask=np.ones((16,16),dtype=bool);mask[1:15,1:15]=False;assert (image[mask]==0).all()
    pos+=n+17
  assert pos==len(out);h=hashlib.sha256(path.read_bytes()).hexdigest();hashes[model].append(h);checks.append({'model':model,'trial':trial['trial'],'sha256':h,'input_prefix_bytes_exact':True,'all_full_oracle_values':True,'all_output_guards_exact':True,'winograd_border_zeros_exact':True,'max_abs_error':float(np.max(abs(out-gold))),'per_output_max_abs_error':errs})
 mx[model]=max(x['max_abs_error'] for x in checks if x['model']==model)
 rows=[t['per_path_times_seconds'] for t in rec['trials']]
 (p/f'path_timing_{model}.json').write_text(json.dumps({'path_order':d['input_contract']['paths'],'all5':rows,'mean_last4':np.mean(np.array(rows)[1:],axis=0).tolist(),'sum_mean_last4':rec['mean_seconds']},indent=2)+'\n')
summary={'configuration_id':d['configuration_id'],'bundle':str(bundle),'prefix_count':len(fixture),'output_count':len(gold),'all20_full_checks':len(checks)==20,'source_convention_oracles':desc['independent_oracles'],'max_cpu_abs_error':mx,'hashes':hashes,'all_Julia_models_bitidentical':len(set(hashes['cuda_jl']+hashes['ka']+hashes['jacc']))==1,'all_native_trials_bitidentical':len(set(hashes['cuda']))==1,'full_checks':checks,'output_statistics':stats,'numpy_version':np.__version__,'near_zero_relative_error':'Large reported max-relative values arise near zero with denominator floor1e-12; acceptance uses predeclared absolute1e-4 plus exactguards/borders. No relative-error accuracy claim.'}
(p/'independent_artifact_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ('max_cpu_abs_error','all20_full_checks','all_Julia_models_bitidentical','all_native_trials_bitidentical')}))
