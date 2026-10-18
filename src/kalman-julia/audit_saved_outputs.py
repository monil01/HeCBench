from pathlib import Path
import json,struct,math,hashlib,sys
p=Path(__file__).resolve().parent;m=json.loads((p/'fixture_description.json').read_text());load=lambda f:list(struct.unpack('<'+'d'*(Path(f).stat().st_size//8),Path(f).read_bytes()));prefix=load(p/'fixture.bin');gold=load(p/'oracle.bin');rows=[]
for arg in sys.argv[1:]:
 bundle=Path(arg);d=json.loads((bundle/'result.json').read_text())
 for model,z in d['models'].items():
  for t in z.get('trials',[]):
   f=Path(t['output_artifact']['saved_path']);v=load(f);assert len(v)==m['artifact_values'];assert f.read_bytes()[:len(prefix)*8]==(p/'fixture.bin').read_bytes();v=v[len(prefix):];pos=0;ma=mr=norm=0.;guards=exact_unwritten=outputs=0
   for path in m['paths']:
    c=m['cases'][path['base_case']]
    for k,n in enumerate(path['output_sizes']):
     a=v[pos:pos+n+16];r=gold[pos:pos+n+16];pos+=n+16;assert a[:8]==r[:8] and a[-8:]==r[-8:];guards+=16;outputs+=n
     for x,y in zip(a[8:-8],r[8:-8]):
      assert math.isfinite(x)
      if (k==4 and not c['confidence']) or(k==2 and path['expected_log_zero']):assert x==y;exact_unwritten+=1
      else:
       er=abs(x-y);assert er<=1e-9+1e-9*abs(y);ma=max(ma,er);mr=max(mr,er/(abs(y)+1e-300));norm=max(norm,er/(1e-9+1e-9*abs(y)))
       if k in [1,4]:assert x>0
    for n in c['array_sizes']:
     a=v[pos:pos+n+16];r=gold[pos:pos+n+16];assert struct.pack('<'+'d'*len(a),*a)==struct.pack('<'+'d'*len(r),*r);pos+=n+16;guards+=16
   assert pos==len(gold);rows.append(dict(model=model,trial=t['trial'],path_count=26,artifact=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),all_checks_pass=True,all_output_scalars=outputs,exact_guard_check_values=guards,exact_unwritten_or_zero_log_scalars=exact_unwritten,max_abs_error=ma,max_rel_error=mr,max_normalized_budget_error=norm))
assert rows
print(json.dumps(dict(all_pass=all(x['all_checks_pass'] for x in rows),oracle=m['oracle'],oracle_precision=m['oracle_precision'],covariance_scope=m['covariance_positive_definiteness'],min_innovation_variance=m['min_innovation_variance'],rows=rows),indent=2))
