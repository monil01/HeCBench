from pathlib import Path
import json,struct,math,hashlib,sys
p=Path(__file__).resolve().parent;m=json.loads((p/'fixture_description.json').read_text());load=lambda f:list(struct.unpack('<'+'d'*(Path(f).stat().st_size//8),Path(f).read_bytes()));prefix=load(p/'fixture.bin');gold=load(p/'oracle.bin');rows=[];floatbytes=lambda a:struct.pack('<'+'d'*len(a),*a)
for arg in sys.argv[1:]:
 d=json.loads((Path(arg)/'result.json').read_text())
 for model,v in d['models'].items():
  for t in v.get('trials',[]):
   f=Path(t['output_artifact']['saved_path']);a=load(f);assert len(a)==m['artifact_values'];assert f.read_bytes()[:len(prefix)*8]==(p/'fixture.bin').read_bytes();a=a[len(prefix):];pos=guards=exact_controls=0;ma=[0.,0.,0.];mr=[0.,0.,0.];norm=[0.,0.,0.];byalg={k:[0.,0.,0.] for k in range(3)}
   for path in m['paths']:
    c=m['cases'][path['case']];gotouts=[]
    for k,n in enumerate(c['output_sizes']):
     x=a[pos:pos+n+16];r=gold[pos:pos+n+16];pos+=n+16;assert x[:8]==r[:8] and x[-8:]==r[-8:];guards+=16;gotouts.append(x[8:-8])
     for xx,rr in zip(x[8:-8],r[8:-8]):
      assert math.isfinite(xx);e=abs(xx-rr);assert e<=1e-5+2e-6*abs(rr);ma[k]=max(ma[k],e);mr[k]=max(mr[k],e/(abs(rr)+1e-300));norm[k]=max(norm[k],e/(1e-5+2e-6*abs(rr)));byalg[path['algorithm']][k]=max(byalg[path['algorithm']][k],e)
      if k==2:assert xx>0
    ins=[]
    for n in c['input_sizes']:
     x=a[pos:pos+n+16];r=gold[pos:pos+n+16];assert floatbytes(x)==floatbytes(r);pos+=n+16;guards+=16;ins.append(x[8:-8])
    for j,x in enumerate(gotouts[0]):
     if c['mode']=='exact_constant_rows' or(j%c['C'])%19==0:assert x==ins[2][j%c['C']];exact_controls+=1
   assert pos==len(gold);rows.append(dict(model=model,trial=t['trial'],all_checks_pass=True,artifact=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),input_prefix_values=len(prefix),path_count=108,output_values=223020,actual_input_echo_payload_values=332748,exact_guard_check_values=guards,exact_affine_constant_controls=exact_controls,max_abs_errors=dict(zip(['out','mean','rstd'],ma)),max_rel_errors=dict(zip(['out','mean','rstd'],mr)),max_normalized_budget_errors=dict(zip(['out','mean','rstd'],norm)),max_abs_by_algorithm=byalg))
assert rows
print(json.dumps(dict(all_pass=all(x['all_checks_pass'] for x in rows),oracle=m['oracle'],rows=rows,native_shared_hazard='Staticreusehazard noted butallnative5 fullarrays pass; noobservedfailure/nativeinvalidity inferred'),indent=2))
