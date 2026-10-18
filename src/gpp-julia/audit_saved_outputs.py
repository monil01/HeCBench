from pathlib import Path
import json,struct,math,hashlib,sys
p=Path(__file__).resolve().parent;r=p.parents[1];meta=json.loads((p/'fixture_description.json').read_text());load=lambda f:list(struct.unpack('<'+'d'*(Path(f).stat().st_size//8),Path(f).read_bytes()));prefix=load(p/'fixture.bin');gold=load(p/'oracle.bin');out=[]
for arg in sys.argv[1:]:
 bundle=Path(arg);d=json.loads((bundle/'result.json').read_text())
 for model,m in d['models'].items():
  for trial,t in enumerate(m.get('trials',[]),1):
   art=t['output_artifact'];f=Path(art.get('saved_path',art['path']));f=f if f.is_absolute() else bundle/f
   if not f.exists():
    matches=list(bundle.rglob(f.name));assert len(matches)==1,(f,matches);f=matches[0]
   v=load(f);assert len(v)==meta['artifact_values'];assert struct.pack('<'+'d'*len(prefix),*v[:len(prefix)])==(p/'fixture.bin').read_bytes()
   v=v[len(prefix):];pos=0;ma=mr=norm=0.;guards=0;echo=0;points=[]
   for case in meta['cases']:
    vals=[]
    for component in range(2):
     for j in range(19):
      x,y=v[pos+j],gold[pos+j];assert math.isfinite(x)
      if 8<=j<11:
       e=abs(x-y);assert e<=1e-11+1e-11*abs(y);ma=max(ma,e);mr=max(mr,e/(abs(y)+1e-300));norm=max(norm,e/(1e-11+1e-11*abs(y)));vals.append(x)
       if case['mode']=='exact_zero':assert x==0.
      else:assert x==y;guards+=1
     pos+=19
    for k in range(12):
     nb,ng,nc=case['nb'],case['ng'],case['nc'];n=[ng,nc+1,3,ng*nc,ng*nc,nb*nc,nb*nc,nb*nc,nb*nc,ng*nc,ng*nc,nc][k]+16
     assert struct.pack('<'+'d'*n,*v[pos:pos+n])==struct.pack('<'+'d'*n,*gold[pos:pos+n]);pos+=n;guards+=16;echo+=n-16
    points.append(vals)
   assert pos==len(gold)
   out.append(dict(model=model,trial=trial,artifact=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),prefix_values=len(prefix),input_echo_payload_values=echo,exact_guard_values=guards,full_complex_outputs=points,oracle_max_abs_error=ma,oracle_max_rel_error=mr,max_normalized_budget_error=norm,all_checks_pass=True))
print(json.dumps(dict(oracle='Decimal80 independent allterm complex expression; varied conditioning and zero controls',rows=out,all_pass=bool(out) and all(x['all_checks_pass'] for x in out)),indent=2))
