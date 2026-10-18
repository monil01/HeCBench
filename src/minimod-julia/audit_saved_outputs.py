"""Read-only field-specific full-artifact audit, with no GPU executions."""
from pathlib import Path
import json,struct,hashlib,argparse,math
ap=argparse.ArgumentParser();ap.add_argument('result');args=ap.parse_args();p=Path(__file__).resolve().parent;d=json.loads(Path(args.result).read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();gb=(p/'oracle.bin').read_bytes();gold=struct.unpack('<'+'d'*(len(gb)//8),gb);LEN=40988;rows=[]
mask=[]
for pos in range(LEN):
 z=pos-20
 if z<0 or z>=40960:mask.append((True,True));continue
 i=z//1024-4;j=(z//32)%32-4;k=z%32-4;active=0<=i<24 and 0<=j<24 and 0<=k<24;pml=active and not(8<=i<16 and 8<=j<16 and 8<=k<16);mask.append((not active,not pml))
for model,m in d['models'].items():
 for t in m['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;v=struct.unpack('<'+'d'*((len(raw)-len(prefix))//8),raw[len(prefix):]);go=0;case_rows=[]
  for ci in range(3):
   errs=[]
   for field in range(5):
    at=1e-10 if field==2 else 2e-6;ma=0.;mr=0.;norm=0.;exact_count=0;nonzero=0
    for pos in range(LEN):
     x=v[go+pos];r=gold[go+pos];e=abs(x-r);assert math.isfinite(x);exact=ci==2 or field>=3 or mask[pos][1 if field==2 else 0]
     assert (x==r) if exact else (e<=at+2e-5*abs(r));exact_count+=int(exact);nonzero+=int(20<=pos<40980 and x!=0);ma=max(ma,e);mr=max(mr,e/(abs(r)+1e-300));norm=max(norm,e/(at+2e-5*abs(r)))
    go+=LEN;errs.append(dict(field=['u_current','v_previous','phi','vp','eta'][field],max_abs_error=ma,max_rel_error=mr,max_normalized_field_budget=norm,exact_mask_count=exact_count,actual_nonzero_payload_values=nonzero))
   n=5*LEN;assert tuple(v[go:go+n])==gold[go:go+n];go+=n;case_rows.append(dict(case=ci,fields=errs))
  assert go==len(gold) and t['correct'];rows.append(dict(model=model,trial=t['trial'],sha256=hashlib.sha256(raw).hexdigest(),exact_input_prefix_bytes=True,all_pass=True,cases=case_rows))
assert len(rows)==20
out=dict(all_pass=True,result=str(Path(args.result).resolve()),result_sha256=hashlib.sha256(Path(args.result).read_bytes()).hexdigest(),rows=rows,all_artifact_sha256_count=len({r['sha256'] for r in rows}),no_GPU_rerun=True)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
