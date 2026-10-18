"""Read-only full normal/flag/input-bit/guard and geometric invariant audit."""
from pathlib import Path
import json,struct,hashlib,argparse,math
ap=argparse.ArgumentParser();ap.add_argument('result');a=ap.parse_args();p=Path(__file__).resolve().parent;res=json.loads(Path(a.result).read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();gb=(p/'oracle.bin').read_bytes();gold=struct.unpack('<'+'d'*(len(gb)//8),gb);pf=struct.unpack('<'+'d'*(len(prefix)//8),prefix);pos=1;points=[]
for case in meta['cases']:
 W,H=map(int,pf[pos:pos+2]);pos+=2;pts=[]
 for _ in range(W*H):pts.append([struct.unpack('<f',struct.pack('<I',int(b)))[0] for b in pf[pos:pos+3]]);pos+=3
 points.append(pts)
rows=[]
for model,m in res['models'].items():
 for tr in m['trials']:
  f=Path(tr['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;v=struct.unpack('<'+'d'*((len(raw)-len(prefix))//8),raw[len(prefix):]);go=0;cases=[];maxerr=0.
  for ci,c in enumerate(meta['cases']):
   N=c['points'];nv=4*(N+24);out=v[go:go+nv];ref=gold[go:go+nv];go+=nv;assert tuple(out[:64])==ref[:64] and tuple(out[-32:])==ref[-32:];unit=0.;camera=-math.inf;plane=0.;err=0.
   for i,status in enumerate(c['status']):
    off=64+4*i;x=out[off:off+3];r=ref[off:off+3];flag=out[off+3];assert flag==ref[off+3] and flag in [0,1] and all(math.isfinite(z) for z in x)
    if status!='normal':assert tuple(x)==tuple(r)==(0.,0.,0.)
    else:
     for xx,rr in zip(x,r):assert abs(xx-rr)<=1e-6+3e-6*abs(rr)
     e=max(abs(xx-rr) for xx,rr in zip(x,r));err=max(err,e);unit=max(unit,abs(sum(z*z for z in x)-1));dot=sum(q*n for q,n in zip(points[ci][i],x));camera=max(camera,dot);assert unit<=1e-5 and dot<=1e-5
     if ci in [0,4]:
      n=[3/256,5/256,-1];length=math.sqrt(sum(z*z for z in n));n=[z/length*(1 if ci==0 else -1) for z in n];plane=max(plane,max(abs(xx-rr) for xx,rr in zip(x,n)));assert plane<=1e-6+3e-6
   ni=3*(N+72);assert tuple(v[go:go+ni])==gold[go:go+ni];go+=ni;maxerr=max(maxerr,err);cases.append(dict(case=ci,full_flags_exact=True,invalid_and_fallback_exact=True,all_input_payload_and_guard_bits_exact=True,max_abs_normal_error=err,max_unit_norm_residual=unit,max_camera_dot=camera if math.isfinite(camera) else None,max_analytic_plane_error=plane if ci in [0,4] else None))
  assert go==len(gold) and tr['correct'];rows.append(dict(model=model,trial=tr['trial'],sha256=hashlib.sha256(raw).hexdigest(),all_pass=True,exact_input_prefix_bytes=True,cases=cases,max_abs_error=maxerr))
assert len(rows)==20
out=dict(result=str(Path(a.result).resolve()),result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),all_pass=True,all_artifact_sha256_count=len({r['sha256'] for r in rows}),rows=rows,no_GPU_rerun=True)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
