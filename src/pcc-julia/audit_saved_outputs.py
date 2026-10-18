"""Read-only full Gram/upper/input/guard and vendor-process audit. No GPU."""
from pathlib import Path
import argparse,json,hashlib,struct,math
ap=argparse.ArgumentParser();ap.add_argument('result');a=ap.parse_args();p=Path(__file__).resolve().parent;res=json.loads(Path(a.result).read_text());m=json.loads((p/'fixture_description.json').read_text());pre=(p/'fixture.bin').read_bytes();gb=(p/'oracle.bin').read_bytes();gold=struct.unpack('<'+'d'*(len(gb)//8),gb);rows=[]
for model,rec in res['models'].items():
 for tr in rec['trials']:
  raw=Path(tr['output_artifact']['saved_path']).read_bytes();assert raw[:len(pre)]==pre;v=struct.unpack('<'+'d'*((len(raw)-len(pre))//8),raw[len(pre):]);go=0;cases=[];uppers=[]
  for case in m['cases']:
   up=[];ma=0.;mr=0.;zero=set(case['constant_rows']);diagerr=0.;cells=0
   for ch in case['chunks']:
    arrays=[]
    for kind,count in [('C',ch['cormat_values']),('U',ch['upper_values'])]:
     got=v[go:go+count+72];ref=gold[go:go+count+72];go+=count+72;assert got[:64]==ref[:64] and got[-8:]==ref[-8:];x=got[64:64+count];r=ref[64:64+count]
     for xx,rr in zip(x,r):
      assert math.isfinite(xx) and abs(xx-rr)<=1e-6+3e-6*abs(rr) and abs(xx)<=1+3e-6;ma=max(ma,abs(xx-rr));mr=max(mr,abs(xx-rr)/(abs(rr)+1e-300))
     arrays.append(x)
    C,U=arrays;up.extend(U);off=ch['offset'];n=ch['Nprime'];n1=ch['rows'];k=0
    for i in range(n1):
     for j in range(n):
      val=C[j*n1+i]
      if off+i in zero or off+j in zero:assert val==0
      if i==j and off+i not in zero:diagerr=max(diagerr,abs(val-1));assert abs(val-1)<=3e-6
     for j in range(i+1,n):
      if off+i in zero or off+j in zero:assert U[k]==0
      k+=1
    assert k==len(U);cells+=len(C)
   ni=case['N']*case['L']+72;assert v[go:go+ni]==gold[go:go+ni];go+=ni;uppers.append(up);cases.append(dict(case=case['case'],max_abs_raw_Pearson_error=ma,max_rel_raw_Pearson_error=mr,constant_zero_policy_exact=True,nonconstant_diagonal_max_error=diagerr,all_input_and_guards_exact=True,computed_Gram_cells=cells,upper_values=len(up)))
  assert go==len(gold) and tr['correct'];assert len(uppers[0])==len(uppers[1]);cross=max(abs(x-y) for x,y in zip(uppers[0],uppers[1]));assert cross<=2e-6+6e-6;rows.append(dict(model=model,trial=tr['trial'],sha256=hashlib.sha256(raw).hexdigest(),all_pass=True,exact_input_prefix_bytes=True,cases=cases,single_vs_multi_same_input_packed_upper_max_abs=cross))
assert len(rows)==20
library='/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224';expected='6571bf8344cf13f3b116e6d613dd234cefca081c2303c3667e71b25d0eba5ec2';vendor=[]
source_runs={'cuda':'20261018T094901745753Z_pcc','cuda_jl':'20261018T095301953606Z_pcc','ka':'20261018T095455309371Z_pcc','jacc':'20261018T095654538542Z_pcc'}
for model,run in source_runs.items():
 log=p.parent.parent/'.porting-state/portable_julia/runs'/run/(model+'_run1.log');lines=log.read_text().splitlines();x=json.loads(next(s[13:] for s in lines if s.startswith('STUDY_VENDOR ')));assert x['cublas_realpath']==library and x['version_integer']==120002 and x['math_mode']==0;assert res['models'][model]['adapter_file_hashes'][library]==expected
 vendor.append(dict(model=model,actual_capture_run=run,log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),printed=x,actual_bound_library_sha256=expected,sha_basis='native printed dladdr realpath plusboundbinaryhash; Julia printedrealpath+SHAandboundbinaryhash',pointer_mode_scope='NativeHOSTscalars passnumerically; JuliaHOSTassertbeforeprintedline. No nativepointermodequeryfabricated.'))
out=dict(result=str(Path(a.result).resolve()),result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),all_pass=True,all_artifact_sha256_count=len({r['sha256'] for r in rows}),rows=rows,vendor_process_evidence=vendor,no_GPU_rerun=True)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['rows','vendor_process_evidence']}))
