from pathlib import Path
import json,hashlib
from fractions import Fraction
import numpy as np
R=Path('/home/imo/HeCBench_julia');B=R/'src/su3-julia';S=R/'.porting-state/portable_julia';P=next(p for p in reversed(sorted((S/'runs').glob('*_su3/result.json')))if json.loads(p.read_text()).get('matched_performance_valid'));d=json.loads(P.read_text());cs=json.loads((B/'cases.json').read_text())['cases'];f=np.memmap(B/'input_u32.bin',dtype='<u4',mode='r');oracle=np.memmap(B/'oracle_u32.bin',dtype='<u4',mode='r');G=np.uint32(0xa5a5a5a5);rows=[];perf=[]
def det(z):return z[...,0,0]*(z[...,1,1]*z[...,2,2]-z[...,1,2]*z[...,2,1])-z[...,0,1]*(z[...,1,0]*z[...,2,2]-z[...,1,2]*z[...,2,0])+z[...,0,2]*(z[...,1,0]*z[...,2,1]-z[...,1,1]*z[...,2,0])
for model,mr in d['models'].items():
 for t in mr['trials']:
  ar=t['output_artifact'];v=np.memmap(ar['saved_path'],dtype='<u4',mode='r');off=0;pre=[];checks=[]
  for c in cs:
   arr=[];pos=c['io']
   for k,n in enumerate([80*c['n'],72,80*c['n']]):
    a=v[off:off+n+64];off+=n+64;assert np.all(a[:32]==G)and np.all(a[-32:]==G)
    if k<2:assert np.array_equal(a[32:-32],f[pos:pos+n]);pos+=n
    else:assert np.all(a[32:-32]==G)
    arr.append(a)
   pre.append(arr)
  assert off==168481208
  for c,before in zip(cs,pre):
   n=c['n'];arr=[]
   for k,size in enumerate([80*n,72,80*n]):
    a=v[off:off+size+64];off+=size+64;assert np.all(a[:32]==G)and np.all(a[-32:]==G)
    if k<2:assert np.array_equal(a,before[k])
    else:assert np.array_equal(a[32:-32],oracle[c['oo']:c['oo']+size])
    arr.append(a[32:-32])
   aa=arr[0].reshape(n,80)[:,:72].view('<f4').reshape(n,4,3,3,2);bb=arr[1].view('<f4').reshape(4,3,3,2);cc=arr[2].reshape(n,80)[:,:72].view('<f4').reshape(n,4,3,3,2);assert np.isfinite(aa).all()and np.isfinite(bb).all()and np.isfinite(cc).all();assert np.all(arr[2].reshape(n,80)[:,72:]==G);assert np.all(arr[2].reshape(n,80)[:,:72]!=G);err=0.;group=0;bound=0
   if c['mode']==1:
    assert np.all(aa[...,0]==1)and np.all(aa[...,1]==0)and np.all(bb[...,0]==np.float32(1/3))and np.all(bb[...,1]==0);assert np.all(cc[...,0]==1)and np.all(cc[...,1]==0);exact=3*Fraction.from_float(float(np.float32(1/3)));err=float(abs(Fraction(1)-exact));assert err==2**-25
   elif c['mode']==4:
    ac=aa[...,0].astype(np.float64)+1j*aa[...,1];bc=bb[...,0].astype(np.float64)+1j*bb[...,1];want=ac@bc;actual=cc[...,0].astype(np.float64)+1j*cc[...,1];assert np.array_equal(actual,want);assert np.all(actual@actual.conj().swapaxes(-1,-2)==np.eye(3));assert np.all(det(actual)==1);group=n*4
   else:
    au=np.rint(aa.astype(np.float64)*8).astype(np.int64);bu=np.rint(bb.astype(np.float64)*16).astype(np.int64);assert np.array_equal(au/8,aa)and np.array_equal(bu/16,bb);real=np.einsum('sdik,dkj->sdij',au[...,0],bu[...,0])-np.einsum('sdik,dkj->sdij',au[...,1],bu[...,1]);imag=np.einsum('sdik,dkj->sdij',au[...,0],bu[...,1])+np.einsum('sdik,dkj->sdij',au[...,1],bu[...,0]);assert np.array_equal(real/128,cc[...,0])and np.array_equal(imag/128,cc[...,1]);bound=6*int(np.abs(au).max())*int(np.abs(bu).max());assert bound<2**24
    if c['mode']==3:assert np.array_equal(cc,aa)
    if c['mode']==0:assert np.all(cc==0)
   checks.append(dict(case=c['case_id'],sites=n,all_actual_A_B_readonly_bits_exact=True,initial_C_all_poison=True,all_defined_C_components_written=True,all_C_metadata_padding_and_guards_exact=True,max_abs_error_against_exact_product_of_represented_FP32_inputs=err,dyadic_partial_sum_integer_units_bound=bound,special_unitary_products_checked=group))
  assert off==len(v);withhash=hashlib.file_digest(Path(ar['saved_path']).open('rb'),'sha256').hexdigest();assert withhash==ar['sha256'];rows.append(dict(model=model,trial=t['trial'],artifact_sha256=ar['sha256'],cases=checks))
  assert len(t['case_times_seconds'])==7;assert abs(sum(t['case_times_seconds'])-t['time_seconds'])<1e-11
  for c,seconds in zip(cs,t['case_times_seconds']):perf.append(dict(model=model,trial=t['trial'],case=c['case_id'],sites=c['n'],threads=c['threads'],time_seconds=seconds))
  del v
means=[]
for model,mr in d['models'].items():
 for c in cs:
  samples=[r['time_seconds']for r in perf if r['model']==model and r['case']==c['case_id']and r['trial']in[2,3,4,5]];assert len(samples)==4;means.append(dict(model=model,case=c['case_id'],sites=c['n'],mean_seconds=sum(samples)/4,samples=samples))
 assert abs(sum(r['mean_seconds']for r in means if r['model']==model)-mr['mean_seconds'])<1e-11
out=dict(result=str(P.relative_to(R)),result_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),checked_saved_artifacts=len(rows),artifact_words=336962416,artifact_bytes=1347849664,input_prefix_words=168481208,mathematical_output_complex_entries_per_trial=sum(c['n']*36 for c in cs),raw_SHA_classes=len({r['artifact_sha256']for r in rows}),interpretation='Complete actual GPU AoS buffers. Independent exactinteger complex products/SU3closure/identity/zerocontrols, source3*representedFloat32third product arithmeticrounding error2^-25 separate from exactbitagreement with roundedFP32oracle. Allphysicalmetadata/padding/read-only/initialpoison/writes/guards exact. Oracle CPU-only, never uploaded. Sourcefullshape/sourcekernel/layout/grid retained; originalR1000 unmeasured, namedR3. Fivecheckedtrials mean2-5; finalonlyinternalrepeat output captured.',trials=rows);(B/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');(B/'per_case_performance.json').write_text(json.dumps(dict(result=out['result'],protocol='five_trials_last_four_v1',samples=perf,means=means,interpretation='Seven-case aggregate and isolated fullL32 rates must be distinguished; source864/site is algorithmic FLOP convention, allocation-byte rate not measured DRAM traffic/roofline intensity.'),indent=2)+'\n');print(out['result'],len(rows),out['raw_SHA_classes'],[m for m in means if m['sites']==1048576])
