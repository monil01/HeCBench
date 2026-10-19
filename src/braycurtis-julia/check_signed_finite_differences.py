"""Supplemental all1848 signed-case input partials from actual saved device-input capture."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;F=R/'.porting-state/portable_julia/runs/20261019T170239306493Z_braycurtis/result.json';r=json.loads(F.read_text());art=r['models']['cuda']['trials'][0]['output_artifact'];path=Path(art['saved_path']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==art['sha256'];x=np.frombuffer(raw,dtype='<f8');plan=json.loads((P/'study_plan.json').read_text());offset=0
for c in plan['cases'][:2]:
 offset+=11;arrays=[]
 for count in [c['x1_count'],c['x2_count'],c['pairs']]*2:
  arrays.append(x[offset+8:offset+8+count].copy());offset+=count+16
c=plan['cases'][1];b,m,n,k=c['b'],c['m'],c['n'],c['k'];a=arrays[0].reshape(b,m,k);bb=arrays[1].reshape(b,n,k);go=arrays[2].reshape(b,m,n);delta=a[:,:,None,:]-bb[:,None,:,:];plus=a[:,:,None,:]+bb[:,None,:,:];D=np.sum(abs(delta),axis=3);S=np.sum(abs(plus),axis=3);w=go[:,:,:,None];S4=S[:,:,:,None];D4=D[:,:,:,None];g1=np.sum(w*(np.sign(delta)*S4-np.sign(plus)*D4)/(S4*S4),axis=2);g2=np.sum(w*(-np.sign(delta)*S4-np.sign(plus)*D4)/(S4*S4),axis=1)
def objective():return np.sum(go*np.sum(abs(a[:,:,None,:]-bb[:,None,:,:]),axis=3)/np.sum(abs(a[:,:,None,:]+bb[:,None,:,:]),axis=3))
h=1e-6;rows=[]
for name,arr,grad in [('x1',a,g1),('x2',bb,g2)]:
 vals=[]
 for q in range(arr.size):
  old=arr.flat[q];arr.flat[q]=old+h;up=objective();arr.flat[q]=old-h;dn=objective();arr.flat[q]=old;vals.append((up-dn)/(2*h))
 err=np.abs(np.asarray(vals)-grad.ravel());assert np.max(err)<1e-7;rows.append(dict(array=name,all_scalar_partials_checked=arr.size,max_absolute_difference=float(np.max(err)),finite_difference_step=h,independent_control_budget=1e-7))
(P/'signed_finite_difference_witness.json').write_text(json.dumps(dict(source_artifact=dict(path=str(path.relative_to(R)),sha256=art['sha256']),case=1,domain='Bothsigns; everyperfeatureabsargumentstrictlynonzero, positivepairdenominator',actual_minimum_abs_difference=float(np.min(abs(delta))),actual_minimum_abs_sum=float(np.min(abs(plus))),checks=rows,total_scalar_partials_checked=sum(z['all_scalar_partials_checked']for z in rows),python_numpy=np.__version__,scope='SupplementalCPUonly finite differences of weighted mathematicalobjective against fullanalyticgradient fromactualdeviceinputreadbacks; noGPUrun/timing/oraclefileload.'),indent=2)+'\n');print('BRAY_SIGNED_ALL1848_FD_PASS',rows)
