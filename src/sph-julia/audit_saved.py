#!/usr/bin/env python3
"""Full saved-array audit; does not launch GPU work or modify evidence."""
import argparse,hashlib,json,pathlib,struct
import numpy as np
P=pathlib.Path(__file__).resolve().parent
A=np.array([2e-6,5e-3,*([5e-9]*3),*([2e-7]*6),*([5e-4]*3)])
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def audit(result,case):
 r=json.load(open(result)); params=np.fromfile(P/(case+'_params_raw.bin'),dtype='<f8'); ints=struct.unpack('<5i',params.tobytes()[72:92]); nf,nb,steps=ints[1:4];n=14*nf
 f=np.r_[np.fromfile(P/(case+'_initial_f64.bin'),dtype='<f8'),np.full(17,-8191.25)];b=np.r_[np.fromfile(P/(case+'_boundary_f64.bin'),dtype='<f8'),np.full(17,-8191.25)];p=np.r_[params,np.full(17,-8191.25)];prefix=np.r_[f,b,p];cpu=np.fromfile(P/(case+'_CPU_final_f64.bin'),dtype='<f8').reshape(nf,14);tr=np.fromfile(P/(case+'_CPU_checkpoints_f64.bin'),dtype='<f8').reshape(-1,nf,14); rows=[]; arrays={}
 for model,m in r['models'].items():
  for t in m.get('trials',[]):
   file=t['output_artifact']['saved_path']; x=np.fromfile(file,dtype='<f8'); assert len(x)==2*len(prefix)+len(tr)*(n+17)
   assert x[:len(prefix)].tobytes()==prefix.tobytes();o=x[len(prefix):];assert np.array_equal(o[n:n+17],f[n:]);assert o[n+17:len(prefix)].tobytes()==np.r_[b,p].tobytes()
   vals=[o[:n].reshape(nf,14)];expected=[cpu];cur=len(prefix)
   for q in tr:
    vals.append(o[cur:cur+n].reshape(nf,14));expected.append(q);assert np.array_equal(o[cur+n:cur+n+17],f[n:]);cur+=n+17
   err=np.zeros(14)
   for v,c in zip(vals,expected):
    assert np.isfinite(v).all() and (v[:,0]>0).all();e=np.abs(v-c);assert (e<=A+1e-9*np.abs(c)).all(),(model,t['trial'],e.max(axis=0));err=np.maximum(err,e.max(axis=0))
   if case=='ballistic':
    z0=f[4];dt=params[5];g=9.8;N=steps;final=vals[0][0];assert abs(final[4]-(z0-0.5*g*dt*dt*N*(N+2)))<=1e-12;assert abs(final[7]-(-(N+1)*g*dt))<=1e-12
   rows.append(dict(model=model,trial=t['trial'],artifact=file,sha256=sha(file),bytes=pathlib.Path(file).stat().st_size,max_abs_error_by_field=err.tolist(),full_input_prefix_exact=True,all_guards_exact=True,readonly_buffers_exact=True,complete_final_and_actual_single_warm_trace_checked=True));arrays[(model,t['trial'])]=x
 dual=[]
 for (model,t),x in arrays.items():
  if model=='cuda':continue
  for ref in ['cuda','cuda_jl']:
   if ref==model or (ref,t) not in arrays:continue
   y=arrays[(ref,t)];e=np.abs(x-y);assert x[:len(prefix)].tobytes()==y[:len(prefix)].tobytes();field_errors=[]
   for start in [len(prefix)]+[2*len(prefix)+j*(n+17) for j in range(len(tr))]:
    a=x[start:start+n].reshape(nf,14);b0=y[start:start+n].reshape(nf,14);assert (np.abs(a-b0)<=A+1e-9*np.abs(b0)).all();field_errors.append(np.abs(a-b0).max(axis=0).tolist())
   dual.append(dict(model=model,reference=ref,trial=t,full_field_budget_pass=True,max_abs_error=float(e.max()),field_errors=field_errors))
 return dict(schema_version=1,case=case,result=str(pathlib.Path(result).resolve()),result_sha256=sha(result),configuration_id=r['configuration_id'],numpy_version=np.__version__,status='pass',atol_by_field=A.tolist(),rtol=1e-9,rows=rows,dual_reference_checks=dual,oracle='independently expressed CPU radial/support-neighbor evaluator of source finite equations; not continuum groundtruth',warm_trace_scope='one actual untimed warmup trace per process, replicated across its five artifacts; not five independent traces')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('result');a.add_argument('case',choices=['default','ballistic']);a.add_argument('--output',required=True);z=a.parse_args();v=audit(z.result,z.case);pathlib.Path(z.output).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'status':v['status'],'rows':len(v['rows']),'dual_checks':len(v['dual_reference_checks'])}))
