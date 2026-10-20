#!/usr/bin/env python3
"""Full saved arrays, exact input bits, actual trace and nonunique SVD invariants."""
import argparse,pathlib,json,hashlib
import numpy as np
P=pathlib.Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def field_check(a,r):
 assert len(a)==len(r)and np.isfinite(a).all()and np.isfinite(r).all();scale=float(np.max(abs(r),initial=0));budget=1e-6+.002*scale;e=float(np.max(abs(a.astype(np.float64)-r),initial=0));assert e<=budget,(e,budget);return {'checked_components':len(a),'max_abs_error':e,'budget':budget,'budget_ratio':e/budget,'pass':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=pathlib.Path,required=True);ap.add_argument('--oracle',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);z=ap.parse_args();r=json.load(open(z.result));o=z.oracle;m=json.load(open(o/'oracle_manifest_completed.json'));tm=json.load(open(o/'trace_iteration3_manifest.json'));n=int(np.prod(m['shape']));nt=m['shape'][2];rows=[];trace_rows=[]
 inputs=np.concatenate([np.fromfile(o/(k+'.bin'),dtype='<f8').astype('<f4')for k in ['input_img','input_mask']]);assert len(inputs)==4*n
 for model,data in r['models'].items():
  if not data.get('trials'):continue
  assert data['provenance_valid']and len(data['trials'])==5
  traced=set()
  for t in data['trials']:
   art=t['output_artifact'];p=pathlib.Path(art['saved_path']);assert p.stat().st_size==m['artifact_float32_count']*4 and sha(p)==art['sha256'];a=np.memmap(p,dtype='<f4',mode='r');assert np.array_equal(a[:4*n].view('<u4'),inputs.view('<u4'));fields={}
   for f in m['serialized_fields']:
    ref=np.memmap(f['oracle']['path'],dtype='<f8',mode='r');x=a[f['float32_offset']:f['float32_offset']+f['float32_count']];fields[f['name']]=field_check(x,ref)
   assert np.all(a[-68:]==np.float32(12345.25));private=pathlib.Path(t['private_output_directory']);svd=private/f'trial{t["trial"]}_raw_SVD_factors.bin';assert svd.stat().st_size==m['raw_SVD_factor_float32_count']*4
   raw=np.fromfile(svd,dtype='<c8').astype(np.complex128);U=raw[:n].reshape((n//nt,nt),order='F');V=raw[n:n+nt*nt].reshape((nt,nt),order='F');SV=raw[n+nt*nt:].reshape((nt,nt),order='F');get=lambda name: next(f for f in m['serialized_fields']if f['name']==name);lf=get('imgl');L=a[lf['float32_offset']:lf['float32_offset']+lf['float32_count']].view('<c8').reshape((n//nt,nt),order='F');lsf=get('ls');LS=np.asarray(a[lsf['float32_offset']:lsf['float32_offset']+nt],dtype=np.float64)
   ug=float(np.max(abs(U.conj().T@U-np.eye(nt))));vg=float(np.max(abs(V@V.conj().T-np.eye(nt))));re=float(np.max(abs(U@SV-L)));se=float(np.max(abs(LS[:,None]*V-SV)));assert ug<1e-4 and vg<1e-4 and re<2e-5 and se<2e-5,(ug,vg,re,se)
   rows.append({'model':model,'trial':t['trial'],'artifact':{'path':str(p),'bytes':p.stat().st_size,'sha256':art['sha256']},'exact_actual_input_prefix':True,'guards_exact':True,'fields':fields,'nonunique_SVD_raw':{'path':str(svd),'bytes':svd.stat().st_size,'sha256':sha(svd),'U_orthogonality_max':ug,'VT_orthogonality_max':vg,'weighted_VT_max_abs_error':se,'L_reconstruction_max_abs_error':re,'phase_identity_claim':False}})
   if str(private)not in traced:
    for name,q in tm['trace_arrays'].items():
     f=private/('warmup_iteration3_'+name+'.bin');assert f.stat().st_size==q['count']*8;v=np.fromfile(f,dtype='<f4');ref=np.memmap(q['path'],dtype='<f8',mode='r');chk=field_check(v,ref);trace_rows.append({'model':model,'trace_iteration':3,'field':name,'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size,**chk})
    traced.add(str(private))
 out={'benchmark':'tgvnn','status':'pass','result':str(z.result.resolve()),'result_sha256':sha(z.result),'oracle_manifest':str(o/'oracle_manifest_completed.json'),'oracle_manifest_sha256':sha(o/'oracle_manifest_completed.json'),'source_auditor_sha256':sha(pathlib.Path(__file__)),'rows':rows,'actual_untimed_third_iteration_trace':trace_rows,'not_claimed':'Every overwritten CUDAkernel temporary/all500intermediate states or raw phase identity of nonuniqueSVDfactors. No GPUexecution/retiming.','artifact_count':len(rows),'artifact_bytes':sum(x['artifact']['bytes']for x in rows)};z.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','artifacts':len(rows),'trace_fields':len(trace_rows),'bytes':out['artifact_bytes']}))
if __name__=='__main__':main()
