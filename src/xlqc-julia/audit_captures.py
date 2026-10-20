#!/usr/bin/env python3
"""CPU-only full raw-buffer, primitive-ERI and coupled-SCF evidence replay."""
from pathlib import Path
import argparse,json,hashlib,ctypes,subprocess,statistics
import numpy as np
P=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(p,d):q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def packed(a,b):return np.maximum(a,b)*(np.maximum(a,b)+1)//2+np.minimum(a,b)
def main():
 a=argparse.ArgumentParser();a.add_argument('--sp-result',type=Path,required=True);a.add_argument('--dp-result',type=Path,required=True);a.add_argument('--output',type=Path,default=P/'all4_full_capture_audit.json');a.add_argument('--cpu-build-root',type=Path,default=Path('/tmp/hecbench-stencil-ports-xlqc-cpu-audit'));z=a.parse_args();z.cpu_build_root.mkdir(parents=True,exist_ok=True);libpath=z.cpu_build_root/'independent_cints.so';I=P/'independent_oracle';cmd=['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(P/'reference_history/int_lib'),str(I/'prefixed_cints.cc'),str(I/'eri_bridge.cc'),'-o',str(libpath),'-lm'];subprocess.run(cmd,check=True);lib=ctypes.CDLL(str(libpath));fn=lib.xl_oracle_eri_table;fn.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(ctypes.c_double)];cache={};reviews=[];filecount=0;bindings=0;resultbindings={};ranges={}
 for precision,path in [('SP',z.sp_result),('DP',z.dp_result)]:
  r=json.load(open(path));assert r['matched_performance_valid']and r['reference_agreement']and all(r['variant_verified'].values());resultbindings[precision]={'path':str(path),'sha256':sha(path),'configuration_id':r['configuration_id']}
  for m,v in r['models'].items():
   assert v['complete']and v['provenance_valid']and v['status']=='native_check_pass'and len(v['trials'])==5;assert v['mean_seconds']==statistics.mean(x['time_seconds']for x in v['trials'][1:])
   for source,h in v['adapter_file_hashes'].items():assert sha(source)==h,(m,source);bindings+=1
   h=hashlib.sha256();base=Path(v['source_path'])
   for source in sorted(base.rglob('*')):
    if source.is_file()and source.suffix in('.jl','.cu','.cuh','.h','.cpp','.toml'):h.update(str(source.relative_to(base)).encode());h.update(source.read_bytes())
   assert h.hexdigest()==v['source_hash'],(m,precision,'source_digest')
   for t in v['trials']:
    assert t['correct']and t['jit_excluded']and t['internal_repeats']==18 and t['iterations']==9
    cap={x['role']:x for x in t['full_physical_captures']};raw={}
    for role,desc in cap.items():
     p=Path(desc['path']);assert p.stat().st_size==desc['bytes']and sha(p)==desc['sha256'];raw[role]=p.read_bytes();filecount+=1
    assert len(cap)==284
    pbf=np.frombuffer(raw['actual_initial_pbf_f64'],dtype='<f8').reshape(21,8);mapping=np.frombuffer(raw['actual_initial_pbf_to_cbf_i32'],dtype='<i4');Q=np.frombuffer(raw['actual_initial_Q_raw_ERI_not_sqrt_f64'],dtype='<f8');assert np.all((mapping>=0)&(mapping<7))
    key=hashlib.sha256(pbf.tobytes()).hexdigest()
    if key not in cache:
     full=np.empty((21,21,21,21));rounded=np.empty_like(full);sp=pbf.astype(np.float32).astype(np.float64);fn(pbf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),21,full.ctypes.data_as(ctypes.POINTER(ctypes.c_double)));fn(sp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),21,rounded.ctypes.data_as(ctypes.POINTER(ctypes.c_double)));cache[key]=(full,rounded)
     alpha=(pbf[:,6,None]+pbf[:,6][None,:]).reshape(-1);center=((pbf[:,6,None,None]*pbf[:,None,:3]+pbf[None,:,6,None]*pbf[None,:,:3])/(pbf[:,6,None,None]+pbf[None,:,6,None])).reshape(-1,3);rho=alpha[:,None]*alpha[None,:]/(alpha[:,None]+alpha[None,:]);X=rho*np.sum((center[:,None,:]-center[None,:,:])**2,axis=2);ranges[key]={'Boys_X_min':float(X.min()),'Boys_X_max':float(X.max()),'actual_angular_four_sum_max':int(4*np.max(np.sum(pbf[:,3:6],axis=1))),'Rys_max_roots':int(2*np.max(np.sum(pbf[:,3:6],axis=1))+1)}
    full,rounded=cache[key];assert np.array_equal(np.frombuffer(raw['independent_fullERI_actualPBF_f64'],dtype='<f8').reshape(full.shape),full)
    desc=t['output_artifact'];artifact=Path(desc['saved_path']);assert sha(artifact)==desc['sha256'];values=np.fromfile(artifact,dtype='<f8');assert len(values)==t['checked_values']==7029 and np.all(np.isfinite(values));prefix=values[:495];assert prefix[0]==int(precision=='DP')and list(prefix[1:4])==[21,7,10];assert prefix[5:173].tobytes()==pbf.tobytes()and np.array_equal(prefix[173:194],mapping)and prefix[194:222].tobytes()==Q.tobytes();S=prefix[222:271].reshape(7,7);H=prefix[369:418].reshape(7,7);assert H.shape==(7,7)
    pair=packed(mapping[:,None],mapping[None,:]);Qfull=Q[pair];eri=full if precision=='DP'else rounded;offset=495;maxratio=0.;maxerr=0.;state={'orthogonality':0.,'idempotency':0.,'electron_count':0.,'eigenproblem':0.}
    for iteration in range(9):
     for stage in range(2):
      op=stage+(2 if precision=='DP'else 0);D=values[offset:offset+28];offset+=28;out=values[offset:offset+231];offset+=231;role=f'iter{iteration}_op{op}';assert D.tobytes()==raw[role+'_actual_D_f64']and out.tobytes()==raw[role+'_fulloutput_f64'];lift=D[pair];expected=[]
      for i in range(21):
       for j in range(i+1):
        if stage==0:mask=np.abs(Qfull[i,j]*Qfull*lift)>=1e-16;value=np.sum(eri[i,j,:,:]*lift*mask)
        else:mask=np.abs(Qfull[i,:,None]*Qfull[j,None,:]*lift)>=1e-16;value=np.sum(eri[i,:,j,:]*lift*mask)
        expected.append(value)
      expected=np.asarray(expected);savedgold=np.frombuffer(raw[role+'_independent_expected_f64'],dtype='<f8');assert np.max(np.abs(savedgold-expected))<1e-11;error=np.abs(out-expected);budget=(5e-10+5e-9*np.abs(expected))if precision=='DP'else(1e-5+3e-5*np.abs(expected));assert np.all(error<=budget);maxratio=max(maxratio,float(np.max(error/budget)));maxerr=max(maxerr,float(error.max()))
      before={};after={}
      for aid in range(1,7):
       b=raw[role+f'_before_allocation{aid}'];a=raw[role+f'_after_allocation{aid}'];assert b[:256]==a[:256]==bytes([0xa5])*256 and b[-256:]==a[-256:]==bytes([0x5a])*256;before[aid]=b[256:-256];after[aid]=a[256:-256]
      assert before[1]==after[1]==pbf.tobytes()and before[2]==after[2]==mapping.tobytes()and before[5]==after[5]==D.tobytes()and before[6]==after[6]==Q.tobytes();assert after[3+stage]==out.tobytes()and before[4-stage]==after[4-stage]
     meta=values[offset:offset+5];offset+=5;assert meta[0]==iteration;d=values[offset:offset+49].reshape(7,7);offset+=49;f=values[offset:offset+49].reshape(7,7);offset+=49;j=values[offset:offset+49].reshape(7,7);offset+=49;k=values[offset:offset+49].reshape(7,7);offset+=49;e=values[offset:offset+7];offset+=7;c=np.frombuffer(raw[f'iter{iteration}_complete_C_raw_phase_f64'],dtype='<f8').reshape(7,7)
     residuals={'orthogonality':float(np.max(np.abs(c.T@S@c-np.eye(7)))),'idempotency':float(np.max(np.abs(d@S@d-2*d))),'electron_count':float(abs(np.trace(d@S)-10)),'eigenproblem':float(np.max(np.abs(f@c-(S@c)*e[None,:])))}
     for name,error in residuals.items():assert error<1e-8;state[name]=max(state[name],error)
    assert offset==len(values)
    lifted=d[mapping[:,None],mapping[None,:]];jp=np.einsum('ijkl,kl->ij',full,lifted);kp=np.einsum('ijkl,jl->ik',full,lifted);jc=np.zeros((7,7));kc=np.zeros((7,7));np.add.at(jc,(mapping[:,None],mapping[None,:]),jp);np.add.at(kc,(mapping[:,None],mapping[None,:]),kp);fresh=H+jc-.5*kc;saved=np.frombuffer(raw['final_fresh_unscreened_physical_F_of_newD_f64'],dtype='<f8').reshape(7,7);assert np.max(np.abs(fresh-saved))<1e-11;comm=float(np.max(np.abs(fresh@d@S-S@d@fresh)));energy=float(prefix[4]+.5*np.sum(d*(H+fresh)));gap=abs(energy-meta[1]);budget=5e-8 if precision=='DP'else 5e-5;assert comm<=budget and gap<=budget
    reviews.append({'precision':precision,'model':m,'trial':t['trial'],'artifact_sha256':desc['sha256'],'full_words':7029,'input_bits_and_raw_readonly_bytes':True,'physical_files_full_hashed':284,'independent_stage_max_abs':maxerr,'independent_stage_max_budget_ratio':maxratio,'state_residuals':state,'fresh_physical_HF_energy':energy,'fresh_physical_commutator_max_abs':comm,'reported_vs_physical_energy_gap':gap,'raw_C_identity_claim':False})
 out={'benchmark':'xlqc','pass_all':True,'artifact_count':len(reviews),'physical_capture_files_hashed':filecount,'source_dependency_hash_bindings_checked':bindings,'results':resultbindings,'CPU_oracle_compiler_command':cmd,'CPU_oracle_library_sha256':sha(libpath),'actual_input_domain':ranges,'numerical_scope':'Complete actual GPU outputs/feedback inputs/full selected SCF states; raw CPU MO phase vectors auxiliary with independentlyrecomputed metric/eigen/projector gates.','native_scientific_algorithm_repaired':False,'reviews':reviews};atomic(z.output,out);print(json.dumps({k:v for k,v in out.items()if k not in('reviews','CPU_oracle_compiler_command')}))
if __name__=='__main__':main()
