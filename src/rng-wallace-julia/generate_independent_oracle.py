"""Independent Float64 permutation plus explicit orthogonal matrices; no GPU runs."""
from pathlib import Path
import numpy as np,json,hashlib
p=Path(__file__).resolve().parent;f=np.fromfile(p/'fixture.bin',dtype='<f4');nb=int(f[0]);n=int(f[1]);pool=f[12:12+n].reshape(nb,2048).astype(float);chi=f[12+n+17:12+n+17+nb].astype(float)
H=np.array([[1,-1,-1,-1],[-1,1,-1,-1],[1,1,-1,1],[1,1,1,-1]],dtype=float)*.5
assert np.array_equal(H@H.T,np.eye(4))
outs=[];stats=[]
for lo,hi,steps in f[3:12].reshape(3,3):
 seed=int(lo)+(int(hi)<<16);m=(1664525*seed+1013904223)&0xffffffff;s=(m+np.arange(256,dtype=np.uint32))&255;indices=[]
 for lane in range(8):
  s=(s*241+59)&255;indices.append(s.astype(np.int64)*8+lane)
 idx=np.array(indices).T;assert np.array_equal(np.sort(idx.ravel()),np.arange(2048))
 z=pool.copy();before=np.sum(z*z,axis=1)
 for _ in range(int(steps)):
  generated=z*chi[:,None];q=z[:,idx]
  rr=np.concatenate([q[:,:,:4]@H.T,q[:,:,4:]@(-H.T)],axis=2)
  z=rr.transpose(0,2,1).reshape(nb,2048)
 after=np.sum(z*z,axis=1)
 outs.extend([np.r_[z.ravel(),np.full(17,-12345.)],np.r_[generated.ravel(),np.full(17,-12345.)]])
 stats.append({'seed':seed,'steps':int(steps),'max_relative_squared_norm_drift_Float64':float(np.max(abs(after-before)/before)),'generated_mean':float(generated.mean()),'generated_variance':float(generated.var()),'pool_mean':float(z.mean()),'pool_variance':float(z.var()),'permutation_is_bijection':True,'orthogonal_matrix_exact_identity':True})
gold=np.concatenate(outs).astype('<f8');gold.tofile(p/'expected_outputs.bin');source=np.fromfile(p/'source_cpu_outputs.bin',dtype='<f4').astype(float);assert len(source)==len(gold);err=float(np.max(abs(source-gold)));assert err<1e-3
(p/'fixture_description.json').write_text(json.dumps({'blocks':nb,'pool_elements':n,'threads_per_block':256,'sourcepool2048':True,'path_order':['seed1_chain1','seed305419896_chain7','seed4294967295_chain31'],'prefix_count':len(f),'output_count':len(gold),'native_source_CPU_vs_independent_Float64_max_abs_error':err,'paths':stats,'initial_pool_mean':float(pool.mean()),'initial_pool_variance':float(pool.var()),'chi2_min':float(chi.min()),'chi2_max':float(chi.max()),'fixture_sha256':hashlib.sha256(f.tobytes()).hexdigest(),'gold_sha256':hashlib.sha256(gold.tobytes()).hexdigest(),'source_CPU_sha256':hashlib.sha256((p/'source_cpu_outputs.bin').read_bytes()).hexdigest(),'scope':'Finite identical-stream statefulWallacetransform, notdistributionalcertification/periodquality, default16384blocks/fullCLIunverified. Originalkernelbodiesunchanged; originalWALLACE_NUM_OUTPUTS_PER_RUN=1 means unusedTOTAL_NUM_THREADS offset has loop0, subset257blocks well-defined. HostsourceKISS/BoxMuller/MakeChi2Scale(n=257*2048) generators create fixtureonce, exactbytes usedallmodels.'},indent=2)+'\n')
