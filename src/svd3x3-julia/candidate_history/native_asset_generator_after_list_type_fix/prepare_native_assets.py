from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/imo/HeCBench_julia/src/svd3x3-julia');D=P/'native_diagnostics';counts=[1,31,256,257,4097,65537];N0=1048576
A=np.memmap(D/'actual_input_f32.bin',dtype='<f4',mode='r')[16:-16].reshape(9,N0)
controls=[np.zeros((3,3)),np.diag([3.,2.,1.]),np.diag([-3.,2.,-1.]),np.diag([1.,1.,1.]),np.diag([5.,0.,0.]),np.diag([4.,2.,0.]),np.ones((3,3)),np.array([[0,3,0],[0,0,-2],[1,0,0.]])]
for j in range(23):
 t=(j+1)*.137;c=np.cos(t);s=np.sin(t);Q=np.array([[c,-s,0],[s,c,0],[0,0,1.]]);v=(j+1)*.179;c=np.cos(v);s=np.sin(v);V=np.array([[1,0,0],[0,c,-s],[0,s,c]]);controls.append(Q@np.diag([2.5,1.+j*.01,.25])@V.T)
assert len(controls)==31
plan={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_native_primary_measurement_and_all_target_authoring':True,'native_reference_repaired':True,'exclude_untouched_native_aggregates':True,'correction':'Jacobi sweep count 4 to 8 only; full1M independent original-budget audit passes; original four-sweep1300reconstruction failures preserved.','counts':counts,'repeats':7,'threads':256,'blocks':'n//256+1, source extra idleCTA preserved','timed_region':'sum six source seven-launch region means, context completion; all allocation/copies/inputchecks/rawI/O/oracle outside ROI','launches_per_trial':42,'scientific_contract':'Original predeclared physical budgets unchanged; independently check every fullfactorization and compare normalized sortedabs singularvalues; raw singularvectors not unique.','scope':'All1M original native diagnostic; primary targets six named batches up to65537 and analyticcontrols, notfull1M target coverage','dual_reference_tolerance':{'atol':2e-5,'rtol':2e-4},'full_output_and_actual_guards_required':True,'cases':[]}
for k,n in enumerate(counts):
 if k==0:B=np.zeros((9,n),dtype='<f4');indices=np.array([],dtype=np.int64);kind='zero'
 elif k==1:B=np.asarray(controls,dtype='<f4').reshape(31,9).T.copy();indices=np.array([],dtype=np.int64);kind='31_analytic_zero_signeddiag_rank1_rank2_repeatedsigma_rotated_known'
 else:
  indices=np.linspace(0,N0-1,n,dtype=np.int64);indices[0]=373132;indices[1]=784340;B=np.asarray(A[:,indices],dtype='<f4');kind='stratified_original_actual_SoA_including_original_worst_reconstruction_and_sigma'
 M=B.T.astype('f8').reshape(n,3,3);S=np.linalg.svd(M,compute_uv=False);(P/f'case{k}_input_f32.bin').write_bytes(np.ascontiguousarray(B).tobytes());(P/f'case{k}_sigma_f64.bin').write_bytes(np.ascontiguousarray(S).tobytes());plan['cases'].append({'index':k,'n':n,'kind':kind,'selection_indices':indices.tolist() if n<300 else None,'selection_recipe':'linspace(0,1048575,n,int64); positions0,1 replaced373132,784340'if k>1 else None,'input_sha256':hashlib.sha256((P/f'case{k}_input_f32.bin').read_bytes()).hexdigest(),'independent_sigma_sha256':hashlib.sha256((P/f'case{k}_sigma_f64.bin').read_bytes()).hexdigest()})
plan['input_prefix_count']=sum(39*n+100 for n in counts);plan['artifact_count']=sum(42*n+138 for n in counts);(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
s=(D/'eight_sweep_diagnostic/kernels.cu').read_text();(P/'corrected_kernels.cuh').write_text(s);(P/'study_params.h').write_text('#pragma once\nconstexpr int NS[6]={'+','.join(map(str,counts))+'};\nconstexpr int PREFIX='+str(plan['input_prefix_count'])+',COUNT='+str(plan['artifact_count'])+',R=7;\n')
print(json.dumps({'prefix':plan['input_prefix_count'],'count':plan['artifact_count'],'n':sum(counts)}))
