from pathlib import Path
import json,hashlib,sys
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;rp=R/sys.argv[1];r=json.load(open(rp));sizes=[('prefix',4556),('stacks',11520),('counts',768),('weights',768),('group',786432),('num',4480),('den',4480),('out',4480),('guards',288)]
starts={};n=0
for key,count in sizes:starts[key]=slice(n,n+count);n+=count
assert n==817772
oracle={key:np.fromfile(P/file,dtype=typ)for key,file,typ in [('stacks','oracle_stacks_u16.bin','<u2'),('counts','oracle_counts_u32.bin','<u4'),('nonzero','oracle_nonzero_u32.bin','<u4'),('weights','oracle_weights_f32.bin','<f4'),('group','oracle_post_f32.bin','<f4'),('num','oracle_num_f32.bin','<f4'),('den','oracle_den_f32.bin','<f4'),('out','oracle_output_u8.bin','u1')]}
noisy=np.fromfile(P/'study_input_u8.bin',dtype='u1');clean=np.fromfile(P/'study_clean_u8.bin',dtype='u1');window=np.fromfile(P/'study_window_f32.bin',dtype='<f4');prefix=np.r_[noisy.astype('f4'),window,np.array([112,40,64,12,19,8,16,3,625,2.7,2500,10],dtype='f4')];guards=np.repeat(np.array([165,1515.25,23063,1234567,1515.25,1515.25,1515.25,1515.25,165],dtype='f4'),32);rows=[];arrays={}
for m,mr in r['models'].items():
 assert len(mr['trials'])==5
 for tr in mr['trials']:
  a=tr['output_artifact'];raw=Path(a['saved_path']).read_bytes();v=np.frombuffer(raw,dtype='<f4');assert len(v)==n and np.all(np.isfinite(v));fields={key:v[s]for key,s in starts.items()};assert np.array_equal(fields['prefix'],prefix);assert np.array_equal(fields['guards'],guards)
  for key in ['stacks','counts','weights']:assert np.array_equal(fields[key],oracle[key]),(m,tr['trial'],key)
  nz=np.zeros(768,dtype=int);valid=fields['weights']>0;nz[valid]=np.rint(1/fields['weights'][valid]).astype(int);assert np.array_equal(nz,oracle['nonzero'])
  scaled={}
  for key in ['group','num','den']:
   err=np.abs(fields[key].astype(float)-oracle[key]);z=float(np.max(err/(.003+.0003*np.abs(oracle[key]))));assert z<=1;scaled[key]=z
  ie=int(np.max(np.abs(fields['out']-oracle['out'])));assert ie<=1;assert np.all(fields['den']>0)
  quality=float(np.mean((fields['out'].astype(float)-clean)**2));arrays[m,tr['trial']]=fields
  rows.append(dict(model=m,trial=tr['trial'],artifact_sha256=hashlib.sha256(raw).hexdigest(),exact_input_prefix=True,exact_discrete_fields=True,exact_288_guards=True,nonzero_counts_exact=True,fieldwise_scaled_CPU_errors=scaled,image_max_gray_error=ie,clean_reference_MSE=quality))
duals=[]
for target in ['cuda_jl','ka','jacc']:
 for reference in ['cuda','cuda_jl']:
  if target==reference:continue
  for trial in range(1,6):
   a=arrays[target,trial];b=arrays[reference,trial]
   for key in ['prefix','stacks','counts','weights','guards']:assert np.array_equal(a[key],b[key]),(target,reference,trial,key)
   scaled={}
   for key in ['group','num','den']:
    z=float(np.max(np.abs(a[key].astype(float)-b[key])/(.003+.0003*np.abs(b[key]))));assert z<=1;scaled[key]=z
   ie=int(np.max(np.abs(a['out']-b['out'])));assert ie<=1;duals.append(dict(target=target,reference=reference,trial=trial,exact_discrete_and_guards=True,scaled_float_errors=scaled,image_max_gray_error=ie))
report=dict(result=str(rp.relative_to(R)),result_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),numpy=np.__version__,all20_complete_artifacts_pass=True,all5_fieldwise_dual_checks_pass=True,central_flat_vector_check_is_weaker_for_discrete_fields=True,fieldwise_contract='Exactmatches/counts/weights/guards/input;postIDCT/numden atol.003 rtol.0003; finalUInt8imageonegray',no_tolerance_relaxation_or_retiming=True,independent_CPU_noisy_MSE=float(np.mean((noisy.astype(float)-clean)**2)),captured_denoised_MSE_range=[min(z['clean_reference_MSE']for z in rows),max(z['clean_reference_MSE']for z in rows)],quality_conclusion='NamedcropdenoisedMSEworsethaninput; agreementnotapplicationimprovement, actualsourcevariance625 retained',artifacts=rows,dual_comparisons=duals)
(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['artifacts','dual_comparisons']}))
