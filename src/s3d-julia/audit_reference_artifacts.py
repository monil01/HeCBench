from pathlib import Path
import json,hashlib,numpy as np
R=Path('/home/imo/HeCBench_julia');P=R/'src/s3d-julia';S=R/'.porting-state/portable_julia';fields=[('WDOT',22),('C',22),('RF',206),('RB',206),('RKLOW',21),('EG_defined',31)];summary={};rows=[]
for tag,log,N in [('f32','s3d_refs_f32_attempt02.log',24**3),('f64','s3d_refs_f64_attempt01.log',16**3)]:
 rp=Path(json.loads((S/log).read_text().splitlines()[-1])['result']);r=json.load(open(rp));assert r['reference_agreement'];e=np.fromfile(P/f'study_cpu_{tag}_f64.bin',dtype='<f8');atol=8*1.401298464324817e-45 if tag=='f32'else 1e-55;rtol=2e-4 if tag=='f32'else 1e-10;models={}
 for m in ('cuda','cuda_jl'):
  md={}
  for trial in r['models'][m]['trials']:
   a=trial['output_artifact'];fn=Path(a['saved_path']);out=np.fromfile(fn,dtype='<f8');assert len(out)==532*N+29;v=out[24*N+29:].reshape(508,N);err=abs(v-e[:,None]);assert np.isfinite(out).all()and np.all(err<=atol+rtol*abs(e[:,None]));assert np.all(v==v[:,0,None]);assert np.all(v[21]==0);c=v[22:44];assert np.all(c>=0);ideal=out[N:2*N]/(out[:N]*8.314510e7);cr=np.max(abs(c.sum(axis=0)-ideal)/abs(ideal));assert cr<=1e-6
   off=0;metrics={}
   for name,count in fields:
    er=err[off:off+count];ex=e[off:off+count];metrics[name]={'max_abs_error_vs_sourceformula_CPU':float(er.max()),'max_rel_error_vs_sourceformula_CPU':float((er/np.maximum(abs(ex[:,None]),1e-300)).max()),'min_actual':float(v[off:off+count].min()),'max_actual':float(v[off:off+count].max()),'subnormal_count_actual':int(np.sum((abs(v[off:off+count])>0)&(abs(v[off:off+count])<np.finfo(np.float32 if tag=='f32'else np.float64).tiny)))};off+=count
   md[str(trial['trial'])]={'field_metrics':metrics,'concentration_sum_coded_ideal_gas_relative_error':float(cr),'WDOT22_exactzero':True,'spatial_uniformity_exact':True};rows.append({'case':tag,'model':m,'trial':trial['trial'],'artifact':str(fn),'sha256':hashlib.sha256(fn.read_bytes()).hexdigest(),'full508N_sourceformula_CPU_pass':True})
  models[m]=md
 summary[tag]={'result':str(rp.relative_to(R)),'N':N,'reference_agreement':True,'models':models}
(P/'reference_saved_output_audit.json').write_text(json.dumps({'numpy':np.__version__,'CPU_oracle_strength':'TranscribedoriginalC++scalarformulas/hostmath, notindependentchemistry.','independent_algebraic_properties':'Cnonnegative/sumC=P/(T*8.314510e7),sourcehomogeneity,sourceWDOT22zero; noelemental/massconservationassertion','all20reference_artifacts_pass':True,'results':summary,'artifacts':rows},indent=2)+'\n');print('20referenceartifacts/sourceformula+independentalgebraicproperties PASS')
