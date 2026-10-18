#!/usr/bin/env python3
"""CPU-only fullstate/independentDoubleobjective audit of all20primary artifacts."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;r=json.loads(Path(sys.argv[1]).read_text());input=np.fromfile(P/'study_input_r10_f32.bin',dtype='<f4');cpu=np.fromfile(P/'study_expected_cpu_r10_f32.bin',dtype='<f4');prefix=np.r_[input,np.array([30,10,.5,1.5,1.5,3.1415],dtype='f4')];Q=900;phi=float(np.float32(3.1415))
def fitness(a):
 a=np.asarray(a,dtype='f8').reshape(-1,30);y=1+(a-1)/4;return np.sin(phi*y[:,0])**2+(y[:,-1]-1)**2+np.sum((y[:,:-1]-1)**2*(1+10*np.sin(phi*y[:,1:])**2),axis=1)
initialfit=fitness(input[:Q]);rows=[]
for model,m in r['models'].items():
 for t in m['trials']:
  path=Path(t['output_artifact']['saved_path']);a=np.fromfile(path,dtype='<f4');assert len(a)==5466 and np.array_equal(a[:2736],prefix) and np.isfinite(a).all();state=a[2736:];d=abs(state.astype('f8')-cpu);assert d.max()<=1e-3;pbfit=fitness(state[2*Q:3*Q]);assert np.all(pbfit<=initialfit+1e-3);globalfit=float(fitness(state[3*Q:])[0]);rows.append({'model':model,'trial':t['trial'],'max_abs_all_state_vs_source_CPU':float(d.max()),'max_abs_personal_best_vs_source_CPU':float(d[2*Q:3*Q].max()),'independent_Double_personal_best_not_worse_than_initial':True,'global_guide_fitness_Double':globalfit,'best_personal_fitness_Double':float(pbfit.min()),'global_guide_is_not_global_argmin_witness':bool(globalfit>pbfit.min()),'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
print(json.dumps({'benchmark':'pso','numpy':np.__version__,'artifacts_checked':len(rows),'all_pass':True,'independent_Doubleobjective_is_diagnostic_not_new_fulltrajectoryoracle':True,'actual_initial_min':float(input[:Q].min()),'actual_initial_max':float(input[:Q].max()),'initial_values_above_declared_START_RANGE_MAX':int(np.sum(input[:Q]>np.float32(5.12))),'original10000_fullstate_not_certified':True,'artifacts':rows},indent=2))
