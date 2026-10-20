#!/usr/bin/env python3
"""Real GISSETTE source input + independently expressed full Gram/RBF oracle.
No GPU data copied into CPU fixtures. Source input is published LIBSVM bz2.
"""
from pathlib import Path
import argparse,bz2,json,hashlib,sys
import numpy as np
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);args=ap.parse_args();raw=args.dataset.read_bytes();assert len(raw)==21902461 and hashlib.sha256(raw).hexdigest()=='26d95cd35c9c51d1b25f5741dc95c0482278aab85a90f1920aa6152616b23d41'
x=np.zeros((513,5001),dtype=np.float32);lines=[]
with bz2.open(args.dataset,'rt')as f:
 for i,line in enumerate(f):
  if i==513:break
  lines.append(line)
  for t in line.split()[1:]:
   k,v=t.split(':');x[i,int(k)]=np.float32(float(v))
  assert int(line.split()[-1].split(':')[0])==5000
(P/'input_rows_f32.bin').write_bytes(x.astype('<f4').tobytes());cases=[];prefix=[];outputs=[];budgets=[];u=2**-24
for n in [1,257,513]:
 d=5001;a=x[:n];gamma=float(np.float32(1/5000));y=a.astype(np.float64);gram=y@y.T;absolute=np.abs(y)@np.abs(y).T;norm=(a*a).astype(np.float64).sum(axis=1);idealnorm=(y*y).sum(axis=1);g=d*u/(1-d*u);db=g*absolute+1e-6
 # Primary exact actual-input prefix layout matches source A columnmajor and sequential row x transfers.
 one=np.r_[[n,d,gamma],[-2047.5]*16,a.T.ravel(order='F'),[-2047.5]*16] # A data source row-major tr_ar? fixed below
 # Source A columnmajor flatten: a row-major input x, transpose into feature-major contiguouspoints.
 one=np.r_[[n,d,gamma],[-2047.5]*16,a.T.ravel(order='C'),[-2047.5]*16,[-2047.5]*16,np.zeros(n),[-2047.5]*16]
 for row in a:one=np.r_[one,[-2047.5]*16,row,[-2047.5]*16]
 prefix.append(one.astype('<f8'))
 # Output dots are row-major GEMV results, guards after every row; CPU RBF and source row IDs follow.
 dotrows=np.pad(gram,((0,0),(16,16)),constant_values=-2047.5);dotbud=np.pad(db,((0,0),(16,16)))
 canonical=np.exp(-gamma*np.maximum(0,idealnorm[:,None]+idealnorm[None,:]-2*gram));eb=gamma*(np.abs(norm-idealnorm)[:,None]+np.abs(norm-idealnorm)[None,:]+2*db)+1e-12;kb=canonical*np.expm1(eb)+2e-13
 krows=np.c_[np.arange(1,n+1),canonical];kbud=np.c_[np.zeros(n),kb]
 out=np.r_[norm,dotrows.ravel(),krows.ravel()];budget=np.r_[np.zeros(n),dotbud.ravel(),kbud.ravel()]
 outputs.append(out);budgets.append(budget);cases.append({'n':n,'d':d,'gamma':gamma,'prefix_count':len(one),'output_count':len(out),'source_row_indices':[0,n-1],'actual_A_layout':'feature-major eachfeatureallpoints; columnmajor(n,d)','dot_budget':'gamma_d*sumabsproducts+1e-6','RBF_budget':'Float32squarednormquantization+2dotbudget propagatedexponential interval+2e-13','source_host_formula_budget':2e-13})
for name,a in [('oracle_prefix_f64.bin',np.concatenate(prefix)),('oracle_output_f64.bin',np.concatenate(outputs)),('oracle_budget_f64.bin',np.concatenate(budgets))]:(P/name).write_bytes(np.asarray(a,dtype='<f8').tobytes())
pre=sum(c['prefix_count']for c in cases);count=pre+sum(c['output_count']for c in cases)
plan={'benchmark':'kmc','cases':cases,'primary_dtype':'float64','prefix_count':pre,'primary_count':count,'primary_bytes':count*8,'dual_atol':2e-5,'dual_rtol':2e-6,'all_actual_input_prefix_bytes_exact':True,'all_row_IDs_and_guards_exact':True,'independent_full_oracle':'CPUFloat64Gram/EuclideandistanceRBF allentries + sourceFloat32normpolicy andsourcehostexpformula separate','predeclared_before_primary_and_targets':True,'budget_not_inferred_from_port_differences':True,'source_timing_scope':'sumall771 rowH2D_SGEMV_dotD2H_completion_intervals_excludes_host_RBF_and_actualinput_readbacks','source_full6000rows_unmeasured':True,'SVM_crossvalidation_not_translated':True,'native_GPU_API_sequence_preserved':True,'native_reference_repaired':False,'native_integration_adapted':True,'exclude_untouched_native_aggregates':True,'numpy':np.__version__,'python':sys.version,'source_dataset_sha256':hashlib.sha256(raw).hexdigest(),'source_dataset_bytes':len(raw),'source_prefix_text_sha256':hashlib.sha256(''.join(lines).encode()).hexdigest()}
(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('KMC',pre,count,count*8,'bytes/trial')
