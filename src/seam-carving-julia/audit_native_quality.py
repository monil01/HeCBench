from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;H=W=128;S=100;I=np.iinfo(np.int32).max
reports={}
for mode in range(3):
 D=P/f'diagnostics/native_mode{mode}';trace=np.fromfile(D/'native_trace_i32.bin','<i4').reshape(S,H);images=np.fromfile(D/'native_preseam_pixels_u32.bin','<u4');assert np.array_equal(trace,np.fromfile(P/f'study_trace_mode{mode}_i32.bin','<i4').reshape(S,H));at=0;rows=[];witness=None
 for step,path in enumerate(trace):
  cw=W-step;n=H*cw;pix=images[at:at+n].reshape(H,cw);at+=n;a=np.stack([pix&255,(pix>>8)&255,(pix>>16)&255],2).astype(np.int32);left=np.concatenate([np.zeros((H,1,3),np.int32),a[:,:-1]],1);right=np.concatenate([a[:,1:],np.zeros((H,1,3),np.int32)],1);above=np.concatenate([np.zeros((1,cw,3),np.int32),a[:-1]],0);u=abs(right-left).sum(2);l=u+abs(above-left).sum(2);r=u+abs(above-right).sum(2);dp=np.empty((H,cw),np.int32);pred=np.zeros((H,cw),np.int32);dp[0]=u[0]
  for row in range(1,H):
   prev=dp[row-1];choices=np.stack([np.concatenate([[I],prev[:-1]+l[row,1:]]),prev+u[row],np.concatenate([prev[1:]+r[row,:-1],[I]])]);k=np.argmin(choices,axis=0);dp[row]=choices[k,np.arange(cw)];pred[row]=np.arange(cw)+k-1
  valid=bool(np.all((path>=0)&(path<cw))and np.all(abs(np.diff(path))<=1));best=int(dp[-1].min());score=None
  if valid:
   score=int(u[0,path[0]])
   for row in range(1,H):score+=int([l,u,r][int(path[row-1]-path[row])+1][row,path[row]])
  gap=None if score is None else score-best;rows.append({'step':step,'width':cw,'source_path_valid':valid,'observed_forward_energy':score,'optimal_forward_energy':best,'gap':gap})
  if witness is None and (not valid or gap>0):
   ideal=np.empty(H,np.int32);col=int(np.argmin(dp[-1]));ideal[-1]=col
   for row in range(H-1,0,-1):col=int(pred[row,col]);ideal[row-1]=col
   ideal_score=int(u[0,ideal[0]])+sum(int([l,u,r][int(ideal[row-1]-ideal[row])+1][row,ideal[row]])for row in range(1,H));assert ideal_score==best
   witness=dict(rows[-1],observed_GPU_path=path.tolist(),lower_energy_valid_path=ideal.tolist(),source_pixel_offset_values=at-n,source_pixel_count_values=n)
 assert at==len(images)
 reports[str(mode)]={'mode':['standard','update','approx'][mode],'actual_GPU_trace_and_preseam_images':True,'not_a_timing_series':True,'trace_equals_CPU_source_emulator':True,'all100_paths_valid':all(z['source_path_valid']for z in rows),'nonoptimal_forward_energy_steps':sum(z['gap']is not None and z['gap']>0 for z in rows),'maximum_gap':max(z['gap']or 0 for z in rows),'is_minimum_energy_contract_failure':mode<2 and any(z['gap']is None or z['gap']>0 for z in rows),'is_intended_approximation_compromise':mode==2,'first_witness':witness,'steps':rows,'artifact_hashes':{f:hashlib.sha256((D/f).read_bytes()).hexdigest()for f in ['native_trace_i32.bin','native_preseam_pixels_u32.bin']}}
(P/'diagnostics/native_quality_witness.json').write_text(json.dumps({'numpy':np.__version__,'scope':'GPUobserved100seams/actualpreseamRGB; independentlyscoreforward-energyobjectiveandprovidevalidlowerenergypath, nothumanperceptualquality.','source_backtracking_defect':'Predecessorselectionignorescurrenttransitioncosts; sourcealsohasoffbyonelogicalrightcondition,but noinvalidseamsobservedatthisfixture.','cases':reports},indent=2)+'\n');print({k:{f:v[f]for f in ['nonoptimal_forward_energy_steps','maximum_gap','is_minimum_energy_contract_failure']}for k,v in reports.items()})
