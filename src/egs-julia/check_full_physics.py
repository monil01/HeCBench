"""Independent complete detector accumulation and full-particle invariant checks.
These invariants do not constitute an independent full Monte Carlo trajectory oracle.
"""
from pathlib import Path
from array import array
import argparse,json,math,hashlib,sys
p=argparse.ArgumentParser();p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=a.capture_dir;M=json.loads((D/'capture_manifest.json').read_text());assert sys.byteorder=='little';desc={d['name']:d for d in M['allocations']};types={'uint32':'I','int32':'i','uint64':'Q','float32':'f','float64':'d'}
def read(label,name):
 d=desc[name];p=D/f'{label}_{name}.bin';raw=p.read_bytes();assert len(raw)==d['bytes'];v=array(types[d['encoding']]);v.frombytes(raw);return v
# Fixed engineering invariant budgets, not posterior tolerances fitted to GPU outputs.
DIRECTION_SQUARED_NORM_BUDGET=1e-3;ENERGY_WEIGHT_ABS_BUDGET=1e-7;ENERGY_WEIGHT_REL_BUDGET=2e-6
edges=read('initial','source_edges');assert len(edges)==146 and M['scalars'][0]==145;emax=max(edges);grand=[0]*9;history=[];max_norm=0.;max_acc=0.;checked_pixels=0;prev={name:read('initial',name)for name in desc if name.startswith('total_count_') or name.startswith('total_energy_')};bytehashes={}
for cap in M['captures']:
 for record in cap['records'].values():
  q=D/record['path'];assert q.stat().st_size==record['bytes'];assert hashlib.sha256(q.read_bytes()).hexdigest()==record['sha256'];bytehashes[record['path']]=record['sha256']
 if cap['label']=='initial':continue
 label=cap['label'];counts=read(label,'step_counts');grand=[x+y for x,y in zip(grand,counts)];stack=[read(label,f'stack_{c}')for c in 'abc'];active=0;directions=[]
 for i in range(32):
  x,y,z=[v[4*i:4*i+4]for v in stack];status=x[0]&255;assert 0<=status<=8;assert (x[0]>>16)&255==0;assert x[3]<=32768;active+=status!=8
  floats=array('f');words=array('I',[x[1],x[2],y[1],y[2],y[3],z[0],z[1],z[2]]);floats.frombytes(words.tobytes());energy,weight,px,py,pz,u,v,w=floats;assert all(math.isfinite(t)for t in floats);assert energy>=0 and weight>=0;assert energy<=emax+1e-6;norm=u*u+v*v+w*w;err=abs(norm-1);max_norm=max(max_norm,err);assert err<=DIRECTION_SQUARED_NORM_BUDGET
  assert z[3]==y[3],'Sourceunused stackC.w mustretain last tmp.w=stackB.z bits'
 started=grand[7];terminated=sum(grand[i]for i in [0,1,5,6]);assert started==terminated+active
 totalsum=energysum=0.
 for c in range(4):
  sc=read(label,f'score_count_{c}');se=read(label,f'score_energy_{c}');tc=read(label,f'total_count_{c}');te=read(label,f'total_energy_{c}');pc=prev[f'total_count_{c}'];pe=prev[f'total_energy_{c}'];assert len(sc)==len(se)==len(tc)==len(te)==512*512
  for k in range(len(sc)):
   assert all(math.isfinite(q)and q>=0 for q in [sc[k],se[k],tc[k],te[k]])
   ec=pc[k]+float(sc[k]);ee=pe[k]+float(se[k]);e=max(abs(tc[k]-ec),abs(te[k]-ee));max_acc=max(max_acc,e);assert tc[k]==ec and te[k]==ee,'Independentfullpixel CPUaccumulation mismatch'
   assert se[k]<=emax*sc[k]+ENERGY_WEIGHT_ABS_BUDGET+ENERGY_WEIGHT_REL_BUDGET*sc[k];assert te[k]<=emax*tc[k]+ENERGY_WEIGHT_ABS_BUDGET+ENERGY_WEIGHT_REL_BUDGET*tc[k]
  totalsum+=math.fsum(tc);energysum+=math.fsum(te);prev[f'total_count_{c}']=tc;prev[f'total_energy_{c}']=te;checked_pixels+=len(tc)*2
 inner=read(label,'inner_iterations')[0];assert 1<=inner<=32768
 history.append({'phase':label,'activity':sum(counts[:8]),'step_counts':list(counts),'started_cumulative':started,'terminated_cumulative':terminated,'active':active,'inner_iterations':inner,'list_depth':read(label,'list_depth')[0],'detector_count_weight':totalsum,'detector_energy_weight':energysum,'total_source_weight':read(label,'total_weight')[0]})
assert history[0]['inner_iterations']==32768;assert history[-1]['activity']==0 and history[-1]['active']==0;assert all(grand[j]>0 for j in [1,2,3,4,5,7]);assert grand[6]==0,'Source80keV belowpairthreshold'
report={'all_pass':True,'kind':'native_readiness_fullbuffer_diagnostic_not_primary_trials','allocations':len(desc),'phase_count':len(history),'history':history,'complete_pixel_values_independently_CPU_accumulated':checked_pixels,'all_pixels_accumulation_max_absolute_error':max_acc,'max_particle_direction_squared_norm_error':max_norm,'direction_squared_norm_budget':DIRECTION_SQUARED_NORM_BUDGET,'energy_weight_budget':{'absolute':ENERGY_WEIGHT_ABS_BUDGET,'relative':ENERGY_WEIGHT_REL_BUDGET},'full_saved_artifact_hashes_verified':len(bytehashes),'population_conservation_exact':True,'unmodified_GPU_physics_kernel_scope':True,'scientific_limitations':['No independent fulltransport/EGSnrc-physicsaccuracy certification','Positivity/particleconservation/photonkinematics/detectoraccumulation are independentinvariants, not unique stochastictrajectory proof','Source1warp/requestedthreshold1 differsoriginal80blocks16warps100MCLI','Histogramhostpreparation corrected; excludeduntouchednative stratum'],'manifest_sha256':hashlib.sha256((D/'capture_manifest.json').read_bytes()).hexdigest()};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
