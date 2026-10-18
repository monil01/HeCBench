from pathlib import Path
from array import array
import struct,json,hashlib,math,sys
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];d=json.loads((r/'.porting-state/portable_julia/runs'/run/'result.json').read_text());pre=array('d',(p/'fixture.bin').read_bytes());gold=array('d',(p/'oracle.bin').read_bytes());kind=(p/'kinds.bin').read_bytes();rows=[];hashes=set();metrics=dict(max_abs_force_error=0,max_rel_nonzero_force_error=0,max_abs_zero_force_error=0,max_budget_fraction=0,maximum_noiseoff_radial_projection_relative_residual=0);pc=1;oc=0;cases=[]
for ci in range(5):
 N,nt,reps=map(int,pre[pc:pc+3]);period=pre[pc+3:pc+6];pc+=6;L=N+1;E=2*N+1;co=pre[pc:pc+4*L];pc+=4*L;vel=pre[pc:pc+4*L];pc+=4*L;nb=pre[pc:pc+L];pc+=L;bonds=pre[pc:pc+2*E];pc+=2*E;l0=pre[pc:pc+E];pc+=E;params=pre[pc:pc+8*(nt+1)];pc+=8*(nt+1);initial=[pre[pc+a*L:pc+(a+1)*L]for a in range(3)];pc+=3*L;force_start=oc+3*(L+128);case=dict(N=N,nt=nt,reps=reps,period=period,co=co,vel=vel,nb=nb,bonds=bonds,params=params,initial=initial,force_start=force_start);cases.append(case);oc+=6*(L+128)+2*(4*L+512)+(L+128)+(2*E+256)+(E+128)+8*(nt+1+128)
assert pc==len(pre)and oc==len(gold)
for m,z in d['models'].items():
 for t in z['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();v=array('d',raw);assert raw[:len(pre)*8]==(p/'fixture.bin').read_bytes()and len(v)==len(pre)+len(gold);a=v[len(pre):]
  for k,x,y in zip(kind,a,gold):
   assert math.isfinite(x)
   if k:assert struct.pack('<d',x)==struct.pack('<d',y)
   else:
    err=abs(x-y);budget=1e-4+2e-4*abs(y);assert err<=budget;metrics['max_abs_force_error']=max(metrics['max_abs_force_error'],err);metrics['max_budget_fraction']=max(metrics['max_budget_fraction'],err/budget)
    if y:
     if err/abs(y)>metrics['max_rel_nonzero_force_error']:metrics['worst_relative_value']=dict(actual=x,reference=y,absolute_error=err,model=m,trial=t['trial'])
     metrics['max_rel_nonzero_force_error']=max(metrics['max_rel_nonzero_force_error'],err/abs(y))
    else:metrics['max_abs_zero_force_error']=max(metrics['max_abs_zero_force_error'],err)
  c=cases[0];L=c['N']+1;start=c['force_start'];out=[a[start+ax*(L+128)+64:start+ax*(L+128)+64+L]for ax in range(3)];gt=c['params'][5]
  for i,j in [(0,1),(1,0)]:
   de=[c['co'][4*i+ax]-c['co'][4*j+ax]for ax in range(3)];dv=[c['vel'][4*i+ax]-c['vel'][4*j+ax]for ax in range(3)];rad=[out[ax][i]-c['initial'][ax][i]+gt*dv[ax]for ax in range(3)];cross=[rad[1]*de[2]-rad[2]*de[1],rad[2]*de[0]-rad[0]*de[2],rad[0]*de[1]-rad[1]*de[0]];res=math.hypot(*cross)/(math.hypot(*rad)*math.hypot(*de));assert res<1e-4;metrics['maximum_noiseoff_radial_projection_relative_residual']=max(metrics['maximum_noiseoff_radial_projection_relative_residual'],res)
  assert abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12 and len(t['path_times_seconds'])==5;h=hashlib.sha256(raw).hexdigest();hashes.add(h);rows.append(dict(model=m,trial=t['trial'],sha256=h,path=str(f),all_input_initial_force_and_guard_bits_exact=True,all_ghost_zero_neighbor_forces_exact=True,noiseoff_radial_projection_checked=True))
assert len(rows)==20 and d['matched_performance_valid']
a=dict(run=run,all_pass=True,full20_artifacts=len(rows),words_per_artifact=len(pre)+len(gold),unique_hashes=sorted(hashes),metrics=metrics,reciprocal_index_pair_has_different_rest_lengths=True,equal_and_opposite_force_or_total_momentum_claim=False,noiseoff_control='For eachdirectedbond, addbackgamt*dv; remainingforceparallelminimum-imageseparation. Pairdifferentrestlengths, noantisymmetryclaim.',native_reference_repaired=False,rows=rows)
(p/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:v for k,v in a.items()if k!='rows'}))
