"""Independent closed-form integer stages and paired-Float32 gradient recurrence."""
from pathlib import Path
import json,sys,struct,math,ctypes,hashlib
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
lib=ctypes.CDLL('libm.so.6');fma=lib.fmaf;fma.argtypes=(ctypes.c_float,ctypes.c_float,ctypes.c_float);fma.restype=ctypes.c_float
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());first=next(m['trials'][0]for m in r['models'].values()if m.get('trials'));raw=Path(first['output_artifact']['saved_path']).read_bytes();firstvals=struct.unpack('<184400f',raw);prefix=[];expected=[];stats=[];off=0
for n,batches,seed in[(1025,4,0),(8193,5,1)]:
 for mode in(0,1):
  init=[[0.]*n,[1. if seed==0 else((i%17)-8)/8 for i in range(n)],[1. if seed==0 else((i%9)-4)/8 for i in range(n)],[0.]*n,[0.]*n]
  for x in init:prefix.extend(x)
  controls=firstvals[off+5*n:off+5*n+10];lo,hi=controls[8:10];assert hi<=lo and list(controls[:8])==[n,batches,seed,mode,256,256,1,f32(.01)];prefix.extend([n,batches,seed,mode,256,256,1,f32(.01),lo,hi]);off+=5*n+10
  pref=[f32(i+batches)for i in range(n)];weights=init[1].copy();gin=init[2];act=[];gout=[];maxfixed=0
  for batch in range(1,batches+1):
   act=[f32(math.sin(0. if batch==1 else float(i+batch-1)))for i in range(n)];gout=[]
   for i,(a,initialg)in enumerate(zip(act,gin)):
    g=initialg;delta=a if i%2 else-a
    for pair in range(128):
     new=f32(f32(g+delta)-delta)
     if new==g:break
     g=new
    maxfixed=max(maxfixed,pair+1);v=f32(f32(math.sin(g))/f32(math.cos(g)));gout.append(v);weights[i]=float(fma(-f32(.01),v,weights[i]))
  for x in(pref,weights,gin,act,gout):expected.extend(x)
  stats.append({'n':n,'batches':batches,'mode':mode,'max_pairs_until_exact_fixed_point_or_128':maxfixed,'zero_gradient_rows':sum(g==0 for g in gin),'zero_gradient_weights_exactly_unchanged':all(weights[i]==init[1][i]for i,g in enumerate(gin)if g==0),'weight_values_changed':sum(a!=b for a,b in zip(weights,init[1])),'max_abs_weight_change':max(abs(a-b)for a,b in zip(weights,init[1]))})
assert len(prefix)==92220 and len(expected)==92180;reports=[]
for m,v in r['models'].items():
 for tr in v['trials']:
  path=Path(tr['output_artifact']['saved_path']);raw=path.read_bytes();got=struct.unpack('<184400f',raw);assert list(got[:92220])==prefix;err=max(abs(a-b)for a,b in zip(got[92220:],expected));assert err<=1e-3,(m,tr['trial'],err);reports.append({'model':m,'trial':tr['trial'],'all5buffer_values':92180,'input_control_values':92220,'max_abs_error_to_independent_recurrence':err,'sha256':hashlib.sha256(raw).hexdigest()})
w={'benchmark':'streamPriority','result':str(rp),'captured_trials_checked':len(reports),'method':'Integer-range cancellation proves forward/prefetch net stage values for measured exact integers. Independent paired Float32 gradient recurrence with exact fixed-point termination and correctlyrounded mathematical sin/cos target; hostlibm fmaf roundsSGDsingle-FMA. Full5finalbuffers, notonlynativegradout.','scope':'Synthetic ML-like scheduling pipeline, not actualgradientofspecifiedloss or learnedmodel accuracy. NamedvalueskeepintegerstagewithinexactFloat32integer range. Source1e-3bound retained; this doesnot prove arbitrary native defaultN2^24precision policy.','cases':stats,'reports':reports};p=Path('.porting-state/portable_julia/scientific_witnesses_300/streamPriority.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w))
