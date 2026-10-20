#!/usr/bin/env python3
"""Mandatory full raw-array comparisons under prospective field-specific budgets; no GPU."""
from pathlib import Path
import argparse,hashlib,json,itertools,numpy as np
P=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('result',type=Path);a.add_argument('--output',type=Path,required=True);x=a.parse_args();d=json.load(open(x.result));plan=json.load(open(P/'study_plan.json'));prefix=plan['input_prefix_count'];pn=plan['prefix_each_rank'];models={m:v for m,v in d['models'].items()if v.get('status')=='native_check_pass'};arrays={};files=[]
 for m,v in models.items():
  assert v['complete']and v['provenance_valid']and len(v['trials'])==5
  for trial in v['trials']:
   f=Path(trial['output_artifact']['saved_path']);h=sha(f);assert h==trial['output_artifact']['sha256']and f.stat().st_size==plan['primary_count']*8;arr=np.memmap(f,dtype='<f8',mode='r');assert np.isfinite(arr).all();arrays[m,trial['trial']]=arr;files.append(dict(model=m,trial=trial['trial'],path=str(f),sha256=h,bytes=f.stat().st_size))
   for rank in [0,1]:
    cursor=0
    for spec in plan['buffers']:
     count=spec['count']+256;before=arr[rank*pn+cursor:rank*pn+cursor+count];after=arr[prefix+rank*pn+cursor:prefix+rank*pn+cursor+count]
     assert np.array_equal(before[:128].view('u1'),after[:128].view('u1'))and np.array_equal(before[-128:].view('u1'),after[-128:].view('u1'))
     if spec['name']in ['D','L','G','S','map']:assert np.array_equal(before.view('u1'),after.view('u1'))
     if spec['name']=='tmp':assert not np.any(after[128+12768:-128])
     cursor+=count
    assert cursor==pn
 comparisons=[]
 for a,b in itertools.combinations(models,2):
  for t in range(1,6):
   av,bv=arrays[a,t],arrays[b,t];assert np.array_equal(av[:prefix].view('u1'),bv[:prefix].view('u1'));fields=[]
   for rank in [0,1]:
    cursor=0
    for spec in plan['buffers']:
     n=spec['count'];start=prefix+rank*pn+cursor+128;aa,bb=av[start:start+n],bv[start:start+n];name=spec['name']
     if name in ['D','L','G','S','map']:assert np.array_equal(aa.view('u1'),bb.view('u1'));budget=np.zeros(n);err=abs(aa-bb);ratio=0.0
     else:
      atol=.002 if name=='rhs'else 3e-5;err=abs(aa-bb);budget=atol+2e-6*np.minimum(abs(aa),abs(bb));assert np.all(err<=budget),(a,b,t,rank,name,float(np.max(err/budget)));ratio=float(np.max(err/budget))
     fields.append(dict(rank=rank,field=name,count=n,max_abs_error=float(err.max()),max_normalized_budget_ratio=ratio,passed=True));cursor+=n+256
   comparisons.append(dict(model_a=a,model_b=b,trial=t,all_full_fields_passed=True,fields=fields))
 report=dict(result=str(x.result),result_sha256=sha(x.result),configuration_id=d['configuration_id'],source_plan_sha256=sha(P/'study_plan.json'),captures=files,comparisons=comparisons,all_full_fields_passed=True,pair_trial_checks=len(comparisons),exact_input_prefix_bytes=True,guards_and_readonly_constants_exact=True,field_budget='Q/res/tmpgather/received3e-5+2e-6minabs, RHS2e-3+2e-6minabs; constants/guards/inputexact; predeclaredbeforetargetdraft',central_envelope_alone_insufficient=True,numerics_only_does_not_adjudicate_timing=True,NumPy=np.__version__,extra_GPU_executions=0);x.output.write_text(json.dumps(report,indent=2)+'\n');print('MANDATORY_FIELD_AUDIT',len(files),'fullcaptures',len(comparisons),'pair-trialspass')
if __name__=='__main__':main()
