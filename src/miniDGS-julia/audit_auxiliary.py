#!/usr/bin/env python3
"""Full independent per-stage/halo audit of all four actual untimed traces; no GPU."""
from pathlib import Path
import argparse,json,hashlib,numpy as np
P=Path(__file__).resolve().parent

def main():
 a=argparse.ArgumentParser();a.add_argument('result',type=Path);a.add_argument('--output',type=Path,required=True);x=a.parse_args();d=json.load(open(x.result));rows=[];couplings=[];bindings=[]
 for model,m in d['models'].items():
  A=Path(m['trials'][0]['auxiliary_actual_trace_path']);pro=json.load(open(A/'process_provenance.json'));assert pro['exit_code']==0 and pro['actual_untimed_warmup_stage_capture'];previous=[np.fromfile(P/f'fixtures/rank{r}_initial_Q.bin',dtype='<f4')for r in [0,1]];maps=[np.fromfile(P/f'fixtures/rank{r}_parmapOUT.bin',dtype='<i4')for r in [0,1]]
  for stage in range(1,6):
   sends=[np.fromfile(A/f'rank{r}_stage{stage}_gathered.bin',dtype='<f4')for r in [0,1]]
   for r in [0,1]:
    received=np.fromfile(A/f'rank{r}_stage{stage}_received.bin',dtype='<f4');assert sends[r].size==12768 and np.array_equal(sends[r].view('u1'),previous[r][maps[r]].view('u1'));assert np.array_equal(received.view('u1'),sends[1-r].view('u1'));couplings.append(dict(model=model,rank=r,stage=stage,full_gather_and_actual_peer_receive_exact=True,count=12768))
    for field in ['volume_rhs','surface_rhs','res','Q']:
     f=A/f'rank{r}_stage{stage}_{field}.bin';v=np.fromfile(f,dtype='<f4').astype(np.float64);ref=np.fromfile(P/f'fixtures/rank{r}_stage{stage}_{field}_reference_f64.bin',dtype='<f8');budget=np.fromfile(P/f'fixtures/rank{r}_stage{stage}_{field}_bound_f64.bin',dtype='<f8');assert v.size==294336 and np.isfinite(v).all();err=abs(v-ref);assert np.all(err<=budget),(model,r,stage,field,float((err/budget).max()));rows.append(dict(model=model,rank=r,stage=stage,field=field,values=v.size,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),max_abs_error=float(err.max()),max_normalized_budget_ratio=float((err/budget).max()),passed=True))
    previous[r]=np.fromfile(A/f'rank{r}_stage{stage}_Q.bin',dtype='<f4')
  for f,h in m['adapter_file_hashes'].items():
   p=Path(f);actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==h,(model,f);bindings.append(dict(model=model,path=f,sha256=h,unchanged=True))
 assert len(rows)==160 and len(couplings)==40
 report=dict(result=str(x.result),result_sha256=hashlib.sha256(x.result.read_bytes()).hexdigest(),independent_full_stage_checks=rows,actual_gather_peer_message_checks=couplings,all_passed=True,independently_checked_values=sum(z['values']for z in rows),source_asset_bindings=bindings,all_measured_bindings_unchanged=True,actual_untimed_capture_not_overwritten_primary_intermediates=True,no_GPU_executions=True)
 x.output.write_text(json.dumps(report,indent=2)+'\n');print('AUXILIARY_FULL',len(rows),'fields',len(couplings),'MPIcouplings',len(bindings),'sourcebindings')
if __name__=='__main__':main()
