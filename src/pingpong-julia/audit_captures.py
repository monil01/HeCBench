#!/usr/bin/env python3
"""Full saved MPI/NCCL buffers, exact pointwise fixture+5 and actual guards."""
from pathlib import Path
import json,argparse,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--result',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();rp=Path(a.result);r=json.loads(rp.read_text());rows=r['models'][a.model]['trials'];sizes=[65536,262144,1048576,257];prefix=5507076;total=3*prefix;checked=0;guards=0
for t in rows:
 d=t['output_artifact'];f=Path(d['saved_path']);assert f.stat().st_size==total*8;assert hashlib.sha256(f.read_bytes()).hexdigest()==d['sha256'];v=np.memmap(f,dtype='<f8',mode='r');k=0;o=prefix
 for variant in range(2):
  for n in sizes:
   ii=np.arange(n,dtype=np.int64);x0=.5+(ii%17-8)/32.;x1=-3.+(ii%13)/64.
   for stage in range(3):
    z=k if stage==0 else o
    for rank in range(2):
     raw=v[z:z+n+64];assert np.all(raw[:32]==-7777.25)and np.all(raw[n+32:]==-7777.25);want=(x0 if rank==0 else x1)if stage==0 else x0+5
     assert np.array_equal(raw[32:n+32],want),(t['trial'],variant,n,stage,rank)
     checked+=n;guards+=64*8;z+=n+64
    if stage==0:k=z
    else:o=z
 assert k==prefix and o==total
out={'benchmark':'pingpong','model':a.model,'result':a.result,'result_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'all5_trials':len(rows),'complete_bothrank_stage_values_checked':checked,'actual_guard_bytes_all5':guards,'input_prefix_count_pertrial':prefix,'artifact_values_pertrial':total,'oracle':'Every actual initialperrankdyadic element, after5incrementsandafter50sourcepingpongroundtrips bothMPI/NCCL. All actual64guardwords/buffer/stage exact.','scope':'Two GPUs, named4sizes; originalup-to1GiBsweep unmeasured, sourcevalidationkerneloutsidecommunicationROI.','issues':[]};Path(a.receipt).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
