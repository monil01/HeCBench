#!/usr/bin/env python3
"""CPU-only complete CAS rehash/shape/oracle/membership audit. Requires NumPy."""
import argparse,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'.porting-state/portable_julia/sparkler_primary');p.add_argument('--model',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
cache={};groups={};full_bytes=0;count=0;guards=0
for ledger in sorted((a.root/a.model).glob('*_trial[1-5].jsonl')):
 nonce,rank,trial=ledger.stem.rsplit('_',2);group=groups.setdefault(nonce,{})
 rows=[json.loads(x)for x in ledger.read_text().splitlines()];rank=int(rank[4:]);trial=int(trial[5:]);group[rank,trial]=len(rows)
 kinds={k:sum(r['kind']==k for r in rows)for k in('input','output','guards')}
 expected={'input':18 if rank<2 else 15,'output':45 if rank<2 else 30,'guards':90 if rank<2 else 60}
 assert kinds==expected,(ledger,kinds,expected)
 for r in rows:
  sha=r['sha256'];kind=r['kind'];count+=1;guards+=512
  # Rehash complete bytes once per distinct exact CAS, then prove all ledger joins.
  if sha not in cache:
   raw=gzip.decompress((a.root/'CAS'/(sha+'.bin.gz')).read_bytes());assert hashlib.sha256(raw).hexdigest()==sha
   cache[sha]=raw;full_bytes+=len(raw)
  raw=cache[sha];assert len(raw)==r.get('bytes',len(raw))
  if kind=='guards':assert raw==b'\xa5'*512;continue
  assert raw[:256]==b'\xa5'*256 and raw[-256:]==b'\xa5'*256
  if kind=='input':
   ld=r['ld'];ac=r['allocated_cols'];mat=np.frombuffer(raw[256:-256],dtype='<f4').reshape((ac,ld));nr=r['rows'];nc=r['cols'];base=r['base']
   # Exact integer pattern and explicit padding; check allactualcells, notsamples.
   if (sha,'input')not in cache:
    for row in range(nr):
     stride=1+(row+base)%499;actual=mat[:nc,row];expected=np.arange(nc,dtype=np.uint64)%stride==0;assert np.array_equal(actual,expected.astype(np.float32))
    assert not np.any(mat[nc:,:])and not np.any(mat[:nc,nr:]);cache[sha,'input']=True
  else:
   if (sha,'output')in cache:continue
   m=r['m'];ld=r['ld'];ac=r['allocated_cols'];mat=np.frombuffer(raw[256:-256],dtype='<f4').reshape((ac,ld));left=r['left_base'];right=r['right_base'];k=r['fields']
   rs=1+(np.arange(m,dtype=np.uint64)+left)%499;cs=1+(np.arange(m,dtype=np.uint64)+right)%499
   for c in range(m):
    l=rs*cs[c]//np.gcd(rs,cs[c]);expected=1+(k-1)//l;assert np.array_equal(mat[c,:m],expected.astype(np.float32))
   assert not np.any(mat[m:,:])and not np.any(mat[:m,m:]);cache[sha,'output']=True
for nonce,g in groups.items():assert set(g)=={(r,t)for r in range(4)for t in range(1,6)},(nonce,g)
out={'model':a.model,'complete_capture_groups':len(groups),'ledger_records':count,'guard_byte_observations':guards,'unique_CAS_frames':sum(isinstance(k,str)for k in cache),'unique_CAS_bytes_rehashed':full_bytes,'complete_input_bit_oracle':True,'complete_C_lcm_integer_oracle':True,'all_four_ranks_all_five_trials':True,'scope':'Repeated internal B generations not individually inputreadbackclaimed. EveryCcall separatelycaptured/checksummed; alluniqueactualbytes rehashed andfullnumericchecked. CPU-only noGPU/timing.'}
a.receipt.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
