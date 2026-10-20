#!/usr/bin/env python3
"""Rehash all physical GPU/guard streams and join every released payload to primary values."""
from pathlib import Path
import argparse,array,hashlib,json,mmap,re,struct
pa=argparse.ArgumentParser();pa.add_argument('result');pa.add_argument('--model',required=True);pa.add_argument('--raw-root',required=True);pa.add_argument('--process-id',type=int,required=True);pa.add_argument('--output',required=True);a=pa.parse_args();r=json.load(open(a.result));receipts=[]
for trial in r['models'][a.model]['trials']:
 pf=open(trial['output_artifact']['saved_path'],'rb');primary=mmap.mmap(pf.fileno(),0,access=mmap.ACCESS_READ);at=int(trial['input_prefix_values'])*8;released={};work={}
 while at<len(primary):
  code=struct.unpack_from('<d',primary,at)[0]
  if code in (1,2,3):
   h=struct.unpack_from('<7d',primary,at);N=int(h[6]);at+=56
   if code==3:at+=8+16*N
   else:at+=8*N*(2 if code==1 else 1)
  elif code==4:
   _,case,id,N=struct.unpack_from('<4d',primary,at);at+=32;released[(int(case),int(id))]=(at,int(N));at+=int(N)*8
  elif code==5:
   _,case,n=struct.unpack_from('<3d',primary,at);assert n==14;at+=24;work[int(case)]=list(map(int,struct.unpack_from('<28d',primary,at)));at+=224
  else:raise AssertionError((code,at))
 assert at==len(primary);cases=[]
 for case in range(5):
  base=Path(a.raw_root)/f'{a.model}_{a.process_id}_trial{trial["trial"]}_case{case}';guards=Path(str(base)+'.guards.u8').read_bytes();assert set(guards)=={165};rf=open(str(base)+'.raw.bin','rb');raw=mmap.mmap(rf.fileno(),0,access=mmap.ACCESS_READ);rows=[json.loads(x)for x in Path(str(base)+'.layout.jsonl').read_text().splitlines()];actual_work=json.load(open(str(base)+'.work.json'));assert actual_work['launches']+actual_work['timed_launches']==work[case];seen=set();guard_at=0;raw_at=0
  for row in rows:
   N=row['payload_bytes'];off=row['raw_offset_bytes'];g=row['guard_offset_bytes'];assert off==raw_at and g==guard_at;actual=raw[off:off+256]+raw[off+256+N:off+512+N];assert actual==guards[g:g+512];raw_at+=N+512;guard_at+=512
   if row['event']=='release':
    id=row['id'];assert id not in seen;seen.add(id);target,count=released[(case,id)]
    if row['dtype']=='float64':assert count*8==N;assert raw[off+256:off+256+N]==primary[target:target+N]
    else:
     assert count*4==N;v=array.array('i');v.frombytes(raw[off+256:off+256+N]);encoded=array.array('d',v).tobytes();assert encoded==primary[target:target+len(encoded)]
  assert raw_at==len(raw)and guard_at==len(guards);assert len(seen)==actual_work['GPU_allocations'];cases.append({'case':case,'GPU_allocations':len(seen),'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_path':str(base)+'.raw.bin','actual_guard_bytes':len(guards),'guard_sha256':hashlib.sha256(guards).hexdigest(),'guard_path':str(base)+'.guards.u8','all_launches':sum(actual_work['launches']),'timed_launches':sum(actual_work['timed_launches'])});raw.close();rf.close()
 receipts.append({'trial':trial['trial'],'primary_sha256':hashlib.sha256(primary).hexdigest(),'complete_primary_values':len(primary)//8,'input_prefix_values':trial['input_prefix_values'],'raw_bytes':sum(c['raw_bytes']for c in cases),'actual_guard_bytes':sum(c['actual_guard_bytes']for c in cases),'cases':cases});primary.close();pf.close()
out={'pass':bool(receipts),'run_id':r['run_id'],'model':a.model,'checked_trials':len(receipts),'raw_bytes':sum(t['raw_bytes']for t in receipts),'actual_guard_bytes':sum(t['actual_guard_bytes']for t in receipts),'trials':receipts,'scope':'EveryactualphysicalGPUbuffer snapshot/guardretained and everyreleasedpayloadjoinedbit-exactlytoprimarydtypeencoding; actualkernelworkcountsjoined. Scientificfullprecisionmoments/outputs/states separatelycomparedtomathoracle/dualrefs. No literalprocesspointeridentityclaim.'};Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k!='trials'})
