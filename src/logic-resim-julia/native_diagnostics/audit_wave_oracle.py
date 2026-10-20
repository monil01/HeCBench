from pathlib import Path
import json,struct,hashlib
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent;R=B.parent.parent;actual=R/'.porting-state/diagnostics/logic-resim/raw_capture_300M/original_all_net_waves.bin';oracle=B/'oracle_waves_u64.bin';names=json.load(open(B/'wire_names.json'));byname={n:i for i,n in enumerate(names)};assert len(byname)==len(names);refs={}
with oracle.open('rb')as f:
 nw=struct.unpack('<Q',f.read(8))[0]
 for i in range(nw):
  wire,n=struct.unpack('<QQ',f.read(16));refs[i]=(n,f.tell());f.seek(n*16,1)
raw=np.memmap(oracle,dtype='<u8',mode='r');mismatch=[];count=0;events=0
with actual.open('rb')as f:
 while True:
  b=f.read(8)
  if not b:break
  size=struct.unpack('<Q',b)[0];name=f.read(size).decode();n=struct.unpack('<Q',f.read(8))[0];values=np.frombuffer(f.read(n*16),dtype='<u8').reshape(-1,2);i=byname[name];rn,off=refs[i];ref=raw[off//8:off//8+rn*2].reshape(-1,2);count+=1;events+=n
  if n!=rn or not np.array_equal(values,ref):
   first=next((k for k in range(min(n,rn))if not np.array_equal(values[k],ref[k])),min(n,rn));mismatch.append({'wire':i,'name':name,'actual_count':n,'oracle_count':rn,'first_mismatch_index':first,'actual_near':values[max(0,first-1):first+3].tolist(),'oracle_near':ref[max(0,first-1):first+3].tolist()})
report={'benchmark':'logic-resim','status':'pass'if not mismatch else'fail','allnets':count,'actual_events':events,'mismatch_wires':len(mismatch),'examples':mismatch[:15],'source_actual':str(actual.relative_to(R)),'independent_oracle':str(oracle.relative_to(R))}
(P/'wave_oracle_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(bool(mismatch))
