#!/usr/bin/env python3
"""Rehash all5 primary files and join every actual physical buffer to canonical outputs."""
from pathlib import Path
from array import array
import sys,json,re,hashlib,struct
result=Path(sys.argv[1]).resolve();model=sys.argv[2];r=Path(__file__).resolve().parent;x=json.loads(result.read_text());m=x['models'][model];assert m['complete']and m['timing_valid'];rows=m['trials'];assert len(rows)==5
meta=json.loads((r/'data/fixture_manifest.json').read_text());log=(result.parent/(model+'_run1.log')).read_text();paths={int(a):Path(b).resolve()for a,b in re.findall(r'RSMT_RAW trial=(\d+) path=(\S+)',log)};receipts=[]
def ints(b,t='i'):
 a=array(t);a.frombytes(b);return a
for row in rows:
 p=Path(row['output_artifact']['saved_path']);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==row['output_artifact']['sha256'];v=ints(data);rawpath=paths[row['trial']];raw=rawpath.read_bytes();off=0;pre=[];outputs=[];guardbytes=0
 for ci,c in enumerate(meta['cases']):
  ns=[c['nets']+1,c['pins'],c['pins'],c['nets']+1,2*c['pins'],2*c['pins'],2*c['pins'],c['nets']];buf=[];guards=[]
  if model!="cuda":ns.append(3)
  for bi,n in enumerate(ns):
   (count,)=struct.unpack_from('<i',raw,off);off+=4;assert count==n*4+512;b=raw[off:off+count];off+=count;assert b[:256]==b[-256:]==bytes([0xa5])*256;guardbytes+=512;buf.append(b[256:-256]);
   if bi<8:guards.extend(ints(b[:256]));guards.extend(ints(b[-256:]))
  fixture=ints((r/'data'/c['fixture']).read_bytes());assert list(fixture[:2])==[c['nets'],c['pins']];pre.extend(fixture[:2]);pre.extend(ints(buf[0]));pre.extend(ints(buf[1]));pre.extend(ints(buf[2]));assert b''.join(buf[:3])==fixture[2:].tobytes()
  out=[]
  for j in [3,4,5]:out.extend(ints(buf[j]))
  edges=ints(buf[6],'h');out.extend(edges[0::2]);out.extend(edges[1::2]);oracle=ints((r/'data'/(c['name']+'_oracle.bin')).read_bytes());live_count=oracle[c['nets']+1+8*c['pins']];wl=ints(buf[7]);assert sorted(wl[:live_count])==list(oracle[-live_count:])if live_count else True;assert all(z==-640034343 for z in wl[live_count:]);out.append(live_count);out.extend(sorted(wl[:live_count]));assert out==list(oracle);outputs.extend(out);outputs.extend(guards)
  if model!='cuda':
   match=re.search(r'RSMT_CASE trial='+str(row['trial'])+r' case='+str(ci)+r' nets=\d+ pins=\d+ large_worklist=\d+ blocks=(\d+)',log);assert match
   blocks=int(match.group(1));assert list(ints(buf[8]))==[c['nets']+blocks*24,live_count+blocks*3,live_count]
 assert off==len(raw);assert pre+outputs==list(v)
 receipts.append({'trial':row['trial'],'primary_path':str(p),'primary_sha256':hashlib.sha256(data).hexdigest(),'raw_physical_path':str(rawpath),'raw_physical_sha256':hashlib.sha256(raw).hexdigest(),'complete_physical_allocations':16 if model=='cuda' else 18,'actual_guard_bytes_checked':guardbytes,'actual_inputs_and_complete_output_join':True,'worklist_complete_live_ID_bijection':True})
out={'benchmark':'rsmt','model':model,'result':str(result),'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'all5_full_raw_primary_guard_join':True,'trials':receipts,'scope':'Full raw worklist order retained; canonical live worklist sorted bijection; no global optimality claim.'};(r/(model+'_complete_capture_audit.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'model':model,'all5_complete':True,'actual_guard_bytes':sum(a['actual_guard_bytes_checked']for a in receipts)}))
