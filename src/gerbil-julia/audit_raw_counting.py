#!/usr/bin/env python3
"""Replay every raw physical word and complete keyed output bijection."""
from pathlib import Path
from collections import Counter
import struct, json, hashlib, argparse
P=Path(__file__).resolve().parent
G=0xa5a5a5a5;G64=0xa5a5a5a5a5a5a5a5

def cases():
 raw=(P/'fixtures.bin').read_bytes();pos=4;out=[]
 for _ in range(struct.unpack_from('<I',raw)[0]):
  k,ip,n,e,threshold,unique,ne,nd=struct.unpack_from('<8I',raw,pos);pos+=32
  keys=struct.unpack_from('<'+'I'*(n*ip),raw,pos);pos+=n*ip*4
  expected=struct.unpack_from('<'+'I'*ne,raw,pos);pos+=ne*4+nd*k
  out.append(dict(k=k,ip=ip,n=n,e=e,threshold=threshold,unique=unique,ne=ne,keys=keys,expected=expected))
 assert pos==len(raw)
 return out

def u64(path):
 b=Path(path).read_bytes();return b,struct.unpack('<'+'Q'*(len(b)//8),b)

def check_case(c,path,artifact,prefix_offset,output_offset):
 raw,v=u64(path);k,ip,n,e,cap,bundles=v[:6]
 assert (k,ip,n,e,cap)==(c['k'],c['ip'],c['n'],c['e'],65536)
 pos=6
 def take(size):
  nonlocal pos
  x=v[pos:pos+size];pos+=size;assert len(x)==size;return x
 table=take(e*(ip+1)+16);keybufs=[take(cap+16) for _ in range(4)]
 failed=take(n*ip+16);echo=take(n*ip+16);nf=take(3);assert pos==len(v)
 for a in [table,*keybufs,failed,echo]:assert a[:8]==(G,)*8 and a[-8:]==(G,)*8
 assert nf==(G64,0,G64)
 assert failed[8:-8]==(G,)*(n*ip)
 assert echo[8:-8]==c['keys']
 expected_buffers=[[G]*cap for _ in range(4)];capacity=cap//ip;nb=0
 for start in range(0,n,capacity):
  stop=min(start+capacity,n);expected_buffers[nb%4][:(stop-start)*ip]=c['keys'][start*ip:stop*ip];nb+=1
 assert nb==bundles
 for actual,expected in zip(keybufs,expected_buffers):assert actual[8:-8]==tuple(expected)
 payload=table[8:-8];rows=[tuple(payload[i*(ip+1):(i+1)*(ip+1)]) for i in range(e)]
 occupied=[i for i,a in enumerate(rows) if a[-1]];empty=[i for i,a in enumerate(rows) if not a[-1]]
 assert all(rows[i]==(0,)*(ip+1) for i in empty)
 got={a[:-1]:a[-1] for a in rows if a[-1]};assert len(got)==len(occupied)
 expected=Counter(tuple(c['keys'][i*ip:(i+1)*ip]) for i in range(n));assert got==expected
 pb=Path(str(path)+'.permutation.u32').read_bytes();perm=struct.unpack('<'+'I'*(len(pb)//4),pb)
 assert sorted(perm)==list(range(e))
 desired=sorted(occupied,key=lambda i:rows[i][:-1])+empty;assert tuple(desired)==perm
 canonical=tuple(x for i in perm for x in rows[i]);pi=prefix_offset;oi=output_offset
 initial=e*(ip+1)+n*ip+1;prefixlen=initial+n*ip
 assert artifact[pi:pi+e*(ip+1)]==(0,)*(e*(ip+1))
 assert artifact[pi+e*(ip+1):pi+e*(ip+1)+n*ip]==(G,)*(n*ip)
 assert artifact[pi+initial-1]==0
 assert artifact[pi+initial:pi+prefixlen]==echo[8:-8]
 assert artifact[oi:oi+e*(ip+1)]==canonical
 assert artifact[oi:oi+c['ne']]==c['expected']
 tail=oi+c['ne'];assert artifact[tail:tail+n*ip]==failed[8:-8];tail+=n*ip
 assert artifact[tail]==nf[1];tail+=1
 guard=[]
 for a in [table,*keybufs,failed,echo]:guard.extend(a[:8]);guard.extend(a[-8:])
 guard.extend([nf[0],nf[2]]);assert artifact[tail:tail+114]==tuple(guard)
 return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'permutation_path':str(path)+'.permutation.u32','permutation_sha256':hashlib.sha256(pb).hexdigest(),'k':k,'full_raw_words_checked':len(v),'full_physical_table_rows_retained':e,'source_stream_bundles_checked':nb,'status':'pass'},pi+prefixlen,tail+114

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--result',required=True);parser.add_argument('--output');args=parser.parse_args();res=json.loads(Path(args.result).read_text());cs=cases();prefixlen=sum(c['e']*(c['ip']+1)+2*c['n']*c['ip']+1 for c in cs);records=[]
 for model,state in res['models'].items():
  if not state.get('complete'):continue
  for trial in state['trials']:
   artifact_path=Path(trial['output_artifact']['saved_path']);ab,art=u64(artifact_path);log=(artifact_path.parent/f'{model}_run1.log').read_text();raws=[]
   for line in log.splitlines():
    if line.startswith('STUDY_RAW_ARTIFACT '):
     x=json.loads(line.split(' ',1)[1])
     if x['trial']==trial['trial']:raws.append(x)
   assert len(raws)==len(cs),(model,trial['trial'],'missing raw case evidence')
   pi=0;oi=prefixlen;rows=[]
   for c in cs:
    item=next(x for x in raws if x['k']==c['k']);row,pi,oi=check_case(c,Path(item['path']),art,pi,oi);rows.append(row)
   assert pi==prefixlen and oi==len(art)
   records.append({'model':model,'trial':trial['trial'],'primary_artifact':str(artifact_path),'primary_sha256':hashlib.sha256(ab).hexdigest(),'complete_raw_cases':rows,'status':'pass'})
 out={'result':args.result,'result_sha256':hashlib.sha256(Path(args.result).read_bytes()).hexdigest(),'checked_model_trials':len(records),'records':records,'all_pass':bool(records),'scope':'Every raw slot/buffer/guard, complete permutation, actual input echo and full canonical mathematical outputs; no physical slot identity claim.'}
 text=json.dumps(out,indent=2)+'\n'
 if args.output:Path(args.output).write_text(text)
 else:print(text)
if __name__=='__main__':main()
