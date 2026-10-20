#!/usr/bin/env python3
"""CPU-only exact source-observable and complete retained-internal audit.
Source known scientific defects remain explicit; this is NOT an IID certificate.
"""
import argparse,pathlib,json,struct,hashlib
p=argparse.ArgumentParser();p.add_argument('--result',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args();R=pathlib.Path(__file__).resolve().parents[2];B=pathlib.Path(__file__).resolve().parent;C=R/'.porting-state/porting_logs/permutate-julia-shared/canonical_inputs_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  while b:=f.read(4*1024**2):h.update(b)
 return h.hexdigest()
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
can=[(C/f'actual_input_case{i}.bin').read_bytes()for i in range(3)];prefix=b''.join(can);gold=[]
for i in range(3):
 data=can[i];cid=struct.unpack_from('<I',data)[0];size,n,N,blocks,threads,blen,mean,median=struct.unpack_from('<6I2d',data,4);baseline=struct.unpack_from('<18d',data,44+n)
 samples=data[44:44+n];histogram=[0]*256
 for x in samples:histogram[x]+=1
 independently_mean=sum(k*v for k,v in enumerate(histogram))/n
 if size==1:independently_median=0.5
 else:
  def order_stat(rank):
   total=0
   for value,count in enumerate(histogram):
    total+=count
    if rank<total:return value
  independently_median=(order_stat(n//2)+order_stat((n-1)//2))/2
 assert mean==independently_mean and median==independently_median

 path=C/f'independent_full_shuffle_and_stats_case{i}.bin'
 with path.open('rb')as f:
  count,ns=struct.unpack('<QQ',f.read(16));binary=struct.unpack('<?',f.read(1))[0];original_counts=struct.unpack('<54I',f.read(216));recomputed=[0]*54
  for t in range(ns):
   stats=struct.unpack('<18d',f.read(144))
   for k,(x,y)in enumerate(zip(stats,baseline)):
    if not k:x,y=f32(x),f32(y)
    recomputed[k*3+(0 if x>y else 1 if x==y else 2)]+=1
  assert tuple(recomputed)==original_counts
  packed_offset=f.tell();h=hashlib.sha256()
  while block:=f.read(4*1024**2):h.update(block)
  packed_sha=h.hexdigest()
 assert ns==2500 and size in (1,8) and cid==i and N==2500 and blocks==10 and threads==250
 gold.append(dict(counts=recomputed,packed_bytes=count,packed_sha256=packed_sha,all_stats_rank_recomputed=True,canonical_sha256=sha(C/f'actual_input_case{i}.bin'),gold_sha256=sha(path)))
r=json.loads(a.result.read_text());records=[];internals=[]
for model,m in r['models'].items():
 for t in m.get('trials',[]):
  art=t['output_artifact'];path=pathlib.Path(art.get('saved_path',art.get('path','')))
  if not path.is_absolute():
   options=[a.result.parent/path,R/path]
   path=next((x for x in options if x.is_file()),path)
  raw=path.read_bytes();assert raw[:len(prefix)]==prefix;offset=len(prefix)
  def take(fmt):
   global offset
   z=struct.unpack_from(fmt,raw,offset);offset+=struct.calcsize(fmt);return z
  cases=[]
  for i in range(3):
   cid,completed,maximum=take('<3I');assert cid==i and 1<=completed<=maximum==4
   histories=[]
   for b in range(completed):
    x=take('<54I');expected=[v*(b+1)for v in gold[i]['counts']];assert list(x)==expected;histories.append(x)
    good=all(x[k*3]+x[k*3+1]>5 and x[k*3+1]+x[k*3+2]>5 for k in range(18))
    assert b==completed-1 or not good
   cid2,=take('<I');stats=take('<19d');final=take('<57I');decision,=take('<B');nbytes,=take('<I');assert cid2==i and tuple(final[:54])==histories[-1] and nbytes%12==0
   baseline=struct.unpack_from('<18d',can[i],44+struct.unpack_from('<I',can[i],8)[0]);assert tuple(stats[:18])==baseline
   compressed=[0,0,0];prev=-1;passed=False
   for j in range(nbytes//12):
    iteration,value=take('<Id');assert iteration==prev+1 and value>=0 and not passed;prev=iteration
    compressed[0 if value>stats[18] else 1 if value==stats[18] else 2]+=1
    passed=compressed[0]+compressed[1]>5 and compressed[1]+compressed[2]>5
   assert list(final[54:])==compressed
   expected_decision=good and passed and prev<9999
   assert bool(decision)==expected_decision
   cases.append(dict(case=i,gpu_batches=completed,compression_evaluations=nbytes//12,source_IID_decision=bool(decision),exact_full_rank_history=True))
  assert offset==len(raw)==art['count'] and sha(path)==art['sha256']
  records.append(dict(model=model,trial=t['trial'],artifact=str(path),sha256=art['sha256'],checked_observable_bytes=len(raw),cases=cases))
  if t['trial']==1:
   for case in t['cases']:
    i=case['case'];side=pathlib.Path(case['actual_internal_packed_capture']);assert side.is_file() and side.stat().st_size==gold[i]['packed_bytes'] and sha(side)==gold[i]['packed_sha256'];internals.append(dict(model=model,case=i,path=str(side),bytes=side.stat().st_size,sha256=sha(side),full_internal_exact_to_CPU=True))
assert records
out=dict(benchmark='permutate',auditor_sha256=sha(pathlib.Path(__file__)),independent_full_sample_mean_median_check=True,result_path=str(a.result),result_sha256=sha(a.result),all_pass=True,full_primary_output_artifacts_rehashed=len(records),complete_retained_internal_captures_rehashed=len(internals),gold_all2500_permutation_statistics_rank_recomputed=gold,records=records,retained_internal_captures=internals,scope='Exact source observable outputs, all ranks, earlystop/compression history; complete retained actual trial1 internal shuffles compare independently generated full CPU permutations. Other4 internal arrays fully checked in original processes but not separately archived. Known source shuffle/statistical defects remain; no IID/scientific randomness certification.',gpu_runs=0)
a.output.parent.mkdir(parents=True,exist_ok=True);q=a.output.with_suffix('.tmp');q.write_text(json.dumps(out,indent=2)+'\n');q.replace(a.output);print(json.dumps({k:v for k,v in out.items() if k not in ['records','retained_internal_captures','gold_all2500_permutation_statistics_rank_recomputed']}))
