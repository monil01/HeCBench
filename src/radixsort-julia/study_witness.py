"""Independent stable digit-group oracle and complete key/scratch audit. No GPU work."""
import array,argparse,collections,hashlib,json,itertools
from pathlib import Path
MASK=2**32-1
def fixture(n):
 if n==32768:return[0 if i%11==0 else MASK if i%13==0 else(0x80000000|(i&255))if i%17==0 else((i*2654435761)&MASK)^(i>>7)^0xa5a55a5a for i in range(n)]
 return[(n-1-i)%1024|(((i//1024)%16)<<28)for i in range(n)]
def grouped(a,bit):
 groups=[[]for _ in range(16)]
 for key in a:groups[(key>>bit)&15].append(key)
 return list(itertools.chain.from_iterable(groups))
def expected(input):
 keys=list(input);n=len(keys)
 for repeat in range(3):
  for bit in range(0,32,4):
   if repeat==2 and bit==28:
    tmp=[]
    for first in range(0,n,512):tmp.extend(grouped(keys[first:first+512],bit))
   keys=grouped(keys,bit)
 assert keys==sorted(input)
 nb=n//256;cn=n//16;counts=[0]*cn;starts=[0]*cn
 for group in range(nb):
  for pos in range(256):
   digit=tmp[group*256+pos]>>28;index=digit*nb+group
   if counts[index]==0:starts[group*16+digit]=pos
   counts[index]+=1
 scan=[];running=0
 for count in counts:scan.append(running);running+=count
 buffer=[scan[i]for i in range(0,cn,1024)]
 return keys,tmp,counts,starts,scan,buffer

def audit(path):
 result=json.loads(Path(path).read_text());cache={};reports=[]
 for model,info in result['models'].items():
  for trial in info.get('trials',[]):
   source=Path(trial['output_artifact']['saved_path']);blob=source.read_bytes();digest=hashlib.sha256(blob).hexdigest()
   if digest not in cache:
    a=array.array('I');a.frombytes(blob);assert a.itemsize==4;off=0;inputs=[]
    for n in (32768,65536):
     cn=n//16;bn=cn//1024;inp=list(a[off:off+n]);off+=n;assert inp==fixture(n);scratch=n+3*cn+bn;assert all(x==0 for x in a[off:off+scratch]);off+=scratch;assert list(a[off:off+7])==[n,32,3,128,256,4,1];off+=7;inputs.append(inp)
    assert off==215060;cases=[]
    for inp in inputs:
     n=len(inp);arrays=expected(inp);counts=[]
     for name,want in zip(['keys','local_temporary_keys','digit_major_counters','block_major_starts','exclusive_counter_scan','top_scan_buffer'],arrays):
      got=list(a[off:off+len(want)]);off+=len(want);assert got==want,(model,trial['trial'],n,name);counts.append({'name':name,'all_values_checked':len(want)})
     assert collections.Counter(arrays[0])==collections.Counter(inp)
     lower=collections.defaultdict(set)
     for key in inp:lower[key&0x0fffffff].add(key>>28)
     report={'n':n,'full_sorted_keys_and_multiset_exact':True,'complete_scratch_arrays':counts,'lower28_groups_with_multiple_upper_nibbles':sum(len(s)>1 for s in lower.values()),'maximum_distinct_upper_nibbles_for_same_lower28':max(map(len,lower.values()))}
     if n==65536:assert len(lower)==1024 and all(len(s)==16 for s in lower.values());assert set(collections.Counter(inp).values())=={4};report['all_exact_keys_duplicated_four_times']=True
     cases.append(report)
    assert off==len(a)==430106;cache[digest]=cases
   reports.append({'model':model,'trial':trial['trial'],'artifact_sha256':digest,'cases':cache[digest]})
 return {'benchmark':'radixsort','result':str(Path(path).resolve()),'captured_trials_checked':len(reports),'distinct_complete_artifacts_independently_checked':len(cache),'verification_contract':'Fullkeys equalitytoindependentsort, exactmultiset, lastlocal512digitblocks/temp,radixcounters/starts,exclusiveprefixandtopbuffer after3nonresetfull32bitsorts.','legacy_checker_blindspot':{'input':[1,2],'wrong_output':[0,0],'monotonic_check_passes':True,'exact_multiset_fails':True,'scope':'Minimalcheckerweakness,notobservedfailureofnativeconfiguredpipeline.'},'scope':'Key-only32bitintegerordering;noassociatedvalue/stabilitycertificate,twovalidpoweroftwosizes,not4MdefaultCLIperf. V1validfixturepreserved;V2intentionallystrengthenslower28/uppernibbletiesbeforeportablefork,notperformance selection.','reports':reports}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--out');args=p.parse_args();out=audit(args.result);text=json.dumps(out,indent=2)+'\n';print(text)
 if args.out:Path(args.out).write_text(text)
