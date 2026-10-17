"""Independent Python arbitrary-integer modular exponentiation oracle."""
from pathlib import Path
import struct,json,hashlib,sys
p=Path(__file__).resolve().parent
bases=[2,7,61]
def mr(n,cnt):
 d=n-1;s=0
 while d%2==0:d//=2;s+=1
 for a in bases[:cnt]:
  if a%n==0:continue
  z=pow(a,d,n)
  if z in (1,n-1):continue
  for j in range(1,s):
   z=z*z%n
   if z==n-1:break
  else:return 0
 return 1
files={}
for mode in ['prime','odd']:
 source=p/f'study_{mode}_u32.bin';ns=struct.unpack('<400000I',source.read_bytes());assert all(n>=5 and n%2 for n in ns)
 cache={};out=[];totals=[0]*4
 for i,n in enumerate(ns):
  if n not in cache:cache[n]=[mr(n,c)for c in [1,2,3]]
  codes=cache[n];out.extend(codes);totals[i//100000]+=sum(codes)
 if mode=='prime':assert all(out)
 out.extend(totals);target=p/f'study_expected_{mode}_u32.bin';target.write_bytes(struct.pack('<'+str(len(out))+'I',*out))
 print(mode,'unique',len(cache),'minmax',min(ns),max(ns),'totals',totals,flush=True)
 for f in [source,target]:files[f.name]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(p/'fixture_provenance.json').write_text(json.dumps({'generator':'Unchangedsource set_nprimes/set_nintegers, exactfixedxorshift128seeds/wheel/nominal8,16,24,32bitmasks,100000each','oracle':'IndependentPythonarbitraryintegerpow Miller-Rabin all3baseprefix decisions perinput; cacheonlyduplicateoraclevalues, GPUfull400000inputs','python':sys.version,'bases':bases,'input_prefix_count':400003,'per_input_classifications':1200000,'timed_aggregate_counts':4,'files':files},indent=2)+'\n')
