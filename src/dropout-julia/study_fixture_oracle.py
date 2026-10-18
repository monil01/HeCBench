from pathlib import Path
import struct,json,math
p=Path(__file__).resolve().parent;MASK=(1<<32)-1;SEED=12345678;OFFSET=87654321
f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]

def philox(counter,seed):
 x0,x1,x2,x3=counter;k0=seed&MASK;k1=seed>>32
 for _ in range(10):
  a=0xd2511f53*x0;b=0xcd9e8d57*x2;x0,x1,x2,x3=((b>>32)^x1^k0)&MASK,b&MASK,((a>>32)^x3^k1)&MASK,a&MASK;k0=(k0+0x9e3779b9)&MASK;k1=(k1+0xbb67ae85)&MASK
 return x0,x1,x2,x3

def init(seq):return ((OFFSET//4)&MASK,(OFFSET//4)>>32,seq&MASK,seq>>32)
def bump(c):
 n=sum(x<<(32*i)for i,x in enumerate(c))+1
 return tuple((n>>(32*i))&MASK for i in range(4))
def uniform(x):return f(f(f(x)*f(2**-32))+f(2**-33))

sizes=[1028,524292];probs=[f(i/4)for i in range(1,5)];fixture=[0xd00f5a11,len(sizes),SEED&MASK,SEED>>32,OFFSET&MASK,OFFSET>>32,len(probs)]+[struct.unpack('<I',struct.pack('<f',x))[0]for x in probs];data_by_n={}
for n in sizes:
 data=[f(.1)if n==1028 else -0.0 if i%61==0 else f(((i%17)-8)/32)for i in range(n)];data_by_n[n]=data;fixture +=[n,0 if n==1028 else 1]+[struct.unpack('<I',struct.pack('<f',x))[0]for x in data]
totalbits=sum(sizes)*3*len(probs);packed=bytearray((totalbits+7)//8);bitoff=0;reviews=[];samples=[]
for n in sizes:
 for vec,blocks in [(1,512),(2,256),(4,128)]:
  print('CPU_PHILOX_ORACLE',n,vec,flush=True);threads=blocks*256;stride=threads*(4 if vec==1 else vec);values=[0.]*n;seen=bytearray(n)
  for idx in range(min(threads,n if vec==1 else (n+vec-1)//vec)):
   counter=init(idx);cache=philox(counter,SEED);linear=idx if vec==1 else idx*vec;iteration=0
   while linear<n:
    counter=bump(counter);nxt=philox(counter,SEED);r=(cache[1],cache[2],cache[3],nxt[0]);cache=nxt
    if idx in [0,1]and iteration<2:samples.append({'n':n,'vec':vec,'subsequence':idx,'iteration':iteration,'uint4':r,'uniform4_bits':[struct.unpack('<I',struct.pack('<f',uniform(x)))[0]for x in r]})
    for lane in range(4 if vec==1 else vec):
     i=linear+(threads*lane if vec==1 else lane)
     if i<n:assert not seen[i];seen[i]=1;values[i]=uniform(r[lane])
    linear+=stride;iteration+=1
  assert all(seen)and all(0<x<=1 for x in values);counts=[]
  for probability in probs:
   count=0
   for i,x in enumerate(values):
    if x<probability:
     pos=bitoff+i;packed[pos//8]|=1<<(pos%8);count+=1
   counts.append(count);bitoff+=n
  assert counts==sorted(counts);reviews.append({'n':n,'vec':vec,'blocks':blocks,'threads':256,'all_output_random_assignments_covered_once':True,'probabilities':probs,'keep_counts':counts,'p1_drop_due_uniform_endpoint':n-counts[-1],'uniform_min':min(values),'uniform_max':max(values),'source_vec2_unused_zw_preserved':vec==2,'oracle_bit_range':[bitoff-4*n,bitoff]})
assert bitoff==totalbits
(p/'fixtures_u32.bin').write_bytes(struct.pack('<'+'I'*len(fixture),*fixture));(p/'oracle_masks_packed.bin').write_bytes(packed);(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent Python unbounded-integer Philox4x32-10 counter rounds,32bitwrap,subsequenceupper64/counteroffsetlower64; offsetSTATE1 cross-blockuint4 composition, Float32uintconversion*2^-32+2^-33, strictuniform<p; storepackedexpectedmask. OutputFloat32source*mask*(1/p) independentlycomputedfromsharedinput bits atverification.','seed':SEED,'offset':OFFSET,'sizes':sizes,'cases':reviews,'raw_integer_uniform_samples':samples,'counter_continuation_scope':'ActualVEC1secondouterloop, VEC2/VEC4fifthouteriteration exercised bylargeN; vector2unreachableflipbranchnotinvented','limits':['Exactseedstreamcontract, notstatisticalRNGcertification orlearnedmodeltrainingquality','Nativealignedvectorloads requireNdivisible4 and16bytepayloadalignment; arbitraryvectailsunmeasured','Float32uniformrange(0,1] andstrictpredicate allowsrare p1drop; sourcepolicy preserved']},indent=2)+'\n');print({'prefix':len(fixture),'mask_bits':totalbits,'mask_bytes':len(packed),'capture_words':len(fixture)+2*totalbits+24*48,'cases':reviews},flush=True)

# Raw device-RNG verification fixture, onceperprocess diagnostic outsidealltiming.
subseqs=[0,1,255,256,131071];raw_probe=[]
for subseq in subseqs:
 counter=init(subseq);cache=philox(counter,SEED)
 for iteration in range(4):
  counter=bump(counter);nxt=philox(counter,SEED);raw_probe.extend([cache[1],cache[2],cache[3],nxt[0]]);cache=nxt
(p/'probe_subseq_u64.bin').write_bytes(struct.pack('<5Q',*subseqs));(p/'oracle_rng_u32.bin').write_bytes(struct.pack('<80I',*raw_probe))
