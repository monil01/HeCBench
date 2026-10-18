from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;rng=np.random.default_rng(20261018);N=1025;L=100
alphabet='ACGT';pairs=[];labels=[]
for i in range(N):
 a=''.join(alphabet[x]for x in rng.integers(0,4,size=L));b=list(a);kind=i%7
 if kind==1:
  j=(i*17)%L;b[j]=alphabet[(alphabet.index(b[j])+1)%4]
 elif kind==2:
  for j in rng.choice(L,size=1+i%45,replace=False):b[j]=alphabet[(alphabet.index(b[j])+1+int(rng.integers(0,3)))%4]
 elif kind==3:
  j=(i*13)%L;b=list((a[:j]+alphabet[i%4]+a[j:])[:L])
 elif kind==4:
  j=(i*11)%L;b=list(a[:j]+a[j+1:]+alphabet[i%4])
 elif kind==5:b=[alphabet[x]for x in rng.integers(0,4,size=L)]
 elif kind==6:a=alphabet[(i//7)%4]*L;b=list(alphabet[((i//7)+(i%2))%4]*L)
 pairs.append((a,''.join(b)));labels.append(kind)
def pack(s):
 w=[0]*8
 for j,c in enumerate(s):w[j//16]|=alphabet.index(c)<<(30-2*(j%16))
 return w
MASK=(1<<32)-1
def lsl(x,s):return (x<<s)&MASK if 0<s<32 else x
def lsr(x,s):return x>>s if 0<s<32 else x
def clz(x):return 32-x.bit_length()
def snake(a,b,T):
 j=shift=glob=err=0
 while j<7 and glob<200:
  r1=lsl(b[j],shift);r2=lsr(b[j+1],32-shift);reference=r1|r2
  best=clz((lsl(a[j],shift)|lsr(a[j+1],32-shift))^reference)
  for e in range(1,T+1):
   corner=0
   if j==0 and shift-2*e<0:
    read=lsr(a[j],2*e-shift)
    for c in range(2*e-shift):corner|=lsl(1,31-c)
   elif shift-2*e<0:read=lsl(a[j-1],32-(2*e-shift))|lsr(a[j],2*e-shift)
   else:read=lsl(a[j],shift-2*e)|lsr(a[j+1],32-(shift-2*e))
   best=max(best,clz((read^reference)|corner))
  for e in range(1,T+1):
   corner=0
   if j<5:
    if shift+2*e<32:read=lsl(a[j],shift+2*e)|lsr(a[j+1],32-(shift+2*e))
    else:read=lsl(a[j+1],(shift+2*e)%32)|lsr(a[j+2],32-(shift+2*e)%32)
   else:
    read=lsl(a[j],shift+2*e)|lsr(a[j+1],32-(shift+2*e))
    if glob+32>200:start=glob+32-200;end=start+2*e
    elif glob+32>=200-2*e:start=0;end=2*e
    else:start=end=0
    for c in range(start,end):corner|=lsl(1,c)
   best=max(best,clz((read^reference)|corner))
  z=(best//2)*2
  if j==6 and z>=8:break
  if (z//2<16 and j<5)or(j==6 and z//2<4):err+=1
  if err>T:return 0
  if shift+z+2>=32:j+=1
  if z==32:glob+=z
  else:shift=(shift+z+2)%32;glob+=z+2
 return 1
def edit_distance(a,b):
 # Independent exact Levenshtein bit-vector dynamic program, not the source diagonal heuristic.
 m=len(a);mask=(1<<m)-1;top=1<<(m-1);peq={c:sum(1<<j for j,v in enumerate(a)if v==c)for c in alphabet};vp=mask;vn=0;score=m
 for c in b:
  x=peq[c]|vn;d0=(((x&vp)+vp)^vp)|x;hp=vn|~(d0|vp);hn=vp&d0
  score+=bool(hp&top)-bool(hn&top);hp=(hp<<1)|1;hn<<=1;vp=(hn|~(d0|hp))&mask;vn=(d0&hp)&mask
 return score
A=np.array([pack(a)for a,b in pairs],dtype='<u4');B=np.array([pack(b)for a,b in pairs],dtype='<u4');dist=np.array([edit_distance(a,b)for a,b in pairs],dtype='<i4');oracle=np.array([[snake(a.tolist(),b.tolist(),T)for a,b in zip(A,B)]for T in range(26)],dtype='<i4')
for i,kind in enumerate(labels):
 if kind==0:assert dist[i]==0
 if kind==1:assert dist[i]==1
 if kind==6:assert dist[i]in[0,100]
arrays={'study_reads_u32.bin':A,'study_refs_u32.bin':B,'oracle_results_i32.bin':oracle,'exact_edit_distance_i32.bin':dist}
for name,x in arrays.items():(P/name).write_bytes(x.tobytes())
(P/'study_sequences.tsv').write_text(''.join(a+'\t'+b+'\n'for a,b in pairs));quality=[]
for T in range(26):
 accepted=oracle[T]==1;true=dist<=T;quality.append({'threshold':T,'accepted':int(accepted.sum()),'false_accepts':int((accepted&~true).sum()),'false_rejects':int((~accepted&true).sum())})
meta={'N':N,'length':L,'thresholds':list(range(26)),'repeat':20,'NumPy':np.__version__,'seed':20261018,'DNA_encoding':'A00/C01/G10/T11 big-endian2-bit basepositions,8UInt32/read paddedzero after100bases','scope':'Named1025readpairs all26source thresholds0..25, partialblock5×256, sourcekernelhardcodes100bases independentCLIReadLength','classes':dict(zip(['identical','one_substitution','1to45_substitutions','insertion_trim','deletion_append','random_unrelated','homopolymer'],np.bincount(labels).tolist())),'independent_quality':quality,'source_operational_oracle':'Explicit32bit customshift identity outside1..31/leadingzero greedy diagonal traversal, noteditdistance','independent_scientific_oracle':'ExactLevenshtein bit-vector dynamic program onoriginalstrings','fixtures':{name:{'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest(),'bytes':(P/name).stat().st_size,'dtype':str(x.dtype),'shape':list(x.shape)}for name,x in arrays.items()},'sequence_text_sha256':hashlib.sha256((P/'study_sequences.tsv').read_bytes()).hexdigest()};(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print('snake fixture',N,'false rejects',sum(x['false_rejects']for x in quality),'false accepts',sum(x['false_accepts']for x in quality))
