#!/usr/bin/env python3
"""Independent nucleotide/k-mer inventories, source-policy filters, full LCS and clustering."""
import array,collections,hashlib,json,math,random,struct,zlib
from pathlib import Path
P=Path(__file__).resolve().parent
def f(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def code(c):return {'A':0,'C':1,'G':2,'T':3,'U':3}.get(c.upper(),4)
def lcs(a,b):
 row=[0]*(len(a)+1)
 for y in b:
  new=[0]
  for j,x in enumerate(a):new.append(max(row[j]+int(x==y),new[-1],row[j+1]))
  row=new
 return row[-1]
def inputs(n):
 rng=random.Random(431);a=''.join(rng.choice('ACGT')for _ in range(192));b=''.join(rng.choice('ACGT')for _ in range(176));swapped=a[96:]+a[:96];mut=a[:37]+('A'if a[37]!='A'else'C')+a[38:];gap=a[:71]+'N'+a[72:]
 patterns=[a,a,a[:160],mut,swapped,b,'A'*128,gap,b.lower()]
 seq=[patterns[i%len(patterns)]for i in range(n)]
 return sorted(enumerate(seq),key=lambda x:(-len(x[1]),x[0]))
def simulate(c,corrected=True):
 n,k,thr=c['n'],c['k'],c['thr'];lengths=c['lengths'];off=c['offsets'];total=off[-1];coded=[code(chr(v))for v in c['raw']];packed=[0]*(total//16);gaps=[];index=[0]*total;words=[];magic=[]
 for i,L in enumerate(lengths):
  bases=coded[off[i]:off[i]+L];canonical=[b for b in bases if b<4];gaps.append(L-len(canonical))
  for j,b in enumerate(canonical):packed[off[i]//16+j//16]|=b<<((15-j%16)*2)
  hist=[4]*k;count=0;mc=[0]*4
  for j,b in enumerate(bases):
   hist=hist[1:]+[b]
   if b<4:mc[b]+=1
   value=0
   for x in hist:value=(value<<2)|(x&3)
   valid=all(x<4 for x in hist);index[off[i]+j]=value if valid else 65535;count+=valid
  words.append(count);magic+=mc
 rawindex=index.copy();cutoff=[L-k+1-math.ceil(f(f(f(L)*f(1-thr))*f(k)))for L in lengths]
 for i,L in enumerate(lengths):
  start=off[i];w=words[i]
  if corrected:index[start:start+w]=[v for v in index[start:start+L]if v!=65535]
  index[start:start+w]=sorted(index[start:start+w])
 orders=[0]*total
 for i,w in enumerate(words):
  assert w>0
  start=off[i];groups=collections.Counter(index[start:start+w])
  for j in range(w):
   if j==w-1 or index[start+j]!=index[start+j+1]:orders[start+j]=groups[index[start+j]]
 cluster=[-1]*n;table=[0]*65536;r=-1;trace=[];reps=[];stages={'magic_pass':0,'word_filter_pass':0,'alignment_accept':0,'alignment_reject':0};true_scores=[]
 while r<n:
  r+=1
  while r<n and cluster[r]>=0:r+=1
  if r<n:cluster[r]=r
  trace+=[r]+cluster.copy()
  if r>=n-1:break
  reps.append(r)
  for j in range(off[r],off[r]+words[r]):
   if orders[j]:table[index[j]]=orders[j]
  trace+=table
  for q in range(n):
   if cluster[q]>=0:continue
   common=sum(min(magic[r*4+t],magic[q*4+t])for t in range(4));required=math.ceil(f(f(lengths[q])*thr))
   if common>required:cluster[q]=-2;stages['magic_pass']+=1
  trace+=cluster.copy()
  for q in range(n):
   if cluster[q]!=-2:continue
   common=sum(min(table[index[j]],orders[j])for j in range(off[q],off[q]+words[q]))
   if common>cutoff[q]:cluster[q]=-3;stages['word_filter_pass']+=1
   else:cluster[q]=-1
  trace+=cluster.copy()
  for q in range(n):
   if cluster[q]!=-3:continue
   a=[x for x in coded[off[r]:off[r]+lengths[r]]if x<4];b=[x for x in coded[off[q]:off[q]+lengths[q]]if x<4];score=lcs(a,b);required=math.ceil(f(f(lengths[q])*thr));accepted=score>=required
   cluster[q]=r if accepted else -1;true_scores.append(dict(representative=r,query=q,score=score,required=required,accepted=accepted));stages['alignment_accept'if accepted else'alignment_reject']+=1
  trace+=cluster.copy()
  for j in range(off[r],off[r]+words[r]):
   if orders[j]:table[index[j]]=0
  trace+=table
 final=coded+packed+gaps+index+orders+words+magic+cutoff+cluster+table+[r]
 text=''.join('>read_'+str(i)+'\n'+c['seq'][i]+'\n'for i,x in enumerate(cluster)if i==x).encode()
 # Final timed buffers and FASTA first; a separately tagged untimed deterministic replay follows.
 replay=rawindex+[len(trace)]+trace
 expected=final+[len(text)]+list(text)+replay
 # Thirteen buffers: lengths,offsets,reads,compressed,gaps,index,orders,words,magic,cutoff,cluster,table,r.
 guards=[-777]*16+[-777]*16+[165]*16+[0xdeadbeef]*16+[-777]*16+[65534]*16+[65534]*16+[-777]*16+[-777]*16+[-777]*16+[-777]*16+[65534]*16+[-777]*16
 expected+=guards
 return dict(expected=expected,representatives=reps,cluster=cluster,stages=stages,alignment_scores=true_scores,trace_values=len(trace),timed_launches=5+6*len(reps)+1,rawindex=rawindex)
def main():
 cs=[]
 for n,k,thr in[(17,4,.81),(17,5,.86),(33,6,.91),(129,7,.96)]:
  pairs=inputs(n);seq=[x[1]for x in pairs];lengths=list(map(len,seq));offset=[0]
  for L in lengths:offset.append(offset[-1]+L//16*16+16)
  raw=bytearray(b'N'*offset[-1])
  for i,s in enumerate(seq):raw[offset[i]:offset[i]+len(s)]=s.encode()
  c=dict(n=n,k=k,thr=f(thr),lengths=lengths,offsets=offset,raw=list(raw),seq=seq);a=simulate(c);old=simulate(c,False)
  c.update(a);c['original_vs_corrected_cluster_differences']=sum(x!=y for x,y in zip(old['cluster'],a['cluster']));cs.append(c)
 out=bytearray(struct.pack('<I',len(cs)));report=[]
 for c in cs:
  expected=array.array('q',c['expected']);compressed=zlib.compress(expected.tobytes(),9)
  out+=struct.pack('<6If',c['n'],c['k'],len(c['raw']),len(c['raw'])//16,len(expected),len(compressed),c['thr'])
  out+=struct.pack('<%di'%c['n'],*c['lengths']);out+=struct.pack('<%dq'%(c['n']+1),*c['offsets']);out+=bytes(c['raw']);out+=compressed
  report.append({k:v for k,v in c.items()if k in['n','k','thr','representatives','stages','trace_values','timed_launches','original_vs_corrected_cluster_differences','alignment_scores']})
 (P/'fixtures.bin').write_bytes(out);(P/'fixture_manifest.json').write_text(json.dumps({'generator':'Independent base/kmer inventories, strict source-policy filters, full integer LCS/representative clustering. Corrected host valid-word compaction.','source_configurations':report,'expected_compression':'Lossless zlib Int64 oracle fixture only; actual GPU trial captures remain full uncompressed arrays.','fixture_bytes':len(out),'sha256':hashlib.sha256(out).hexdigest(),'scope':'Forward sequence clustering with source strict filters and threshold on original query length; ambiguous bases dropped only for compressed LCS. Not exhaustive all-pairs threshold clustering or reverse-complement matching.'},indent=2)+'\n')
 print(json.dumps([dict(n=c['n'],k=c['k'],representatives=c['representatives'],stages=c['stages'],values=len(c['expected']),timed_launches=c['timed_launches'],cluster_changes=c['original_vs_corrected_cluster_differences'])for c in cs],indent=2));print('fixture',len(out),'bytes')
if __name__=='__main__':main()
