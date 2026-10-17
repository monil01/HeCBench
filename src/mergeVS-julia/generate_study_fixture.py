from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent
GUARD=-777777
fixtures=[]
for M,H,qs,kvs in [(64,4,[193,4097],[257,8192]),(128,3,[257,8193],[513,8193])]:
 B=2;C=8193;R=(C+M-1)//M;V=S=8
 vv=[];ss=[]
 for b in range(B):
  v=[0,1,63,64,127,255,kvs[b]//2,kvs[b]-1];v=sorted(v)
  for h in range(H):vv+=v;ss+=[20000,16000,8192,4160,4096,128,100,0]
 vc=[1,3,5,8][:H];sc=[1,2,4,8][:H]
 if H==3:vc=[2,4,8];sc=[2,5,8]
 fixtures.append(dict(B=B,H=H,R=R,M=M,N=M,V=V,S=S,C=C,q=qs,kv=kvs,vv=vv,ss=ss,vc=vc,sc=sc))
def oracle(f,causal):
 B,H,R,M,N,V,S=[f[k] for k in ('B','H','R','M','N','V','S')];ns=B*H*R
 bc=[0]*ns+[GUARD]*17;bo=[0]*(ns*S)+[GUARD]*17;cc=[0]*ns+[GUARD]*17;ci=[0]*(ns*V)+[GUARD]*17
 for b in range(B):
  for h in range(H):
   vert=f['vv'][(b*H+h)*V:(b*H+h)*V+f['vc'][h]];slash=f['ss'][(b*H+h)*S:(b*H+h)*S+f['sc'][h]]
   assert vert==sorted(vert) and slash==sorted(slash,reverse=True)
   for row in range(R):
    start=row*M;end=start+M
    if start>=f['q'][b]:continue
    kv=f['kv'][b];delta=kv-f['q'][b] if causal else kv;threshold=end+delta
    sv=0
    while sv+1<len(slash) and slash[sv]>=threshold:sv+=1
    has=slash[sv]<=threshold;si=max(threshold-slash[sv],M);sv+=1
    rs,re=(si-M,si) if has else ((delta+end,delta+end+N) if causal else (kv,kv+N))
    iv=1;vi=vert[0];finished=False;blocks=[];columns=[]
    def save():
     if rs<kv:blocks.extend(range(rs,min(re,kv),N))
    steps=0
    while True:
     steps+=1;assert steps<1000
     if vi<re:
      if vi<rs:columns.append(vi)
      if iv<len(vert):vi=vert[iv];iv+=1
      else:vi=end+N+delta
     else:
      if sv<len(slash) and (causal or slash[sv]>=start):
       si=max(delta+end-slash[sv],M);sv+=1
      else:
       if iv==len(vert) or (vi>rs and causal):
        if iv==len(vert) and not causal and vi<kv:columns.append(vi)
        save();break
       if causal:rs,re=delta+end,delta+end+N
       else:save();rs,re=kv,kv+N
       finished=True
      if not finished:
       if si>re+M:save();rs,re=si-M,si
       elif si>re:re+=M
    off=(b*H+h)*R+row;assert len(blocks)<=S and len(columns)<=V;assert all(0<=x<kv for x in blocks);assert all(0<=x<kv for x in columns)
    bc[off]=len(blocks);cc[off]=len(columns);bo[off*S:off*S+len(blocks)]=blocks;ci[off*V:off*V+len(columns)]=columns
 return bc+bo+cc+ci
prefix=[];expected=[]
for f in fixtures:
 prefix += [f[k] for k in ('B','H','R','M','N','V','S','C')]
 for k in ('q','kv','vv','ss','vc','sc'):prefix+=f[k]
 for causal in (True,False):expected+=oracle(f,causal)
(p/'fixture_bits.bin').write_bytes(struct.pack('<'+'i'*len(prefix),*prefix));(p/'expected_metadata.bin').write_bytes(struct.pack('<'+'i'*len(expected),*expected))
(p/'fixture_definition.json').write_text(json.dumps({'configurations':fixtures,'input_prefix_count':len(prefix),'physical_output_count':len(expected),'path_order':['M64causal','M64noncausal','M128causal','M128noncausal'],'guard':GUARD,'guards_per_output_array':17,'oracle':'Independent Python row-wise source-contract interval/block bookkeeping, initialskip uniformbothmodes. NOTvalidation offull attentionoutputs orminimalmathematicalmaskunion; originalblockcover policy expands overlappingrangebyM, saveclipsstarts/ends toKV.','boundaries':'Positive variable perheadNNZcount, ascendingvertical anddescendingslash, inactivequeryrows/partialquery+KVtail, firstslashoutofrange, slashfallback+merged/disjoint intervals. Paddingbeyondvalidperheadcounts initialized butnotread.'},indent=2)+'\n')
print(len(prefix),len(expected))
