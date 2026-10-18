from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;rng=np.random.default_rng(20261018);allin=[];allout=[];cases=[]
for ci,(m,n,z)in enumerate([(20082,20082,150616),(257,509,4097),(1,17,1)],1):
 if ci==1:counts=rng.multinomial(z,np.full(m,1/m))
 elif ci==2:
  counts=np.zeros(m,dtype='i4');counts[0]=n;rest=z-n;slots=[i for i in range(1,m)if i%17];q,rem=divmod(rest,len(slots))
  for i,row in enumerate(slots):counts[row]=q+(i<rem)
 else:counts=np.array([1],dtype='i4')
 assert int(counts.sum())==z and int(counts.max())<=n
 rp=np.r_[0,np.cumsum(counts)].astype('<i4');cols=np.empty(z,dtype='<i4');values=np.empty(z,dtype='<f4')
 for row in range(m):
  s,e=rp[row:row+2];cols[s:e]=rng.choice(n,size=e-s,replace=False)
  ids=np.arange(s,e,dtype='i4');values[s:e]=((ids+1)*np.where(ids%2,1,-1)).astype('f4')
 perm=np.empty(z,dtype='<i4')
 for row in range(m):
  s,e=rp[row:row+2];perm[s:e]=np.argsort(cols[s:e],kind='stable').astype('i4')+s
 io=sum(len(a)for a in allin);oo=sum(len(a)for a in allout);arrays=[rp.view('<u4'),cols.view('<u4'),values.view('<u4')];allin.extend(arrays);out=[rp.view('<u4'),cols[perm].view('<u4'),perm.view('<u4'),values[perm].view('<u4')];allout.extend(out);cases.append(dict(case=ci,m=m,n=n,nnz=z,io=io,oo=oo,input_words=m+1+2*z,output_words=m+1+3*z,empty_rows=int(np.sum(counts==0)),max_row_nnz=int(counts.max()),R=1000))
a=np.concatenate(allin);b=np.concatenate(allout);(P/'input_u32.bin').write_bytes(a.tobytes());(P/'oracle_u32.bin').write_bytes(b.tobytes());(P/'study_cases.json').write_text(json.dumps({'cases':cases},indent=2)+'\n');h=['struct Case{int m,n,z,io,oo;};','const Case cases[]={']
for c in cases:h.append('{'+','.join(map(str,[c[k]for k in ['m','n','nnz','io','oo']]))+'},')
h.append('};');(P/'study_cases.h').write_text('\n'.join(h)+'\n');jl=['const CASES=[']
for c in cases:jl.append('('+','.join(k+'='+str(c[v])for k,v in [('m','m'),('n','n'),('z','nnz'),('io','io'),('oo','oo')])+',),')
jl.append(']');(P/'study_cases.jl').write_text('\n'.join(jl)+'\n');prefix=len(a)+sum(c['nnz']for c in cases)+16*len(cases);guard=5*16*len(cases);meta=dict(NumPy=np.__version__,seed=20261018,cases=cases,input_words=len(a),oracle_words=len(b),prefix_count=prefix,guard_count=guard,artifact_count=prefix+len(b)+guard,scope='Originalfirstgeometry20082²/150616nnz butdirectsyntheticCSR, notdenseoriginalrandominitializer. Secondrectangular/empty/fullrow, thirdsingleton; originalFloat32main only. Uniqueperrowcolumns, signedexactFloat32payloadindices. OriginalsourceR1000statefulsort+gatherpreserved.',repeat_semantics='Resetoriginalunsortedcolumns/identityPonceperoutertrial outsideROI, then1000sourceXcsrsort+Gathercallswithoutinter-reset; dominatesalreadysortedinputafterfirstcall, notfreshunsortedsortingthroughput.',oracle='IndependentCPUper-rowargsort+exactpermutation/payloadassociation; fullrowptr/columns/globalP/valuebits, guards. ValuesexactFloat32integeridentities up150616.',fixtures={n:dict(sha256=hashlib.sha256((P/n).read_bytes()).hexdigest(),bytes=(P/n).stat().st_size)for n in ['input_u32.bin','oracle_u32.bin','study_cases.json','study_cases.h','study_cases.jl']});(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['prefix_count'],meta['oracle_words'],meta['guard_count'],meta['artifact_count'])
