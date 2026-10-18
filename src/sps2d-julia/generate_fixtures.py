"""Source-compatible sorted unique Int64 CSR/F32 values, exact independent dense oracle."""
from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;inputs=[];oracles=[];cases=[]
def add(a):
 global inputs
 a=np.ascontiguousarray(a).ravel();off=len(inputs);inputs.extend(a.view('<u4'));return off
for ci,(m,n,nnz)in enumerate([(1,1,1),(1,17,5),(17,1,5),(3,7,4),(257,513,1280),(2048,4096,150616)]):
 if ci==3:idx=np.array([0,6,15,20],dtype='i8')
 else:
  # Odd stride for power-of-two large fixture; gcd checked for every shape.
  stride=65537
  while np.gcd(stride,m*n)!=1:stride+=2
  idx=np.sort((np.arange(nnz,dtype='i8')*stride+min(97,m*n-1))%(m*n))
 assert len(np.unique(idx))==nnz and idx.min()>=0 and idx.max()<m*n
 rows=idx//n;cols=(idx%n).astype('<i8');rp=np.r_[0,np.cumsum(np.bincount(rows,minlength=m))].astype('<i8');vals=(1+((7*np.arange(nnz)+3)%1024)/1024).astype('<f4');dense=np.zeros(m*n,dtype='<f4');dense[idx]=vals
 c=dict(id=ci,m=m,n=n,nnz=nnz,rp=add(rp),col=add(cols),val=add(vals),oo=len(oracles));oracles.extend(dense.view('<u4'));cases.append(c)
a=np.array(inputs,dtype='<u4');o=np.array(oracles,dtype='<u4');a.tofile(P/'input_u32.bin');o.tofile(P/'oracle_u32.bin');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();prefix=len(a)+12*len(cases);count=prefix+len(o)+4*16*len(cases)
plan=dict(cases=cases,repeat=1000,input_words=len(a),oracle_words=len(o),input_prefix_count=prefix,artifact_words=count,source_original_cases=[[20082,20082,150616,1000],[41490,41490,17590850,1000]],source_original_square_extents_measured=False,named_largest_shape=[2048,4096],source_first_nnz_preserved=150616,input_initializer='sortedunique deterministicmodularstrideCSR/dyadicF32values[1,2); notsourceglibcRNGshuffle',precision='Int64zero-basedCSR/F32rowmajordense',empty_rows_and_boundaries_checked=True,NumPy=np.__version__,files=[dict(path=n,bytes=(P/n).stat().st_size,sha256=sha(P/n))for n in ['input_u32.bin','oracle_u32.bin']]);(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');fields=['id','m','n','nnz','rp','col','val','oo'];(P/'study_cases.h').write_text('struct Case{int '+','.join(fields)+';};\nconst Case cases[]={\n'+''.join('{'+','.join(str(c[f])for f in fields)+'},\n'for c in cases)+'};\n');(P/'study_cases.jl').write_text('const CASES=[\n'+''.join('('+','.join(f'{f}={c[f]}'for f in fields)+',),\n'for c in cases)+']\n');print('SPS2D',len(a),len(o),prefix,count,flush=True)
