from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;rng=np.random.default_rng(20261018);cases=[];allinput=[];alloracle=[];denseoracles=[]
A=np.fromfile(P/'source_A_f32.bin',dtype='<f4').reshape(1024,1024);B=np.fromfile(P/'source_B_f32.bin',dtype='<f4').reshape(1024,1024)
for ci,(m,k,n)in enumerate([(1024,1024,1024),(129,97,33)],1):
 if ci==2:
  A=np.zeros((m,k),dtype='<f4')
  for row in range(m):
   if row%17:
    cols=np.sort(rng.choice(k,size=1+row%13,replace=False));v=rng.integers(-64,65,size=len(cols));v[v==0]=1;A[row,cols]=v.astype('f4')/16
  B=(rng.integers(-32,33,size=(k,n)).astype('f4')/32).astype('<f4')
 rows,cols=np.nonzero(A);rows=rows.astype('<i4');cols=cols.astype('<i4');values=A[rows,cols].astype('<f4');rp=np.r_[0,np.cumsum(np.bincount(rows,minlength=m))].astype('<i4');nnz=len(rows)
 # Independent dense BLAS and sparse row expansion oracle methods.
 C=A.astype('f8')@B.astype('f8');sparse=np.zeros_like(C)
 for row,col,val in zip(rows,cols,values):sparse[row]+=float(val)*B[col].astype('f8')
 assert np.max(np.abs(sparse-C))<1e-10
 arrays=[rows.view('<u4'),cols.view('<u4'),rp.view('<u4'),values.view('<u4'),B.ravel().view('<u4')];start=sum(len(x)for x in allinput);cursor=start;offsets=[]
 for a in arrays:offsets.append(cursor);allinput.append(a);cursor+=len(a)
 oo=sum(len(x)for x in alloracle);alloracle.append(C.ravel().astype('<f8'));cases.append(dict(case=ci,m=m,k=k,n=n,nnz=nnz,input_offsets=dict(zip(['rows','columns','row_offsets','values','B'],offsets)),input_end=cursor,oracle_offset=oo,oracle_count=m*n,source_initializer=ci==1,zero_rows=np.flatnonzero(np.diff(rp)==0).tolist(),full_dense_oracle_disagreement=float(np.max(np.abs(sparse-C)))))
input=np.concatenate(allinput);oracle=np.concatenate(alloracle);(P/'input_u32.bin').write_bytes(input.tobytes());(P/'oracle_f64.bin').write_bytes(oracle.tobytes());(P/'study_cases.json').write_text(json.dumps({'cases':cases},indent=2)+'\n')
h=['struct Case{int m,k,n,nnz,ro,co,rp,vo,bo,oo;};','const Case cases[]={']
for c in cases:
 o=c['input_offsets'];h.append('{'+','.join(map(str,[c['m'],c['k'],c['n'],c['nnz'],o['rows'],o['columns'],o['row_offsets'],o['values'],o['B'],c['oracle_offset']]))+'},')
h+=['};'];(P/'study_cases.h').write_text('\n'.join(h)+'\n')
# Prefix includes actualsparsearrays/denseB +16controls percase, once for shared COO/CSR input.
prefix=len(input)+16*len(cases);output=2*len(oracle);guards=(5+2)*16*len(cases);meta={'NumPy':np.__version__,'seed':20261018,'cases':cases,'input_words':len(input),'prefix_count':prefix,'output_count':output,'guard_count':guards,'artifact_count':prefix+output+guards,'repeat':20,'atol':1e-4,'rtol':2e-5,'oracle':'IndependentCPUFloat64densematrixmultiplicationandsparse-rowexpansion alloutputs; declaredFP32mixedbudget. Signeddyadicprobeexactrepresentable; noalgorithm/treeISAclaim.','source_scope':'FirstcaseexactoriginalCOOinitializerfirstA/Bdraw; formatscontrolsameA/BforCOO/CSR ratherthanoriginalseparatefreshdrand48draws. BothsourceSpMMAPIformats/roworder/defaultalg/alpha1beta0preserved; originalR1000reduced20, secondarysignedrectangularcase.','fixtures':{n:{'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest(),'bytes':(P/n).stat().st_size}for n in ['input_u32.bin','oracle_f64.bin','study_cases.json','study_cases.h','source_A_f32.bin','source_B_f32.bin']}};(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print('spgemmfixture',meta['prefix_count'],meta['output_count'],meta['guard_count'],meta['artifact_count'])
