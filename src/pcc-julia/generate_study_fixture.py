"""Source C++ host normalization plus independent high-precision Pearson oracle."""
from pathlib import Path
import ctypes,decimal,struct,json,hashlib
p=Path(__file__).resolve().parent;D=decimal.Decimal;decimal.getcontext().prec=80;R=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];lib=p.parent.parent/'.porting-state/porting_logs/pcc-julia-native/normalization_native.so';dll=ctypes.CDLL(str(lib));dll.study_preprocessing.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int]
pre=[5];gold=[];cases=[];maxnormerror=0.
for ci,(N,L,avail) in enumerate([(9,17,0),(9,17,54),(17,33,160),(5,1,0),(7,64,0)]):
 raw=[]
 for i in range(N):
  base=[R(((j*7)%19-9)/8) for j in range(L)]
  row=([2.]*L if i%9==0 else base if i%9 in [1,2] else [-x for x in base] if i%9==3 else [R(x+10) for x in base] if i%9==4 else [R((j%7-3)/4) for j in range(L)] if i%9==5 else [R((j-L//2)/8) for j in range(L)] if i%9==6 else [(-1. if j%2 else 1.) for j in range(L)] if i%9==7 else [R(((j*j+i*3)%23-11)/8) for j in range(L)])
  raw+=row
 arr=(ctypes.c_float*(N*L))(*raw);dll.study_preprocessing(arr,N,L);norm=list(arr);high=[];zero=[]
 for i in range(N):
  x=[D.from_float(a) for a in raw[i*L:(i+1)*L]];mean=sum(x)/D(L);diff=[a-mean for a in x];ss=sum(a*a for a in diff);den=ss.sqrt();z=ss==0;zero.append(z);h=[D(0)]*L if z else [a/den for a in diff];high.append(h)
  maxnormerror=max(maxnormerror,max(abs(float(a)-b) for a,b in zip(h,norm[i*L:(i+1)*L])))
 def pearson(i,j):return float(sum(a*b for a,b in zip(high[i],high[j])))
 pre +=[N,L,avail]+raw+norm;chunks=[];off=0
 while off<N:
  n=N-off;n1=n if avail==0 else min(n,avail//(2*n));assert n1>0;C=[pearson(off+i,off+j) for j in range(n) for i in range(n1)];U=[pearson(off+i,off+j) for i in range(n1) for j in range(i+1,n)];assert len(U)==n1*n-n1*(n1+1)//2
  gold +=[-12345.]*64+C+[-12345.]*8+[-12345.]*64+U+[-12345.]*8;chunks.append(dict(offset=off,Nprime=n,rows=n1,cormat_values=len(C),upper_values=len(U),singlepass=avail==0));off+=n1
 gold +=[-12345.]*64+norm+[-12345.]*8
 cases.append(dict(case=ci,N=N,L=L,available_float_slots=avail,chunks=chunks,constant_rows=[i for i,z in enumerate(zero) if z],policy='Zero-variance time series normalizedzero/correlationszero, not mathematicallydefinedPearsoncoefficient forconstantrows.',identical_and_anticorrelated_controls=N>=4 and L>1))
assert maxnormerror<1e-7
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(a),*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),atol=1e-6,rtol=3e-6,normalization_source=str(p/'normalization_native.cpp'),normalization_shared_library=dict(path=str(lib),sha256=hashlib.sha256(lib.read_bytes()).hexdigest()),source_normalized_F32_vs_Decimal80_max_error=maxnormerror,oracle='IndependentDecimal80rawtime-series centeredPearsoncoefficients, constant-rowzeroconventionexplicit. ActualGPUinputs sourceC++Float64normalization thenF32storagecommonfixture, allscalar/raw/normedinputsretained.',timing='Eachchunk SGEMM+completion thenker/ker2+completion exactlymatching sourceGPUregions. CPUpreprocessing/handlecreation/alloc/warmup/reset/readbacks outside; perchunkstagetimes retained. Namedforcedmemorybudgetsnotdefault4GBCLI.',library='All4calltypedcublasSgemm_v2 withowneddefaultMathhandle, actualprocesslibrarypath/versioncheckedoutsideROI. KA/JACC authoronlyextractionkernels, not vendorGEMM.'),indent=2)+'\n');print(len(pre),len(gold),maxnormerror)
