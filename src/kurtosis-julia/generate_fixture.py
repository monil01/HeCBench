from pathlib import Path
import struct,math,json
B=Path(__file__).resolve().parent
specs=[(1,'singleton'),(257,'symmetric'),(4099,'source_positive'),(4099,'outlier'),(513,'constant_zero'),(513,'constant_nonzero'),(1024,'signed')];data=bytearray();cases=[];oracle=[]
for n,kind in specs:
 values=[]
 for i in range(n):
  if kind=='singleton':v=1.5
  elif kind=='symmetric':v=0 if i==n-1 else (-1 if i%2 else 1)
  elif kind=='source_positive':v=1+((i*37)%1024)/1024
  elif kind=='outlier':v=4 if i==n-1 else 0
  elif kind=='constant_zero':v=0
  elif kind=='constant_nonzero':v=1.5
  else:v=((i*53)%2047-1023)/512
  values.append(v)
 offset=len(data)//24
 for i,v in enumerate(values):data.extend(struct.pack('<iiQfI',i,(i*7)%13,i*1000003,v,0))
 mean=math.fsum(values)/n;dev=[v-mean for v in values];sums={p:math.fsum(abs(v)**p for v in dev) for p in range(1,5)};m=[math.fsum(v**p for v in dev) for p in [2,3,4]];mb=1e-5+4e-6*max(1,max(abs(v) for v in values));bud=[0.,mb]+[3e-5*(1+sums[p])+p*mb*(n if p==1 else sums[p-1]) for p in [2,3,4]]
 cases.append(dict(n=n,kind=kind,input_record_offset=offset,expected=[n,mean,*m],absolute_budgets=bud,constant=kind in ['singleton','constant_zero','constant_nonzero'],pearson_kurtosis=(n*m[2]/m[0]**2 if m[0]>0 else None)))
 oracle.extend([n,mean,*m])
(B/'fixtures_aos24.bin').write_bytes(data);(B/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(oracle),*oracle))
(B/'fixture_description.json').write_text(json.dumps(dict(cases=cases,record_bytes=24,result_bytes=20,guard_records=16,guard_record=dict(tag=-777,metric=19,time=65261,value=-1234.5,padding=0),internal_repeats=10,input_prefix_count=sum(6*(c['n']+32) for c in cases),complete_capture_count=sum(6*(c['n']+32) for c in cases)+6*len(cases),operator='Count/mean/UNNORMALIZED central sums M2/M3/M4. Not standardized or excess kurtosis; derived Pearson coefficient separate/undefined onzero variance.',oracle='Independent Float64 fsum direct mean/central powers, not parallel merge recurrence; actualdyadicFloat32 values.',budget='Predeclared finite-case: count exact, mean delta=1e-5+4e-6 max(1,maxabsx); Mp absolute3e-5*(1+sumabsdev^p)+p*delta*sumabsdev^(p-1). Fixed Float32 rounding/recentering allowance, not formalall-inputproof. Constant/singletonmoments exactlyzero.',global_comparison_envelope='atol.05+rtol1e-4 is only genericcontainer envelope; strongerdeclaredfieldwise dual budgets/count/size/constant exact required in selfchecks andindependent captureaudit.',timing='Sum7 complete reduction-API hostwindows/R10: native unchanged Thrusttransform_reduce inclallocations/scalarD2H andlast hostnewresult; Julia authoredhierarchicalreduce incltempalloc/readback/free andhostresultRef; contextcompletion included. Inputsetup/JIT/copies/fullcapture/oracle outside. Notkernel-only; algorithms/allocator/runtime are covariates.',scope='Positive source-valued domain and additional signed/constant/outliercontrols, n1/257/4099/513/1024, not original100million CLI or trainedscientifictime-seriesprediction.'),indent=2)+'\n')
