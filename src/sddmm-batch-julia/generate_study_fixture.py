"""Independent exact integer-scaled sampled dense matrix products."""
from pathlib import Path
import struct,json,math
p=Path(__file__).resolve().parent;pre=[5.];gold=[];cases=[]
for ci,(b,m,k,n,nnz) in enumerate([(1,1,1,1,1),(3,17,11,19,37),(4,33,17,35,67),(32,32,33,31,64),(2,1,7,9,4)]):
    A=[];B=[];off=[];col=[];initial=[];values=[];maxsum=0;empty=0
    for batch in range(b):
        a=[((i*5+j*3+batch*7+ci)%17)-8 for i in range(m) for j in range(k)]
        bb=[((i*7+j*11+batch*3+2*ci)%19)-9 for i in range(k) for j in range(n)]
        step=3
        while math.gcd(step,m*n)!=1:step+=1
        selected=sorted(((q*step+ci*7)%(m*n)) for q in range(nnz));assert len(set(selected))==nnz
        offsets=[0];cols=[]
        for i in range(m):
            row=[j%n for j in selected if j//n==i];empty+=not row;cols.extend(row);offsets.append(len(cols))
            for j in row:
                terms=[a[i*k+h]*bb[h*n+j] for h in range(k)]
                scaled=sum(terms);maxsum=max(maxsum,sum(map(abs,terms)));values.append(scaled/32.)
        assert len(cols)==nnz and offsets[-1]==nnz
        A.extend(x/4. for x in a);B.extend(x/8. for x in bb);off.extend(offsets);col.extend(cols);initial.extend(100+batch+j/4. for j in range(nnz))
    assert maxsum<2**24
    pre.extend([b,m,k,n,nnz,5]);pre.extend(A+B+off+col+initial)
    # Every actual input remains read-only, including sparse structure and dense
    # strides; output C overwrites source initial values under alpha1,beta0.
    out=values+A+B+off+col+initial+[16384]*80
    gold.extend(out)
    cases.append(dict(case=ci+1,batches=b,m=m,k=k,n=n,nnz_per_batch=nnz,repeat=5,A_values=len(A),B_values=len(B),offsets=len(off),columns=len(col),C_values=len(values),empty_C_rows=empty,maximum_absolute_sum_units_1_over32=maxsum,oracle_values=len(out)))
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]: (p/name).write_bytes(struct.pack('<'+str(len(a))+'d',*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_values=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),independent_oracle='Integer-scaled row/column dot product at every sampled CSR entry: A quarters, B eighths, exactproducts/partialsums inunits1/32 with absolute product sum<2^24, order-independent Float32 exactness. Distinct dense content perbatch, repeated CSR pattern matching original source initializer policy, empty rows, non-square/odd shapes, batch32retained.',operator='cusparseSDDMM rowmajor A(M,K)/B(K,N), fullperbatchzero-basedsorteduniqueCSR C(M,N), alpha1beta0/F32/default/NONTRANSPOSE; values overwritten, offsets/columns immutable.',unverified='Original1024/4096dense sizes, originalrandom inputs/1000repeats, otherdtypes/operations/algorithms/broadcastbatchstrides/CPUperformance/portabledevicekernelimplementation.'),indent=2)+'\n')
print(len(pre),len(gold),len(pre)+len(gold))
