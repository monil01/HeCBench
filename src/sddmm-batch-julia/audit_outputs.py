"""Independent outer-product replay of all sampled products and saved fields."""
from pathlib import Path
import json,struct,hashlib,sys
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];bundle=r/'.porting-state/portable_julia/runs'/run;d=json.loads((bundle/'result.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(f,n):
 b=Path(f).read_bytes();assert len(b)==8*n;return list(struct.unpack('<'+str(n)+'d',b))
pre=load(p/'fixture.bin',78637);gold=load(p/'oracle.bin',81442);pos=1;independent=[];cases=[]
for ci in range(5):
 b,m,k,n,nnz,repeat=map(int,pre[pos:pos+6]);pos+=6
 A=pre[pos:pos+b*m*k];pos+=len(A);B=pre[pos:pos+b*k*n];pos+=len(B);off=list(map(int,pre[pos:pos+b*(m+1)]));pos+=len(off);col=list(map(int,pre[pos:pos+b*nnz]));pos+=len(col);initial=pre[pos:pos+b*nnz];pos+=len(initial)
 expected=[];batchinputs=set();patterns=set()
 for batch in range(b):
  aa=A[batch*m*k:(batch+1)*m*k];bb=B[batch*k*n:(batch+1)*k*n];oo=off[batch*(m+1):(batch+1)*(m+1)];cc=col[batch*nnz:(batch+1)*nnz]
  assert oo[0]==0 and oo[-1]==nnz and oo==sorted(oo)
  for row in range(m):assert all(0<=j<n for j in cc[oo[row]:oo[row+1]]) and cc[oo[row]:oo[row+1]]==sorted(set(cc[oo[row]:oo[row+1]]))
  patterns.add(tuple(oo+cc));batchinputs.add(hashlib.sha256(struct.pack('<'+str(len(aa)+len(bb))+'f',*(aa+bb))).hexdigest())
  # Different loop organization from fixture's sampled inner products: construct
  # the whole integer-scaled dense product as a sum of rank-one outer products.
  dense=[[0]*n for _ in range(m)]
  for h in range(k):
   for row in range(m):
    a=int(aa[row*k+h]*4);assert a==aa[row*k+h]*4
    for j in range(n):
     bv=int(bb[h*n+j]*8);assert bv==bb[h*n+j]*8;dense[row][j]+=a*bv
  for row in range(m):
   for j in cc[oo[row]:oo[row+1]]:expected.append(dense[row][j]/32.)
 assert len(patterns)==1 and len(batchinputs)==b
 vals=expected+A+B+off+col+initial+[16384]*80;independent.extend(vals)
 cases.append(dict(case=ci+1,batches=b,sampled_values=len(expected),distinct_dense_input_pairs=len(batchinputs),distinct_CSR_patterns=len(patterns),all_values_exact=True,all_CSR_rows_columns_checked=True,guard_words=80))
assert pos==len(pre) and independent==gold
rows=[];hashes=set()
for model,v in d['models'].items():
 assert v['complete'] and len(v['trials'])==5
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);a=load(f,160079);assert a[:78637]==pre and a[78637:]==independent and t['correct'];h=sha(f);hashes.add(h);rows.append(dict(model=model,trial=t['trial'],path=str(f.relative_to(r)),sha256=h,bytes=f.stat().st_size,all_sampled_outputs_and_input_bits_and_guards_exact=True))
out=dict(benchmark='sddmm-batch',configuration_id=d['configuration_id'],all_pass=True,all20_bitwise_identical=len(hashes)==1,artifact_values=160079,input_prefix_values=78637,sampled_values_per_trial=sum(x['sampled_values'] for x in cases),guard_words_per_trial=400,independent_oracle='Fullinteger-scaled dense outer-product accumulation, then independently traversed sorteduniqueCSR samples, differentorganizationfromfixtureinnerdot; allboundedbinaryrationalvaluesexactF32.',source_pattern_scope='OneCSRstructurepercase repeatedacrossbatches, distinctdensepairsall42batches. OriginalsourceinitializerCPUwitness andfailedvariedoffsetvendorcontrols separatelyretained.',authored_device_kernel_entrypoints=0,actual_vendor_GPU_kernel_count=None,actual_API_calls_per_trial=25,cases=cases,rows=rows,result_sha256=sha(bundle/'result.json'),config_sha256=sha(bundle/'config.json'))
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ['cases','rows']})
