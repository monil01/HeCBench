from pathlib import Path
import struct,json,hashlib,difflib
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;w=json.loads((p/'reduction_witness.json').read_text());f=Path(w['artifact']);raw=f.read_bytes();assert len(raw)==4*3078;v=list(struct.unpack('<3078f',raw));N=513;a=[float(i%31+1) for i in range(N)];b=[x*x for x in a];assert v[:N]==a and v[N:2*N]==b;stage=0;no=N;mul=1;blocks=(N+255)//256;pos=2*N;stages=[]
while blocks:
 aa=list(a);bb=list(b);nf=256-(blocks*256-no)
 for block in range(blocks):
  busy=min(256,no-block*256);pa=[aa[(block*256+i)*mul] for i in range(busy)];pb=[bb[(block*256+i)*mul] for i in range(busy)]
  if block!=blocks-1 or nf==256:df=256
  else:df=0 if nf==1 else 2**(nf.bit_length()-1)
  span=2
  while span<=df:
   for tx in range(span-1,df,span):pa[tx]+=pa[tx-span//2];pb[tx]+=pb[tx-span//2]
   span*=2
  if df:
   for i in range(block*256+df,block*256+busy):pa[df-1]+=aa[i];pb[df-1]+=bb[i] # Original bug, deliberately not *mul.
   a[block*mul*256]=pa[df-1];b[block*mul*256]=pb[df-1]
 actuala=v[pos:pos+N];actualb=v[pos+N:pos+2*N];pos+=2*N;assert actuala==a and actualb==b
 stages.append(dict(stage=stage+1,no=no,mul=mul,blocks=blocks,actual_sum=a[0],actual_sum2=b[0],full_arrays_exactly_match_original_wrong_index_emulation=True));stage+=1;no=blocks
 if blocks==1:blocks=0
 else:mul*=256;blocks=(blocks+255)//256
assert pos==len(v) and a[0]==8075 and b[0]==168161 and sum(v[:N])==8089 and sum(v[N:2*N])==168441
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();s=(r/'src/srad-cuda/reduce_kernel.cu').read_text();corrected=s.replace('d_sums[i];','d_sums[i*d_mul];').replace('d_sums2[i];','d_sums2[i*d_mul];');assert corrected!=s and s.count('d_sums[i];')==1 and s.count('d_sums2[i];')==1
(p/'corrected_reduce_kernel.cuh').write_text(corrected);(p/'native_correction.patch').write_text(''.join(difflib.unified_diff(s.splitlines(True),corrected.splitlines(True),fromfile='original_reduce_kernel',tofile='corrected_reduce_kernel')))
out=dict(original_GPU_kernel_unchanged=True,full_capture=dict(path=str(f.relative_to(r)),sha256=sha(f),bytes=len(raw),dtype='Float32',count=3078),stages=stages,exact_mathematical_sum=8089,actual_original_sum=8075,exact_mathematical_sum_squared=168441,actual_original_sum_squared=168161,root_cause='Lastpartialblockremainder reads physicalindexi insteadof i*d_mul. N513secondlevelno3/mul256 readsinputpixel2(value3) insteadofpartial512(value17), sumerror−14/squaredsum−280. Allvalues/partialsumsexactFloat32integers, notroundoff.',source_cli_relevance='N502x458 has899firstpartials; laternonpower-of-two partials/mul256 exercisebadread, butfulloriginalCLIoutputnotcaptured bythisminimalwitness.',correction='Two globalremainder reads only multiplyindexbyd_mul; allotherreductiontopology/loops/nf1inplaceidentity retained.',original_source=dict(path='src/srad-cuda/reduce_kernel.cu',sha256=sha(r/'src/srad-cuda/reduce_kernel.cu')),corrected_source=dict(path='src/srad-julia/corrected_reduce_kernel.cuh',sha256=sha(p/'corrected_reduce_kernel.cuh')),patch=dict(path='src/srad-julia/native_correction.patch',sha256=sha(p/'native_correction.patch')),framework_unsupported_claim=False)
(p/'reduction_witness_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(out['root_cause'])
