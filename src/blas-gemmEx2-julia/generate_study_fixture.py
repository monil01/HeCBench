from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent;pre=[27];gold=[];kind=bytearray();paths=[]
for shape,(m,n,k)in enumerate([(16,16,16),(8,12,16),(32,48,64)]):
 for dtype in range(9):
  mode={1:0,2:1,3:2,4:3}.get(dtype,0);scale=1 if dtype==8 else 4;A=[((2*i+3*q+shape)%5-2)/scale for i in range(m)for q in range(k)];B=[((4*q+2*j+2*shape)%5-2)/scale for q in range(k)for j in range(n)];C=[((i*3+j*2)%9-4)if dtype==8 else((i*3+j*2)%9-4)/8 for i in range(m)for j in range(n)];out=[]
  for i in range(m):
   for j in range(n):
    numerator=sum(round(A[i*k+q]*scale)*round(B[q*n+j]*scale)for q in range(k));assert abs(numerator)<=4*k<=256;out.append(numerator/(scale*scale))
  pre.extend([dtype,m,n,k,mode]);pre.extend(A);pre.extend(B);pre.extend(C)
  inbytes=8 if dtype==0 else 4 if dtype in range(1,5)else 2 if dtype in [5,6,7]else 1;outbytes=4 if dtype==8 else inbytes;leadA=256//inbytes;leadC=256//outbytes;gA=73 if dtype==8 else 16384;gC=16384
  for v,lead,g in [(C,leadC,gC),(out,leadC,gC),(A,leadA,gA),(B,leadA,gA)]:
   gold.extend([g]*lead+v+[g]*lead);kind.extend(bytes([1])*lead);kind.extend(bytes([0 if v is out else 1])*len(v));kind.extend(bytes([1])*lead)
  paths.append(dict(dtype=dtype,m=m,n=n,k=k,mode_flag=mode,actual_FP32_compute_for_all_flags=68 if dtype in range(1,5)else None,input_element_bytes=inbytes,output_element_bytes=outbytes,input_guard=gA,input_guard_elements=leadA,output_guard_elements=leadC,precision='BF16/F16/F32/F64dyadicintegermantissabound<=256units1/16, orderexactforallintermediates; int8exactInt32bound256',periodic_input_pattern=5))
gold.extend([165]*512);kind.extend(bytes([1])*512)
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'oracle.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold));(p/'kinds.bin').write_bytes(kind);(p/'fixture_description.json').write_text(json.dumps(dict(paths=paths,input_prefix_count=len(pre),oracle_count=len(gold),artifact_count=len(pre)+len(gold),oracle='Independentintegerdotproducts withboundedquarterinputs,exactallreductionordersinFP16/BF16/FP32/FP64;allfloatingstoragevaluesandproductsrepresentable, computeinternalprecisionnotinferred. IntegerpathInt32nooverflow.',all_FP32_source_modes_fallthrough_to_compute32=True,source_alpha_beta=[1,0],native_pointer_order_corrected=True,workspace_bytes=33554432,workspace_guard_bytes=512,managed_allocation='SourcecudaMallocManaged preserved;allpayload256Baligned,explicitprefetchGPUoutsideROI afterresets/warmup. InitialCnonzero butbeta0 nohiddenaccumulation.',scope='Smallperiod5boundedmatrices,notarbitrarylowprecisionrounding/overflow/largeGEMMthroughput.'),indent=2)+'\n');print(len(pre),len(gold))
