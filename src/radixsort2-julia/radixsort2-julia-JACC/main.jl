using CUDA,JACC
JACC.@init_backend
include("../study_common.jl")
function local_rank_kernel(index,keys,localzero,blockzero,N,bit,template)
 tid=Int32((index-1)%1024);i=Int32(index)-Int32(1);scratch=JACC.shared(template);flag=UInt32(0)
 @inbounds if i<N;flag=UInt32(((ordered_bits(keys[i+Int32(1)])>>bit)&UInt32(1))==0);end
 scratch[tid+Int32(1)]=flag;JACC.sync_workgroup();stride=Int32(1)
 while stride<Int32(1024)
  addition=tid>=stride ? scratch[tid-stride+Int32(1)] : UInt32(0);JACC.sync_workgroup();scratch[tid+Int32(1)]+=addition;JACC.sync_workgroup();stride<<=Int32(1)
 end
 @inbounds if i<N;localzero[i+Int32(1)]=scratch[tid+Int32(1)]-flag;end
 @inbounds if tid==Int32(1023);blockzero[Int32((index-1)÷1024)+Int32(1)]=scratch[1024];end

end
function block_scan_kernel(index,blockzero,blockoffset,totalzero,template)
 tid=Int32((index-1)%1024);scratch=JACC.shared(template);@inbounds value=blockzero[tid+Int32(1)];scratch[tid+Int32(1)]=value;JACC.sync_workgroup();stride=Int32(1)
 while stride<Int32(1024)
  addition=tid>=stride ? scratch[tid-stride+Int32(1)] : UInt32(0);JACC.sync_workgroup();scratch[tid+Int32(1)]+=addition;JACC.sync_workgroup();stride<<=Int32(1)
 end
 @inbounds blockoffset[tid+Int32(1)]=scratch[tid+Int32(1)]-value
 @inbounds if tid==Int32(1023);totalzero[1]=scratch[1024];end

end
function scatter_keys_kernel(index,src,dst,localzero,blockoffset,totalzero,N,bit)
 i=Int32(index)-Int32(1)
 @inbounds if i<N;key=src[i+Int32(1)];j=radix_destination(i,key,localzero,blockoffset,totalzero,bit);dst[Int(j)+1]=key;end

end
function scatter_pairs_kernel(index,src,dst,values,outvalues,localzero,blockoffset,totalzero,N,bit)
 i=Int32(index)-Int32(1)
 @inbounds if i<N;key=src[i+Int32(1)];j=radix_destination(i,key,localzero,blockoffset,totalzero,bit);dst[Int(j)+1]=key;outvalues[Int(j)+1]=values[i+Int32(1)];end

end
function factory(keys,values,N,only)
 temp=similar(keys);tempvalues=similar(values);localzero=CUDA.zeros(UInt32,N);blockzero=CUDA.zeros(UInt32,1024);blockoffset=similar(blockzero);totalzero=CUDA.zeros(UInt32,1);template=CUDA.zeros(UInt32,1024);rankSpec=JACC.launch_spec(;threads=1024,blocks=1024,shmem_size=4096,sync=false);scanSpec=JACC.launch_spec(;threads=1024,blocks=1,shmem_size=4096,sync=false);scatterSpec=JACC.launch_spec(;threads=1024,blocks=1024,shmem_size=0,sync=false)
 ()->begin
  src=keys;dst=temp;vals=values;outvals=tempvalues
  for bit in Int32(0):Int32(31)
   JACC.parallel_for(rankSpec,1048576,local_rank_kernel,src,localzero,blockzero,N,bit,template)
   JACC.parallel_for(scanSpec,1024,block_scan_kernel,blockzero,blockoffset,totalzero,template)
   if only;JACC.parallel_for(scatterSpec,1048576,scatter_keys_kernel,src,dst,localzero,blockoffset,totalzero,N,bit)
   else;JACC.parallel_for(scatterSpec,1048576,scatter_pairs_kernel,src,dst,vals,outvals,localzero,blockoffset,totalzero,N,bit);end
   src,dst=dst,src;vals,outvals=outvals,vals
  end
 end
end
main()=radixsort_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
