using CUDA
include("../study_common.jl")
function local_rank_kernel(keys,localzero,blockzero,N,bit)
 tid=threadIdx().x-Int32(1);i=(blockIdx().x-Int32(1))*Int32(1024)+tid;scratch=CuStaticSharedArray(UInt32,1024);flag=UInt32(0)
 @inbounds if i<N;flag=UInt32(((ordered_bits(keys[i+Int32(1)])>>bit)&UInt32(1))==0);end
 scratch[tid+Int32(1)]=flag;sync_threads();stride=Int32(1)
 while stride<Int32(1024)
  addition=tid>=stride ? scratch[tid-stride+Int32(1)] : UInt32(0);sync_threads();scratch[tid+Int32(1)]+=addition;sync_threads();stride<<=Int32(1)
 end
 @inbounds if i<N;localzero[i+Int32(1)]=scratch[tid+Int32(1)]-flag;end
 @inbounds if tid==Int32(1023);blockzero[blockIdx().x]=scratch[1024];end
 return
end
function block_scan_kernel(blockzero,blockoffset,totalzero)
 tid=threadIdx().x-Int32(1);scratch=CuStaticSharedArray(UInt32,1024);@inbounds value=blockzero[tid+Int32(1)];scratch[tid+Int32(1)]=value;sync_threads();stride=Int32(1)
 while stride<Int32(1024)
  addition=tid>=stride ? scratch[tid-stride+Int32(1)] : UInt32(0);sync_threads();scratch[tid+Int32(1)]+=addition;sync_threads();stride<<=Int32(1)
 end
 @inbounds blockoffset[tid+Int32(1)]=scratch[tid+Int32(1)]-value
 @inbounds if tid==Int32(1023);totalzero[1]=scratch[1024];end
 return
end
function scatter_keys_kernel(src,dst,localzero,blockoffset,totalzero,N,bit)
 i=(blockIdx().x-Int32(1))*Int32(1024)+threadIdx().x-Int32(1)
 @inbounds if i<N;key=src[i+Int32(1)];j=radix_destination(i,key,localzero,blockoffset,totalzero,bit);dst[Int(j)+1]=key;end
 return
end
function scatter_pairs_kernel(src,dst,values,outvalues,localzero,blockoffset,totalzero,N,bit)
 i=(blockIdx().x-Int32(1))*Int32(1024)+threadIdx().x-Int32(1)
 @inbounds if i<N;key=src[i+Int32(1)];j=radix_destination(i,key,localzero,blockoffset,totalzero,bit);dst[Int(j)+1]=key;outvalues[Int(j)+1]=values[i+Int32(1)];end
 return
end
function factory(keys,values,N,only)
 temp=similar(keys);tempvalues=similar(values);localzero=CUDA.zeros(UInt32,N);blockzero=CUDA.zeros(UInt32,1024);blockoffset=similar(blockzero);totalzero=CUDA.zeros(UInt32,1)
 ()->begin
  src=keys;dst=temp;vals=values;outvals=tempvalues
  for bit in Int32(0):Int32(31)
   @cuda threads=1024 blocks=1024 local_rank_kernel(src,localzero,blockzero,N,bit)
   @cuda threads=1024 blocks=1 block_scan_kernel(blockzero,blockoffset,totalzero)
   if only;@cuda threads=1024 blocks=1024 scatter_keys_kernel(src,dst,localzero,blockoffset,totalzero,N,bit)
   else;@cuda threads=1024 blocks=1024 scatter_pairs_kernel(src,dst,vals,outvals,localzero,blockoffset,totalzero,N,bit);end
   src,dst=dst,src;vals,outvals=outvals,vals
  end
 end
end
main()=radixsort_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
