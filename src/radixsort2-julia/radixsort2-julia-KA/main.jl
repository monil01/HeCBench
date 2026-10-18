using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function local_rank_kernel(keys,localzero,blockzero,N,bit)
 tid=Int32(@index(Local,Linear))-Int32(1);i=Int32(@index(Global,Linear))-Int32(1);scratch=@localmem UInt32 (1024,);flag=UInt32(0)
 @inbounds if i<N;flag=UInt32(((ordered_bits(keys[i+Int32(1)])>>bit)&UInt32(1))==0);end
 scratch[tid+Int32(1)]=flag;@synchronize;stride=Int32(1)
 while stride<Int32(1024)
  addition=tid>=stride ? scratch[tid-stride+Int32(1)] : UInt32(0);@synchronize;scratch[tid+Int32(1)]+=addition;@synchronize;stride<<=Int32(1)
 end
 @inbounds if i<N;localzero[i+Int32(1)]=scratch[tid+Int32(1)]-flag;end
 @inbounds if tid==Int32(1023);blockzero[Int32(@index(Group,Linear))]=scratch[1024];end

end
@kernel function block_scan_kernel(blockzero,blockoffset,totalzero)
 tid=Int32(@index(Local,Linear))-Int32(1);scratch=@localmem UInt32 (1024,);@inbounds value=blockzero[tid+Int32(1)];scratch[tid+Int32(1)]=value;@synchronize;stride=Int32(1)
 while stride<Int32(1024)
  addition=tid>=stride ? scratch[tid-stride+Int32(1)] : UInt32(0);@synchronize;scratch[tid+Int32(1)]+=addition;@synchronize;stride<<=Int32(1)
 end
 @inbounds blockoffset[tid+Int32(1)]=scratch[tid+Int32(1)]-value
 @inbounds if tid==Int32(1023);totalzero[1]=scratch[1024];end

end
@kernel function scatter_keys_kernel(src,dst,localzero,blockoffset,totalzero,N,bit)
 i=Int32(@index(Global,Linear))-Int32(1)
 @inbounds if i<N;key=src[i+Int32(1)];j=radix_destination(i,key,localzero,blockoffset,totalzero,bit);dst[Int(j)+1]=key;end

end
@kernel function scatter_pairs_kernel(src,dst,values,outvalues,localzero,blockoffset,totalzero,N,bit)
 i=Int32(@index(Global,Linear))-Int32(1)
 @inbounds if i<N;key=src[i+Int32(1)];j=radix_destination(i,key,localzero,blockoffset,totalzero,bit);dst[Int(j)+1]=key;outvalues[Int(j)+1]=values[i+Int32(1)];end

end
function factory(keys,values,N,only)
 temp=similar(keys);tempvalues=similar(values);localzero=CUDA.zeros(UInt32,N);blockzero=CUDA.zeros(UInt32,1024);blockoffset=similar(blockzero);totalzero=CUDA.zeros(UInt32,1);backend=CUDABackend();ranklaunch=local_rank_kernel(backend,1024);scanlaunch=block_scan_kernel(backend,1024);keylaunch=scatter_keys_kernel(backend,1024);pairlaunch=scatter_pairs_kernel(backend,1024)
 ()->begin
  src=keys;dst=temp;vals=values;outvals=tempvalues
  for bit in Int32(0):Int32(31)
   ranklaunch(src,localzero,blockzero,N,bit;ndrange=1048576)
   scanlaunch(blockzero,blockoffset,totalzero;ndrange=1024)
   if only;keylaunch(src,dst,localzero,blockoffset,totalzero,N,bit;ndrange=1048576)
   else;pairlaunch(src,dst,vals,outvals,localzero,blockoffset,totalzero,N,bit;ndrange=1048576);end
   src,dst=dst,src;vals,outvals=outvals,vals
  end
 end
end
main()=radixsort_study(factory)
if abspath(PROGRAM_FILE)==@__FILE__;main();end
