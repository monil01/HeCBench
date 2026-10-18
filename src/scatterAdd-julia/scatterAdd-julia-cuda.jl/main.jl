using CUDA
function scatter_kernel!(indexes,source,outputs,n::Int32,o::Int32,v::Int32,::Val{MODE}) where MODE
 tid=threadIdx().x-Int32(1);lane=tid%Int32(32);warp=tid÷Int32(32);base=(blockIdx().x-Int32(1))*Int32(8192);nout=o*v
 outbuf=CuDynamicSharedArray(Float32,nout);idxbuf=CuDynamicSharedArray(Int32,512,sizeof(Float32)*Int(nout))
 pos=tid
 while pos<nout;@inbounds outbuf[pos+Int32(1)]=0f0;pos+=Int32(512);end
 if MODE==1;sync_threads();end
 start=Int32(0)
 while start<Int32(8192)
  bi=base+start+tid
  if bi<n;@inbounds idxbuf[tid+Int32(1)]=indexes[bi+Int32(1)];end
  sync_threads();bs=Int32(0)
  while bs<Int32(512)
   slot=bs+warp;bi=base+start+slot
   if bi<n
    index=@inbounds idxbuf[slot+Int32(1)];j=lane
    while j<v
     value=@inbounds source[bi*v+j+Int32(1)];CUDA.@atomic outbuf[index*v+j+Int32(1)]+=value;j+=Int32(32)
    end
   end
   bs+=Int32(16)
  end
  sync_threads();start+=Int32(512)
 end
 pos=tid
 while pos<nout
  value=@inbounds outbuf[pos+Int32(1)];CUDA.@atomic outputs[pos+Int32(1)]+=value;pos+=Int32(512)
 end
 return
end
