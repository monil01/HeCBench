using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function histogram_gmem_julia!(input,parts,width::Int32,height::Int32,stride::Int32,::Val{CH}) where {CH}
 tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);gx=Int32(blockIdx().x)-Int32(1);gy=Int32(blockIdx().y)-Int32(1)
 tid=tx+ty*Int32(32);group=gx+gy*Int32(16);x=gx*Int32(32)+tx;y=gy*Int32(4)+ty;bins=Int32(CH*256)
 i=tid
 while i<bins;@inbounds parts[group*Int32(1024)+i+Int32(1)]=UInt32(0);i+=Int32(128);end
 sync_threads()
 col=x
 while col<width
  row=y
  while row<height
   pixel=(row*width+col)*stride
   for channel in Int32(0):Int32(CH-1)
    value=@inbounds input[pixel+channel+Int32(1)];bin=histogram_bin(value)
    CUDA.@atomic parts[group*Int32(1024)+channel*Int32(256)+bin+Int32(1)]+=UInt32(1)
   end
   row+=Int32(64)
  end
  col+=Int32(512)
 end
return
end
function histogram_smem_julia!(input,parts,width::Int32,height::Int32,stride::Int32,::Val{CH}) where {CH}
 tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1);gx=Int32(blockIdx().x)-Int32(1);gy=Int32(blockIdx().y)-Int32(1)
 tid=tx+ty*Int32(32);group=gx+gy*Int32(16);x=gx*Int32(32)+tx;y=gy*Int32(4)+ty;bins=Int32(CH*256)
 scratch=@cuDynamicSharedMem(UInt32,CH*256+3)
 i=tid
 while i<bins+Int32(3);@inbounds scratch[i+Int32(1)]=UInt32(0);i+=Int32(128);end
 sync_threads()
 col=x
 while col<width
  row=y
  while row<height
   pixel=(row*width+col)*stride
   for channel in Int32(0):Int32(CH-1)
    value=@inbounds input[pixel+channel+Int32(1)];bin=histogram_bin(value)
    CUDA.@atomic scratch[channel*Int32(256)+bin+channel+Int32(1)]+=UInt32(1)
   end
   row+=Int32(64)
  end
  col+=Int32(512)
 end
 sync_threads()
 i=tid
 while i<Int32(256)
  for channel in Int32(0):Int32(CH-1)
   @inbounds parts[group*Int32(1024)+i+channel*Int32(256)+Int32(1)]=scratch[i+channel*Int32(256)+channel+Int32(1)]
  end
  i+=Int32(128)
 end
return
end
function histogram_accum_julia!(parts,out,bins::Int32)
 index=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1)
 if index<bins
  total=UInt32(0)
  for group in Int32(0):Int32(255);@inbounds total+=parts[index+group*Int32(1024)+Int32(1)];end
  @inbounds out[index+Int32(1)]=total
 end
return
end
function factory(input,parts,out,width,height,channels,stride,method)
 tag=Val(channels);bins=channels*256
 return ()->begin
  if method=="smem"
   @cuda threads=(32,4) blocks=(16,16) shmem=(bins+3)*4 histogram_smem_julia!(input,parts,Int32(width),Int32(height),Int32(stride),tag)
  else
   @cuda threads=(32,4) blocks=(16,16) histogram_gmem_julia!(input,parts,Int32(width),Int32(height),Int32(stride),tag)
  end
  @cuda threads=128 blocks=div(bins,128) histogram_accum_julia!(parts,out,Int32(bins))
 end
end
if abspath(PROGRAM_FILE)==@__FILE__;histogram_study(factory);end
