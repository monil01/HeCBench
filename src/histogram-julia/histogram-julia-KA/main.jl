using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function histogram_gmem_ka!(input,parts,width::Int32,height::Int32,stride::Int32,::Val{CH}) where {CH}
 tx=Int32(@index(Local,NTuple)[1])-Int32(1);ty=Int32(@index(Local,NTuple)[2])-Int32(1);gx=Int32(@index(Group,NTuple)[1])-Int32(1);gy=Int32(@index(Group,NTuple)[2])-Int32(1)
 tid=tx+ty*Int32(32);group=gx+gy*Int32(16);x=gx*Int32(32)+tx;y=gy*Int32(4)+ty;bins=Int32(CH*256)
 i=tid
 while i<bins;@inbounds parts[group*Int32(1024)+i+Int32(1)]=UInt32(0);i+=Int32(128);end
 @synchronize
 col=x
 while col<width
  row=y
  while row<height
   pixel=(row*width+col)*stride
   for channel in Int32(0):Int32(CH-1)
    value=@inbounds input[pixel+channel+Int32(1)];bin=histogram_bin(value)
    Atomix.@atomic parts[group*Int32(1024)+channel*Int32(256)+bin+Int32(1)]+=UInt32(1)
   end
   row+=Int32(64)
  end
  col+=Int32(512)
 end
end
@kernel function histogram_smem_ka!(input,parts,width::Int32,height::Int32,stride::Int32,::Val{CH}) where {CH}
 tx=Int32(@index(Local,NTuple)[1])-Int32(1);ty=Int32(@index(Local,NTuple)[2])-Int32(1);gx=Int32(@index(Group,NTuple)[1])-Int32(1);gy=Int32(@index(Group,NTuple)[2])-Int32(1)
 tid=tx+ty*Int32(32);group=gx+gy*Int32(16);x=gx*Int32(32)+tx;y=gy*Int32(4)+ty;bins=Int32(CH*256)
 scratch=@localmem UInt32 (CH*256+3,)
 i=tid
 while i<bins+Int32(3);@inbounds scratch[i+Int32(1)]=UInt32(0);i+=Int32(128);end
 @synchronize
 col=x
 while col<width
  row=y
  while row<height
   pixel=(row*width+col)*stride
   for channel in Int32(0):Int32(CH-1)
    value=@inbounds input[pixel+channel+Int32(1)];bin=histogram_bin(value)
    Atomix.@atomic scratch[channel*Int32(256)+bin+channel+Int32(1)]+=UInt32(1)
   end
   row+=Int32(64)
  end
  col+=Int32(512)
 end
 @synchronize
 i=tid
 while i<Int32(256)
  for channel in Int32(0):Int32(CH-1)
   @inbounds parts[group*Int32(1024)+i+channel*Int32(256)+Int32(1)]=scratch[i+channel*Int32(256)+channel+Int32(1)]
  end
  i+=Int32(128)
 end
end
@kernel function histogram_accum_ka!(parts,out,bins::Int32)
 index=Int32(@index(Global,Linear))-Int32(1)
 if index<bins
  total=UInt32(0)
  for group in Int32(0):Int32(255);@inbounds total+=parts[index+group*Int32(1024)+Int32(1)];end
  @inbounds out[index+Int32(1)]=total
 end
end
function factory(input,parts,out,width,height,channels,stride,method)
 tag=Val(channels);bins=channels*256
 firstpass=(method=="smem" ? histogram_smem_ka! : histogram_gmem_ka!)(get_backend(input),(32,4))
 accum=histogram_accum_ka!(get_backend(input),128)
 return ()->begin
  firstpass(input,parts,Int32(width),Int32(height),Int32(stride),tag;ndrange=(512,64))
  accum(parts,out,Int32(bins);ndrange=bins)
 end
end
histogram_study(factory)
