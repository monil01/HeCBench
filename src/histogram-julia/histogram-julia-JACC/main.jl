using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function histogram_gmem_jacc!(ix,iy,input,parts,width::Int32,height::Int32,stride::Int32,template,::Val{CH}) where {CH}
 tx=Int32(mod(ix-1,32));ty=Int32(mod(iy-1,4));gx=Int32(div(ix-1,32));gy=Int32(div(iy-1,4))
 tid=tx+ty*Int32(32);group=gx+gy*Int32(16);x=gx*Int32(32)+tx;y=gy*Int32(4)+ty;bins=Int32(CH*256)
 i=tid
 while i<bins;@inbounds parts[group*Int32(1024)+i+Int32(1)]=UInt32(0);i+=Int32(128);end
 JACC.sync_workgroup()
 col=x
 while col<width
  row=y
  while row<height
   pixel=(row*width+col)*stride
   for channel in Int32(0):Int32(CH-1)
    value=@inbounds input[pixel+channel+Int32(1)];bin=histogram_bin(value)
    JACC.@atomic parts[group*Int32(1024)+channel*Int32(256)+bin+Int32(1)]+=UInt32(1)
   end
   row+=Int32(64)
  end
  col+=Int32(512)
 end
return
end
function histogram_smem_jacc!(ix,iy,input,parts,width::Int32,height::Int32,stride::Int32,template,::Val{CH}) where {CH}
 tx=Int32(mod(ix-1,32));ty=Int32(mod(iy-1,4));gx=Int32(div(ix-1,32));gy=Int32(div(iy-1,4))
 tid=tx+ty*Int32(32);group=gx+gy*Int32(16);x=gx*Int32(32)+tx;y=gy*Int32(4)+ty;bins=Int32(CH*256)
 scratch=JACC.shared(template)
 i=tid
 while i<bins+Int32(3);@inbounds scratch[i+Int32(1)]=UInt32(0);i+=Int32(128);end
 JACC.sync_workgroup()
 col=x
 while col<width
  row=y
  while row<height
   pixel=(row*width+col)*stride
   for channel in Int32(0):Int32(CH-1)
    value=@inbounds input[pixel+channel+Int32(1)];bin=histogram_bin(value)
    JACC.@atomic scratch[channel*Int32(256)+bin+channel+Int32(1)]+=UInt32(1)
   end
   row+=Int32(64)
  end
  col+=Int32(512)
 end
 JACC.sync_workgroup()
 i=tid
 while i<Int32(256)
  for channel in Int32(0):Int32(CH-1)
   @inbounds parts[group*Int32(1024)+i+channel*Int32(256)+Int32(1)]=scratch[i+channel*Int32(256)+channel+Int32(1)]
  end
  i+=Int32(128)
 end
return
end
function histogram_accum_jacc!(ix,parts,out,bins::Int32)
 index=Int32(ix)-Int32(1)
 if index<bins
  total=UInt32(0)
  for group in Int32(0):Int32(255);@inbounds total+=parts[index+group*Int32(1024)+Int32(1)];end
  @inbounds out[index+Int32(1)]=total
 end
return
end
function factory(input,parts,out,width,height,channels,stride,method)
 tag=Val(channels);bins=channels*256
 template=CUDA.zeros(UInt32,bins+3)
 firstspec=JACC.launch_spec(;threads=(32,4),blocks=(16,16),shmem_size=method=="smem" ? (bins+3)*4 : 0,sync=false)
 accumspec=JACC.launch_spec(;threads=128,blocks=div(bins,128),shmem_size=0,sync=false)
 firstpass=method=="smem" ? histogram_smem_jacc! : histogram_gmem_jacc!
 return ()->begin
  JACC.parallel_for(firstspec,(512,64),firstpass,input,parts,Int32(width),Int32(height),Int32(stride),template,tag)
  JACC.parallel_for(accumspec,bins,histogram_accum_jacc!,parts,out,Int32(bins))
 end
end
histogram_study(factory)
