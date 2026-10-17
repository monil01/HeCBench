using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function gemv_int8_julia!(mat,vec,res,n::Int32,scale::Float16,zp::Float16)
 tx=Int32(threadIdx().x);ty=Int32(threadIdx().y);row=(Int32(blockIdx().y)-Int32(1))*Int32(4)+ty-Int32(1)
 scratch=CuStaticSharedArray(Float32,(32,4));acc=0.f0
 for iteration in Int32(0):div(n,Int32(32*8))-Int32(1)
  base=(tx-Int32(1)+iteration*Int32(32))*Int32(8)
  for off in Int32(0):Int32(7)
   term=q8(mat,row,base+off,n)-Float32(zp)
   acc=muladd(Float32(@inbounds vec[base+off+Int32(1)]),term,acc)
  end
 end
 @inbounds scratch[tx,ty]=acc*Float32(scale)
 sync_threads()
 for stride in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tx<=stride;@inbounds scratch[tx,ty]+=scratch[tx+stride,ty];end
  sync_threads()
 end
 if tx==Int32(1);@inbounds res[row+Int32(1)]=Float16(scratch[1,ty]);end
return
end
function gemv_int4_julia!(mat,vec,res,n::Int32,scale::Float16,zp::Float16)
 tx=Int32(threadIdx().x);ty=Int32(threadIdx().y);row=(Int32(blockIdx().y)-Int32(1))*Int32(4)+ty-Int32(1)
 scratch=CuStaticSharedArray(Float32,(32,4));acc=0.f0
 for iteration in Int32(0):div(n,Int32(32*16))-Int32(1)
  base=(tx-Int32(1)+iteration*Int32(32))*Int32(16)
  for off in Int32(0):Int32(15)
   term=q4(mat,row,base+off,n)-Float32(zp)
   acc=muladd(Float32(@inbounds vec[base+off+Int32(1)]),term,acc)
  end
 end
 @inbounds scratch[tx,ty]=acc*Float32(scale)
 sync_threads()
 for stride in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tx<=stride;@inbounds scratch[tx,ty]+=scratch[tx+stride,ty];end
  sync_threads()
 end
 if tx==Int32(1);@inbounds res[row+Int32(1)]=Float16(scratch[1,ty]);end
return
end
function factory(mat,vec,res,n,scale,zp,packed)
 kernel=packed ? gemv_int4_julia! : gemv_int8_julia!
 return ()->(@cuda threads=(32,4) blocks=(1,128) kernel(mat,vec,res,Int32(n),scale,zp))
end
if abspath(PROGRAM_FILE)==@__FILE__;gemv_study(factory);end
