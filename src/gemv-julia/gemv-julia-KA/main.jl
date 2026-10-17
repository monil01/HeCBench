using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function gemv_int8_ka!(mat,vec,res,n::Int32,scale::Float16,zp::Float16)
 tx=Int32(@index(Local,NTuple)[1]);ty=Int32(@index(Local,NTuple)[2]);row=(Int32(@index(Group,NTuple)[2])-Int32(1))*Int32(4)+ty-Int32(1)
 scratch=@localmem Float32 (32,4);acc=0.f0
 for iteration in Int32(0):div(n,Int32(32*8))-Int32(1)
  base=(tx-Int32(1)+iteration*Int32(32))*Int32(8)
  for off in Int32(0):Int32(7)
   term=q8(mat,row,base+off,n)-Float32(zp)
   acc=muladd(Float32(@inbounds vec[base+off+Int32(1)]),term,acc)
  end
 end
 @inbounds scratch[tx,ty]=acc*Float32(scale)
 @synchronize
 for stride in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tx<=stride;@inbounds scratch[tx,ty]+=scratch[tx+stride,ty];end
  @synchronize
 end
 if tx==Int32(1);@inbounds res[row+Int32(1)]=Float16(scratch[1,ty]);end
end
@kernel function gemv_int4_ka!(mat,vec,res,n::Int32,scale::Float16,zp::Float16)
 tx=Int32(@index(Local,NTuple)[1]);ty=Int32(@index(Local,NTuple)[2]);row=(Int32(@index(Group,NTuple)[2])-Int32(1))*Int32(4)+ty-Int32(1)
 scratch=@localmem Float32 (32,4);acc=0.f0
 for iteration in Int32(0):div(n,Int32(32*16))-Int32(1)
  base=(tx-Int32(1)+iteration*Int32(32))*Int32(16)
  for off in Int32(0):Int32(15)
   term=q4(mat,row,base+off,n)-Float32(zp)
   acc=muladd(Float32(@inbounds vec[base+off+Int32(1)]),term,acc)
  end
 end
 @inbounds scratch[tx,ty]=acc*Float32(scale)
 @synchronize
 for stride in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tx<=stride;@inbounds scratch[tx,ty]+=scratch[tx+stride,ty];end
  @synchronize
 end
 if tx==Int32(1);@inbounds res[row+Int32(1)]=Float16(scratch[1,ty]);end
end
function factory(mat,vec,res,n,scale,zp,packed)
 kernel=packed ? gemv_int4_ka! : gemv_int8_ka!
 launch=kernel(get_backend(mat),(32,4))
 return ()->launch(mat,vec,res,Int32(n),scale,zp;ndrange=(32,n))
end
gemv_study(factory)
