using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function gemv_int8_jacc!(indexlane,indexrow,mat,vec,res,n::Int32,scale::Float16,zp::Float16,template)
 tx=Int32(indexlane);ty=Int32(mod(indexrow-1,4)+1);row=Int32(indexrow-1)
 scratch=JACC.shared(template);acc=0.f0
 for iteration in Int32(0):div(n,Int32(32*8))-Int32(1)
  base=(tx-Int32(1)+iteration*Int32(32))*Int32(8)
  for off in Int32(0):Int32(7)
   term=q8(mat,row,base+off,n)-Float32(zp)
   acc=muladd(Float32(@inbounds vec[base+off+Int32(1)]),term,acc)
  end
 end
 @inbounds scratch[tx,ty]=acc*Float32(scale)
 JACC.sync_workgroup()
 for stride in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tx<=stride;@inbounds scratch[tx,ty]+=scratch[tx+stride,ty];end
  JACC.sync_workgroup()
 end
 if tx==Int32(1);@inbounds res[row+Int32(1)]=Float16(scratch[1,ty]);end
end
function gemv_int4_jacc!(indexlane,indexrow,mat,vec,res,n::Int32,scale::Float16,zp::Float16,template)
 tx=Int32(indexlane);ty=Int32(mod(indexrow-1,4)+1);row=Int32(indexrow-1)
 scratch=JACC.shared(template);acc=0.f0
 for iteration in Int32(0):div(n,Int32(32*16))-Int32(1)
  base=(tx-Int32(1)+iteration*Int32(32))*Int32(16)
  for off in Int32(0):Int32(15)
   term=q4(mat,row,base+off,n)-Float32(zp)
   acc=muladd(Float32(@inbounds vec[base+off+Int32(1)]),term,acc)
  end
 end
 @inbounds scratch[tx,ty]=acc*Float32(scale)
 JACC.sync_workgroup()
 for stride in (Int32(16),Int32(8),Int32(4),Int32(2),Int32(1))
  if tx<=stride;@inbounds scratch[tx,ty]+=scratch[tx+stride,ty];end
  JACC.sync_workgroup()
 end
 if tx==Int32(1);@inbounds res[row+Int32(1)]=Float16(scratch[1,ty]);end
end
function factory(mat,vec,res,n,scale,zp,packed)
 kernel=packed ? gemv_int4_jacc! : gemv_int8_jacc!
 template=CUDA.zeros(Float32,32,4)
 spec=JACC.launch_spec(;threads=(32,4),blocks=(1,128),shmem_size=512,sync=false)
 return ()->JACC.parallel_for(spec,(32,n),kernel,mat,vec,res,Int32(n),scale,zp,template)
end
gemv_study(factory)
