using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
function ssim_kernel(i,j,k,A,B,O,bx,by,bz,gx,gy)
 x=Int32(i-1);y=Int32(j-1);z=Int32(k-1)
 if x<bx&&y<by&&z<bz
  @inbounds O[x+y*bx+z*bx*by+Int32(1)]=ssim_value(x,y,z,A,B,gx,gy)
 end
 nothing
end
function makearray(a)
 d=JACC.array(a)
 d isa CUDA.CuArray{Float32}||error("ExpectedJACCCUDAstorage")
 d
end
function factory(A,B,O)
 function launch(t)
  spec=JACC.launch_spec(;threads=(8,8,8),blocks=(cld(t.bx,8),cld(t.by,8),cld(t.bz,8)),shmem_size=0,sync=false)
  JACC.parallel_for(spec,(t.bx,t.by,t.bz),ssim_kernel,A,B,O,Int32(t.bx),Int32(t.by),Int32(t.bz),Int32(t.gx),Int32(t.gy))
  nothing
 end
 launch
end
include("../study_common.jl")
ssim_study(makearray,factory)
