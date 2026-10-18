using CUDA
include("../study_device.jl")
include("../study_common.jl")
function dot_tree!(a,b,out,n,::Val{T}) where T
    tid=Int32(threadIdx().x);index=Int32((blockIdx().x-1)*blockDim().x+tid-1);stride=Int32(gridDim().x*blockDim().x)
    x,y=f16sp_acc(a,b,n,index,stride,T);shared=CUDA.CuStaticSharedArray(T,(2,128))
    @inbounds shared[1,tid]=x;shared[2,tid]=y
    CUDA.sync_threads()
    off=Int32(64)
    while off>=1
        if tid<=off
            @inbounds shared[1,tid]+=shared[1,tid+off];shared[2,tid]+=shared[2,tid+off]
        end
        CUDA.sync_threads();off÷=Int32(2)
    end
    if tid==1
        value=Float32(shared[1,1])+Float32(shared[2,1]);CUDA.@atomic out[1]+=value
    end
    nothing
end
function dot_cub_style!(a,b,out,n)
    tid=Int32(threadIdx().x);index=Int32((blockIdx().x-1)*blockDim().x+tid-1);stride=Int32(gridDim().x*blockDim().x)
    x,y=f16sp_acc(a,b,n,index,stride,Float32);lane=(tid-Int32(1))%Int32(32);warp=(tid-Int32(1))÷Int32(32);value=f16sp_warp(x+y,lane);shared=CUDA.CuStaticSharedArray(Float32,4)
    if lane==0;@inbounds shared[warp+1]=value;end
    CUDA.sync_threads()
    if tid==1
        for w in 2:4;@inbounds value+=shared[w];end
        CUDA.@atomic out[1]+=value
    end
    nothing
end
function factory(a,b,out,n)
 function run(mode,grid)
  if mode==1
   CUDA.@cuda threads=128 blocks=grid dot_tree!(a,b,out,n,Val(Float16))
  elseif mode==2
   CUDA.@cuda threads=128 blocks=grid dot_tree!(a,b,out,n,Val(Float32))
  else
   CUDA.@cuda threads=128 blocks=grid dot_cub_style!(a,b,out,n)
  end
  nothing
 end
end
f16sp_study(factory)
