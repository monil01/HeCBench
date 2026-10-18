using CUDA,JACC
JACC.@init_backend
include("../study_device.jl")
include("../study_common.jl")
function dot_tree!(i,a,b,out,n,grid,::Val{T}) where T
    tid=Int32(threadIdx().x);index=Int32(i-1);stride=grid*Int32(128)
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
function dot_cub_style!(i,a,b,out,n,grid)
    tid=Int32(threadIdx().x);index=Int32(i-1);stride=grid*Int32(128)
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
  spec=JACC.launch_spec(;threads=128,blocks=grid,shmem_size=0,sync=false)
  if mode==1
   JACC.parallel_for(spec,grid*128,dot_tree!,a,b,out,n,Int32(grid),Val(Float16))
  elseif mode==2
   JACC.parallel_for(spec,grid*128,dot_tree!,a,b,out,n,Int32(grid),Val(Float32))
  else
   JACC.parallel_for(spec,grid*128,dot_cub_style!,a,b,out,n,Int32(grid))
  end
  nothing
 end
end
f16sp_study(factory)
