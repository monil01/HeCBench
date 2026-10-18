using CUDA,KernelAbstractions,Atomix
include("../study_device.jl")
include("../study_common.jl")
@kernel function dot_tree!(a,b,out,n,grid,::Val{T}) where T
    tid=Int32(@index(Local,Linear));index=Int32(@index(Global,Linear)-1);stride=grid*Int32(128)
    x,y=f16sp_acc(a,b,n,index,stride,T);shared=@localmem T (2,128)
    @inbounds shared[1,tid]=x;shared[2,tid]=y
    @synchronize
    for off in (64,32,16,8,4,2,1)
        if tid<=off
            @inbounds shared[1,tid]+=shared[1,tid+off];shared[2,tid]+=shared[2,tid+off]
        end
        @synchronize
    end
    if tid==1
        value=Float32(shared[1,1])+Float32(shared[2,1]);Atomix.@atomic out[1]+=value
    end
end
@kernel function dot_cub_style!(a,b,out,n,grid)
    tid=Int32(@index(Local,Linear));index=Int32(@index(Global,Linear)-1);stride=grid*Int32(128)
    x,y=f16sp_acc(a,b,n,index,stride,Float32);lane=(tid-Int32(1))%Int32(32);warp=(tid-Int32(1))÷Int32(32);value=f16sp_warp(x+y,lane);shared=@localmem Float32 (4,)
    if lane==0;@inbounds shared[warp+1]=value;end
    @synchronize
    if tid==1
        for w in 2:4;@inbounds value+=shared[w];end
        Atomix.@atomic out[1]+=value
    end
end
function factory(a,b,out,n)
    backend=CUDABackend();tree=dot_tree!(backend,128);warp=dot_cub_style!(backend,128)
    function run(mode,grid)
        if mode==1
            tree(a,b,out,n,Int32(grid),Val(Float16);ndrange=grid*128)
        elseif mode==2
            tree(a,b,out,n,Int32(grid),Val(Float32);ndrange=grid*128)
        else
            warp(a,b,out,n,Int32(grid);ndrange=grid*128)
        end
        nothing
    end
end
f16sp_study(factory)
