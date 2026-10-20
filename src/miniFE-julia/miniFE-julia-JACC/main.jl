using CUDA,Libdl,SHA,JACC
include(joinpath(@__DIR__,"..","mini_device.jl"))
@inline function vector_entry!(index,x,y,w,n,a,b,op)
    mini_vector!(Int32(index),n,x,y,w,a,b,op);nothing
end
@inline function matvec_entry!(index,rp,c,a,x,w,n)
    mini_matvec!(Int32(index),n,rp,c,a,x,w);nothing
end
@inline function dot_entry!(index,x,y,d,n,stride)
    t=Int32(threadIdx().x);i=Int32(index);s=0.0
    @inbounds while i<=n
        s=muladd(x[i],y[i],s);i+=stride
    end
    # CUDA-backend extension retains exactly the source 256-double tile.
    red=CuStaticSharedArray(Float64,256);@inbounds red[t]=s
    Base.Cartesian.@nexprs 8 k -> begin
        offset=Int32(256>>k);sync_threads()
        if t<=offset
            @inbounds s+=red[t+offset]
            @inbounds red[t]=s
        end
    end
    if t==1;@inbounds d[blockIdx().x]=s;end
    nothing
end
@inline function final_entry!(index,d)
    t=Int32(threadIdx().x);@inbounds s=d[t]
    red=CuStaticSharedArray(Float64,256);@inbounds red[t]=s
    Base.Cartesian.@nexprs 8 k -> begin
        offset=Int32(256>>k);sync_threads()
        if t<=offset
            @inbounds s+=red[t+offset]
            @inbounds red[t]=s
        end
    end
    if t==1;@inbounds d[1]=s;end
    nothing
end
const MINI_JACC_SPECS=Dict{Tuple{Int,Int},Any}()
function mini_spec(g)
    get!(MINI_JACC_SPECS,(256,g)) do
        # Cache complete geometry; never reuse a dimension inferred for another mesh.
        JACC.launch_spec(;threads=(256,),blocks=(g,),shmem_size=0,sync=false,stream=CUDA.legacy_stream())
    end
end
function mini_vector_launch(x,y,w,n,a,b,g,op)
    JACC.parallel_for(mini_spec(g),g*256,vector_entry!,x,y,w,n,a,b,op)
end
function mini_matvec_launch(rp,c,a,x,w,n,g)
    JACC.parallel_for(mini_spec(g),g*256,matvec_entry!,rp,c,a,x,w,n)
end
function mini_dot_launch(x,y,w,n,g)
    JACC.parallel_for(mini_spec(g),g*256,dot_entry!,x,y,w,n,Int32(g*256))
end
function mini_final_launch(w)
    JACC.parallel_for(mini_spec(1),256,final_entry!,w)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mini_study()
