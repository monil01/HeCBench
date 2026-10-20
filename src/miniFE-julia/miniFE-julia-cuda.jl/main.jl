using CUDA,Libdl,SHA
include(joinpath(@__DIR__,"..","mini_device.jl"))
function vector_entry!(x,y,w,n,a,b,op)
    i=Int32(threadIdx().x)+Int32(blockIdx().x-1)*Int32(blockDim().x)
    mini_vector!(i,n,x,y,w,a,b,op);nothing
end
function matvec_entry!(rp,c,a,x,w,n)
    i=Int32(threadIdx().x)+Int32(blockIdx().x-1)*Int32(blockDim().x)
    mini_matvec!(i,n,rp,c,a,x,w);nothing
end
function dot_entry!(x,y,d,n)
    t=Int32(threadIdx().x);i=t+Int32(blockIdx().x-1)*Int32(blockDim().x)
    stride=Int32(gridDim().x)*Int32(blockDim().x);s=0.0
    @inbounds while i<=n
        s=muladd(x[i],y[i],s);i+=stride
    end
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
function final_entry!(d)
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
function mini_vector_launch(x,y,w,n,a,b,g,op)
    CUDA.@cuda threads=256 blocks=g stream=CUDA.legacy_stream() vector_entry!(x,y,w,n,a,b,op)
end
function mini_matvec_launch(rp,c,a,x,w,n,g)
    CUDA.@cuda threads=256 blocks=g stream=CUDA.legacy_stream() matvec_entry!(rp,c,a,x,w,n)
end
function mini_dot_launch(x,y,w,n,g)
    CUDA.@cuda threads=256 blocks=g stream=CUDA.legacy_stream() dot_entry!(x,y,w,n)
end
function mini_final_launch(w)
    CUDA.@cuda threads=256 blocks=1 stream=CUDA.legacy_stream() final_entry!(w)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
mini_study()
