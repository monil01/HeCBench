using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_find!(img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(@index(Global,Linear))
    if idx<=n
        @inbounds begin
            d1=abs(Int32(img[idx])-Int32(h1[idx]));d2=abs(Int32(img[idx])-Int32(h2[idx]));t=Int32(tn[idx])
            mp[idx]=(d1>t||d2>t) ? UInt8(255) : UInt8(0)
        end
    end
    nothing
end
@kernel function portable_background!(img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(@index(Global,Linear))
    if idx<=n
        @inbounds if mp[idx]==UInt8(0)
            bn[idx]=UInt8(unsafe_trunc(Int32,.92*Float64(bn[idx])+.08*Float64(img[idx])))
        end
    end
    nothing
end
@kernel function portable_threshold!(img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(@index(Global,Linear))
    if idx<=n
        @inbounds if mp[idx]==UInt8(0)
            th=Float32(.92*Float64(tn[idx])+.24*Float64(Int32(img[idx])-Int32(bn[idx])))
            tn[idx]=UInt8(unsafe_trunc(Int32,max(th,20f0)))
        end
    end
    nothing
end
@kernel function portable_merge!(img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(@index(Global,Linear))
    if idx<=n
        @inbounds if abs(Int32(img[idx])-Int32(h1[idx]))<=tn[idx] && abs(Int32(img[idx])-Int32(h2[idx]))<=tn[idx]
            bn[idx]=UInt8(unsafe_trunc(Int32,.92*Float64(bn[idx])+.08*Float64(img[idx])))
            th=Float32(.92*Float64(tn[idx])+.24*Float64(Int32(img[idx])-Int32(bn[idx])))
            tn[idx]=UInt8(unsafe_trunc(Int32,max(th,20f0)))
        end
    end
    nothing
end
function study_prepare()
    backend=CUDABackend()
    k_find=portable_find!(backend,256)
    k_background=portable_background!(backend,256)
    k_threshold=portable_threshold!(backend,256)
    k_merge=portable_merge!(backend,256)
    function launch!(mode,img,h1,h2,tn,bn,mp)
        n=Int32(length(img));args=(img,h1,h2,tn,bn,mp,n)
        if mode==1
            k_merge(args...;ndrange=Int(n))
        else
            k_find(args...;ndrange=Int(n));k_background(args...;ndrange=Int(n));k_threshold(args...;ndrange=Int(n))
        end
    end
end
exit(study_main(study_prepare))
