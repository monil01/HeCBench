using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_find!(i,img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(i)
    if idx<=n
        @inbounds begin
            d1=abs(Int32(img[idx])-Int32(h1[idx]));d2=abs(Int32(img[idx])-Int32(h2[idx]));t=Int32(tn[idx])
            mp[idx]=(d1>t||d2>t) ? UInt8(255) : UInt8(0)
        end
    end
    nothing
end
function portable_background!(i,img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(i)
    if idx<=n
        @inbounds if mp[idx]==UInt8(0)
            bn[idx]=UInt8(unsafe_trunc(Int32,.92*Float64(bn[idx])+.08*Float64(img[idx])))
        end
    end
    nothing
end
function portable_threshold!(i,img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(i)
    if idx<=n
        @inbounds if mp[idx]==UInt8(0)
            th=Float32(.92*Float64(tn[idx])+.24*Float64(Int32(img[idx])-Int32(bn[idx])))
            tn[idx]=UInt8(unsafe_trunc(Int32,max(th,20f0)))
        end
    end
    nothing
end
function portable_merge!(i,img,h1,h2,tn,bn,mp,n::Int32)
    idx=Int32(i)
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
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=256,shmem_size=0,sync=false)
    function launch!(mode,img,h1,h2,tn,bn,mp)
        n=length(img);args=(img,h1,h2,tn,bn,mp,Int32(n))
        if mode==1
            JACC.parallel_for(portable_merge!,spec,n,args...)
        else
            JACC.parallel_for(portable_find!,spec,n,args...);JACC.parallel_for(portable_background!,spec,n,args...);JACC.parallel_for(portable_threshold!,spec,n,args...)
        end
    end
end
exit(study_main(study_prepare))
