using CUDA
text=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(text,r"\nmain\(\)\s*$"=>"\n"),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(op,a,b,c,partial,n)
    if op=="copy";return ()->(@cuda threads=256 blocks=cld(n,256) copy_kernel!(a,c))
    elseif op=="mul";return ()->(@cuda threads=256 blocks=cld(n,256) mul_kernel!(b,c,0.4f0))
    elseif op=="add";return ()->(@cuda threads=256 blocks=cld(n,256) add_kernel!(a,b,c))
    elseif op=="triad";return ()->(@cuda threads=256 blocks=cld(n,256) triad_kernel!(a,b,c,0.4f0))
    elseif op=="nstream";return ()->(@cuda threads=256 blocks=cld(n,256) nstream_kernel!(a,b,c,0.4f0))
    elseif op=="dot";return ()->(@cuda threads=256 blocks=256 dot_kernel!(a,b,partial,Int32(n)))
    else;error("unknown operation");end
end
stream_study(factory)
