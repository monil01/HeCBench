using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
# Common defines unchanged scalar helper; only original device wrapper loaded here.
source=source[findfirst("function zero_point_kernel!",source).start:findfirst("function reference!",source).start-1]
include(joinpath(@__DIR__,"..","study_common.jl"));include_string(Main,source,joinpath(@__DIR__,"main.jl"))
factory(lo,hi,s,z,n)=()->(@cuda threads=256 blocks=cld(n,256) zero_point_kernel!(lo,hi,Int32(-127),Int32(127),n,true,s,z))
zeropoint_study(factory)
