using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String);lastmain=findlast("\nmain()",source);include_string(Main,source[1:first(lastmain)-1],joinpath(@__DIR__,"main.jl"))
include("../study_common.jl")
function factory(a)
 p=Params(2,3,19,23,9,37,29)
 ()->CUDA.@cuda threads=256 blocks=11 car_kernel!(a[1],a[2],a[3],a[4],a[5],p,Int32(1),Int32(1),Int64(2622))
end
car_study(factory)
