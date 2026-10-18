using CUDA
path=joinpath(@__DIR__,"tridiagonal-julia-cuda.jl","main.jl")
source=first(split(read(path,String),"\ninclude(joinpath(@__DIR__,\"..\",\"study_common.jl\"))"))
include_string(Main,source,path)
D=CUDA.CuDeviceVector{Float32,1}
open(ARGS[1],"w")do io
 CUDA.code_ptx(io,sweep!,Tuple{D,D,D,D,D,D,Val{128},Int32,Val{6}};kernel=true,dump_module=true)
 CUDA.code_ptx(io,pcr!,Tuple{D,D,D,D,D,Val{128},Int32,Val{0}};kernel=true,dump_module=true)
end
println("Two compile-only CUDAjl PTX reflections, zero kernel launches")
