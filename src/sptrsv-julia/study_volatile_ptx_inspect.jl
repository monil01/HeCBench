using CUDA
path=joinpath(@__DIR__,"sptrsv-julia-cuda.jl","main.jl")
source=first(split(read(path,String),"\ninclude(joinpath(@__DIR__,\"..\",\"study_common.jl\"))"))
include_string(Main,source,path)
I=CUDA.CuDeviceVector{Int32,1};D=CUDA.CuDeviceVector{Float64,1}
open(ARGS[1],"w") do io
 CUDA.code_ptx(io,solve!,Tuple{I,I,D,I,Int32,D,D,I,Int32};kernel=true,dump_module=true)
end
println("PTX compile-only volatile i32/fence/warp synchronization inspection; zero kernel launches")
