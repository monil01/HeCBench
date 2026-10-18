using CUDA
path=joinpath(@__DIR__,"mis-julia-cuda.jl","main.jl")
source=first(split(read(path,String),"\ninclude(joinpath(@__DIR__,\"..\",\"study_common.jl\"))"))
include_string(Main,source,path)
types=Tuple{Int32,CUDA.CuDeviceVector{Int32,1},CUDA.CuDeviceVector{Int32,1},CUDA.CuDeviceVector{UInt8,1}}
open(ARGS[1],"w") do io
 CUDA.code_ptx(io,findmins!,types;kernel=true,dump_module=true)
end
println("PTX compile-only volatile memory inspection; no kernel launch")
