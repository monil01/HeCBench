using CUDA
path=joinpath(@__DIR__,"nbody-julia-cuda.jl","main.jl")
source=first(split(read(path,String),"\ninclude(joinpath(@__DIR__,\"..\",\"study_common.jl\"))"))
include_string(Main,source,path)
types=Tuple{CUDA.CuDeviceVector{StudyParticle,1},Int32,Float32,Float32}
open(ARGS[1],"w") do io
    CUDA.code_ptx(io,accelerate!,types;kernel=true,dump_module=true)
end
println("Compile-only SFU inspection; no GPU kernel launched")
