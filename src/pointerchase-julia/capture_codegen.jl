# Separate static reflection phase; not a timing trial or translation candidate.
using CUDA
model=ARGS[1];suffix=Dict("cuda_jl"=>"cuda.jl","ka"=>"KA","jacc"=>"JACC")[model]
source=joinpath(@__DIR__,"pointerchase-julia-"*suffix,"main.jl")
text=first(split(read(source,String),"include(\"../study_common.jl\")"))
text=replace(text,"include(\"../pointer_device.jl\")"=>"include("*repr(joinpath(@__DIR__,"pointer_device.jl"))*")")
include_string(Main,text,source)
d=CuArray(UInt64[0]);base=UInt64(pointer(d));copyto!(d,UInt64[base])
path=ARGS[2]
open(path,"w") do io
 CUDA.@device_code_ptx io=io launch(d,base,UInt32(1),UInt32(0),1)
 CUDA.synchronize()
end
println("Static PTX reflection complete: ",path)
