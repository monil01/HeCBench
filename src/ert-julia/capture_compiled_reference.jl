# Compile-only reflection of the measured CUDA.jl function/types, no timed series.
using CUDA
source=joinpath(@__DIR__,"ert-julia-cuda.jl","main.jl")
text=read(source,String)
prefix=first(split(text,"include(joinpath(@__DIR__,\"..\",\"study_common.jl\"))"))
include_string(Main,prefix,source)
folder=ARGS[1];mkpath(folder)
for (T,label) in [(UInt32,"packedhalf2"),(Float32,"f32"),(Float64,"f64")]
 a=CuArray{T}(undef,32);viewed=@view a[9:24];d=CUDA.cudaconvert(viewed)
 open(joinpath(folder,"cuda_jl_$(label).ptx"),"w") do io
  CUDA.code_ptx(io,ert_kernel!,Tuple{typeof(d),UInt32,UInt32,UInt32};kernel=true,raw=true)
 end
 open(joinpath(folder,"cuda_jl_chain_$(label).ptx"),"w") do io
  CUDA.code_ptx(io,arithmetic_chain,Tuple{T};kernel=false,raw=true)
 end
 println(label," ",typeof(d))
end
