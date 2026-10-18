module CUDA
synchronize()=nothing
end
include("/home/imo/HeCBench_julia/src/lulesh-julia/study_device.jl")
include("/home/imo/HeCBench_julia/src/lulesh-julia/study_common.jl")
function launch(::Val{name},n,args...)where{name}
 f=getfield(Main,Symbol(string(name),"_body!"))
 for i in Int32(0):Int32(n-1);f(args...,i);end
end
states=prepare_states(copy)
values=Float64[]
for s in states;capture!(values,s,true);end
for s in states;pipeline!(s,launch,values,true);end
oracle=collect(reinterpret(Float64,read("/home/imo/HeCBench_julia/src/lulesh-julia/source_cpu_oracle_f64.bin")))
@assert length(values)==length(oracle)
maxerror=maximum(abs.(values.-oracle));bad=findfirst(i->!isfinite(values[i])||abs(values[i]-oracle[i])>1e-7+3e-8*abs(oracle[i]),eachindex(values))
println("Full Julia CPU source-pipeline check words=",length(values)," maxabs=",maxerror," firstbad=",bad)
if bad!==nothing;println("actual/expected ",values[bad]," / ",oracle[bad]);error("CPU translated operator mismatch");end
