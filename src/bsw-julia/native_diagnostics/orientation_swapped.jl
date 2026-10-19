# Supplemental one-pipeline diagnostic; excludes all primary timing statistics.
using CUDA
include(joinpath(@__DIR__,"orientation_swapped_params.jl"))
model=ARGS[1];suffix=model=="cuda_jl" ? "cuda.jl" : model=="ka" ? "KA" : "JACC"
entry=joinpath(@__DIR__,"..","bsw-julia-"*suffix,"main.jl")
src=first(split(read(entry,String),"bsw_study("));src=replace(src,"include(joinpath(@__DIR__,\"..\",\"study_common.jl\"))"=>"")
include_string(Main,src,entry)
allocate=model=="cuda_jl" ? CuArray : model=="ka" ? allocate_ka : JACC.array
function orientation_diagnostic(allocate,launch)
 wrap(v,g)=allocate([fill(eltype(v)(g),16);v;fill(eltype(v)(g),16)])
 av(a)=view(a,17:length(a)-16)
 a=wrap(zeros(UInt8,NA),211);b=wrap(zeros(UInt8,NB),211);oa=wrap(zeros(UInt32,N),24681357);ob=wrap(zeros(UInt32,N),24681357);sc=wrap(SCORING,-12345);en=wrap(ENCODING,-12345);outs=[wrap(fill(Int16(-12345),N),-12345) for _ in 1:5]
 inputs=(a,b,oa,ob,sc,en);before=Int32[];initial=Int32[]
 for x in inputs;append!(before,Int32.(Array(x)));end
 for x in outs;append!(initial,Int32.(Array(x)));end
 copyto!(av(oa),OFF_A);copyto!(av(ob),OFF_B);copyto!(av(a),SEQ_A);copyto!(av(b),SEQ_B)
 args=(av(a),av(b),av(oa),av(ob),(av(x) for x in outs)...,Int16(-6),Int16(-1),av(sc),av(en))
 launch(false,FORWARD,args...);ea=Array(av(outs[2]));eb=Array(av(outs[4]));CUDA.device_synchronize(blocking=true);reverse=Int(min(maximum(ea),maximum(eb)));launch(true,reverse,args...);CUDA.device_synchronize(blocking=true)
 actual=Int32[N,128,160,-6,-1,FORWARD,reverse];append!(actual,before)
 for x in inputs;append!(actual,Int32.(Array(x)));end
 append!(actual,Int32.(ea));append!(actual,Int32.(eb));append!(actual,initial)
 for x in outs;append!(actual,Int32.(Array(x)));end
 expected=collect(reinterpret(Int32,read(joinpath(@__DIR__,"orientation_swapped_expected_i32.bin"))));write(ARGS[2],actual)
 println("DIAGNOSTIC swapped_orientation model=",model," actual_values=",length(actual)," correct=",actual==expected," timing_excluded=true");flush(stdout);actual==expected||error("independent fullmatrix swappedorientation mismatch")
end
if model=="cuda_jl"
 orientation_diagnostic(allocate,(inverse,threads,args...)->CUDA.@cuda(threads=threads,blocks=N,shmem=SHMEM,sequence_cuda!(inverse,args...)))
elseif model=="ka"
 orientation_diagnostic(allocate,launch_ka!)
else
 orientation_diagnostic(allocate,launch_jacc!)
end
