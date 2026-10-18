# Diagnostic only: compile/launch the exact primary entrypoints outside study ROI.
using CUDA,SHA
model=ARGS[1]
folder=Dict("cuda_jl"=>"cuda.jl","ka"=>"KA","jacc"=>"JACC")[model]
file=joinpath(@__DIR__,"kernelLaunch-julia-"*folder,"main.jl")
source=read(file,String)
source=replace(source, "include(\"../study_types.jl\")"=>"include("*repr(joinpath(@__DIR__,"study_types.jl"))*")", "include(\"../study_common.jl\")"=>"include("*repr(joinpath(@__DIR__,"study_common.jl"))*")")
@assert occursin("run_study(launch,launch_echo)",source)
include_string(Main,replace(source,"run_study(launch,launch_echo)"=>""),file)
base=joinpath(@__DIR__,"..","..",".porting-state","porting_logs","kernelLaunch-julia-"*model,"argument_abi")
mkpath(base)
for n in (16,256,4096)
 args=KLArgs(ntuple(i->UInt8(mod(37*i,256)),n))
 out=reinterpret(Core.LLVMPtr{UInt8,1},UInt64(0))
 fileout=joinpath(base,"primary_$(n).ptx")
 open(fileout,"w") do io
  CUDA.@device_code_ptx io=io launch(args,out)
  CUDA.device_synchronize()
 end
 println("ABI_PTX ",model," ",n," ",bytes2hex(sha256(read(fileout)))," ",fileout)
end
