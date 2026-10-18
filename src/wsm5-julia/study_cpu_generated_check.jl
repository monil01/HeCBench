using StaticArrays
module GeneratedCPU
using StaticArrays
module CUDA
rsqrt(x::Float32)=inv(sqrt(x))
end
include(joinpath(@__DIR__,"study_device.jl"))
end
p=@__DIR__
open(joinpath(p,"fixtures_f32.bin"),"r")do io
 nc=read(io,UInt32);nc==3||error("case count")
 open(joinpath(p,"oracle_generated_cpu_f32.bin"),"w")do out
  for ci in 1:Int(nc)
   nx=read(io,UInt32);ny=read(io,UInt32);dt=read(io,Float32);a=[read!(io,Vector{Float32}(undef,Int(nx*ny*(f<10 ? 3 : 1)))) for f in 0:14]
   bounds=Int32[1,nx,1,ny,1,3,1,nx,1,ny,1,3,1,nx,1,ny,1,3]
   for j in UInt32(0):ny-UInt32(1),i in UInt32(0):nx-UInt32(1)
    GeneratedCPU.wsm_body!(a...,dt,bounds...,i÷UInt32(8),j÷UInt32(8),UInt32(8),UInt32(8),i%UInt32(8),j%UInt32(8))
   end
   for v in a;all(isfinite,v)||error("nonfinitegeneratedCPUstate");write(out,v);end
   println("GENERATED_CPU_CASE ",ci," full15fields finite")
  end
 end
 eof(io)||error("fixture trailing")
end
x=open(io->read!(io,Vector{Float32}(undef,filesize(joinpath(p,"oracle_cpu_f32.bin"))÷4)),joinpath(p,"oracle_cpu_f32.bin"));y=open(io->read!(io,similar(x)),joinpath(p,"oracle_generated_cpu_f32.bin"));length(x)==length(y)||error("oracle length");err=maximum(abs.(Float64.(x).-Float64.(y)));scaled=maximum(abs.(Float64.(x).-Float64.(y))./(1e-7.+5e-5.*abs.(Float64.(x))));println("CPU_FULL_COMPARE maxabs=",err," maxnormalizedbudget=",scaled);scaled<=1||error("generatedCPUvsoriginalCPU drift")
