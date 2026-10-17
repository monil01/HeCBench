using CUDA
function matern_cuda!(n::Int32,l::Float32,sources,targets,weights,result)
 tx=threadIdx().x-Int32(1);ty=threadIdx().y-Int32(1);px=(blockIdx().x-Int32(1))*Int32(16)+tx
 if px>=n||ty>=Int32(50);return;end
 shared=@cuStaticSharedMem(Float32,(1048,))
 @inbounds begin
  if ty==0
   for i=Int32(0):Int32(2);shared[Int32(800)+tx*Int32(3)+i+Int32(1)]=targets[px*Int32(3)+i+Int32(1)];end
  end
  if tx==0
   for i=Int32(0):Int32(2);shared[Int32(848)+ty*Int32(3)+i+Int32(1)]=sources[ty*Int32(3)+i+Int32(1)];end
   shared[Int32(998)+ty+Int32(1)]=weights[ty+Int32(1)]
  end
 end
 sync_threads();squared=0f0
 @inbounds for i=Int32(0):Int32(2)
  delta=shared[Int32(800)+tx*Int32(3)+i+Int32(1)]-shared[Int32(848)+ty*Int32(3)+i+Int32(1)];squared+=delta*delta
 end
 diff=sqrt(squared)
 @inbounds shared[tx*Int32(50)+ty+Int32(1)]=(1f0+sqrt(5f0)*diff/l+5f0*squared/(3f0*l*l))*exp(-sqrt(5f0)*diff/l)*shared[Int32(998)+ty+Int32(1)]
 sync_threads()
 if ty==0
  res=0f0
  @inbounds for i=Int32(0):Int32(49);res+=shared[tx*Int32(50)+i+Int32(1)];end
  @inbounds result[px+Int32(1)]=res
 end
 return
end
function launch_reference(n,l,s,t,w,out)
 @cuda threads=(16,64) blocks=cld(Int(n),16) matern_cuda!(n,l,s,t,w,out)
end
include("../study_common.jl")
function main(args)
 isempty(args)||error("This entry runs the verified named17cube study. Historical CLI was a placeholder and is archived in original_main.jl.")
 matern_study(launch_reference)
 return 0
end
if abspath(PROGRAM_FILE)==@__FILE__;exit(main(ARGS));end
