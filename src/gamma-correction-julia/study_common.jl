@inline function transform(q)
 v=(0.3f0*Float32(q[3])+0.59f0*Float32(q[2])+0.11f0*Float32(q[1]))/255f0
 x=unsafe_trunc(UInt8,255f0*v*v)
 return (x,x,x,x)
end
function study(run!)
 n=4096;repeat=10
 input=NTuple{4,UInt8}[(UInt8(mod(17i+3,256)),UInt8(mod(31i+7,256)),UInt8(mod(13i+19,256)),UInt8(mod(7i,256))) for i in 0:n-1]
 expected=transform.(input);device=CuArray(input)
 run!(device,n);CUDA.synchronize()
 for trial in 1:5
  total=0.0
  for _ in 1:repeat
   copyto!(device,input);CUDA.synchronize();start=time_ns();run!(device,n);CUDA.synchronize();total+=(time_ns()-start)*1e-9
  end
  got=Array(device);correct=got==expected;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total/repeat,",\"correct\":",correct,",\"checked_values\":",4n,",\"max_abs_error\":",maximum(abs.(Int.(reinterpret(UInt8,got)).-Int.(reinterpret(UInt8,expected)))),",\"timing_region\":\"mean_individually_synchronized_kernel_excluding_reset\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint8\",\"count\":",4n,",\"byte_order\":\"little\"}}")
  correct||error("bytewise CPU oracle failed")
 end
end
