function directed(a,b)
 largest=0f0
 for point in a
  smallest=floatmax(Float32)
  for other in b
   dx=point[1]-other[1];dy=point[2]-other[2];smallest=min(smallest,dx*dx+dy*dy)
  end
  largest=max(largest,smallest)
 end
 return largest
end
function study(run!)
 na=512;nb=768;repeat=10
 a=NTuple{2,Float32}[(Float32(mod(17i+3,1024))/1024f0,Float32(mod(29i+7,1024))/1024f0) for i in 0:na-1]
 b=NTuple{2,Float32}[(Float32(mod(31i+11,1024))/1024f0,Float32(mod(43i+19,1024))/1024f0) for i in 0:nb-1]
 expected=Float32[directed(a,b),directed(b,a)];da=CuArray(a);db=CuArray(b);initial=collect(reinterpret(UInt32,fill(-1f0,2)));out=CuArray(initial)
 run!(da,db,out,Int32(na),Int32(nb));CUDA.synchronize()
 for trial in 1:5
  total=0.
  for _ in 1:repeat
   copyto!(out,initial);CUDA.synchronize();start=time_ns();run!(da,db,out,Int32(na),Int32(nb));CUDA.synchronize();total+=(time_ns()-start)*1e-9
  end
  got=collect(reinterpret(Float32,Array(out)));err=maximum(abs.(got.-expected));correct=all(isfinite,got)&&err<=1f-3;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total/repeat,",\"correct\":",correct,",\"checked_values\":2,\"max_abs_error\":",err,",\"timing_region\":\"mean_two_directional_kernels_excluding_reset\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":2,\"byte_order\":\"little\"}}")
  correct||error("Directed squared Hausdorff CPU oracle failed")
 end
end
