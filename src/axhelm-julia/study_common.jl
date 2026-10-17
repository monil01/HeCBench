include(joinpath(@__DIR__,"study_fixture.jl"))
const STUDY_REPEATS=100
function studycase(run!,fields)
 offset=STUDY_ELEMENTS*512;input=STUDY_Q[1:fields*offset];output=Vector{Float32}(undef,fields*offset)
 CUDA.synchronize();t=time_ns();G=CuArray(STUDY_G);Q=CuArray(input);A=CuArray{Float32}(undef,length(output));D=CuArray(STUDY_D);L=CuArray(STUDY_LAMBDA)
 for repeat in 1:STUDY_REPEATS;run!(fields,Int32(offset),G,D,L,Q,A);end
 copyto!(output,A)
 for a in (G,Q,A,D,L);CUDA.unsafe_free!(a);end
 CUDA.synchronize();elapsed=(time_ns()-t)*1e-9/STUDY_REPEATS
 expected=STUDY_EXPECTED[1:fields*offset];maxerr=maximum(abs.(Float64.(output).-Float64.(expected)));ok=maxerr<1e-3
 return output,ok,maxerr,elapsed
end
function study(run!)
 CUDA.functional()||error("CUDAunavailable");get(ENV,"JULIA_CUDA_MEMORY_POOL","")=="none"||error("Poolmustbedisabledfortimedfree")
 for fields in (1,3);studycase(run!,fields);end
 for trial in 1:5
  got=Float32[];total=0.;ok=true;maxerr=0.
  for fields in (1,3);output,good,err,elapsed=studycase(run!,fields);append!(got,output);total+=elapsed;ok&=good;maxerr=max(maxerr,err);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":14336,\"max_abs_error\":",maxerr,",\"timing_region\":\"sum_scalar_fused3_allocate_h2d_100kernels_d2h_free_perrepeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":14336,\"byte_order\":\"little\"}}")
  ok||error("Helmholtzfulloutputoraclefailed")
 end
end
