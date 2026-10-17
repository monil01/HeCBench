include(joinpath(@__DIR__,"study_expected.jl"))
const STUDY_REPEATS=3
const STUDY_CASES=((.25f0,.2f0),(.125f0,.15f0),(.375f0,.25f0))
function study(run!)
 maturities=Int32[4,4,4,8,8,8,20,20,20,28,28,28,40,40,40];rates=Float32[.045,.05,.055,.045,.05,.055,.045,.05,.055,.045,.05,.055,.045,.05,.055]
 cases=[];prefix=Float32[]
 for (delta,lam) in STUDY_CASES
  lambda=fill(lam,NN);append!(prefix,lambda);append!(prefix,Float32.(maturities));append!(prefix,rates);push!(prefix,delta)
  push!(cases,(CUDA.zeros(Float32,NPATH),CUDA.zeros(Float32,NPATH),CuArray(lambda),CuArray(maturities),CuArray(rates),delta,CUDA.zeros(Float32,NPATH*NN),CUDA.zeros(Float32,NPATH*L2_SIZE),CUDA.zeros(Float32,NPATH*NMAT),CUDA.zeros(Float32,NPATH*NMAT),CUDA.zeros(Float32,NPATH*NMAT),CUDA.zeros(Float32,NPATH*NMAT)))
 end
 function case!(args,c;repeats=STUDY_REPEATS)
  outputs=Float32[];total=0.;maxerr=0.
  for mode in (false,true)
   CUDA.synchronize();t=time_ns()
   for _ in 1:repeats;run!(Val(mode),args...);end
   CUDA.synchronize();total+=(time_ns()-t)*1e-9/repeats
   value=Array(args[1]);append!(outputs,value);maxerr=max(maxerr,maximum(abs.(Float64.(value).-EXPECTED_V[c])))
   if mode
    greek=Array(args[2]);append!(outputs,greek);maxerr=max(maxerr,maximum(abs.(Float64.(greek).-EXPECTED_G[c])))
   end
  end
  return outputs,total,maxerr
 end
 for (c,args) in enumerate(cases);case!(args,c;repeats=1);end
 for trial in 1:5
  output=copy(prefix);elapsed=0.;maxerr=0.
  for (c,args) in enumerate(cases);o,t,e=case!(args,c);append!(output,o);elapsed+=t;maxerr=max(maxerr,e);end
  ok=all(isfinite,output)&&maxerr<=1e-3;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,output),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":2646,\"max_abs_error\":",maxerr,",\"timing_region\":\"sum_three_cases_value_and_greek_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":2646,\"byte_order\":\"little\"}}")
  ok||Base.error("IndependentLIBORprice/finite-differenceGreekfailed")
 end
end
