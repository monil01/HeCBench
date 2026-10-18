
using CUDA,Printf,Libdl
const BASE=dirname(@__FILE__);const HOST="/home/imo/HeCBench_julia/.porting-state/portable_julia/runtime/myocyte"
const SOLVER=joinpath(HOST,"libstudy_solver.so");const ORACLE=joinpath(HOST,"libstudy_oracle.so")
const DY=CuArray(fill(12345.5f0,107));const DP=CuArray(fill(12345.5f0,34));const DF=CuArray(fill(12345.5f0,107));const DC=CuArray(fill(12345.5f0,19))
const LAUNCH=Ref{Any}(nothing)
function evaluator_callback(t::Float32,y::Ptr{Float32},p::Ptr{Float32},out::Ptr{Float32},com::Ptr{Float32})::Float64
 unsafe_copyto!(pointer(DY),y,91);unsafe_copyto!(pointer(DP),p,18)
 CUDA.device_synchronize(blocking=true);start=time_ns();LAUNCH[](Int32(trunc(t)));CUDA.device_synchronize(blocking=true);elapsed=(time_ns()-start)*1e-9
 unsafe_copyto!(out,pointer(DF),91);unsafe_copyto!(com,pointer(DC),3)
 elapsed
end
const CALLBACK=@cfunction(evaluator_callback,Float64,(Float32,Ptr{Float32},Ptr{Float32},Ptr{Float32},Ptr{Float32}))
function myocyte_study(launch)
 LAUNCH[]=launch;y=parse.(Float32,split(read(joinpath(BASE,"study_y.txt"),String)));p=parse.(Float32,split(read(joinpath(BASE,"study_params.txt"),String)));while length(p)<18;push!(p,last(p));end;pd=Float64.(p)
 function solve(xmax)
  elapsed=ccall((:study_solve,SOLVER),Float64,(Ptr{Cvoid},Ptr{Float32},Ptr{Float32},Cint),CALLBACK,y,p,xmax)
  n=ccall((:study_count,SOLVER),Csize_t,());result=Vector{Float64}(undef,n);ccall((:study_copy,SOLVER),Cvoid,(Ptr{Float64},),result)
  metrics=zeros(Float64,6);ccall((:study_check,ORACLE),Cvoid,(Ptr{Float64},Csize_t,Cint,Ptr{Float64},Ptr{Float64}),result,n,xmax,pd,metrics)
  elapsed,result,metrics
 end
 solve(2)
 for trial=1:5
  values=vcat(Float64.(y),Float64.(p),[1.,2.,2.,32.,91.,18.]);times=Float64[];calls=Int[];ok=true;checked=0;ma=0.;mr=0.
  for xmax in (1,2)
   elapsed,result,metrics=solve(xmax);push!(times,elapsed);push!(calls,Int(metrics[5]));append!(values,result)
   ok &= metrics[1]==0;checked+=Int(metrics[2]);ma=max(ma,metrics[3]);mr=max(mr,metrics[4])
  end
  for (a,n)in ((DY,91),(DP,18),(DF,91),(DC,3))
   guards=Array(a)[n+1:end];append!(values,Float64.(guards));ok &= all(==(12345.5f0),guards);checked+=length(guards)
  end
  path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,values))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"relative_error_denominator_floor\":1e-12,\"timing_region\":\"sum_original_gpu_evaluator_times_full_RKF_xmax1_and2\",\"jit_excluded\":true,\"internal_repeats\":1,\"evaluator_calls\":[%s],\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",checked,ma,mr,join(calls,","),join(times,","),path,length(values));flush(stdout);ok||error("Fullmyocyteequations/hostsolver/CPUprecisioncrosscheckfailed")
 end
end
