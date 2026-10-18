
const BASE=dirname(@__FILE__);const G=32;const REPEATS=3
const CASE_DIMS=[(5,5,5,30,4,33),(2,3,2,2,4,5)]
function minisweep_study(factory)
 states=map(enumerate(CASE_DIMS))do (case,dims)
  nx,ny,nz,ne,nm,na=dims;ns=nx*ny*nz*ne*nm*4;am=nm*na*8;fxy=nx*ny*ne*na*4*8;fxz=nx*nz*ne*na*4*8;fyz=ny*nz*ne*na*4*8
  input=collect(reinterpret(Float64,read(joinpath(BASE,"study_case$(case)_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(BASE,"oracle_case$(case)_f64.bin"))))
  lens=(am,am,ns,ns,fxy,fxz,fyz,fxy);arrays=map(lens)do n;CuArray(fill(12345.5,n+G));end
  k=1
  for idx=1:3;n=lens[idx];copyto!(arrays[idx],1,input,k,n);k+=n;end
  d=Int32.(dims);launch=factory(d,arrays)
  function execute(iterations)
   captured=Vector{Float64}(undef,ns*iterations);elapsed=0.
   for r=1:iterations
    CUDA.device_synchronize(blocking=true);t=time_ns()
    CUDA.memset(reinterpret(CUDA.CuPtr{UInt8},pointer(arrays[4])),UInt8(0),ns*8)
    launch();CUDA.device_synchronize(blocking=true);elapsed+=(time_ns()-t)*1e-9
    GC.@preserve captured unsafe_copyto!(pointer(captured,(r-1)*ns+1),pointer(arrays[4]),ns)
   end
   elapsed/iterations,captured
  end
  execute(1);(dims,lens,arrays,oracle,execute)
 end
 for trial=1:5
  prefix=Float64[];values=Float64[];guards=Float64[];componenttimes=Float64[];ma=0.;mr=0.;ok=true;mismatches=0
  for (case,(dims,lens,arrays,oracle,execute))in enumerate(states)
   elapsed,captured=execute(REPEATS);push!(componenttimes,elapsed)
   actual=[Array(a) for a in arrays]
   for idx=1:3;append!(prefix,actual[idx][1:lens[idx]]);end
   nx,ny,nz,ne,nm,na=dims;append!(prefix,[nx,ny,nz,ne,na,4,4,8,REPEATS,1,1,1,1,case,1,0])
   v=copy(captured)
   for idx=5:8;append!(v,actual[idx][1:lens[idx]]);end
   append!(values,v);for idx=1:8;append!(guards,actual[idx][lens[idx]+1:end]);end
   for i in eachindex(v)
    diff=abs(v[i]-oracle[i]);ma=max(ma,diff);mr=max(mr,diff/max(abs(oracle[i]),1e-12));pass=isfinite(v[i])&&diff<=1e-8+1e-10*abs(oracle[i]);ok &= pass;mismatches+=!pass
   end
  end
  ok &= all(==(12345.5),guards);out=vcat(prefix,values,guards);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR","/tmp"),"trial$(trial).bin");write(path,reinterpret(UInt8,out))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":3356704,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"relative_error_denominator_floor\":1e-12,\"mismatch_values\":%d,\"timing_region\":\"sum2_mean_memset_three_face_inits_persistent_sweep_complete\",\"jit_excluded\":true,\"internal_repeats\":3,\"untimed_state_readbacks_per_case\":3,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(componenttimes),ok ? "true" : "false",ma,mr,mismatches,join(componenttimes,","),path,length(out));flush(stdout);ok||error("Fulltransportstate/face/scratch/guardcheckfailed")
 end
end
