using CUDA
const ROW_SIZE=17
@inline f(x)=exp(x)*sin(x)
function romberg_kernel!(result,a,b,nwg,workspace)
 tid=Int(threadIdx().x)-1;bid=Int(blockIdx().x)-1
 shared=CuStaticSharedArray(Float64,544)
 diff=(b-a)/Float64(nwg);hi=a+Float64(bid+1)*diff;lo=a+Float64(bid)*diff
 max_eval=65536;step=(hi-lo)/Float64(max_eval)
 local_col=ntuple(_->0.,Val(17))
 k=tid==0 ? 32 : tid
 if tid==0;local_col=Base.setindex(local_col,f(lo)+f(hi),1);end
 while k<max_eval
  pos=17-trailing_zeros(k)
  local_col=Base.setindex(local_col,local_col[pos]+2.0*f(lo+step*Float64(k)),pos);k+=32
 end
 for i in 1:17;shared[17tid+i]=local_col[i];end
 CUDA.sync_threads()
 if tid<17
  total=0.
  for thread in 0:31;total+=shared[17thread+tid+1];end
  shared[tid+1]=total
 end
 CUDA.sync_threads()
 if tid==0
  table=local_col;table=Base.setindex(table,shared[1],1)
  for k in 2:17;table=Base.setindex(table,table[k-1]+shared[k],k);end
  for k in 1:17;table=Base.setindex(table,table[k]*(hi-lo)/Float64(1<<k),k);end
  for col in 0:15
   for row in 17:-1:col+2
    table=Base.setindex(table,table[row]+(table[row]-table[row-1])/Float64((1<<(2col+1))-1),row)
   end
  end
  result[bid+1]=table[17]
 end
 return nothing
end
function launch(result,nwg,workspace)
 @cuda threads=32 blocks=nwg romberg_kernel!(result,0.,15.,nwg,nothing)
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",(isempty(case_times) ? "[]" : string(case_times)),",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 nwg=length(args)>0 ? parse(Int,args[1]) : 4;repeat=length(args)>1 ? parse(Int,args[2]) : 3
 # Independent analytic antiderivative exp(x)*(sin(x)-cos(x))/2.
 integral(x)=exp(x)*(sin(x)-cos(x))/2
 expected=Float64[integral(15.0*i/nwg)-integral(15.0*(i-1)/nwg) for i in 1:nwg]
 result=CUDA.zeros(Float64,nwg);workspace=nothing;launch(result,nwg,workspace);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();start=time_ns();for _ in 1:repeat;launch(result,nwg,workspace);end;CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
  values=Array(result);abs(sum(values)-sum(expected))<=1e-7||Base.error("OriginalsumEPSfailed")
  emit(trial,seconds,values,expected,repeat;atol=1e-7)
 end
end
main(ARGS)
