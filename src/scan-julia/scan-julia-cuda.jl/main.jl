using CUDA
@inline physical(i,::Val{PADDED}) where PADDED = i+(PADDED ? div(i,32) : 0)+1
function scan_kernel!(output,input,nblocks,::Val{N},mode::Val{PADDED},workspace) where {N,PADDED}
 t=Int(threadIdx().x)-1;bid=Int(blockIdx().x)-1
 shared=CuStaticSharedArray(eltype(output),PADDED ? 2N : N)
 while bid<nblocks
  a=PADDED ? t : 2t;b=PADDED ? t+div(N,2) : 2t+1
  shared[physical(a,mode)]=input[bid*N+a+1];shared[physical(b,mode)]=input[bid*N+b+1]
  offset=1;d=div(N,2)
  while d>0
   CUDA.sync_threads()
   if t<d
    ai=offset*(2t+1)-1;bi=offset*(2t+2)-1
    shared[physical(bi,mode)]+=shared[physical(ai,mode)]
   end
   offset*=2;d=div(d,2)
  end
  if t==0;shared[physical(N-1,mode)]=zero(eltype(output));end
  d=1
  while d<N
   offset=div(offset,2)
   CUDA.sync_threads()
   if t<d
    ai=physical(offset*(2t+1)-1,mode);bi=physical(offset*(2t+2)-1,mode)
    temp=shared[ai];shared[ai]=shared[bi];shared[bi]+=temp
   end
   d*=2
  end
  if PADDED
   CUDA.sync_threads()
  end
  output[bid*N+a+1]=shared[physical(a,mode)];output[bid*N+b+1]=shared[physical(b,mode)]
  bid+=1280
 end
 return nothing
end
function launch(output,input,nblocks,::Val{N},mode,workspace) where N
 @cuda threads=div(N,2) blocks=1280 scan_kernel!(output,input,nblocks,Val(N),mode,nothing)
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function cpu_scan(input,N)
 output=similar(input)
 for base in 0:N:length(input)-1
  total=zero(eltype(input))
  for j in 1:N;output[base+j]=total;total+=input[base+j];end
 end
 output
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 4096;repeat=length(args)>1 ? parse(Int,args[2]) : 3
 cases=Any[];expected=Float64[]
 for N in (128,256,512,1024,2048),T in (Int8,Int16,Int32,Int64),padded in (false,true)
  input=T[mod(i*17,5)+1 for i in 0:n-1];out=CUDA.zeros(T,n);di=CuArray(input);workspace=nothing
  push!(cases,(out,di,div(n,N),Val(N),Val(padded),workspace));append!(expected,Float64.(cpu_scan(input,N)))
 end
 for case in cases;launch(case...);end;CUDA.synchronize()
 for trial in 1:5
  result=Float64[];times=Float64[]
  for case in cases
   CUDA.synchronize();start=time_ns();for _ in 1:repeat;launch(case...);end;CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);append!(result,Float64.(Array(case[1])))
  end
  emit(trial,sum(times),result,expected,repeat;case_times=times)
 end
end
main(ARGS)
