using CUDA,JACC
@inline physical(i,::Val{PADDED}) where PADDED = i+(PADDED ? (i>>5) : zero(i))+one(i)
function scan_kernel!(index,output,input,nblocks,::Val{N},mode::Val{PADDED},workspace) where {N,PADDED}
 @inbounds begin
 t=Int32(mod(index-1,div(N,2)));bid=Int32(div(index-1,div(N,2)))
 while bid<nblocks
  shared=JACC.shared(workspace)
  a=PADDED ? t : t+t;b=PADDED ? t+Int32(div(N,2)) : t+t+Int32(1)
  shared[physical(a,mode)]=input[bid*Int32(N)+a+Int32(1)];shared[physical(b,mode)]=input[bid*Int32(N)+b+Int32(1)]
  offset=Int32(1);d=Int32(div(N,2))
  while d>0
   JACC.sync_workgroup()
   if t<d
    ai=offset*(t+t+Int32(1))-Int32(1);bi=offset*(t+t+Int32(2))-Int32(1)
    shared[physical(bi,mode)]+=shared[physical(ai,mode)]
   end
   offset*=Int32(2);d>>=1
  end
  if t==0;shared[physical(Int32(N-1),mode)]=zero(eltype(output));end
  d=Int32(1)
  while d<N
   offset>>=1
   JACC.sync_workgroup()
   if t<d
    ai=physical(offset*(t+t+Int32(1))-Int32(1),mode);bi=physical(offset*(t+t+Int32(2))-Int32(1),mode)
    temp=shared[ai];shared[ai]=shared[bi];shared[bi]+=temp
   end
   d<<=1
  end
  if PADDED
   JACC.sync_workgroup()
  end
  output[bid*Int32(N)+a+Int32(1)]=shared[physical(a,mode)];output[bid*Int32(N)+b+Int32(1)]=shared[physical(b,mode)]
  bid+=Int32(1280)
 end
 end
end
function launch(output,input,nblocks,::Val{N},mode::Val{PADDED},workspace) where {N,PADDED}
 JACC.parallel_for(JACC.launch_spec(threads=div(N,2),blocks=1280,shmem_size=sizeof(eltype(output))*(PADDED ? 2N : N),sync=false),1280*div(N,2),scan_kernel!,output,input,Int32(nblocks),Val(N),mode,workspace)
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
  input=T[mod(i*17,5)+1 for i in 0:n-1];out=CUDA.zeros(T,n);di=CuArray(input);workspace=CUDA.zeros(T,padded ? 2N : N)
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
