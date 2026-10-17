using CUDA,KernelAbstractions
using Printf

const THREADS_PER_BLOCK = 256

@kernel function reduce_kernel!(input, partial, input_size::Int32)
    bid=@index(Group,Linear)
    tid=@index(Local,Linear)
    global_tid = Int32((bid - 1) * 256 + tid)
    stride = Int32(256 * 512)
    acc = 0.0
    i = global_tid
    while i <= input_size
        @inbounds acc += Float64(input[i])
        i += stride
    end
    sh = @localmem Float64 (256,)
    sh[tid] = acc
    @synchronize
    offset = 256 ÷ 2
    while offset > 0
        if tid <= offset
            sh[tid] += sh[tid + offset]
        end
        @synchronize
        offset ÷= 2
    end
    if tid == 1
        @inbounds partial[bid] = sh[1]
    end
end

@kernel function reduce_final_kernel!(partial, result, nblocks::Int32)
    tid=@index(Local,Linear)
    acc = 0.0
    i = tid
    while i <= nblocks
        @inbounds acc += partial[i]
        i += 256
    end
    sh = @localmem Float64 (256,)
    sh[tid] = acc
    @synchronize
    offset = 256 ÷ 2
    while offset > 0
        if tid <= offset
            sh[tid] += sh[tid + offset]
        end
        @synchronize
        offset ÷= 2
    end
    if tid == 1
        result[1] = sh[1]
    end
end

function launch_reductions(input,partial,result,n,nblocks)
 reduce_kernel!(CUDA.CUDABackend(),256)(input,partial,Int32(n);ndrange=nblocks*256)
 reduce_final_kernel!(CUDA.CUDABackend(),256)(partial,result,Int32(nblocks);ndrange=256)
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 262144;inner=length(args)>1 ? parse(Int,args[2]) : 100;nblocks=512
 input=Float32[Float32(mod(i*17+3,256))/Float32(2147483647) for i in 0:n-1];result_h=zeros(Float64,1);CUDA.pin(input);CUDA.pin(result_h)
 input_d=CUDA.zeros(Float32,n);partial=CUDA.zeros(Float64,nblocks);result=CUDA.zeros(Float64,1);stream=CuStream();CUDA.synchronize()
 rawinput=CUDA.cudaconvert(input_d);rawpartial=CUDA.cudaconvert(partial);rawresult=CUDA.cudaconvert(result)
 expectedpartial=zeros(Float64,nblocks)
 for i in 0:n-1;expectedpartial[div(mod(i,nblocks*256),256)+1]+=Float64(input[i+1]);end
 expectedone=vcat(expectedpartial,sum(Float64,input));expected=vcat(expectedone,expectedone)
 function segment!()
  unsafe_copyto!(pointer(input_d),pointer(input),n;async=true)
  for _ in 1:inner
   fill!(partial,0.);launch_reductions(rawinput,rawpartial,rawresult,n,nblocks)
  end
  unsafe_copyto!(pointer(result_h),pointer(result),1;async=true)
 end
 CUDA.stream!(segment!,stream);CUDA.synchronize(stream)
 graph=CUDA.stream!(stream) do
  CUDA.capture(segment!)
 end
 executable=CUDA.instantiate(graph);CUDA.launch(executable,stream);CUDA.synchronize(stream)
 owners=(input,input_d,partial,result,result_h)
 GC.@preserve owners graph executable begin
  for trial in 1:5
   values=Float64[];times=Float64[]
   for mode in 1:2
    CUDA.synchronize(stream);start=time_ns()
    if mode==1;CUDA.launch(executable,stream);else;CUDA.stream!(segment!,stream);end
    CUDA.synchronize(stream);push!(times,(time_ns()-start)*1e-9)
    hpartial=zeros(Float64,nblocks);unsafe_copyto!(pointer(hpartial),pointer(partial),nblocks;async=false);append!(values,hpartial);append!(values,result_h)
   end
   emit(trial,sum(times),values,expected,inner;atol=1e-6,region="graph_and_stream_segment_total_including_h2d_d2h",case_times=times)
  end
 end
 finalize(executable);finalize(graph);finalize(stream)
end
main(ARGS)
