using CUDA
import JACC
JACC.@init_backend
float_to_uchar(x::Float32)=UInt8(unsafe_trunc(Int32,x)&Int32(0xff))
function blockaccess_kernel!(tid,output,input,n,total_threads,::Val{PHASE}) where PHASE
    if PHASE==1
        i=tid
        while i<=n
            @inbounds output[i]=float_to_uchar(input[i]);i+=total_threads
        end
    else
        base=(tid-1)*4
        while base<n
            for item in 1:4
                if base+item<=n
                    @inbounds output[base+item]=float_to_uchar(input[base+item])
                end
            end
            base+=total_threads*4
        end
    end
end
function launch!(out,input,n,blocks,phase)
    spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=0,sync=false)
    JACC.parallel_for(spec,blocks*256,blockaccess_kernel!,out,input,Int32(n),Int32(blocks*256),Val(phase))
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 65536
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    blocks=cld(n,1024);input=Float32[Float32(mod((i-1)*37,1024)-512)/4f0 for i in 1:n]
    expected=float_to_uchar.(input);dx=CuArray(input);dy=CUDA.zeros(UInt8,n)
    for phase in 1:2
        launch!(dy,dx,n,blocks,phase)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=UInt32[];correct=true
        for phase in 1:2
            fill!(dy,UInt8(0));CUDA.synchronize();start=time_ns()
            for _ in 1:repeat
                launch!(dy,dx,n,blocks,phase)
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            result=Array(dy);correct &= result==expected;append!(values,UInt32.(result))
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":",2n,",\"max_abs_error\":0.0,\"timing_region\":\"sum_two_float_to_byte_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"uint32\",\"count\":",2n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))
