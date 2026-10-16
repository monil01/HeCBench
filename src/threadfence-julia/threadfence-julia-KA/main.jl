using CUDA,KernelAbstractions,Atomix
@inline function volatile_load(array,i)
    ptr=pointer(array,i)
    CUDA.CUDACore.@asmcall("ld.volatile.global.f32 \$0, [\$1];", "=f,l,~{memory}", true,
        Float32,Tuple{Core.LLVMPtr{Float32,1}},ptr)
end
@inline function volatile_store!(array,i,value)
    ptr=pointer(array,i)
    CUDA.CUDACore.@asmcall("st.volatile.global.f32 [\$0], \$1;", "l,f,~{memory}", true,
        Cvoid,Tuple{Core.LLVMPtr{Float32,1},Float32},ptr,value)
end
@kernel function sum_kernel!(input,output,counter,n,grids,workspace)
    lid=@index(Local,Linear);group=@index(Group,Linear);gid=@index(Global,Linear)
    shared=@localmem Float32 (2,)
    if lid==1;shared[1]=0f0;shared[2]=0f0;end
    @synchronize
    if gid<=n;Atomix.@atomic shared[1]+=input[gid];end
    @synchronize
    if lid==1
        volatile_store!(output,group,shared[1]);CUDA.threadfence()
        previous=first(Atomix.@atomic counter[1]+UInt32(1))
        shared[2]=previous==UInt32(grids-1) ? 1f0 : 0f0
    end
    @synchronize
    if shared[2]==1f0
        if lid==1;shared[1]=0f0;end
        @synchronize
        i=lid
        while i<=grids
            value=volatile_load(output,i);Atomix.@atomic shared[1]+=value;i+=256
        end
        @synchronize
        if lid==1
            volatile_store!(output,1,shared[1]);counter[1]=0
        end
    end
end
function launch(input,output,counter,n,grids)
    sum_kernel!(get_backend(input),256)(input,output,counter,n,grids,nothing;ndrange=grids*256)
end
function main(args)
    repeat=length(args)>0 ? parse(Int,args[1]) : 10
    n=length(args)>1 ? parse(Int,args[2]) : 4096
    grids=cld(n,256);input=CUDA.fill(-1f0,n);output=CUDA.zeros(Float32,grids);counter=CUDA.zeros(UInt32,1)
    launch(input,output,counter,n,grids);CUDA.synchronize()
    expected=fill(-256f0,grids);expected[end]=-Float32(n-(grids-1)*256);expected[1]=-Float32(n);push!(expected,0f0)
    for trial in 1:5
        fill!(output,0f0);fill!(counter,0);CUDA.synchronize();start=time_ns()
        for _ in 1:repeat
            launch(input,output,counter,n,grids)
        end
        CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
        result=vcat(Array(output),Float32.(Array(counter)));error=maximum(abs.(result.-expected));correct=result==expected
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,result);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
         ",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"synchronized_complete_reduction_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))
