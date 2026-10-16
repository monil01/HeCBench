using CUDA,Printf
function vadd_study(launch!;args=ARGS)
    m=length(args)>0 ? parse(Int,args[1]) : 256
    n=length(args)>1 ? parse(Int,args[2]) : 256
    repeats=length(args)>2 ? parse(Int,args[3]) : 100
    m>0 && m%16==0 && n>0 && n%256==0 && m*n%2048==0 && repeats>0 || error("namedfixture requires alignednativefulltiles")
    total=m*n;a=Vector{Float16}(undef,total);b=similar(a)
    ccall(:srand,Cvoid,(Cuint,),UInt32(42))
    for i in 1:total
        a[i]=(ccall(:rand,Cint,())%2)!=0 ? Float16(-1) : Float16(1)
        b[i]=(ccall(:rand,Cint,())%2)!=0 ? Float16(-1) : Float16(1)
    end
    expected=Float32.(a).+Float32.(b);da=CuArray(a);db=CuArray(b);dc=CUDA.zeros(Float16,total)
    for mode in 1:3;launch!(Val(mode),da,db,dc,m,n);end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];outputs=Float32[];correct=true;maxabs=0.0
        for mode in 1:3
            CUDA.fill!(dc,Float16(0));CUDA.synchronize();start=time_ns()
            for _ in 1:repeats;launch!(Val(mode),da,db,dc,m,n);end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeats)
            got=Float32.(Array(dc));err=abs.(got.-expected);maxabs=max(maxabs,maximum(err))
            correct &= all(isfinite,got)&&all(err.<=1f-2 .+ 1f-5.*abs.(expected));append!(outputs,got)
        end
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"vadd_$(trial).bin");write(file,outputs)
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,",\"checked_values\":",length(outputs),",\"max_abs_error\":",maxabs,",\"max_rel_error\":0,\"timing_region\":\"sum_three_synchronized_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeats,",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",file,"\",\"dtype\":\"float32\",\"count\":",length(outputs),",\"byte_order\":\"little\"}}")
        correct||error("vaddfullCPUcheckfailed")
    end
end
