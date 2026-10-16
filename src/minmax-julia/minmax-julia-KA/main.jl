using CUDA,KernelAbstractions
struct Extremum
    magnitude::Float32
    index::Int32
end
@inline choose_min(a,b)=a.magnitude<b.magnitude || (a.magnitude==b.magnitude&&a.index<b.index) ? a : b
@inline choose_max(a,b)=a.magnitude>b.magnitude || (a.magnitude==b.magnitude&&a.index<b.index) ? a : b
@kernel function extrema_stage!(x,y,input_min,input_max,output_min,output_max,n,total_threads,::Val{FIRST},::Val{MODE}) where {FIRST,MODE}
    t=@index(Local,Linear);global_i=@index(Global,Linear);group=@index(Group,Linear)
    mins=@localmem Extremum (256,)
    maxs=@localmem Extremum (256,)
    low=Extremum(Inf32,typemax(Int32));high=Extremum(-Inf32,typemax(Int32));i=global_i
    while i<=n
        if FIRST
            @inbounds score=Extremum(x[i]*x[i]+y[i]*y[i],Int32(i))
            if MODE!==:max
                low=choose_min(low,score)
            end
            if MODE!==:min
                high=choose_max(high,score)
            end
        else
            if MODE!==:max
                @inbounds low=choose_min(low,input_min[i])
            end
            if MODE!==:min
                @inbounds high=choose_max(high,input_max[i])
            end
        end
        i+=total_threads
    end
    @inbounds mins[t]=low;maxs[t]=high
    @synchronize
    offset=128
    while offset>0
        if t<=offset
            if MODE!==:max
                @inbounds mins[t]=choose_min(mins[t],mins[t+offset])
            end
            if MODE!==:min
                @inbounds maxs[t]=choose_max(maxs[t],maxs[t+offset])
            end
        end
        @synchronize
        offset÷=2
    end
    if t==1
        if MODE!==:max
            @inbounds output_min[group]=mins[1]
        end
        if MODE!==:min
            @inbounds output_max[group]=maxs[1]
        end
    end
end
function make_state(n)
    blocks=min(cld(n,256),256)
    return (blocks=blocks,mins=CuArray{Extremum}(undef,blocks),maxs=CuArray{Extremum}(undef,blocks),low=CuArray{Extremum}(undef,1),high=CuArray{Extremum}(undef,1))
end
function reduce_extrema!(state,dx,dy,n,mode)
    backend=get_backend(dx);blocks=state.blocks
    extrema_stage!(backend,256)(dx,dy,state.mins,state.maxs,state.mins,state.maxs,Int32(n),Int32(blocks*256),Val(true),Val(mode);ndrange=blocks*256)
    extrema_stage!(backend,256)(dx,dy,state.mins,state.maxs,state.low,state.high,Int32(blocks),Int32(256),Val(false),Val(mode);ndrange=256)
    low=mode===:max ? 0 : Int(Array(state.low)[1].index)
    high=mode===:min ? 0 : Int(Array(state.high)[1].index)
    return low,high
end
function evaluate!(state,dx,dy,x,y,n,phase)
    if phase==1
        imin,_=reduce_extrema!(state,dx,dy,n,:min)
        _,imax=reduce_extrema!(state,dx,dy,n,:max)
    else
        imin,imax=reduce_extrema!(state,dx,dy,n,:both)
    end
    return Float32[x[imin],y[imin],x[imax],y[imax]]
end
function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 10
    x=Float32[mod((i-1)*37,n) for i in 1:n]
    y=Float32[Float32(mod((i-1)*19,n))/Float32(n) for i in 1:n]
    dx=CuArray(x);dy=CuArray(y);state=make_state(n)
    magnitudes=x.*x.+y.*y;imin=argmin(magnitudes);imax=argmax(magnitudes)
    expected=Float32[x[imin],y[imin],x[imax],y[imax]]
    for phase in 1:2
        evaluate!(state,dx,dy,x,y,n,phase)
    end
    CUDA.synchronize()
    for trial in 1:5
        times=Float64[];values=Float32[];correct=true
        for phase in 1:2
            CUDA.synchronize();start=time_ns();result=Float32[]
            for _ in 1:repeat
                result=evaluate!(state,dx,dy,x,y,n,phase)
            end
            CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
            correct &= result==expected;append!(values,result)
        end
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,values);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,
         ",\"checked_values\":8,\"max_abs_error\":0.0,\"timing_region\":\"sum_two_extrema_operations_host_results_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":8,\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))
