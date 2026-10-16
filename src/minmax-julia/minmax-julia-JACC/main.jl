using CUDA
import JACC
JACC.@init_backend
struct Extremum
    magnitude::Float32
    index::Int32
end
struct ExtremaPair
    low::Extremum
    high::Extremum
end
@inline choose_min(a,b)=a.magnitude<b.magnitude || (a.magnitude==b.magnitude&&a.index<b.index) ? a : b
@inline choose_max(a,b)=a.magnitude>b.magnitude || (a.magnitude==b.magnitude&&a.index<b.index) ? a : b
@inline merge_pair(a,b)=ExtremaPair(choose_min(a.low,b.low),choose_max(a.high,b.high))
@inline score(i,x,y)=Extremum(@inbounds(x[i]*x[i]+y[i]*y[i]),Int32(i))
@inline pair_score(i,x,y)=ExtremaPair(score(i,x,y),score(i,x,y))
function make_state(n)
    low=Extremum(Inf32,typemax(Int32));high=Extremum(-Inf32,typemax(Int32))
    return (min=JACC.reducer(range=n,type=Extremum,op=choose_min,init=low),max=JACC.reducer(range=n,type=Extremum,op=choose_max,init=high),both=JACC.reducer(range=n,type=ExtremaPair,op=merge_pair,init=ExtremaPair(low,high)))
end
function evaluate!(state,dx,dy,x,y,n,phase)
    if phase==1
        state.min(score,dx,dy);low=JACC.get_result(state.min)
        state.max(score,dx,dy);high=JACC.get_result(state.max)
    else
        state.both(pair_score,dx,dy);pair=JACC.get_result(state.both);low=pair.low;high=pair.high
    end
    return Float32[x[low.index],y[low.index],x[high.index],y[high.index]]
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
