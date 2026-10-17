using CUDA
using Printf
function rain_histories()
    fixed=([0.,1.],[0.,1.,0.],[0.,3.,1.,2.,0.],[0.,0.,1.,1.,0.,0.],[2.,2.,2.,2.],[-3.,-2.,0.,1.,4.]);history=Float64[];offsets=Int32[0]
    for id in 0:256
        if id<6;h=fixed[id+1]
        else
            h=Float64[]
            for j in 0:16+(id*13)%113-1;push!(h,j>0&&j%7==0 ? h[end] : Float64((j*17+id*7)%37-18)/8.0+Float64((id%3)-1)*j/64.0);end
        end
        append!(history,h);push!(offsets,Int32(length(history)))
    end
    history,offsets
end
function list_oracle(history)
    clean=Float64[];for v in history;(isempty(clean)||v!=clean[end])&&push!(clean,v);end
    extrema=length(clean)==1 ? [clean[1],clean[1]] : vcat(clean[1],[clean[j] for j in 2:length(clean)-1 if (clean[j]-clean[j-1])*(clean[j+1]-clean[j])<0],clean[end]);stack=Float64[];cycles=NTuple{3,Float64}[]
    for v in extrema
        push!(stack,v)
        while length(stack)>=3&&abs(stack[end]-stack[end-1])>=abs(stack[end-1]-stack[end-2])
            range=abs(stack[end-1]-stack[end-2]);mean=0.5*(stack[end-1]+stack[end-2])
            if length(stack)==3;push!(cycles,(.5,range,mean));deleteat!(stack,1)
            else;push!(cycles,(1.,range,mean));deleteat!(stack,length(stack)-2:length(stack)-1);end
        end
    end
    for j in 1:length(stack)-1;push!(cycles,(.5,abs(stack[j]-stack[j+1]),.5*(stack[j]+stack[j+1])));end
    2sum(c[1] for c in cycles)==length(extrema)-1||error("half/fullcyclebalance");all(c[1] in (.5,1.)&&c[2]>=0 for c in cycles)||error("cycleinvariants");cycles,length(extrema)
end
function study_main(prepare,args=ARGS)
    repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positiverepeat");history,offsets=rain_histories();L=length(history);n=257;gold=fill(-123.25,3L);counts=zeros(Int32,n);valid=0
    for i in 1:n
        base=Int(offsets[i]);cycles,extlen=list_oracle(history[base+1:Int(offsets[i+1])]);counts[i]=length(cycles);valid+=length(cycles)
        i==2&&cycles!=[(.5,1.,.5),(.5,1.,.5)]&&error("knowntrianglehistory")
        i==3&&cycles!=[(1.,1.,1.5),(.5,3.,1.5),(.5,3.,1.5)]&&error("knownnestedcyclehistory")
        for (j,cycle) in enumerate(cycles);gold[3*(base+j-1)+1:3*(base+j-1)+3]=collect(cycle);end
    end
    dh,do_=CuArray(history),CuArray(offsets);extrema=CUDA.zeros(Float64,L);points=CUDA.zeros(Int32,L);results=CUDA.fill(-123.25,3L);lengths=CUDA.fill(Int32(-1),n);launch! = prepare(dh,do_,extrema,points,results,lengths,n);launch!();CUDA.synchronize();prefix=vcat(history,Float64.(offsets));allok=true
    for trial in 1:5
        CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds=(time_ns()-t0)*1e-9/repeat;got=Array(results);gl=Array(lengths);ok=got==gold&&gl==counts&&all(isfinite,got);allok &=ok;output=vcat(prefix,Float64.(gl),got);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"rainflow_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"valid_cycles\":%d,\"computed_output_values\":%d,\"defined_padding_values\":%d,\"input_prefix_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"complete257_variable_length_histories_full_rainflow_counts_ranges_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),valid,n+3valid,3*(L-valid),length(prefix),repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
