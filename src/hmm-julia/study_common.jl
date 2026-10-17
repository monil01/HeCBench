using CUDA
using Printf
function make_hmm(kind)
    n=kind<2 ? 3 : 257;ne=kind<2 ? 3 : 17;nt=kind<2 ? 5 : 19
    init=Float32[kind==1 ? 1f0/16f0 : kind==0 ? Float32(i+1)/16f0 : Float32(i%7+1)/32f0 for i in 0:n-1]
    state=Float32[kind==1 ? 1f0/8f0 : kind==0 ? Float32((d*3+q*5)%7+1)/16f0 : Float32((d*13+q*7)%31+1)/64f0 for d in 0:n-1 for q in 0:n-1]
    emit=Float32[kind==1 ? 1f0/32f0 : kind==0 ? Float32((o*2+s*3)%5+1)/32f0 : Float32((o*5+s*11)%23+1)/128f0 for o in 0:ne-1 for s in 0:n-1]
    obs=Int32[kind==0 ? (t*2)%ne : kind==1 ? t%ne : (t*7)%ne for t in 0:nt-1]
    init,state,emit,obs,n,ne,nt
end
function score_oracle(init,state,emit,obs,n,nt)
    old=Float64.(init);ptr=fill(Int32(-1),(nt-1)*n)
    for t in 1:nt-1
        new=zeros(Float64,n)
        for dest in 0:n-1
            best=-Inf;which=Int32(-1)
            for prev in 0:n-1
                v=old[prev+1]+Float64(state[dest*n+prev+1])
                if v>best;best=v;which=Int32(prev);end
            end
            new[dest+1]=best+Float64(emit[Int(obs[t+1])*n+dest+1]);ptr[(t-1)*n+dest+1]=which
        end
        old=new
    end
    score=maximum(old);last=Int32(findfirst(==(score),old)-1);path=Vector{Int32}(undef,nt);path[end]=last
    for t in nt-2:-1:0;path[t+1]=ptr[t*n+Int(path[t+2])+1];end
    old,ptr,path,score
end
function exhaustive_oracle(init,state,emit,obs,n,nt)
    best=-Inf;pathbest=Int32[]
    for code in 0:n^nt-1
        path=Int32.(digits(code;base=n,pad=nt));score=Float64(init[path[1]+1])
        for t in 1:nt-1;score+=Float64(state[Int(path[t+1])*n+Int(path[t])+1])+Float64(emit[Int(obs[t+1])*n+Int(path[t+1])+1]);end
        if score>best;best=score;pathbest=copy(path);end
    end
    pathbest,best
end
function study_main(prepare,args=ARGS)
    isempty(args)||args==["1"]||error("onecompleteforwardsequencepertrial");fixtures=[];prefix=Float64[]
    for kind in 0:2
        init,state,emit,obs,n,ne,nt=make_hmm(kind);gold=score_oracle(init,state,emit,obs,n,nt)
        if kind<2;path,score=exhaustive_oracle(init,state,emit,obs,n,nt);path==gold[3]&&score==gold[4]||error("exhaustivepathoracle");end
        kind==1&&(!all(==(0),gold[3]))&&error("alltieearliestpath");for x in (init,state,emit,obs);append!(prefix,Float64.(x));end
        old,new=CuArray(init),CUDA.zeros(Float32,n);ds,de,do_=CuArray(state),CuArray(emit),CuArray(obs);dp=CUDA.fill(Int32(-1),(nt-1)*n);launch! = prepare(old,ds,de,do_,new,dp,n)
        for t in 1:nt-1;launch!(t);copyto!(old,new);end
        push!(fixtures,(init,old,new,dp,launch!,gold,n,nt))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=copy(prefix);seconds=0.0;ok=true
        for (init,old,new,dp,launch!,gold,n,nt) in fixtures
            copyto!(old,init);CUDA.synchronize();t0=time_ns()
            for t in 1:nt-1;launch!(t);copyto!(old,new);end
            CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9;terminal=Array(new);ptr=Array(dp);best=maximum(terminal);last=Int32(findfirst(==(best),terminal)-1);path=Vector{Int32}(undef,nt);path[end]=last
            for t in nt-2:-1:0;path[t+1]=ptr[t*n+Int(path[t+2])+1];end
            ok &=Float64.(terminal)==gold[1]&&ptr==gold[2]&&path==gold[3]&&Float64(best)==gold[4];append!(output,Float64.(terminal));append!(output,Float64.(ptr));append!(output,Float64.(path));push!(output,Float64(best))
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"hmm_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"computed_output_values\":4945,\"input_prefix_values\":70746,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum_three_complete_positive_maxplus_forward_sequences_including_D2D_copy_eachstep\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
