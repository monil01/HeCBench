using CUDA
using Printf
const FW_INF=UInt32(1000000)
function graph_fixture(kind,n)
    if kind==0
        ccall(:srand,Cvoid,(Cuint,),2);d=UInt32[UInt32(ccall(:rand,Cint,())%201) for _ in 1:n*n]
    else
        d=fill(FW_INF,n*n)
        for i in 0:n-1,j in 0:n-1
            if i÷32==j÷32&&i!=j&&((j-i)%32 in (1,3,7)||((i*17+j*13)%11==0))
                d[i*n+j+1]=UInt32((i*7+j*11)%31+1)
            end
        end
    end
    for i in 0:n-1;d[i*n+i+1]=0;end
    d
end
function floyd_cpu(input,n)
    d=copy(input);path=UInt32[UInt32(i) for i in 0:n-1 for _ in 0:n-1]
    for k in 0:n-1,y in 0:n-1,x in 0:n-1
        c=d[y*n+k+1]+d[k*n+x+1]
        if c<d[y*n+x+1];d[y*n+x+1]=c;path[y*n+x+1]=UInt32(k);end
    end
    d,path
end
function dijkstra_all(input,n)
    out=Vector{UInt32}(undef,n*n)
    for source in 0:n-1
        d=fill(Int64(FW_INF),n);d[source+1]=0;used=falses(n)
        for _ in 1:n
            best=Int64(FW_INF);node=-1
            for i in 0:n-1;if !used[i+1]&&d[i+1]<best;best=d[i+1];node=i;end;end
            node<0&&break;used[node+1]=true
            for dest in 0:n-1;d[dest+1]=min(d[dest+1],best+Int64(input[node*n+dest+1]));end
        end
        out[source*n+1:(source+1)*n]=UInt32.(d)
    end
    out
end
function route_check(input,d,path,n)
    function route(i,j,depth)
        depth>2n&&error("intermediatepathcycle");i==j&&return Int[i]
        k=Int(path[i*n+j+1]);k==i&&return Int[i,j]
        k==j&&error("invalidendpointintermediate");left=route(i,k,depth+1);right=route(k,j,depth+1);vcat(left,right[2:end])
    end
    for i in 0:n-1,j in 0:n-1
        if d[i*n+j+1]<FW_INF
            p=route(i,j,0);cost=sum(Int64(input[p[q]*n+p[q+1]+1]) for q in 1:length(p)-1;init=Int64(0));cost==Int64(d[i*n+j+1])||return false
        end
    end
    all(d[i*n+i+1]==0 for i in 0:n-1)&&all(Int64(d[i*n+j+1])<=Int64(d[i*n+k+1])+Int64(d[k*n+j+1]) for i in 0:n-1 for j in 0:n-1 for k in 0:n-1)
end
function study_main(prepare,args=ARGS)
    isempty(args)||args==["1"]||error("onecompleteAPSPpertrial");n=64;fixtures=[];prefix=UInt32[]
    for kind in 0:1
        input=graph_fixture(kind,n);initial=UInt32[UInt32(i) for i in 0:n-1 for _ in 0:n-1];expected,path=floyd_cpu(input,n);expected==dijkstra_all(input,n)||error("independentDijkstradistance");route_check(input,expected,path,n)||error("CPUroutewitness");append!(prefix,input);append!(prefix,initial);d,dp=CuArray(input),CuArray(initial);launch! = prepare(d,dp,n);for k in 0:n-1;launch!(k);end;push!(fixtures,(input,initial,d,dp,launch!,expected,path))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=copy(prefix);seconds=0.0;ok=true
        for (input,initial,d,dp,launch!,expected,path) in fixtures
            copyto!(d,input);copyto!(dp,initial);CUDA.synchronize();t0=time_ns();for k in 0:n-1;launch!(k);end;CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9;got=Array(d);gp=Array(dp);ok &=got==expected&&gp==path&&route_check(input,got,gp,n);append!(output,got);append!(output,gp)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"floyd_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"computed_output_values\":16384,\"input_prefix_values\":16384,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum_two_complete64pass_APSP_distance_and_intermediate_matrices\",\"jit_excluded\":true,\"internal_repeats\":1,\"route_reconstruction_pass\":%s,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),string(ok),file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
