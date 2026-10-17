using CUDA
using Printf
function fixture(batch,n)
    xyz=Int32[]
    for row in 0:batch-1,i in 0:n-1
        x=(i*7+row*3)%16-8;y=(i÷16*5+row)%16-8;z=(i÷128*3+i%3)%16-8
        if i==0;x=0;y=0;z=0;elseif i==1;x=16;y=16;z=0;elseif i==n-1;x=100;y=100;z=100;end
        append!(xyz,Int32[x,y,z])
    end
    xyz
end
function sorted_oracle(xyz,b,n,ranks)
    out=Int32[]
    for row in 0:b-1,i in 0:n-1
        pos=3*(row*n+i);x,y,z=xyz[pos+1:pos+3];buckets=[Tuple{Int64,Int}[] for _ in 1:8]
        for j in 0:n-1
            i==j&&continue;q=3*(row*n+j);tx,ty,tz=xyz[q+1:q+3];dist=(Int64(x)-tx)^2+(Int64(y)-ty)^2+(Int64(z)-tz)^2
            # Native strict insertion into radius-initialized slots excludes equality.
            dist>=512&&continue;octant=(tx>x ? 4 : 0)+(ty>y ? 2 : 0)+(tz>z ? 1 : 0);push!(buckets[octant+1],(dist,j))
        end
        for bucket in buckets
            sort!(bucket)
            for k in 1:ranks;push!(out,k<=length(bucket) ? Int32(bucket[k][2]) : Int32(i));end
        end
    end
    out
end
function study_main(prepare,args=ARGS)
    b=2;n=520;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");xyz=fixture(b,n);dx=CuArray(xyz);fixtures=[]
    for ranks in (1,2,4)
        expected=sorted_oracle(xyz,b,n,ranks);out=CUDA.zeros(Int32,length(expected));launch! = prepare(dx,out,b,n,ranks);launch!();push!(fixtures,(out,launch!,expected))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Int32[];seconds=0.0;ok=true
        for (out,launch!,expected) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9/repeat;got=Array(out);ok &=got==expected;append!(output,got)
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"s8n_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum_all_three_complete_octant_neighbor_selections\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
