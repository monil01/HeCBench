using Printf
const FW_GUARD=Int32(1234567890)
const FW_INF=Int32(1073741823)
function f_read_cases()
    open(joinpath(@__DIR__,"fixtures.bin"),"r") do io
        read(io,UInt32)==7 || error("fixture count")
        cases=NamedTuple[]
        for _ in 1:7
            n=read(io,Int32);e=read(io,Int32);u=read(io,Int32);r=read(io,Int32)
            1<=n<=u<=1088 && u%64==0 && e>0 && r==3 || error("fixture geometry")
            idx=read!(io,Vector{Int32}(undef,n+1));nei=read!(io,Vector{Int32}(undef,e));wt=read!(io,Vector{Int32}(undef,e));oracle=read!(io,Vector{Int32}(undef,u*u))
            idx[1]==0 && idx[end]==e || error("CSRbounds")
            push!(cases,(n=n,e=e,u=u,r=r,idx=idx,nei=nei,wt=wt,oracle=oracle))
        end
        eof(io) || error("fixture trailingdata");cases
    end
end
function f_guarded(v,alloc)
    a=alloc(vcat(fill(FW_GUARD,64),v,fill(FW_GUARD,64)))
    UInt(pointer(a))%256==0 || error("allocationalignment")
    (device=a,payload=view(a,65:length(v)+64))
end
function f_read(g)
    a=Array(g.device)
    all(==(FW_GUARD),view(a,1:64)) && all(==(FW_GUARD),view(a,length(a)-63:length(a))) || error("exact guard")
    a
end
function f_check(g,want,label)
    a=f_read(g)
    if view(a,65:length(a)-64)!=want
        for k in eachindex(want)
            if a[k+64]!=want[k];error("$label at$k actual$(a[k+64]) expected$(want[k])");end
        end
        error("$label length")
    end
    nothing
end
function f_execute(trial,cases,alloc,initialize,compute,observe)
    prefix=Int32[];post=Int32[];seconds=0.0
    for (ci,c) in enumerate(cases)
        initial=fill(FW_INF,c.u*c.u)
        gi=f_guarded(c.idx,alloc);gn=f_guarded(c.nei,alloc);gw=f_guarded(c.wt,alloc)
        a=f_guarded(initial,alloc);kr=f_guarded(initial,alloc);kc=f_guarded(initial,alloc);param=f_guarded(zeros(Int32,4),alloc)
        buffers=(gi,gn,gw,a,kr,kc,param)
        observe(param.payload,c.n,c.e,c.u,c.r);CUDA.device_synchronize()
        f_check(gi,c.idx,"actualCSRidx");f_check(gn,c.nei,"actualCSRnei");f_check(gw,c.wt,"actualCSRwt");f_check(param,Int32[c.n,c.e,c.u,c.r],"actualparameters")
        for g in (a,kr,kc);f_check(g,initial,"actualinitialmatrices");end
        if trial>0;for g in buffers;append!(prefix,f_read(g));end;end
        CUDA.device_synchronize();t=time_ns()
        for repeat in 1:c.r;initialize(c.n,gi.payload,gn.payload,gw.payload,a.payload,c.u);end
        CUDA.device_synchronize();init=(time_ns()-t)*1e-9/c.r
        t=time_ns()
        for repeat in 1:c.r;compute(a.payload,c.u,kr.payload,kc.payload);end
        CUDA.device_synchronize();comptime=(time_ns()-t)*1e-9/c.r;seconds+=init+comptime
        f_check(gi,c.idx,"readonlyCSRidx");f_check(gn,c.nei,"readonlyCSRnei");f_check(gw,c.wt,"readonlyCSRwt");f_check(param,Int32[c.n,c.e,c.u,c.r],"readonlyparameters")
        f_check(a,c.oracle,"independentDijkstraAPSP");f_check(kr,c.oracle,"independentrowworkspace")
        cols=similar(c.oracle)
        for i in 0:c.u-1,j in 0:c.u-1
            cols[i*c.u+j+1]=c.oracle[((i÷64)*64+j%64)*c.u+(j÷64)*64+i%64+1]
        end
        f_check(kc,cols,"independentblocktransposecolumnworkspace")
        if trial>0
            for g in buffers;append!(post,f_read(g));end
            @printf("STUDY_REGION trial%d case%d init_mean_s%.12g compute_mean_s%.12g repeat%d\n",trial,ci-1,init,comptime,c.r)
        end
        for g in buffers;CUDA.unsafe_free!(g.device);end
    end
    if trial>0
        length(prefix)==5062239 && length(post)==5062239 || error("artifactlayout")
        name="floydwarshall2_trial$(trial).bin"
        open(name,"w") do io;write(io,prefix);write(io,post);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":10124478,\"max_abs_error\":0,\"max_rel_error\":0,\"jit_excluded\":true,\"timing_region\":\"sum7_source_init_and_compute_repeat3_region_means\",\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":10124478,\"byte_order\":\"little\"}}\n",trial,seconds,name)
        flush(stdout)
    end
end
function floyd_study(alloc,initialize,compute,observe)
    CUDA.allowscalar(false);cases=f_read_cases()
    println("\nECL_APSP_STUDY named7graphs original64tile1024threads sourceR3closedstatepipeline");flush(stdout)
    f_execute(0,cases,alloc,initialize,compute,observe)
    for t in 1:5;f_execute(t,cases,alloc,initialize,compute,observe);GC.gc();end
end
