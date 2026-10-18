const QS_SG=UInt32(0xdeadbeef)
qs_bits(x::UInt32)=UInt64(x)
qs_bits(x::Float32)=UInt64(reinterpret(UInt32,x))
qs_bits(x::Float64)=reinterpret(UInt64,x)
qs_unbits(::Type{UInt32},x)=UInt32(x)
qs_unbits(::Type{Float32},x)=reinterpret(Float32,UInt32(x))
qs_unbits(::Type{Float64},x)=reinterpret(Float64,x)
qs_empty(::Type{T}) where T=QSWork{T}(0,0,zero(T),42)
qs_workguard(::Type{T},g) where T=QSWork{T}(QS_SG,QS_SG,g,QS_SG)
qs_blockguard(::Type{T},g) where T=QSBlock{T}(QS_SG,QS_SG,g,QS_SG,QS_SG)
function qs_makecase(T,n,rounded,initial,dn,allocate)
    ng=256÷sizeof(T);guard=T===UInt32 ? typemax(UInt32) : T(-1234567)
    draw=allocate(fill(guard,rounded+2ng));nraw=allocate(fill(guard,rounded+2ng))
    br=allocate(fill(qs_blockguard(T,guard),n+2));pr=allocate(fill(QS_SG,5(n+2)))
    nr=allocate(fill(qs_workguard(T,guard),2n+2));dr=allocate(fill(qs_workguard(T,guard),n+2))
    work=QSWork{T}[];done=QSWork{T}[];news=QSWork{T}[];blocks=QSBlock{T}[];parents=UInt32[];trace=Int[]
    for (v,cap) in ((work,n),(done,n),(news,2n),(blocks,n),(parents,5n),(trace,n))
        sizehint!(v,cap)
    end
    (T=T,n=n,rounded=rounded,ng=ng,guard=guard,initial=initial,hostdn=dn,host=copy(initial),draw=draw,nraw=nraw,
     braw=br,praw=pr,nrraw=nr,drraw=dr,d=view(draw,ng+1:ng+rounded),dn=view(nraw,ng+1:ng+rounded),
     blocks=view(br,2:n+1),parents=view(pr,6:5(n+1)),news=view(nr,2:2n+1),done=view(dr,2:n+1),
     hwork=work,hdone=done,hnews=news,hblocks=blocks,hparents=parents,trace=trace,
     global_template=allocate(zeros(UInt64,131)),local_template=allocate(zeros(UInt64,(3712*sizeof(T)+1136)÷8)))
end
function qs_reset(c)
    filldata=fill(c.guard,c.rounded+2c.ng);copyto!(c.draw,filldata);copyto!(c.nraw,filldata)
    copyto!(c.braw,fill(qs_blockguard(c.T,c.guard),c.n+2));copyto!(c.praw,fill(QS_SG,5(c.n+2)))
    copyto!(c.nrraw,fill(qs_workguard(c.T,c.guard),2c.n+2));copyto!(c.drraw,fill(qs_workguard(c.T,c.guard),c.n+2))
    copyto!(c.host,c.initial);CUDA.device_synchronize(blocking=true)
end
function qs_pipeline(c,launchglobal,launchlocal)
    copyto!(c.d,c.host);copyto!(c.dn,c.hostdn)
    maxseq=2^floor(Int,log(c.n*0.00009516+203)/log(2.0)+0.5)
    work=c.hwork;done=c.hdone;news=c.hnews;blocks=c.hblocks;parents=c.hparents
    empty!(work);empty!(done);empty!(news);empty!(blocks);empty!(parents);empty!(c.trace)
    pivot=qs_u32(qs_median(c.host[1],c.host[c.n÷2+1],c.host[c.n]))
    push!(work,QSWork{c.T}(0,UInt32(c.n),c.T(pivot),1))
    while !isempty(work)
        blocksize=0
        for w in work
            blocksize+=max(Int(w.stop-w.start)÷maxseq,1)
        end
        for w in work
            start=Int(w.start);stop=Int(w.stop);piv=qs_u32(w.pivot);count=cld(stop-start,blocksize)
            parent=UInt32(length(parents)÷5)
            append!(parents,(w.start,w.stop,w.start,w.stop,UInt32(count-1)))
            for i in 0:count-1
                bs=start+blocksize*i;be=i==count-1 ? stop : bs+blocksize
                push!(blocks,QSBlock{c.T}(UInt32(bs),UInt32(be),c.T(piv),w.direction,parent))
            end
        end
        resize!(news,2length(blocks));fill!(news,qs_empty(c.T))
        copyto!(c.blocks,1,blocks,1,length(blocks));copyto!(c.parents,1,parents,1,length(parents));copyto!(c.news,1,news,1,length(news))
        launchglobal(c,length(blocks));CUDA.device_synchronize(blocking=true)
        copyto!(news,1,c.news,1,length(news));push!(c.trace,length(blocks))
        empty!(work);empty!(parents);empty!(blocks)
        for w in news
            if w.direction!=42
                if w.stop-w.start<=1728
                    w.stop>w.start && push!(done,w)
                else
                    push!(work,w)
                end
            end
        end
        empty!(news)
    end
    if !isempty(done)
        copyto!(c.done,1,done,1,length(done));launchlocal(c,length(done));CUDA.device_synchronize(blocking=true)
    end
    copyto!(c.host,c.d)
    nothing
end
function qs_capture(c,payload)
    ok=c.host==c.hostdn
    append!(payload,qs_bits.(c.host));append!(payload,qs_bits.(c.hostdn))
    for (which,raw) in enumerate((c.draw,c.nraw))
        host=Array(raw)
        for range in (1:c.ng,c.ng+c.rounded+1:length(host))
            ok&=all(==(c.guard),@view host[range]);append!(payload,qs_bits.(@view host[range]))
        end
        if which==2
            range=c.ng+c.n+1:c.ng+c.rounded
            ok&=all(==(c.guard),@view host[range]);append!(payload,qs_bits.(@view host[range]))
        end
    end
    for raw in (c.braw,)
        host=Vector{eltype(raw)}(undef,2);copyto!(host,1,raw,1,1);copyto!(host,2,raw,length(raw),1)
        for x in host
            ok&=x==qs_blockguard(c.T,c.guard)
            append!(payload,UInt64[x.start,x.stop,qs_bits(x.pivot),x.direction,x.parent])
        end
    end
    hostp=Vector{UInt32}(undef,10);copyto!(hostp,1,c.praw,1,5);copyto!(hostp,6,c.praw,length(c.praw)-4,5)
    ok&=all(==(QS_SG),hostp);append!(payload,UInt64.(hostp))
    for raw in (c.nrraw,c.drraw)
        host=Vector{eltype(raw)}(undef,2);copyto!(host,1,raw,1,1);copyto!(host,2,raw,length(raw),1)
        for x in host
            ok&=x==qs_workguard(c.T,c.guard)
            append!(payload,UInt64[x.start,x.stop,qs_bits(x.pivot),x.direction])
        end
    end
    ok&=all(iszero,Array(c.global_template))&&all(iszero,Array(c.local_template))
    ok
end
function study(allocate,launchglobal,launchlocal)
    f=open(joinpath(@__DIR__,"fixtures_u64.bin")) do io
        read!(io,Vector{UInt64}(undef,filesize(io)÷8))
    end
    pos=2;cases=Any[]
    for _ in 1:Int(f[1])
        tid,n,mode,rounded=Int.(f[pos:pos+3]);pos+=4;T=(UInt32,Float32,Float64)[tid+1]
        initial=[qs_unbits(T,x) for x in f[pos:pos+rounded-1]];pos+=rounded
        dn=[qs_unbits(T,x) for x in f[pos:pos+rounded-1]];pos+=rounded
        c=qs_makecase(T,n,rounded,initial,dn,allocate);push!(cases,c);qs_reset(c);qs_pipeline(c,launchglobal,launchlocal)
    end
    for trial in 1:5
        payload=copy(f);total=0.;times=Float64[];counts=Int[];ok=true
        for c in cases
            qs_reset(c);start=time_ns();qs_pipeline(c,launchglobal,launchlocal);tm=(time_ns()-start)*1e-9
            total+=tm;push!(times,tm);ok&=qs_capture(c,payload)
            push!(counts,length(c.trace));push!(counts,length(c.hdone));append!(counts,c.trace)
        end
        path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : 1,",\"max_rel_error\":null,\"timing_region\":\"sum24_preallocated_source_custom_quicksort_H2D_hostqueue_kernels_D2H\",\"jit_excluded\":true,\"internal_repeats\":1,\"case_times_seconds\":[",join(times,","),"],\"stage_count_encoding\":\"percase:gq_rounds,local_sequences,gq_blocks_eachround\",\"stage_counts\":[",join(counts,","),"],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
        ok||error("Fullsort/permutation/guard checkfailed")
    end
end
