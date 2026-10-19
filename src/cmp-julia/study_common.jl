using Printf
const STUDY_ROOT=@__DIR__;const ATOL=5e-5;const RTOL=2e-4
function read64(path)
    open(path,"r")do f
        @assert filesize(path)%8==0
        x=Vector{Float64}(undef,div(filesize(path),8));read!(f,x);@assert eof(f);x
    end
end
struct Guarded{T}
    base::CuArray{T,1};initial::Vector{T};payload::CuArray{T,1}
end
function Guarded(x::Vector{T})where T
    guard=T<:Integer ? T(-9001) : T(-9001.25)
    initial=vcat(fill(guard,32),x,fill(guard,32));base=CuArray(initial)
    payload=unsafe_wrap(CuArray,pointer(base,33),length(x);own=false)
    Guarded(base,initial,payload)
end
reset!(g)=copyto!(g.base,g.initial)
take(g)=Array(g.base)
function loadcases()
    x=read64(joinpath(STUDY_ROOT,"fixture.bin"));@assert x[1]==2;k=2;cases=[]
    for ci in 0:1
        head=x[k:k+10];nc,ns,ng,nt,mt=Int.(head[1:5]);c0,inc,dt,idt=Float32.(head[6:9]);tau,w=Int.(head[10:11]);k+=11
        counts=Int.(x[k:k+ng-1]);k+=ng;geom=[]
        for q in 1:5;push!(geom,Float32.(x[k:k+nt-1]));k+=nt;end
        samples=Float32.(x[k:k+nt*ns-1]);k+=nt*ns
        push!(cases,(head=head,nc=nc,ns=ns,ng=ng,nt=nt,mt=mt,c0=c0,inc=inc,dt=dt,idt=idt,tau=tau,w=w,counts=counts,samples=samples,sc=Guarded(geom[1]),gx=Guarded(geom[2]),gy=Guarded(geom[3]),sx=Guarded(geom[4]),sy=Guarded(geom[5]),curv=Guarded(zeros(Float32,nc)),half=Guarded(zeros(Float32,nt)),sample=Guarded(zeros(Float32,mt*ns)),num=Guarded(zeros(Float32,ns*nc)),stt=Guarded(zeros(Float32,ns*nc)),ctr=Guarded(zeros(Int32,ng*ns)),str=Guarded(zeros(Float32,ng*ns)),stk=Guarded(zeros(Float32,ng*ns)),expected_sample=Ref(Float32[]),refctr=zeros(Int32,ng*ns),refstr=zeros(Float64,ng*ns),refstk=zeros(Float64,ng*ns)))
    end
    @assert k==length(x)+1;cases
end
mutable struct Check
    gold::Vector{Float64};index::Int;ok::Bool;checked::Int;maxerr::Float64;maxbudget::Float64
end
Check(gold)=Check(gold,1,true,0,0.,0.)
function equal!(c,v,expected)
    e=abs(Float64(v)-expected);budget=ATOL+RTOL*abs(expected);c.maxerr=max(c.maxerr,e);c.maxbudget=max(c.maxbudget,e/budget);c.checked+=1
    isfinite(v)&&e<=budget
end
function input!(c,g,out,expected=g.initial)
    x=take(g);c.ok &= x==expected;append!(out,x)
end
function takeout!(c,g,out)
    x=take(g);append!(out,x);c.ok &= x[1:32]==g.initial[1:32]&&x[end-31:end]==g.initial[end-31:end]
    x[33:end-32]
end
function output!(c,g,out)
    x=takeout!(c,g,out)
    for v in x;c.ok &= equal!(c,v,c.gold[c.index]);c.index+=1;end
end
function prefix!(c,b,out)
    append!(out,b.head);append!(out,b.counts)
    for g in (b.sc,b.gx,b.gy,b.sx,b.sy,b.curv,b.half,b.num,b.stt,b.ctr,b.str,b.stk,b.sample);input!(c,g,out);end
end
function initialize!(b)
    launch_c(b.nc,b.curv.payload,b.inc,b.c0)
    launch_half(b.nt,b.sc.payload,b.gx.payload,b.gy.payload,b.sx.payload,b.sy.payload,b.half.payload)
end
bounds(b,group)=(group==0 ? 0 : b.counts[group],b.counts[group+1])
function upload!(b,group)
    first,last=bounds(b,group);copyto!(b.sample.payload,1,b.samples,first*b.ns+1,(last-first)*b.ns)
end
function update_expected!(b,group)
    first,last=bounds(b,group);copyto!(b.expected_sample[],33,b.samples,first*b.ns+1,(last-first)*b.ns)
end
function compute!(b,group)
    first,last=bounds(b,group)
    launch_scan(b.half.payload,b.curv.payload,b.sample.payload,b.num.payload,b.stt.payload,first,last,b.idt,b.dt,b.tau,b.w,b.nc,b.ns)
    launch_redux(b.num.payload,b.stt.payload,b.ctr.payload,b.str.payload,b.stk.payload,b.nc,group,b.ns)
end
function checkgroup!(c,b,group,out)
    output!(c,b.num,out);output!(c,b.stt,out)
    for i in 1:b.ns;b.refctr[group*b.ns+i]=Int32(c.gold[c.index]);c.index+=1;end
    for i in 1:b.ns;b.refstr[group*b.ns+i]=c.gold[c.index];c.index+=1;end
    for i in 1:b.ns;b.refstk[group*b.ns+i]=c.gold[c.index];c.index+=1;end
    ctr=takeout!(c,b.ctr,out);str=takeout!(c,b.str,out);stk=takeout!(c,b.stk,out)
    for i in eachindex(ctr)
        c.ok &= ctr[i]==b.refctr[i];c.ok &= equal!(c,str[i],b.refstr[i]);c.ok &= equal!(c,stk[i],b.refstk[i])
    end
    num=take(b.num);stt=take(b.stt)
    for t in 0:b.ns-1
        best=0f0;ix=-1
        for q in 0:b.nc-1
            value=num[33+t*b.nc+q];c.ok &= value==-1f0||(0<=value<=1.0001f0)
            if value>best;best=value;ix=q;end
        end
        pos=group*b.ns+t+1;c.ok &= ctr[pos]==ix&&str[pos]==best&&stk[pos]==(ix>=0 ? stt[33+t*b.nc+ix] : 0f0)
    end
end
complete()=CUDA.synchronize(CUDA.context())
function measure!(segments,work)
    complete();t=time_ns();work();complete();push!(segments,(time_ns()-t)*1e-9)
end
function main()
    cases=loadcases();gold=read64(joinpath(STUDY_ROOT,"oracle.bin"))
    for b in cases;initialize!(b);upload!(b,0);compute!(b,0);end;complete()
    for trial in 1:5
        c=Check(gold);prefix=Float64[2];out=Float64[];times=Float64[]
        for b in cases
            for g in (b.curv,b.half,b.sample,b.num,b.stt,b.ctr,b.str,b.stk);reset!(g);end
            b.expected_sample[]=copy(b.sample.initial);fill!(b.refctr,0);fill!(b.refstr,0);fill!(b.refstk,0);prefix!(c,b,prefix)
        end
        for b in cases
            measure!(times,()->initialize!(b));output!(c,b.curv,out);output!(c,b.half,out);frozenc=take(b.curv);frozenh=take(b.half)
            for group in 0:b.ng-1
                measure!(times,()->upload!(b,group));update_expected!(b,group);input!(c,b.sample,prefix,b.expected_sample[])
                measure!(times,()->compute!(b,group));checkgroup!(c,b,group,out);c.ok &= take(b.sample)==b.expected_sample[]
            end
            for g in (b.sc,b.gx,b.gy,b.sx,b.sy);input!(c,g,out);end
            input!(c,b.curv,out,frozenc);input!(c,b.half,out,frozenh);input!(c,b.sample,out,b.expected_sample[])
        end
        @assert c.index==length(gold)+1&&length(prefix)==128743
        append!(prefix,out);@assert length(prefix)==296645
        path="output_trial$(trial).bin";open(path,"w")do f;write(f,prefix);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_fraction_of_predeclared_budget\":%.17g,\"timing_region\":\"sum_init_and_perCDP_sampleH2D_and_scan_redux_contextcompleted_segments_excludes_captures\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"segment_seconds\":[%s]}\n",trial,sum(times),string(c.ok),c.checked,c.maxerr,c.maxbudget,path,length(prefix),join(times,","));flush(stdout)
    end
end
