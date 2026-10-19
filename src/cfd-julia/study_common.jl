using Printf
const STUDY_ROOT=@__DIR__
const ATOL=2e-4;const RTOL=2e-3
read64(p)=open(p,"r")do f;s=filesize(p);@assert s%8==0;x=Vector{Float64}(undef,div(s,8));read!(f,x);@assert eof(f);x;end
struct Guarded{T}
    base::CuArray{T,1}
    initial::Vector{T}
    payload::CuArray{T,1}
end
function Guarded(x::Vector{T})where T
    guard=T<:Integer ? T(-9001) : T(-9001.25)
    initial=vcat(fill(guard,32),x,fill(guard,32))
    # CUDA array view is preserved as a contiguous CuArray with actual device offset.
    base=CuArray(initial);payload=unsafe_wrap(CuArray,pointer(base,33),length(x);own=false)
    Guarded(base,initial,payload)
end
reset!(g)=copyto!(g.base,g.initial)
take(g)=Array(g.base)
function loadcases()
    z=read64(joinpath(STUDY_ROOT,"fixture.bin"));k=1;@assert z[k]==2;k+=1;cases=[]
    for ci in 0:1
        n,steps,id=Int.(z[k:k+2]);k+=3;@assert steps==2000&&id==ci&&n==(ci==0 ? 192 : 97152)
        chunks=[]
        for size in (n,4n,12n,5,3,3,3,3)
            @assert k+size-1<=length(z);push!(chunks,z[k:k+size-1]);k+=size
        end
        a,nb,normal,ff,mx,my,mz,de=chunks
        push!(cases,(n=n,id=id,physical=ci==0 ? 192 : 97046,a=Guarded(Float32.(a)),nb=Guarded(Int32.(nb)),normal=Guarded(Float32.(normal)),ff=Guarded(Float32.(ff)),mx=Guarded(Float32.(mx)),my=Guarded(Float32.(my)),mz=Guarded(Float32.(mz)),de=Guarded(Float32.(de)),u=Guarded(zeros(Float32,5n)),old=Guarded(zeros(Float32,5n)),flux=Guarded(zeros(Float32,5n)),step=Guarded(zeros(Float32,n))))
    end
    @assert k==length(z)+1;cases
end
mutable struct Check
    gold::Vector{Float64};index::Int;correct::Bool;checked::Int;maxabs::Float64;minrho::Float64;minp::Float64;ffdrift::Float64
end
Check(gold)=Check(gold,1,true,0,0,Inf,Inf,0)
function input!(c,g,out)
    x=take(g);c.correct &= x==g.initial;append!(out,x)
end
function inputs!(c,b,out,working)
    working&&append!(out,[b.n,2000,b.id])
    for g in (b.a,b.nb,b.normal,b.ff,b.mx,b.my,b.mz,b.de);input!(c,g,out);end
    if working;for g in (b.u,b.old,b.flux,b.step);input!(c,g,out);end;end
end
function output!(c,g,out)
    x=take(g);append!(out,x);n=length(x)-64
    c.correct &= x[1:32]==g.initial[1:32]&&x[end-31:end]==g.initial[end-31:end]
    for i in 1:n
        v=Float64(x[i+32]);expected=c.gold[c.index];c.index+=1;e=abs(v-expected)
        c.maxabs=max(c.maxabs,e);c.correct &= isfinite(v)&&e<=ATOL+RTOL*abs(expected);c.checked+=1
    end
end
function outputs!(c,b,out)
    for g in (b.u,b.old,b.flux,b.step);output!(c,g,out);end
end
function physical!(c,b)
    x=take(b.u);n=b.n
    for i in 1:n
        rho=Float64(x[i+32]);vsq=0.
        for d in 1:3;v=Float64(x[i+32+d*n])/rho;vsq+=v*v;end
        p=(Float64(1.4f0)-1)*(Float64(x[i+32+4n])-.5*rho*vsq)
        c.minrho=min(c.minrho,rho);c.minp=min(c.minp,p);c.correct &= rho>0&&p>0&&isfinite(rho)&&isfinite(p)
        if b.id==0
            for k in 0:4;c.ffdrift=max(c.ffdrift,abs(Float64(x[i+32+k*n])-b.ff.initial[33+k]));end
        elseif i>b.physical
            for k in 0:4;c.correct &= x[i+32+k*n]==x[b.physical+32+k*n];end
        end
    end
    c.correct &= c.ffdrift<=3e-6
end
function initialize!(b)
    for g in (b.u,b.old,b.flux);launch_init(b.n,g.payload,b.ff.payload);end
    launch_zero(b.n,b.step.payload)
end
function advance!(b,count)
    for it in 1:count
        copyto!(b.old.payload,b.u.payload)
        launch_step(b.n,b.u.payload,b.a.payload,b.step.payload)
        for stage in 0:2
            launch_flux(b.n,b.nb.payload,b.normal.payload,b.u.payload,b.ff.payload,b.flux.payload,b.de.payload,b.mx.payload,b.my.payload,b.mz.payload)
            launch_update(stage,b.n,b.old.payload,b.u.payload,b.step.payload,b.flux.payload)
        end
    end
end
# Explicit whole-context fence, matching native cudaDeviceSynchronize.
complete()=CUDA.synchronize(CUDA.context())
function measured!(segments,work)
    complete();t=time_ns();work();complete();push!(segments,(time_ns()-t)*1e-9)
end
function main()
    cases=loadcases();gold=read64(joinpath(STUDY_ROOT,"oracle.bin"))
    for b in cases;initialize!(b);advance!(b,1);end;complete()
    for trial in 1:5
        c=Check(gold);prefix=Float64[2];out=Float64[];segments=Float64[]
        for b in cases
            for g in (b.u,b.old,b.flux,b.step);reset!(g);end
            inputs!(c,b,prefix,true)
        end
        @assert length(prefix)==3213929
        for b in cases
            measured!(segments,()->initialize!(b));outputs!(c,b,out);physical!(c,b)
            for segment in 1:20
                measured!(segments,()->advance!(b,100));output!(c,b.u,out);physical!(c,b)
            end
            outputs!(c,b,out);inputs!(c,b,out,false)
        end
        @assert c.index==length(gold)+1
        append!(prefix,out);@assert length(prefix)==17722827
        path="output_trial$(trial).bin";open(path,"w")do f;write(f,prefix);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"min_density\":%.17g,\"min_pressure\":%.17g,\"freestream_max_abs\":%.17g,\"timing_region\":\"sum_initialization_and_20_100iteration_segments_percase_context_completion_excludes_observation\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"segment_seconds\":[%s]}\n",trial,sum(segments),string(c.correct),c.checked,c.maxabs,c.minrho,c.minp,c.ffdrift,path,length(prefix),join(segments,","));flush(stdout)
    end
end
