using Printf
const P=@__DIR__
include("study_params.jl")
function read_values(name,T,n)
    path=joinpath(P,name);filesize(path)==sizeof(T)*n||error("Wrong asset size $name")
    open(path) do io;read!(io,Vector{T}(undef,n));end
end
guarded(a,g)=vcat(fill(g,16),a,fill(g,16))
function make_work(k)
    nx=NXS[k];ny=NYS[k];steps=TS[k];n=nx*ny;g=n÷128
    f=read_values("initial$(k-1)_f32.bin",Float32,9*n)
    ia=ntuple(q->guarded(f[(q-1)*n+1:q*n],-99f0),9)
    ib=ntuple(q->guarded(zeros(Float32,n),-99f0),9)
    io=guarded(read_values("obstacles$(k-1)_i32.bin",Int32,n),Int32(-12345))
    ip=guarded(zeros(Float32,g*steps),-99f0);ic=guarded(zeros(Int32,g*steps),Int32(-12345))
    a=map(allocate_model,ia);b=map(allocate_model,ib);ob=allocate_model(io);ps=allocate_model(ip);pc=allocate_model(ic)
    (;nx,ny,steps,n,g,rho=RHOS[k],accel=ACCS[k],omega=OMS[k],da=RHOS[k]*ACCS[k],ia,ib,io,ip,ic,a,b,ob,ps,pc)
end
function reset!(w)
    for q=1:9;copyto!(w.a[q],w.ia[q]);copyto!(w.b[q],w.ib[q]);end
    copyto!(w.ob,w.io);copyto!(w.ps,w.ip);copyto!(w.pc,w.ic);nothing
end
function prefix!(out,w)
    append!(out,Float64[w.nx,w.ny,w.steps,w.rho,w.accel,w.omega,w.da])
    for v in (w.a...,w.b...,w.ob,w.ps,w.pc);append!(out,Array(v));end
end
function evolve(w,launch::F) where F
    a=w.a;b=w.b
    for t=0:998
        launch(a,b,w.ob,w.ps,w.pc,w.da,w.omega,Int32(w.nx),Int32(w.ny),Int32(t));a,b=b,a
    end
    CUDA.device_synchronize();start=time_ns()
    for t=999:w.steps-1
        launch(a,b,w.ob,w.ps,w.pc,w.da,w.omega,Int32(w.nx),Int32(w.ny),Int32(t));a,b=b,a
    end
    CUDA.device_synchronize();secs=(time_ns()-start)*1e-9
    secs,a,b
end
function main()
    expected=read_values("oracle_f64.bin",Float64,COUNT);exact=read_values("exact_mask_u8.bin",UInt8,COUNT)
    work=map(make_work,1:3)
    for w in work;reset!(w);evolve(w,launch_step);end
    for trial=1:5
        out=Float64[]
        for w in work;reset!(w);prefix!(out,w);end
        length(out)==PREFIX||error("prefix mismatch")
        seconds=0.0;states=Any[]
        for w in work
            sec,a,b=evolve(w,launch_step);seconds+=sec;push!(states,(a,b))
        end
        for (w,(a,b)) in zip(work,states)
            for v in (a...,b...,w.ps,w.pc);append!(out,Array(v));end
        end
        length(out)==COUNT||error("capture mismatch")
        good=true;err=0.0;budget=0.0
        for i in eachindex(out)
            e=abs(out[i]-expected[i]);tol=exact[i]!=0 ? 0.0 : 2e-5+2e-4*abs(expected[i])
            good &= isfinite(out[i])&&e<=tol;err=max(err,e);tol>0&&(budget=max(budget,e/tol))
        end
        path=joinpath(ENV["STUDY_OUTPUT_DIR"],"d2q9_$(MODEL)_trial$(trial).bin")
        open(path,"w") do io;write(io,out);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"oracle_budget_fraction\":%.17g,\"jit_excluded\":true,\"timing_region\":\"source_t999_tail_sum3resident_LBM_cases_complete_context\",\"iterations\":3202,\"internal_repeats\":1,\"kernel_launches\":205,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,good ? "true" : "false",length(out),err,budget,path,length(out));flush(stdout)
        good||error("independent numerical contract rejected")
    end
end
