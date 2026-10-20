using CUDA
const P = @__DIR__
function readvec(io,T,n)
    v=Vector{T}(undef,n);read!(io,v);v
end
function loadcases(path)
    io=open(path);n=Int(read(io,UInt32));@assert n==4;cs=Any[]
    for i in 1:n
        h=Int.(readvec(io,UInt32,9));d,a,v,w,neg,ns,blocks,t,cold=h
        host=(readvec(io,Float32,v*a+17),readvec(io,Float32,v*a+17),readvec(io,Int32,6150+17),readvec(io,UInt32,1024+17),readvec(io,Int32,t+17),readvec(io,Float32,1000))
        gpu=map(CuArray,host);push!(cs,(h=h,host=host,gpu=gpu))
    end
    @assert eof(io);close(io);cs
end
function reset!(c)
    for (g,h)in zip(c.gpu,c.host);copyto!(g,h);end
    if c.h[9]!=0;launch_zero!(c.gpu[2],c.h[3]*c.h[2]);end
    CUDA.device_synchronize(blocking=true)
end
function capture!(out,c)
    for g in c.gpu;append!(out,Float64.(Array(g)));end
end
function main()
    CUDA.stream!(CUDA.legacy_stream());cs=loadcases(ARGS[1]);gold=reinterpret(Float64,read(ARGS[2]));@assert length(gold)==5377048
    for c in cs;reset!(c);launch_cbow!(c);CUDA.device_synchronize(blocking=true);end
    CUDA.device_synchronize(blocking=true)
    for tr in 1:5
        out=Float64[];sizehint!(out,length(gold))
        for c in cs;reset!(c);capture!(out,c);end
        prefix=length(out);@assert prefix==2688524
        CUDA.device_synchronize(blocking=true);start=time_ns()
        for c in cs;launch_cbow!(c);end
        CUDA.device_synchronize(blocking=true);secs=(time_ns()-start)/1e9
        for c in cs;capture!(out,c);end
        @assert length(out)==length(gold)
        ok=all(isequal.(view(out,1:prefix),view(gold,1:prefix)));ma=0.;mr=0.;at=prefix+1
        for c in cs
            for (fld,g)in enumerate(c.gpu)
                n=length(g)
                for j in 1:n
                    x=out[at];y=gold[at];e=abs(x-y);ma=max(ma,e);if y!=0;mr=max(mr,e/abs(y));end
                    exact=fld>=3||j>n-17
                    ok &= isfinite(x)&&(exact ? isequal(x,y) : e<=1e-6+2e-5*abs(y));at+=1
                end
            end
        end
        file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"julia_trial$(tr).f64");open(f->write(f,out),file,"w")
        println("STUDY_TRIAL {\"trial\":",tr,",\"correct\":",ok,",\"time_seconds\":",secs,",\"timing_valid\":true,\"jit_excluded\":true,\"timing_region\":\"completed_four_named_CBOW_kernel_cases\",\"checked_values\":",length(out),",\"max_abs_error\":",ma,",\"max_rel_error\":",mr,",\"actual_device_inputs_checked\":true,\"tail_guards_checked\":true,\"output_artifact\":{\"path\":\"",file,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
        flush(stdout);@assert ok "fullindependentCBOWoraclefailed"
    end
    CUDA.device_synchronize(blocking=true)
end
