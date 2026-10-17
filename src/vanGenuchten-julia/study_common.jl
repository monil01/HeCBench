function study(run!,n=4096,repeat=10)
    sat=Float64[1e-6+(1.0-1e-6)*(i-1)/n for i in 1:n]
    psi=Float64[-100.0+101.0*(i-1)/n for i in 1:n]
    ds=CuArray(sat); dp=CuArray(psi);dc=CUDA.zeros(Float64,n);dt=similar(dc);dk=similar(dc)
    cr=zeros(n);tr=zeros(n);kr=zeros(n)
    for i in 1:n
        a=0.02; ts=0.45; rr=0.1; nn=1.8; m=(nn-1.0)/nn
        @inbounds h=psi[i]*100.0
        th=h<0.0 ? (ts-rr)/(1.0+(a*(-h))^nn)^m+rr : ts
        se=(th-rr)/(ts-rr)
        t=1.0-(1.0-se^(1.0/m))^m
        @inbounds tr[i]=th
        @inbounds kr[i]=sat[i]*sqrt(se)*t*t
        @inbounds cr[i]=h<0.0 ? 100.0*a*nn*(1.0/nn-1.0)*(a*abs(h))^(nn-1.0)*(rr-ts)*((a*abs(h))^nn+1.0)^(1.0/nn-2.0) : 0.0
    end
    expected=vcat(cr,tr,kr)
    run!(ds,dp,dc,dt,dk,n);CUDA.synchronize()
    for trial in 1:5
        CUDA.synchronize();start=time_ns()
        for _ in 1:repeat;run!(ds,dp,dc,dt,dk,n);end
        CUDA.synchronize();elapsed=(time_ns()-start)*1e-9/repeat
        got=vcat(Array(dc),Array(dt),Array(dk));err=maximum(abs.(got.-expected));correct=all(isfinite,got)&&err<=1e-3
        path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",correct,",\"checked_values\":",3n,",\"max_abs_error\":",err,",\"timing_region\":\"synchronized_kernel_loop_mean\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",3n,",\"byte_order\":\"little\"}}")
        correct||error("CPU formula oracle failed")
    end
end
