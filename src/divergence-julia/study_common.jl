using CUDA
using Printf
const DIV_RREARTH=1.5683814303638645e-7
function study_fixture(kind)
    np=4;nodes=[-1.0,-1.0/sqrt(5.0),1.0/sqrt(5.0),1.0];weights=ones(Float64,np);D=zeros(Float64,np*np)
    for i in 1:np
        for j in 1:np
            if i!=j;weights[i]*=nodes[i]-nodes[j];end
        end
        weights[i]=1.0/weights[i]
    end
    for i in 1:np
        s=0.0
        for j in 1:np
            if i!=j
                value=weights[j]/(weights[i]*(nodes[i]-nodes[j]));D[(i-1)*np+j]=value;s+=value
            end
        end
        D[(i-1)*np+i]=-s
    end
    a,b,c,d=kind==0 ? (1.,0.,0.,1.) : (2.,.25,.5,1.5);det=a*d-b*c;met=fill(det,np*np);rmet=fill(1.0/det,np*np);dinv=Float64[];v=Float64[];gold=Float64[]
    for j in 1:np,i in 1:np
        x=a*nodes[i]+b*nodes[j];y=c*nodes[i]+d*nodes[j];append!(v,[x*x+2.0*y,x*y+y*y]);append!(dinv,[d/det,-b/det,-c/det,a/det]);push!(gold,(3.0*x+2.0*y)*DIV_RREARTH)
    end
    (np=np,v=v,met=met,dinv=dinv,rmet=rmet,D=D,gold=gold,det=det)
end
function full_operation(launch!,f)
    np=f.np;gv=Vector{Float64}(undef,2np*np)
    for j in 0:np-1,i in 0:np-1,k in 0:1
        q=j*np+i;gv[2q+k+1]=f.met[q+1]*(f.dinv[4q+2k+1]*f.v[2q+1]+f.dinv[4q+2k+2]*f.v[2q+2])
    end
    dg=CuArray(gv);dd=CuArray(f.D);out=CuArray{Float64}(undef,np*np);temp=CuArray{Float64}(undef,np*np);dr=CuArray(f.rmet)
    launch!(dg,dd,out,temp,dr,Int32(np));host=Array(out)
    for buffer in (dg,dd,out,temp,dr);CUDA.unsafe_free!(buffer);end
    host
end
function study_main(launch!,args=ARGS)
    repeat=isempty(args) ? 10 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[study_fixture(0),study_fixture(1)];prefix=vcat([vcat(f.v,f.met,f.dinv,f.rmet,f.D) for f in fixtures]...);length(prefix)==288||error("inputprefix")
    for f in fixtures;full_operation(launch!,f);end
    CUDA.synchronize();allok=true
    for f in fixtures
        wrongsign=2maximum(abs.(f.gold));wrongscale=maximum(abs.(f.gold.*(f.det-1.0)));@printf("STUDY_ORACLE_SENSITIVITY {\"determinant\":%.12g,\"wrong_sign_max_error\":%.12g,\"omitted_inverse_metric_max_error\":%.12g,\"kind\":\"CPU_algebraic_negative_control_not_GPU_failure\"}\n",f.det,wrongsign,wrongscale)
    end
    for trial in 1:5
        output=copy(prefix);times=Float64[];ma=0.;mr=0.;ok=true;gct=0.0
        for f in fixtures
            results=Vector{Vector{Float64}}(undef,repeat);CUDA.synchronize();gc0=Base.gc_num();t0=time_ns()
            for rep in 1:repeat;results[rep]=full_operation(launch!,f);end
            CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);gct+=(Base.gc_num().total_time-gc0.total_time)*1e-9
            for got in results
                err=abs.(got.-f.gold);ma=max(ma,maximum(err));mr=max(mr,maximum(err./max.(abs.(f.gold),1e-300)));ok &=all(isfinite,got)&&all(err.<=1e-20.+1e-12.*abs.(f.gold));append!(output,got)
            end
        end
        allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"divergence_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_two_NP4_full_native_workflows_CPU_contravariant_allocation_H2D_kernel_D2H_free\",\"jit_excluded\":true,\"internal_repeats\":%d,\"input_prefix_values\":288,\"divergence_output_values\":%d,\"gc_time_seconds\":%.12g,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,mr,repeat,32repeat,gct,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
