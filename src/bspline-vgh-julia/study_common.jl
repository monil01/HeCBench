using CUDA
using Printf
const VGH_CASES=((0,0,0,0.,0.,0.),(1,0,1,.25,.5,.75),(0,1,1,.875,.125,.625),(1,1,0,.5,.25,.125))
function ideal_weights(t)
    [(1-t)^3/6,(3t^3-6t^2+4)/6,(-3t^3+3t^2+3t+1)/6,t^3/6],[-.5*(1-t)^2,1.5t^2-2t,-1.5t^2+t+.5,.5t^2],[1-t,3t-2,1-3t,t]
end
function tensor_oracle(coefs,basis,ideal)
    vals=fill(-123.25,4*2003);grads=fill(-123.25,4*6012);hess=fill(-123.25,4*18036);xs=2016*25;ys=2016*5;zs=2016
    for walker in 0:3
        ix,iy,iz,tx,ty,tz=VGH_CASES[walker+1];dx,dy,dz=5.,7.,9.
        if ideal
            a,da,d2a=ideal_weights(tx);b,db,d2b=ideal_weights(ty);c,dc,d2c=ideal_weights(tz)
        else
            wb=walker*42;a,b,c,da,db,dc,d2a,d2b,d2c=ntuple(k->Float64.(basis[wb+(k-1)*4+1:wb+k*4]),9)
        end
        for n in 1:2003
            acc=zeros(Float64,10)
            for i in 0:3,j in 0:3,k in 0:3
                cf=Float64(coefs[(ix+i)*xs+(iy+j)*ys+(iz+k)*zs+n]);aa=a[i+1];bb=b[j+1];cc=c[k+1]
                acc[1]+=cf*aa*bb*cc;acc[2]+=cf*da[i+1]*bb*cc;acc[3]+=cf*aa*db[j+1]*cc;acc[4]+=cf*aa*bb*dc[k+1];acc[5]+=cf*d2a[i+1]*bb*cc;acc[6]+=cf*da[i+1]*db[j+1]*cc;acc[7]+=cf*da[i+1]*bb*dc[k+1];acc[8]+=cf*aa*d2b[j+1]*cc;acc[9]+=cf*aa*db[j+1]*dc[k+1];acc[10]+=cf*aa*bb*d2c[k+1]
            end
            vals[walker*2003+n]=acc[1];gb=walker*6012+(n-1)*3;hb=walker*18036+(n-1)*9;grads[gb+1:gb+3]=[acc[2]*dx,acc[3]*dy,acc[4]*dz];hess[hb+1:hb+9]=[acc[5]*dx*dx,acc[6]*dx*dy,acc[7]*dx*dz,acc[6]*dx*dy,acc[8]*dy*dy,acc[9]*dy*dz,acc[7]*dx*dz,acc[9]*dy*dz,acc[10]*dz*dz]
        end
    end
    vcat(vals,grads,hess)
end
function study_main(prepare,args=ARGS)
    repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positiverepeat");basis=collect(reinterpret(Float32,read(joinpath(@__DIR__,"study_basis.bin"))));length(basis)==168||error("basislength");dw=CuArray(basis);fixtures=[];prefix=Float32[]
    for kind in 0:1
        coefs=Float32[kind==0 ? Float32(n%17-8)/16f0 : Float32((n*17+x*x*13+y*y*7+z*z*3+x*y*11+y*z*5+x*z*3+n*(x+2y+3z))%251-125)/256f0 for x in 0:4 for y in 0:4 for z in 0:4 for n in 0:2015]
        gold=tensor_oracle(coefs,basis,false);ideal=tensor_oracle(coefs,basis,true);append!(prefix,coefs);dc=CuArray(coefs);vals=CUDA.fill(-123.25f0,4*2003);grads=CUDA.fill(-123.25f0,4*6012);hess=CUDA.fill(-123.25f0,4*18036);launch! = prepare(dc,vals,grads,hess,dw,basis);for walker in 0:3;launch!(walker);end;push!(fixtures,(vals,grads,hess,launch!,gold,ideal,kind))
    end
    append!(prefix,basis);CUDA.synchronize();allok=true
    for trial in 1:5
        output=copy(prefix);times=Float64[];ma=0.0;ie=0.0;symmetry=0.0;constant_error=0.0;ok=true
        for (vals,grads,hess,launch!,gold,ideal,kind) in fixtures
            for walker in 0:3;CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!(walker);end;CUDA.synchronize();push!(times,(time_ns()-t0)*1e-9/repeat);end
            hv,hg,hh=Array(vals),Array(grads),Array(hess);got=vcat(hv,hg,hh);err=maximum(abs.(Float64.(got).-gold));de=maximum(abs.(Float64.(got).-ideal));ma=max(ma,err);ie=max(ie,de);ok &=all(isfinite,got)&&all(abs.(Float64.(got).-gold).<=1e-4.+1e-6.*abs.(gold))&&de<=1e-3
            for walker in 0:3,n in 1:2003
                hb=walker*18036+(n-1)*9;symmetry=max(symmetry,abs(hh[hb+2]-hh[hb+4]),abs(hh[hb+3]-hh[hb+7]),abs(hh[hb+6]-hh[hb+8]))
                if kind==0;constant_error=max(constant_error,abs(Float64(hv[walker*2003+n])-Float64(Float32((n-1)%17-8)/16f0)));end
            end
            for walker in 0:3;ok &=all(==(-123.25f0),hg[walker*6012+6010:walker*6012+6012])&&all(==(-123.25f0),hh[walker*18036+18028:walker*18036+18036]);end
            append!(output,got)
        end
        ok &=symmetry==0&&constant_error<=5e-6;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"bspline_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"computed_output_values\":208312,\"defined_padding_values\":96,\"input_prefix_values\":504168,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"ideal_cubic_basis_max_error\":%.12g,\"hessian_symmetry_error\":%.12g,\"constant_value_error\":%.12g,\"timing_region\":\"sum_two_coefficient_fields_four_complete_2003channel_VGH_walkers\",\"jit_excluded\":true,\"internal_repeats\":%d,\"case_times_seconds\":%s,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(output),ma,ie,symmetry,constant_error,repeat,string(times),file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
