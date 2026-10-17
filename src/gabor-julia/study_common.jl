using CUDA
using Printf
function study_main(prepare,args=ARGS)
    h=97;w=131;repeat=isempty(args) ? 100 : parse(Int,args[1]);repeat>0||error("positive repeat");fixtures=[]
    sx=13.0/(2.0*sqrt(2.0*log(2.0)));sy=2.65*sx;scale=1.0/(2.0*pi*sx*sy);sx2=sx*sx;sy2=sy*sy;fx=1.0/13.0
    for theta in (45.0,0.0)
        ct=ccall(:cos,Cdouble,(Cdouble,),theta);st=ccall(:sin,Cdouble,(Cdouble,),theta);cy=h/2.0;cx=w/2.0
        expected=Float64[];gold=Float64[]
        setprecision(BigFloat,256) do
            for y in 0:h-1,x in 0:w-1
                dx=Float64(x)-cx;dy=Float64(y)-cy;u=ct*dx-st*dy;v=ct*dy+st*dx
                push!(expected,scale*exp(-0.5*(u*u/sx2+v*v/sy2))*cos(2.0*pi*fx*u))
                # Re-evaluate complete expression at256bits from exact Float64 parameters.
                bu=BigFloat(ct)*BigFloat(dx)-BigFloat(st)*BigFloat(dy);bv=BigFloat(ct)*BigFloat(dy)+BigFloat(st)*BigFloat(dx)
                push!(gold,Float64(BigFloat(scale)*exp(-BigFloat(0.5)*(bu*bu/BigFloat(sx2)+bv*bv/BigFloat(sy2)))*cos(BigFloat(2.0*pi)*BigFloat(fx)*bu)))
            end
        end
        out=CUDA.zeros(Float64,h*w);launch! = prepare(out,h,w,cy,cx,ct,st,scale,sx2,sy2,fx);launch!();push!(fixtures,(out,launch!,expected,gold))
    end
    CUDA.synchronize();allok=true
    for trial in 1:5
        output=Float64[];seconds=0.0;ma=0.0;be=0.0;ok=true
        for (out,launch!,expected,gold) in fixtures
            CUDA.synchronize();t0=time_ns();for rep in 1:repeat;launch!();end;CUDA.synchronize();seconds+=(time_ns()-t0)*1e-9/repeat;got=Array(out)
            ma=max(ma,maximum(abs.(got.-expected)));be=max(be,maximum(abs.(got.-gold)));ok &=all(isfinite,got)&&all(abs.(got.-expected).<=1e-16.+1e-12.*abs.(expected))&&all(abs.(got.-gold).<=1e-16.+1e-12.*abs.(gold));append!(output,got)
        end
        # theta0 envelope/carrier are reflection symmetric around half-grid center.
        theta0=reshape(output[h*w+1:end],w,h)
        symmetry=maximum(abs(theta0[x+1,y+1]-theta0[w-x+1,h-y+1]) for y in 1:h-1 for x in 1:w-1)
        ok &=symmetry<=1e-16;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"gabor_trial$(trial).bin");write(file,output)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"bigfloat_oracle_max_error\":%.12g,\"central_reflection_error\":%.12g,\"timing_region\":\"sum_two_full_Gabor_kernels\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,string(ok),length(output),ma,be,symmetry,repeat,file,length(output))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
