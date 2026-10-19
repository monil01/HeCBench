using Printf
const TONE_WS=(16,32,64,256)
const TONE_HS=(16,16,32,256)
const TONE_GUARD=4096.25f0
function tone_guarded(v,alloc)
    a=alloc(vcat(TONE_GUARD,v,TONE_GUARD))
    (device=a,payload=view(a,2:length(v)+1),expected=v)
end
function tone_read(g)
    a=Array(g.device);a[1]==TONE_GUARD && a[end]==TONE_GUARD || error("device guard")
    a[2:end-1]
end
function tone_guards!(a,g)
    v=Array(g.device);v[1]==TONE_GUARD && v[end]==TONE_GUARD || error("device guard")
    push!(a,v[1],v[end])
end
function tone_compare!(e,got,want)
    err=abs(Float64(got)-want);e[1]=max(e[1],err);e[2]=max(e[2],err/max(abs(want),1e-5))
    isfinite(got) && err<=1e-5+2e-5*abs(want) || error("Float64 physical oracle mismatch")
end
function tone_lum64(a,j)
    Float64(.2126f0)*a[4j+1]+Float64(.7152f0)*a[4j+2]+Float64(.0722f0)*a[4j+3]
end
function tone_execute(alloc,launch,observe)
    prefix=Float32[];outputs=Float32[];seconds=0.0;errors=zeros(Float64,2)
    for case in 1:4
        w=TONE_WS[case];h=TONE_HS[case];n=4*w*h;input=Vector{Float32}(undef,n)
        for j in 0:w*h-1
            for channel in 0:2
                value=.5f0
                if case==2;value=1f0/64+Float32((j*(channel+3)+7*channel)%97)/16;end
                if case==3;value=.125f0;end
                if case==4;value=ldexp(1f0,(j*(channel+3)+7*channel)%13-6)+1f0/64;end
                input[4j+channel+1]=value
            end
            input[4j+4]=Float32(j%17)/16
        end
        if case==3;j=(h÷2)*w+w÷2;input[4j+1]=64;input[4j+2]=32;input[4j+3]=16;end
        av=0f0
        for j in 0:w*h-1;av+=tone_lum(input[4j+1],input[4j+2],input[4j+3]);end
        av/=Float32(w*h);gamma=.4f0;c=.25f0;delta=.000002f0
        params=Float32[av,gamma,c,delta,w,4,h]
        gi=tone_guarded(input,alloc);go=tone_guarded(zeros(Float32,n),alloc);gp=tone_guarded(zeros(Float32,7),alloc)
        observe(gp.payload,av,gamma,c,delta,Int32(w),Int32(h));CUDA.synchronize()
        actual=tone_read(gi);actual==input || error("actual input mismatch");append!(prefix,actual);append!(prefix,tone_read(go));actualparam=tone_read(gp);actualparam==params || error("actual scalar input mismatch");append!(prefix,actualparam)
        for g in (gi,go,gp);tone_guards!(prefix,g);end
        t=time_ns();launch(gi.payload,go.payload,av,gamma,c,delta,Int32(w),Int32(h));CUDA.synchronize();seconds+=(time_ns()-t)*1e-9
        out=tone_read(go)
        for yy in 0:h-1, xx in 0:w-1
            j=yy*w+xx;l=tone_lum64(input,j);mean=l
            if xx>0 && yy>0 && xx<w-1 && yy<h-1
                mean=0.0
                for dy in -1:1, dx in -1:1
                    if dx!=0 || dy!=0;mean+=tone_lum64(input,(yy+dy)*w+xx+dx);end
                end
                mean/=8
            end
            cp=mean*log(Float64(delta)+mean/l)+Float64(c)*av;display=l/(l+cp)
            for channel in 0:2
                want=(Float64(input[4j+channel+1])/l)^Float64(gamma)*display
                tone_compare!(errors,out[4j+channel+1],want)
                out[4j+channel+1]>=0 || error("positive output")
            end
            out[4j+4]==input[4j+4] || error("alpha identity")
            if case==1
                closed=1/(1+log(1+Float64(delta))+Float64(c))
                for channel in 0:2;tone_compare!(errors,out[4j+channel+1],closed);end
            end
        end
        append!(outputs,out);actual=tone_read(gi);actual==input || error("input mutation");append!(outputs,actual)
        for g in (gi,go,gp);tone_guards!(outputs,g);CUDA.unsafe_free!(g.device);end
    end
    length(prefix)==546868 && length(outputs)==546840 || error("artifact count")
    (prefix=prefix,outputs=outputs,seconds=seconds,errors=errors)
end
function tone_study(alloc,launch,observe)
    CUDA.allowscalar(false)
    tone_execute(alloc,launch,observe)
    for trial in 1:5
        a=tone_execute(alloc,launch,observe);name="tonemapping_trial$(trial).bin"
        open(name,"w") do io;write(io,a.prefix);write(io,a.outputs);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":1093708,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"timing_region\":\"sum4_source16x16_tonemapping_kernel_launch_completion\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":1093708,\"byte_order\":\"little\"}}\n",trial,a.seconds,a.errors[1],a.errors[2],name)
    end
end
