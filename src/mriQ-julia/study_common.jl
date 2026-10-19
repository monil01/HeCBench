using Printf
const MRIQ_KS=(1,2,257,1024,1026)
const MRIQ_XS=(256,256,512,256,512)
const MRIQ_GUARD=4096.25f0
function mriq_guarded(values,alloc)
    host=vcat(MRIQ_GUARD,values,MRIQ_GUARD)
    device=alloc(host)
    (device=device,payload=view(device,2:length(host)-1),expected=values)
end
function mriq_read(g)
    a=Array(g.device)
    a[1]==MRIQ_GUARD && a[end]==MRIQ_GUARD || error("device guard")
    a[2:end-1]
end
function mriq_guards!(a,g)
    v=Array(g.device)
    v[1]==MRIQ_GUARD && v[end]==MRIQ_GUARD || error("device guard")
    push!(a,v[1],v[end])
end
function mriq_compare!(errors,got,want)
    e=abs(Float64(got)-want)
    errors[1]=max(errors[1],e);errors[2]=max(errors[2],e/max(abs(want),1e-4))
    isfinite(got) && e<=1e-4+2e-5*abs(want) || error("Float64 scientific oracle mismatch: got=$got expected=$want error=$e")
end
function mriq_execute(alloc,launchphi,launchq)
    prefix=Float32[];outputs=Float32[];seconds=0.0;errors=zeros(Float64,2)
    for c in 1:5
        k=MRIQ_KS[c];n=MRIQ_XS[c]
        kr=Float32[(i%13-6)/32 for i in 0:k-1];ki=Float32[((i*3)%11-5)/32 for i in 0:k-1];kz=Float32[((i*7)%9-4)/32 for i in 0:k-1]
        pr=Float32[(i%7+1)/16 for i in 0:k-1];pii=Float32[((i*5)%7-3)/16 for i in 0:k-1]
        if c==1;kr[1]=.125f0;ki[1]=-.0625f0;kz[1]=.03125f0;end
        if c==2;kr[2]=-kr[1];ki[2]=-ki[1];kz[2]=-kz[1];pr[2]=pr[1];pii[2]=pii[1];end
        x=Float32[(i%17)/16 for i in 0:n-1];y=Float32[((i*3)%19)/16 for i in 0:n-1];z=Float32[((i*7)%23)/16 for i in 0:n-1];x[1]=y[1]=z[1]=0f0
        hostinputs=(kr,ki,kz,pr,pii,x,y,z)
        inputs=map(v->mriq_guarded(v,alloc),hostinputs)
        gm=mriq_guarded(zeros(Float32,k),alloc);gr=mriq_guarded(zeros(Float32,n),alloc);gi=mriq_guarded(zeros(Float32,n),alloc)
        allg=(inputs...,gm,gr,gi)
        for g in inputs
            actual=mriq_read(g);actual==g.expected || error("actual input mismatch");append!(prefix,actual)
        end
        append!(prefix,mriq_read(gm));append!(prefix,mriq_read(gr));append!(prefix,mriq_read(gi));for g in allg;mriq_guards!(prefix,g);end
        t=time_ns();launchphi(inputs[4].payload,inputs[5].payload,gm.payload,Int32(k));CUDA.synchronize();seconds+=(time_ns()-t)*1e-9
        mag=mriq_read(gm);kv=Vector{Float32}(undef,4*k)
        for i in 1:k
            mriq_compare!(errors,mag[i],Float64(pr[i])*pr[i]+Float64(pii[i])*pii[i])
            kv[4i-3]=kr[i];kv[4i-2]=ki[i];kv[4i-1]=kz[i];kv[4i]=mag[i]
        end
        tile=alloc(zeros(Float32,4096))
        for base in 0:1024:k-1
            count=min(1024,k-base);hosttile=kv[4base+1:4(base+count)]
            t=time_ns();copyto!(view(tile,1:4count),hosttile);launchq(tile,Int32(k),Int32(base),inputs[6].payload,inputs[7].payload,inputs[8].payload,gr.payload,gi.payload,n);CUDA.synchronize();seconds+=(time_ns()-t)*1e-9
            actual=Array(view(tile,1:4count));actual==hosttile || error("actual tile input mismatch");append!(prefix,actual)
        end
        qr=mriq_read(gr);qi=mriq_read(gi)
        for j in 1:n
            rr=0.0;ii=0.0
            for i in 1:k
                m=Float64(pr[i])*pr[i]+Float64(pii[i])*pii[i]
                angle=2*pi*(Float64(kr[i])*x[j]+Float64(ki[i])*y[j]+Float64(kz[i])*z[j])
                rr+=m*cos(angle);ii+=m*sin(angle)
            end
            mriq_compare!(errors,qr[j],rr);mriq_compare!(errors,qi[j],ii)
            if c==1;mriq_compare!(errors,Float32(hypot(Float64(qr[j]),Float64(qi[j]))),Float64(pr[1])*pr[1]+Float64(pii[1])*pii[1]);end
            if c==2;abs(qi[j])<=1e-6 || error("paired cancellation");end
        end
        Float64(qr[1])==sum(Float64,mag) && qi[1]==0 || error("zero spatial sample")
        append!(outputs,mag);append!(outputs,qr);append!(outputs,qi)
        for g in inputs
            actual=mriq_read(g);actual==g.expected || error("input mutation");append!(outputs,actual)
        end
        for g in allg;mriq_guards!(outputs,g);end
        for g in allg;CUDA.unsafe_free!(g.device);end
        CUDA.unsafe_free!(tile)
    end
    length(prefix)==32170 && length(outputs)==22930 || error("artifact count")
    (prefix=prefix,outputs=outputs,seconds=seconds,errors=errors)
end
function mriq_study(alloc,launchphi,launchq)
    CUDA.allowscalar(false)
    mriq_execute(alloc,launchphi,launchq)
    for trial in 1:5
        a=mriq_execute(alloc,launchphi,launchq)
        name="mriq_trial$(trial).bin"
        open(name,"w") do io;write(io,a.prefix);write(io,a.outputs);end
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":true,\"checked_values\":55100,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"timing_region\":\"sum11_Phi_and_Q_upload_launch_completion_fragments\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":55100,\"byte_order\":\"little\"}}\n",trial,a.seconds,a.errors[1],a.errors[2],name)
    end
end
