using CUDA, Printf
const CW=64;const CH=8;const CP=CW*CH;const COUNTS=(1,2,127,4001,64)
const GUARD=1024.25f0;const CATOL=2e-4;const CRTOL=5e-5
function atom_fixture(which)
    n=COUNTS[which];a=Vector{NTuple{4,Float32}}(undef,n)
    for i in 0:n-1
        x=Float32(mod(17*i,97))/16f0;y=Float32(mod(11*i,29))/32f0;z=.25f0+Float32(mod(i,13))/32f0;q=Float32(mod(i,7)+1)/8f0*(isodd(i) ? -1f0 : 1f0)
        which==1&&(q=1f0)
        if which==2;x=0f0;y=0f0;z=.5f0;q=i==0 ? 1f0 : -1f0;end
        which==5&&(q=0f0)
        a[i+1]=(x,y,z,q)
    end
    a
end
function physical_oracle(a)
    ref=Vector{Float64}(undef,CP)
    for y in 0:CH-1,x in 0:CW-1
        xx=Float64(Float32(x)*.1f0);yy=Float64(Float32(y)*.1f0);s=0.0
        for t in a
            dx=xx-Float64(t[1]);dy=yy-Float64(t[2]);z=Float64(t[3]);s+=Float64(t[4])/sqrt(dx*dx+dy*dy+z*z)
        end
        ref[y*CW+x+1]=s
    end
    ref
end
function prepared_atoms(a,start,count)
    NTuple{4,Float32}[(t[1],t[2],(0f0-t[3])*(0f0-t[3]),t[4]) for t in @view a[start:start+count-1]]
end
function clenergy_study(launch)
    aa=[atom_fixture(k) for k in 1:5];refs=physical_oracle.(aa)
    raw=CUDA.zeros(Float32,CP+2);out=view(raw,2:CP+1)
    # One device buffer per actual source chunk, all outside the kernel-only ROI.
    chunks=[[(prep=prepared_atoms(aa[k],start,min(4000,COUNTS[k]-start+1)),) for start in 1:4000:COUNTS[k]] for k in 1:5]
    device_chunks=[[CuArray(c.prep) for c in cs] for cs in chunks]
    init=zeros(Float32,CP+2);init[1]=init[end]=GUARD
    copyto!(raw,init);launch(device_chunks[1][1],out,Int32(1));CUDA.synchronize()
    allok=true
    for trial in 1:5
        seconds=0.0;ok=true;maxabs=0.0;maxrel=0.0;prefix=Float32[];outputs=Float32[]
        for k in 1:5
            copyto!(raw,init);actual=Array(raw);ok &= reinterpret(UInt32,actual)==reinterpret(UInt32,init);append!(prefix,@view actual[2:end-1])
            for (j,di) in enumerate(device_chunks[k])
                pre=Array(di);ok &= reinterpret(UInt32,pre)==reinterpret(UInt32,chunks[k][j].prep);append!(prefix,reinterpret(Float32,pre))
                CUDA.synchronize();t=time_ns();launch(di,out,Int32(length(di)));CUDA.synchronize();seconds+=(time_ns()-t)*1e-9
            end
            actual=Array(raw);ok &= actual[1]==GUARD&&actual[end]==GUARD;push!(prefix,actual[1],actual[end])
            for idx in 1:CP
                got=Float64(actual[idx+1]);err=abs(got-refs[k][idx]);maxabs=max(maxabs,err);maxrel=max(maxrel,err/max(abs(refs[k][idx]),CATOL));ok &= isfinite(got)&&err<=CATOL+CRTOL*abs(refs[k][idx]);if k==2||k==5;ok &= got==0.;end
                push!(outputs,Float32(got))
            end
        end
        length(prefix)==19350||error("input/guard prefix count");length(outputs)==2560||error("output count")
        artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact);allok &= ok
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"sum_six_kernel_launch_completion_fragments_five_fixtures\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(artifact),maxabs,maxrel,path,length(artifact))
    end
    allok||error("full Coulomb/input/guard/analytic checks failed")
end
