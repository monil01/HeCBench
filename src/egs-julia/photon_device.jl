# Translation of CUDA EGS 1.0.0 photon transport; original GPL-2.0-or-later notice
# is preserved in reference_history/COPYRIGHT and configured_native/kernels.cu.
# All device work is Julia. CUDA backend collectives are explicit extension use.
using CUDA, StaticArrays
const EGS_MASK = typemax(UInt32)
struct Photon
    words::MVector{14,UInt32}
end
@inline function Base.getproperty(p::Photon, name::Symbol)
    v = getfield(p, :words)
    name === :status && return reinterpret(Int32, v[1])
    name === :reserved && return reinterpret(Int32, v[2])
    name === :charge && return reinterpret(Int32, v[3])
    name === :process && return v[4] != 0
    name === :e && return reinterpret(Float32, v[5])
    name === :wt && return reinterpret(Float32, v[6])
    name === :region && return reinterpret(Int32, v[7])
    name === :latch && return v[8]
    name === :x && return reinterpret(Float32, v[9])
    name === :y && return reinterpret(Float32, v[10])
    name === :z && return reinterpret(Float32, v[11])
    name === :u && return reinterpret(Float32, v[12])
    name === :v && return reinterpret(Float32, v[13])
    name === :w && return reinterpret(Float32, v[14])
    return v
end
@inline function Base.setproperty!(p::Photon, name::Symbol, x)
    v = getfield(p, :words)
    name === :status && (v[1] = UInt32(x); return x)
    name === :reserved && (v[2] = UInt32(x); return x)
    name === :charge && (v[3] = UInt32(x); return x)
    name === :process && (v[4] = UInt32(x); return x)
    name === :e && (v[5] = reinterpret(UInt32, Float32(x)); return x)
    name === :wt && (v[6] = reinterpret(UInt32, Float32(x)); return x)
    name === :region && (v[7] = UInt32(x); return x)
    name === :latch && (v[8] = UInt32(x); return x)
    name === :x && (v[9] = reinterpret(UInt32, Float32(x)); return x)
    name === :y && (v[10] = reinterpret(UInt32, Float32(x)); return x)
    name === :z && (v[11] = reinterpret(UInt32, Float32(x)); return x)
    name === :u && (v[12] = reinterpret(UInt32, Float32(x)); return x)
    name === :v && (v[13] = reinterpret(UInt32, Float32(x)); return x)
    name === :w && (v[14] = reinterpret(UInt32, Float32(x)); return x)
    return x
end
@inline egs_all(x) = CUDA.vote_all_sync(EGS_MASK, x)
@inline egs_any(x) = CUDA.vote_any_sync(EGS_MASK, x)
@inline egs_count(x) = count_ones(CUDA.vote_ballot_sync(EGS_MASK, x))
@inline egs_warp() = CUDA.sync_warp(EGS_MASK)
@inline function egs_shared()
    (state=CUDA.CuStaticSharedArray(UInt32,101), params=CUDA.CuStaticSharedArray(UInt32,4),
     rec=CUDA.CuStaticSharedArray(UInt32,16), tmp=CUDA.CuStaticSharedArray(UInt32,16),
     random=CUDA.CuStaticSharedArray(Float32,96), ri=CUDA.CuStaticSharedArray(Int32,1),
     counts=CUDA.CuStaticSharedArray(UInt32,9), weights=CUDA.CuStaticSharedArray(Float32,32),
     combined=CUDA.CuStaticSharedArray(Float64,1), depth=CUDA.CuStaticSharedArray(UInt32,1),
     iterations=CUDA.CuStaticSharedArray(UInt32,1))
end
@inline function egs_mt_generate(s, lane)
    M=Int32(s.params[1]); sh1=s.params[2]; sh2=s.params[3]; mask=s.params[4]
    bound=min(Int32(101)-M,Int32(96))
    # All lanes fence between source SIMT batches; no native defect is inferred.
    for base in (Int32(0),Int32(32),Int32(64))
        i=base+lane; x=UInt32(0); result=0f0
        if i<bound
            x=(s.state[i+1]&mask)⊻s.state[i+2]; x⊻=x<<sh1
            x⊻=s.state[i+M+1]>>sh2; x⊻=s.rec[Int32(x&15)+1]
            t=s.state[i+M]; t⊻=t>>16; t⊻=t>>8
            result=reinterpret(Float32,((x>>9)⊻s.tmp[Int32(t&15)+1])|UInt32(1))-1f0
        end
        egs_warp()
        if i<bound
            s.random[i+1]=result; s.state[i+1]=x
        end
        egs_warp()
    end
    i=bound+lane; x=UInt32(0); result=0f0
    if i<101
        x=(s.state[i+1]&mask)⊻s.state[(i+1)%Int32(101)+1]; x⊻=x<<sh1
        x⊻=s.state[(i+M)%Int32(101)+1]>>sh2; x⊻=s.rec[Int32(x&15)+1]
        if i<96
            t=s.state[(i+M-1)%Int32(101)+1]; t⊻=t>>16; t⊻=t>>8
            result=reinterpret(Float32,((x>>9)⊻s.tmp[Int32(t&15)+1])|UInt32(1))-1f0
        end
    end
    egs_warp()
    if i<101
        s.state[i+1]=x
        if i<96; s.random[i+1]=result; end
    end
    egs_warp()
    nothing
end
@inline function egs_mt_read(c,s,lane)
    if lane<4; s.params[lane+1]=c.mt_params[lane+1]; end
    if lane<16
        s.rec[lane+1]=c.mt_tables[lane+1]; s.tmp[lane+1]=c.mt_tables[lane+17]
    end
    for i in lane:Int32(32):Int32(100); s.state[i+1]=c.mt_state[i+1]; end
    if lane==0; s.ri[1]=0; end
    egs_warp(); egs_mt_generate(s,lane)
    nothing
end
@inline function egs_rand(s,lane)
    egs_warp(); i=s.ri[1]+Int32(1)
    if i>=3; egs_mt_generate(s,lane); i=Int32(0); end
    if lane==0; s.ri[1]=i; end
    egs_warp()
    return s.random[i*Int32(32)+lane+1]
end
@inline function egs_region(c,r)
    i=r*Int32(4); word=c.regions[i+1]
    (med=Int32(word&0xffff), flags=word>>16, rhof=reinterpret(Float32,c.regions[i+2]),
     pcut=reinterpret(Float32,c.regions[i+3]), ecut=reinterpret(Float32,c.regions[i+4]))
end
@inline egs_interp(a,i,x) = muladd(a[2*i+2],x,a[2*i+1])
@inline function egs_where(x,bounds)
    (x<bounds[1] || x>bounds[33]) && return Int32(-1)
    lo=Int32(0); hi=Int32(32)
    while hi-lo>1
        mid=(lo+hi)÷Int32(2)
        if x<=bounds[mid+1]; hi=mid; else; lo=mid; end
    end
    return hi-Int32(1)
end
@inline function egs_howfar(c,p,t)
    if p.region>0
        ir=p.region-Int32(1); iz=ir÷Int32(1024); ir-=iz*Int32(1024)
        iy=ir÷Int32(32); ix=ir-iy*Int32(32); next=p.region
        for axis in (1,2,3)
            coord=axis==1 ? p.x : axis==2 ? p.y : p.z
            direction=axis==1 ? p.u : axis==2 ? p.v : p.w
            i=axis==1 ? ix : axis==2 ? iy : iz
            stride=axis==1 ? Int32(1) : axis==2 ? Int32(32) : Int32(1024)
            bounds=axis==1 ? c.x_bounds : axis==2 ? c.y_bounds : c.z_bounds
            if direction>0f0
                d=(bounds[i+2]-coord)/direction
                if d<=t; t=d; next=i+1<32 ? p.region+stride : Int32(0); end
            elseif direction<0f0
                d=(bounds[i+1]-coord)/direction
                if d<=t; t=d; next=i-1>=0 ? p.region-stride : Int32(0); end
            end
        end
        return next,t
    end
    for axis in (1,2,3)
        coord=axis==1 ? p.x : axis==2 ? p.y : p.z
        direction=axis==1 ? p.u : axis==2 ? p.v : p.w
        bounds=axis==1 ? c.x_bounds : axis==2 ? c.y_bounds : c.z_bounds
        plane=Int32(-1); dist=0f0
        if coord<=bounds[1] && direction>0f0
            dist=(bounds[1]-coord)/direction; plane=Int32(0)
        elseif coord>=bounds[33] && direction<0f0
            dist=(bounds[33]-coord)/direction; plane=Int32(31)
        end
        if plane>=0 && dist<=t
            x=muladd(p.u,dist,p.x); y=muladd(p.v,dist,p.y); z=muladd(p.w,dist,p.z)
            ix=axis==1 ? plane : egs_where(x,c.x_bounds)
            iy=axis==2 ? plane : egs_where(y,c.y_bounds)
            iz=axis==3 ? plane : egs_where(z,c.z_bounds)
            if ix>=0 && iy>=0 && iz>=0
                return ix+iy*Int32(32)+iz*Int32(1024)+Int32(1),dist
            end
        end
    end
    return Int32(0),t
end
@inline function egs_uphi(c,s,lane,p,ct,st)
    r=egs_rand(s,lane)
    if p.process
        phi=(2f0*3.14159265f0)*r
        sp=Base.FastMath.sin_fast(phi); cp=Base.FastMath.cos_fast(phi)
        sinps2=muladd(p.u,p.u,p.v*p.v)
        if sinps2<1f-20
            p.u=st*cp; p.v=st*sp; p.w=p.w*ct
        else
            sinpsi=sqrt(sinps2); us=st*cp; vs=st*sp
            sindel=p.v/sinpsi; cosdel=p.u/sinpsi
            u=muladd(p.u,ct,muladd(p.w*cosdel,us,-sindel*vs))
            v=muladd(p.v,ct,muladd(p.w*sindel,us,cosdel*vs))
            w=muladd(-sinpsi,us,p.w*ct)
            p.u=u; p.v=v; p.w=w
        end
    end
    nothing
end
@inline function egs_new(c,s,lane,p)
    r1=egs_rand(s,lane); r2=egs_rand(s,lane); r3=egs_rand(s,lane); r4=egs_rand(s,lane)
    if p.process
        p.charge=0; p.reserved=0; p.status=2; p.region=0
        aj=r3*145f0; j=unsafe_trunc(Int32,aj); aj-=Float32(j)
        if aj>c.source_weights[j+1]; j=c.source_alias[j+1]; end
        x=c.source_edges[j+1]; dx=c.source_edges[j+2]-x; p.e=muladd(dx,r4,x)
        p.x=0f0; p.y=0f0; p.z=-30f0
        p.u=muladd(12.8f0,r1,-6.4f0); p.v=muladd(12.8f0,r2,-6.4f0); p.w=-6.4f0-p.z
        d2i=1f0/muladd(p.u,p.u,muladd(p.v,p.v,p.w*p.w)); di=sqrt(d2i)
        p.u*=di; p.v*=di; p.w*=di
        p.wt=(12.8f0*12.8f0)*abs(p.w)*d2i; s.weights[lane+1]=p.wt; p.latch=0
    else
        s.weights[lane+1]=0f0
    end
    egs_warp()
    nothing
end
@inline function egs_almost(a,b)
    ia=reinterpret(Int32,a); ib=reinterpret(Int32,b)
    if ia<0; ia=reinterpret(Int32,UInt32(0x80000000)-reinterpret(UInt32,ia)); end
    if ib<0; ib=reinterpret(Int32,UInt32(0x80000000)-reinterpret(UInt32,ib)); end
    return abs(ia-ib)<=20
end
@inline function egs_score(c,x,y,cat,weight,energy)
    if 0<=x<512 && 0<=y<512
        i=y*Int32(512)+x+Int32(1)
        CUDA.atomic_add!(pointer(c.sc_count[cat+1],i),weight)
        CUDA.atomic_add!(pointer(c.sc_energy[cat+1],i),energy)
    end
    nothing
end
@inline function egs_user(c,p)
    if !p.process; return; end
    p.status=8
    if p.charge!=0 || p.region!=0 || p.w==0f0; return; end
    delta=(30f0-p.z)/p.w
    if delta<0f0; return; end
    posx=muladd(delta,p.u,p.x); posy=muladd(delta,p.v,p.y)
    pixx=posx/0.1f0+256f0; pixy=posy/0.1f0+256f0
    lx=floor(pixx); ux=ceil(pixx); ly=floor(pixy); uy=ceil(pixy)
    split=Int32(0)
    if egs_almost(pixx,lx); split+=1; ix=unsafe_trunc(Int32,lx)
    elseif egs_almost(pixx,ux); split+=1; ix=unsafe_trunc(Int32,ux)
    else; ix=unsafe_trunc(Int32,lx); end
    if egs_almost(pixy,ly); split+=2; iy=unsafe_trunc(Int32,ly)
    elseif egs_almost(pixy,uy); split+=2; iy=unsafe_trunc(Int32,uy)
    else; iy=unsafe_trunc(Int32,ly); end
    nc=p.latch&UInt32(0xffff); nr=p.latch>>16
    cat=(nc==0 && nr==0) ? Int32(0) : (nc==1 && nr==0) ? Int32(1) : (nc==0 && nr==1) ? Int32(2) : Int32(3)
    if split==0
        egs_score(c,ix,iy,cat,p.wt,p.e*p.wt)
    elseif split==1
        p.wt/=2f0; p.e*=p.wt
        egs_score(c,ix,iy,cat,p.wt,p.e); egs_score(c,ix-Int32(1),iy,cat,p.wt,p.e)
    elseif split==2
        p.wt/=2f0; p.e*=p.wt
        egs_score(c,ix,iy,cat,p.wt,p.e); egs_score(c,ix,iy-Int32(1),cat,p.wt,p.e)
    else
        p.wt/=4f0; p.e*=p.wt
        egs_score(c,ix,iy,cat,p.wt,p.e); egs_score(c,ix-Int32(1),iy,cat,p.wt,p.e)
        egs_score(c,ix,iy-Int32(1),cat,p.wt,p.e); egs_score(c,ix-Int32(1),iy-Int32(1),cat,p.wt,p.e)
    end
    nothing
end
@inline function egs_photon(c,s,lane,p)
    reg=egs_region(c,p.region)
    if p.process
        if p.e<=reg.pcut; p.status=0
        elseif p.wt<=0f0; p.status=1; end
    end
    r1=egs_rand(s,lane); dpmfp=-log(r1); gle=log(p.e)
    lgle=Int32(0); tstep=0f0; gmfp=0f0; cohfac=0f0; oldmed=reg.med
    if p.process && p.status==2
        if reg.med==65535
            tstep=1f8
        else
            lgle=unsafe_trunc(Int32,egs_interp(c.ge,reg.med,gle))
            gmfp=egs_interp(c.gmfp,reg.med*Int32(200)+lgle,gle)/reg.rhof
            if (reg.flags&UInt32(1))!=0
                cohfac=egs_interp(c.cohe,reg.med*Int32(200)+lgle,gle); gmfp*=cohfac
            end
            tstep=gmfp*dpmfp
        end
        next,tstep=egs_howfar(c,p,tstep)
        discard=next==0 ? (reg.med==65535 ? Int32(1) : Int32(-1)) : Int32(0)
        if discard>0
            p.region=0; p.status=1
        else
            p.x=muladd(p.u,tstep,p.x); p.y=muladd(p.v,tstep,p.y); p.z=muladd(p.w,tstep,p.z)
            if reg.med!=65535; dpmfp=max(0f0,dpmfp-tstep/gmfp); end
            oldmed=reg.med
            if next!=p.region; p.region=next; reg=egs_region(c,next); end
            if p.e<=reg.pcut; p.status=0
            elseif discard<0; p.status=1; end
        end
    end
    process=p.process && p.status==2
    if process && (reg.med!=oldmed || reg.med==65535 || dpmfp>=1f-5)
        p.status=2; process=false
    end
    r2=egs_rand(s,lane)
    if process && (reg.flags&UInt32(1))!=0 && r2<1f0-cohfac
        p.status=3; process=false
    end
    r3=egs_rand(s,lane)
    if process
        index=reg.med*Int32(200)+lgle
        b1=egs_interp(c.gbr1,index,gle); b2=egs_interp(c.gbr2,index,gle)
        if r3<=b1 && p.e>2f0*0.5110034f0; p.status=6
        else; p.status=r3<b2 ? 4 : 5; end
    end
    nothing
end
@inline function egs_rayleigh(c,s,lane,p)
    med=Int32(0); xmax=0f0; pmax=0f0
    if p.process
        reg=egs_region(c,p.region); med=reg.med; p.status=2; p.latch+=UInt32(1)<<16
        gle=log(p.e); lgle=unsafe_trunc(Int32,egs_interp(c.ge,med,gle))
        pmax=egs_interp(c.pmax,med*Int32(200)+lgle,gle); xmax=80.65506856998f0*p.e
    end
    xv=0f0; ct=0f0; ct2=0f0; done=!p.process
    while true
        inner=done
        while true
            r1=egs_rand(s,lane)
            if !inner
                temp=r1*pmax; ibin=unsafe_trunc(Int32,temp*99f0)
                ib=Int32(c.rayleigh_indices[med*Int32(100)+ibin+1])-Int32(1)
                next=Int32(c.rayleigh_indices[med*Int32(100)+ibin+2])-Int32(1)
                if next>ib
                    while true
                        idx=(med*Int32(100)+ib+Int32(1))*Int32(4)
                        if temp<c.rayleigh[idx+2] || ib>=98; break; end
                        ib+=Int32(1)
                    end
                end
                idx=(med*Int32(100)+ib)*Int32(4)
                temp=(temp-c.rayleigh[idx+2])*c.rayleigh[idx+4]
                xv=c.rayleigh[idx+1]*exp(log(1f0+temp)*c.rayleigh[idx+3])
                if xv<xmax; inner=true; end
            end
            if egs_all(inner); break; end
        end
        r2=egs_rand(s,lane)
        if !done
            xv/=p.e; ct=muladd(-(0.000307444456f0*xv),xv,1f0); ct2=ct*ct
            if 2f0*r2<1f0+ct2; done=true; end
        end
        if egs_all(done); break; end
    end
    egs_uphi(c,s,lane,p,ct,sqrt(1f0-ct2))
    nothing
end
@inline function egs_compton(c,s,lane,p)
    if p.process; p.status=2; p.latch+=UInt32(1); end
    ko=p.e/0.5110034f0; broi=muladd(2f0,ko,1f0); bro=1f0/broi
    done=!p.process; st=0f0; ct=0f0; br=0f0
    while true
        r1=egs_rand(s,lane); r2=egs_rand(s,lane); r3=egs_rand(s,lane)
        if !done
            if ko>2f0
                broi2=broi*broi; a1=log(broi); a2=ko*(broi+1f0)*bro*bro
                if r1*(a1+a2)<a1; br=exp(a1*r2)*bro
                else; br=sqrt(muladd(r2,broi2,1f0-r2))*bro; end
                ct=(1f0-br)/(ko*br); st=max(0f0,ct*(2f0-ct)); aux=muladd(br,br,1f0)
                if r3*aux<muladd(-br,st,aux); done=true; end
            else
                bro1=1f0-bro; maximum=broi+bro; br=muladd(bro1,r1,bro)
                ct=(1f0-br)/(ko*br); st=max(0f0,ct*(2f0-ct))
                rejection=muladd(-br,st,muladd(br,br,1f0))
                if r2*br*maximum<rejection; done=true; end
            end
        end
        if egs_all(done); break; end
    end
    ct=1f0-ct; st=sqrt(st)
    if p.process; p.e*=br; end
    egs_uphi(c,s,lane,p,ct,st)
    nothing
end
@inline function egs_transport(c,lane,initialize,limit)
    s=egs_shared()
    for i in lane:Int32(32):Int32(512*512-1)
        for cat in 1:4; c.sc_count[cat][i+1]=0f0; c.sc_energy[cat][i+1]=0f0; end
    end
    CUDA.sync_threads()
    if lane<9; s.counts[lane+1]=0; end
    if lane==0; s.combined[1]=0.; s.depth[1]=0; s.iterations[1]=0; end
    egs_mt_read(c,s,lane)
    p=Photon(MVector{14,UInt32}(undef));j=lane*Int32(4)
    a=c.stack_a[j+1]; p.status=Int32(a&255); p.reserved=Int32((a>>8)&255)
    p.charge=Int32((a>>16)&255); p.process=((a>>24)&255)!=0
    p.e=reinterpret(Float32,c.stack_a[j+2]); p.wt=reinterpret(Float32,c.stack_a[j+3]); p.region=Int32(c.stack_a[j+4])
    p.latch=c.stack_b[j+1]; p.x=reinterpret(Float32,c.stack_b[j+2]); p.y=reinterpret(Float32,c.stack_b[j+3]); p.z=reinterpret(Float32,c.stack_b[j+4])
    p.u=reinterpret(Float32,c.stack_c[j+1]); p.v=reinterpret(Float32,c.stack_c[j+2]); p.w=reinterpret(Float32,c.stack_c[j+3])
    if initialize; p.status=7; end
    for iteration in Int32(1):Int32(32768)
        if lane==0; s.iterations[1]+=UInt32(1); end
        done=false
        if p.status==8 && !limit; p.status=7; end
        for status in (Int32(2),Int32(7),Int32(1),Int32(4),Int32(5),Int32(3),Int32(6),Int32(0))
            p.process=p.status==status; done|=p.process
            if egs_any(p.process)
                if status==2; egs_photon(c,s,lane,p)
                elseif status==7
                    egs_new(c,s,lane,p)
                    if lane==0
                        combined=0.
                        for k in 1:32; combined+=Float64(s.weights[k]); end
                        s.combined[1]+=combined
                    end
                elseif status==1; egs_user(c,p)
                elseif status==4; egs_compton(c,s,lane,p)
                elseif status==3; egs_rayleigh(c,s,lane,p)
                elseif p.process; p.status=8
                end
                n=egs_count(p.process)
                if lane==0; s.counts[status+1]+=UInt32(n); s.depth[1]+=UInt32(1); end
            end
            if status!=0 && egs_all(done); break; end
        end
        if egs_all(!done) && limit; break; end
    end
    CUDA.sync_threads()
    if lane<9; c.step_counts[lane+1]=UInt64(s.counts[lane+1]); end
    if lane==0
        if s.combined[1]>0.; c.total_weight[1]+=s.combined[1]; end
        c.list_depth[1]=UInt64(s.depth[1]); c.inner_iterations[1]=UInt64(s.iterations[1])
    end
    c.stack_a[j+1]=UInt32(p.status)|(UInt32(p.reserved)<<8)|(UInt32(p.charge)<<16)|(UInt32(p.process)<<24)
    c.stack_a[j+2]=reinterpret(UInt32,p.e); c.stack_a[j+3]=reinterpret(UInt32,p.wt); c.stack_a[j+4]=UInt32(p.region)
    c.stack_b[j+1]=p.latch; c.stack_b[j+2]=reinterpret(UInt32,p.x); c.stack_b[j+3]=reinterpret(UInt32,p.y); c.stack_b[j+4]=reinterpret(UInt32,p.z)
    c.stack_c[j+1]=reinterpret(UInt32,p.u); c.stack_c[j+2]=reinterpret(UInt32,p.v); c.stack_c[j+3]=reinterpret(UInt32,p.w); c.stack_c[j+4]=reinterpret(UInt32,p.z)
    for i in lane:Int32(32):Int32(100); c.mt_state[i+1]=s.state[i+1]; end
    nothing
end
@inline function egs_accumulate(c,block,lane)
    cat=block%Int32(4)+Int32(1); docount=block÷Int32(4)!=0
    scores=docount ? c.sc_count[cat] : c.sc_energy[cat]
    total=docount ? c.det_count[cat] : c.det_energy[cat]
    for i in lane:Int32(1024):Int32(512*512-1); total[i+1]+=Float64(scores[i+1]); end
    nothing
end
