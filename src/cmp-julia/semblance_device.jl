# Source four-stage semblance scan; private window bound is the original MAX_W=16.
@inline function init_c_at(i,c,inc,c0)
    @inbounds c[i]=muladd(inc,Float32(i-1),c0)
    nothing
end
@inline function init_half_at(i,scalco,gx,gy,sx,sy,h)
    @inbounds begin
        scale=scalco[i]
        if -1e-13<scale<1e-13;scale=1f0
        elseif scale<0;scale=1f0/scale;end
        hx=(gx[i]-sx[i])*scale;hy=(gy[i]-sy[i])*scale
        # FACTOR is an unsuffixed double C literal; retain source promotion.
        h[i]=Float32(Float64(0.25f0*muladd(hy,hy,hx*hx))/1.0e6)
    end
    nothing
end
@inline function semblance_at(i,h,c,samples,num,stt,t_id0,t_idf,idt,dt,tau,w,nc,ns)
    i0=i-1
    if i0<ns*nc
        t0=div(i0,nc);cid=rem(i0,nc)
        den=0f0;linear=0f0;squared=0f0;m=0f0;err=0
        window=MVector{16,Float32}(undef)
        @inbounds begin
            curvature=c[cid+1];tzero=dt*Float32(t0);tzero=tzero*tzero
            for j in 1:w;window[j]=0f0;end
            for tr in t_id0:t_idf-1
                travel=sqrt(muladd(curvature,h[tr+1],tzero))*idt
                it=Base.unsafe_trunc(Int32,travel);ittau=it-tau;fraction=travel-Float32(it)
                if ittau>=0&&it+tau+1<ns
                    k1=ittau+(tr-t_id0)*ns
                    nextsample=samples[k1+1]
                    for j in 1:w
                        k1+=1;previous=nextsample;nextsample=samples[k1+1]
                        v=muladd(nextsample-previous,fraction,previous)
                        window[j]+=v;den=muladd(v,v,den);linear+=v
                    end
                    m+=1f0
                else
                    err+=1
                end
            end
            for j in 1:w;squared=muladd(window[j],window[j],squared);end
            if den>1e-13&&m>1e-13&&w>1e-13&&err<2
                num[i]=squared/(den*m);stt[i]=linear/(Float32(w)*m)
            else
                num[i]=-1f0;stt[i]=-1f0
            end
        end
    end
    nothing
end
@inline function redux_at(i,num,stt,ctr,str,stk,nc,cdp_id,ns)
    t0=i-1
    if t0<ns
        @inbounds begin
            best=0f0;best_flat=-1
            for q in t0*nc:(t0+1)*nc-1
                value=num[q+1]
                if value>best;best=value;best_flat=q;end
            end
            pos=cdp_id*ns+i
            ctr[pos]=Int32(rem(best_flat,nc)) # C remainder retains -1 sentinel.
            str[pos]=best;stk[pos]=best_flat>=0 ? stt[best_flat+1] : 0f0
        end
    end
    nothing
end
