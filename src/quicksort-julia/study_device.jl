# Source Intel 2014-2019 custom GPU quicksort algorithm; see LICENSE.
# Shared storage uses aligned UInt64 words reinterpreted as source T / UInt32 regions.
struct QSWork{T}
    start::UInt32
    stop::UInt32
    pivot::T
    direction::UInt32
end
struct QSBlock{T}
    start::UInt32
    stop::UInt32
    pivot::T
    direction::UInt32
    parent::UInt32
end
@inline qs_u32(x::UInt32)=x
@inline qs_u32(x::Float64)=unsafe_trunc(UInt32,x)
@inline function qs_u32(x::Float32)
    # Actual native sm70 cast probe: Float32(UINT_MAX)=2^32 saturates back to UINT_MAX.
    if x>=4294967296f0
        return typemax(UInt32)
    end
    unsafe_trunc(UInt32,x)
end
@inline qs_median(a,b,c)=a<b ? (b<c ? b : (a<c ? c : a)) : (a<c ? a : (b<c ? c : b))
@inline function qs_oldadd!(a,i,x)
    pair=Atomix.@atomic :monotonic a[i]+x
    pair[1]
end
@inline function qs_scan!(c,lb,gb,ls,gs,tx,sync)
    width=Int32(1)
    while width<128
        mask=2width-1
        if (tx&mask)==mask
            @inbounds c[lb+tx+1]+=c[lb+tx-width+1]
            @inbounds c[gb+tx+1]+=c[gb+tx-width+1]
        end
        sync()
        width<<=1
    end
    if tx==127
        @inbounds begin
            c[lb+129]=c[lb+128];c[ls+1]=c[lb+128];c[lb+128]=UInt32(0)
            c[gb+129]=c[gb+128];c[gs+1]=c[gb+128];c[gb+128]=UInt32(0)
        end
    end
    width=Int32(64)
    while width>=1
        mask=2width-1
        if (tx&mask)==mask
            @inbounds begin
                a=c[lb+tx-width+1];b=c[lb+tx+1]
                c[lb+tx-width+1]=b;c[lb+tx+1]=b+a
                a=c[gb+tx-width+1];b=c[gb+tx+1]
                c[gb+tx-width+1]=b;c[gb+tx+1]=b+a
            end
        end
        sync()
        width>>=1
    end
    nothing
end
@inline function qs_global_body!(d,dn,blocks,parents,news,shared,bid,tx,sync)
    T=eltype(d);ctrl=reinterpret(UInt32,shared)
    @inbounds block=blocks[bid+1]
    start=Int32(block.start);stop=Int32(block.stop);piv=block.pivot;direction=block.direction
    src=direction==1 ? d : dn;dst=direction==1 ? dn : d
    @inbounds begin ctrl[tx+1]=UInt32(0);ctrl[129+tx+1]=UInt32(0) end
    sync()
    less=UInt32(0);greater=UInt32(0)
    i=start+tx
    while i<stop
        @inbounds value=qs_u32(src[i+1])
        less+=ifelse(T(value)<piv,UInt32(1),UInt32(0))
        greater+=ifelse(T(value)>piv,UInt32(1),UInt32(0))
        i+=128
    end
    @inbounds begin ctrl[tx+1]=less;ctrl[129+tx+1]=greater end
    sync()
    qs_scan!(ctrl,Int32(0),Int32(129),Int32(258),Int32(259),tx,sync)
    parent=Int32(block.parent)*5
    if tx==0
        @inbounds begin
            ls=ctrl[259];gs=ctrl[260]
            ctrl[261]=qs_oldadd!(parents,parent+1,ls)
            ctrl[262]=qs_oldadd!(parents,parent+2,-gs)-gs
        end
    end
    sync()
    @inbounds lfrom=Int32(ctrl[261]+ctrl[tx+1])
    @inbounds gfrom=Int32(ctrl[262]+ctrl[129+tx+1])
    i=start+tx
    while i<stop
        @inbounds value=qs_u32(src[i+1])
        if T(value)<piv
            @inbounds dst[lfrom+1]=value;lfrom+=1
        end
        if T(value)>piv
            @inbounds dst[gfrom+1]=value;gfrom+=1
        end
        i+=128
    end
    sync()
    if tx==0 && qs_oldadd!(parents,parent+5,typemax(UInt32))==0
        @inbounds begin
            ss=parents[parent+1];se=parents[parent+2]
            os=parents[parent+3];oe=parents[parent+4]
        end
        i=Int32(ss)
        while i<Int32(se)
            @inbounds d[i+1]=piv;i+=1
        end
        @inbounds lp=qs_u32(dst[Int32(os)+1])
        @inbounds gp=qs_u32(dst[Int32(oe)])
        if os<ss
            @inbounds lp=qs_median(lp,qs_u32(dst[Int32((os+ss)>>1)+1]),qs_u32(dst[Int32(ss)]))
        end
        if se<oe
            @inbounds gp=qs_median(qs_u32(dst[Int32(se)+1]),qs_u32(dst[Int32((oe+se)>>1)+1]),gp)
        end
        T=eltype(d);nextdir=direction⊻UInt32(1)
        @inbounds news[2bid+1]=QSWork{T}(os,ss,T(lp),nextdir)
        @inbounds news[2bid+2]=QSWork{T}(se,oe,T(gp),nextdir)
    end
    nothing
end
@inline function qs_bitonic!(data,base,tx,sync)
    level=Int32(1)
    while level<128
        width=level
        while width>0
            pos=2tx-(tx&(width-1));direction=tx&level
            @inbounds a=qs_u32(data[base+pos+1])
            @inbounds b=qs_u32(data[base+pos+width+1])
            hi=max(a,b);lo=min(a,b)
            @inbounds data[base+pos+1]=direction!=0 ? hi : lo
            @inbounds data[base+pos+width+1]=direction!=0 ? lo : hi
            sync();width>>=1
        end
        level<<=1
    end
    width=Int32(128)
    while width>0
        pos=2tx-(tx&(width-1))
        @inbounds a=qs_u32(data[base+pos+1])
        @inbounds b=qs_u32(data[base+pos+width+1])
        @inbounds data[base+pos+1]=min(a,b)
        @inbounds data[base+pos+width+1]=max(a,b)
        sync();width>>=1
    end
    nothing
end
@inline function qs_threshold!(data,sbase,d,doffset,start,stop,tx,sync)
    count=stop-start
    if count==256
        qs_bitonic!(data,sbase+start,tx,sync)
        i=tx
        while i<256
            @inbounds d[doffset+start+i+1]=data[sbase+start+i+1]
            i+=128
        end
    elseif count>1
        i=tx
        while i<256
            @inbounds data[3456+i+1]=i<count ? data[sbase+start+i+1] : eltype(data)(typemax(UInt32))
            i+=128
        end
        sync();qs_bitonic!(data,Int32(3456),tx,sync)
        i=tx
        while i<count
            @inbounds d[doffset+start+i+1]=data[3456+i+1]
            i+=128
        end
    elseif count==1 && tx==0
        @inbounds d[doffset+start+1]=data[sbase+start+1]
    end
    nothing
end
@inline function qs_push!(ctrl,sp,stack,start,stop,direction,tx,sync)
    if tx==0
        @inbounds pointer=reinterpret(Int32,ctrl[sp+1])+Int32(1)
        @inbounds begin
            ctrl[sp+1]=reinterpret(UInt32,pointer)
            ctrl[stack+3pointer+1]=UInt32(start)
            ctrl[stack+3pointer+2]=UInt32(stop)
            ctrl[stack+3pointer+3]=direction⊻UInt32(1)
        end
    end
    sync()
    nothing
end
@inline function qs_local_body!(d,dn,seqs,shared,bid,tx,sync)
    T=eltype(d);data=reinterpret(T,shared);ctrl=reinterpret(UInt32,shared)
    control=Int32(3712*sizeof(T)÷4);lb=control;gb=control+129;ls=control+258;gs=control+259;sp=control+260;stack=control+261
    @inbounds record=seqs[bid+1]
    doffset=Int32(record.start);initialend=Int32(record.stop-record.start)
    if tx==0
        @inbounds begin
            ctrl[sp+1]=UInt32(0)
            ctrl[stack+1]=UInt32(0);ctrl[stack+2]=UInt32(initialend);ctrl[stack+3]=UInt32(1)
        end
    end
    src=record.direction==1 ? d : dn
    i=tx
    while i<initialend
        @inbounds data[i+1]=src[i+doffset+1]
        i+=128
    end
    sync()
    @inbounds pointer=reinterpret(Int32,ctrl[sp+1])
    while pointer>=0
        @inbounds begin
            start=Int32(ctrl[stack+3pointer+1]);stop=Int32(ctrl[stack+3pointer+2]);direction=ctrl[stack+3pointer+3]
        end
        sync()
        if tx==0
            @inbounds ctrl[sp+1]=reinterpret(UInt32,pointer-Int32(1))
            @inbounds begin ctrl[ls+1]=UInt32(0);ctrl[gs+1]=UInt32(0) end
        end
        sbase=direction==1 ? Int32(0) : Int32(1728)
        nbase=direction==1 ? Int32(1728) : Int32(0)
        @inbounds begin ctrl[lb+tx+1]=UInt32(0);ctrl[gb+tx+1]=UInt32(0) end
        less=UInt32(0);greater=UInt32(0)
        sync()
        @inbounds piv=qs_u32(data[sbase+start+1])
        if start<stop
            @inbounds piv=qs_median(piv,qs_u32(data[sbase+((start+stop)>>1)+1]),qs_u32(data[sbase+stop]))
        end
        i=start+tx
        while i<stop
            @inbounds value=qs_u32(data[sbase+i+1])
            less+=ifelse(value<piv,UInt32(1),UInt32(0));greater+=ifelse(value>piv,UInt32(1),UInt32(0));i+=128
        end
        @inbounds begin ctrl[lb+tx+1]=less;ctrl[gb+tx+1]=greater end
        sync();qs_scan!(ctrl,lb,gb,ls,gs,tx,sync)
        @inbounds lfrom=start+Int32(ctrl[lb+tx+1])
        @inbounds gfrom=stop-Int32(ctrl[gb+tx+2])
        i=start+tx
        while i<stop
            @inbounds value=qs_u32(data[sbase+i+1])
            if value<piv
                @inbounds data[nbase+lfrom+1]=value;lfrom+=1
            end
            if value>piv
                @inbounds data[nbase+gfrom+1]=value;gfrom+=1
            end
            i+=128
        end
        sync()
        @inbounds lesscount=Int32(ctrl[ls+1])
        @inbounds greatercount=Int32(ctrl[gs+1])
        i=start+lesscount+tx
        while i<stop-greatercount
            @inbounds d[doffset+i+1]=piv;i+=128
        end
        sync()
        if lesscount<=256
            qs_threshold!(data,nbase,d,doffset,start,start+lesscount,tx,sync)
        else
            qs_push!(ctrl,sp,stack,start,start+lesscount,direction,tx,sync)
        end
        if greatercount<=256
            qs_threshold!(data,nbase,d,doffset,stop-greatercount,stop,tx,sync)
        else
            qs_push!(ctrl,sp,stack,stop-greatercount,stop,direction,tx,sync)
        end
        @inbounds pointer=reinterpret(Int32,ctrl[sp+1])
    end
    nothing
end
