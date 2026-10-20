# Source-faithful CUDA backend leaves. These scalar volatile operations are explicit
# backend extensions, not delegation of a native compression kernel.
@inline function vload(p::Core.LLVMPtr{Int32,A}) where A
    CUDA.CUDACore.@asmcall("ld.volatile.global.u32 \$0, [\$1];", "=r,l,~{memory}",true,Int32,Tuple{Core.LLVMPtr{Int32,A}},p)
end
@inline function vstore(p::Core.LLVMPtr{Int32,A},v::Int32) where A
    CUDA.CUDACore.@asmcall("st.volatile.global.u32 [\$0], \$1;", "l,r,~{memory}",true,Cvoid,Tuple{Core.LLVMPtr{Int32,A},Int32},p,v)
    nothing
end
@inline function getlong(S,base::Int32,i::Int32)
    @inbounds lo=S[base+Int32(2)*i+Int32(1)]
    @inbounds hi=S[base+Int32(2)*i+Int32(2)]
    UInt64(lo)|(UInt64(hi)<<32)
end
@inline function setlong!(S,base::Int32,i::Int32,v::UInt64)
    @inbounds S[base+Int32(2)*i+Int32(1)]=UInt32(v&0xffffffff)
    @inbounds S[base+Int32(2)*i+Int32(2)]=UInt32(v>>32)
    nothing
end
@inline function scan32(v::Int32,S,tid::Int32)
    warp=tid>>5;lane=tid&Int32(31)
    for d in (Int32(1),Int32(2),Int32(4),Int32(8),Int32(16))
        x=CUDA.shfl_up_sync(typemax(UInt32),v,d)
        if lane>=d;v+=x;end
    end
    if lane==31;@inbounds S[warp+Int32(1)]=reinterpret(UInt32,v);end
    CUDA.sync_threads()
    if warp==0
        @inbounds w=reinterpret(Int32,S[lane+Int32(1)])
        for d in (Int32(1),Int32(2),Int32(4),Int32(8),Int32(16))
            x=CUDA.shfl_up_sync(typemax(UInt32),w,d)
            if lane>=d;w+=x;end
        end
        @inbounds S[lane+Int32(1)]=reinterpret(UInt32,w)
    end
    CUDA.sync_threads()
    if warp>0;@inbounds v+=reinterpret(Int32,S[warp]);end
    v
end
@inline function scan64(v::UInt64,S,tid::Int32)
    warp=tid>>5;lane=tid&Int32(31)
    for d in (Int32(1),Int32(2),Int32(4),Int32(8),Int32(16))
        x=CUDA.shfl_up_sync(typemax(UInt32),v,d)
        if lane>=d;v+=x;end
    end
    if lane==31;setlong!(S,Int32(0),warp,v);end
    CUDA.sync_threads()
    if warp==0
        w=getlong(S,Int32(0),lane)
        for d in (Int32(1),Int32(2),Int32(4),Int32(8),Int32(16))
            x=CUDA.shfl_up_sync(typemax(UInt32),w,d)
            if lane>=d;w+=x;end
        end
        setlong!(S,Int32(0),lane,w)
    end
    CUDA.sync_threads()
    if warp>0;v+=getlong(S,Int32(0),warp-Int32(1));end
    v
end
@inline function scandim64(v::UInt64,S,tid::Int32,dim::Int32)
    warp=tid>>5;lane=tid&Int32(31);tix=warp*dim+rem(tid,dim)
    d=dim
    while d<32
        x=CUDA.shfl_up_sync(typemax(UInt32),v,d)
        if lane>=d;v+=x;end
        d*=Int32(2)
    end
    if lane+dim>31;setlong!(S,Int32(0),tix,v);end
    CUDA.sync_threads()
    if warp<dim
        idx=lane*dim+warp;w=getlong(S,Int32(0),idx)
        for d in (Int32(1),Int32(2),Int32(4),Int32(8),Int32(16))
            x=CUDA.shfl_up_sync(typemax(UInt32),w,d)
            if lane>=d;w+=x;end
        end
        setlong!(S,Int32(0),idx,w)
    end
    CUDA.sync_threads()
    if warp>0;v+=getlong(S,Int32(0),tix-dim);end
    v
end
@inline function compress_body!(S,n::Int32,original,compressed,goffset,dim8::UInt8)
    tid=Int32(CUDA.threadIdx().x)-Int32(1);bid=Int32(CUDA.blockIdx().x)-Int32(1);gdim=Int32(CUDA.gridDim().x);dim=Int32(dim8)
    lanex=tid&Int32(63);warpx=tid&Int32(0x3c0);bid1=bid+Int32(1)==gdim ? Int32(0) : bid+Int32(1)
    initial=Int32(1)+div(n+Int32(63),Int32(64));last=div(n+Int32(1023),Int32(1024))-Int32(1);chunk=bid
    while chunk<=last
        idx=tid+chunk*Int32(1024);v1=UInt64(0)
        if idx<n
            @inbounds v1=original[idx+Int32(1)]
            setlong!(S,Int32(0),tid,v1)
        end
        CUDA.sync_threads()
        if tid>=dim && idx<n;v1-=getlong(S,Int32(0),tid-dim);end
        setlong!(S,Int32(2048),tid,v1)
        CUDA.sync_threads()
        v2=UInt64(0)
        for i=Int32(63):-Int32(1):Int32(0)
            v2=(v2<<1)+((getlong(S,Int32(2048),warpx+i)>>lanex)&UInt64(1))
        end
        setlong!(S,Int32(0),tid,v2)
        CUDA.sync_threads()
        if tid>0;v2-=getlong(S,Int32(0),tid-Int32(1));end
        loc=v2==0 ? Int32(0) : Int32(1)
        bitmap=CUDA.vote_ballot_sync(typemax(UInt32),loc!=0)
        if lanex==32;setlong!(S,Int32(2048),tid,UInt64(bitmap));end
        CUDA.sync_threads()
        if lanex==0 && idx<n
            @inbounds compressed[Int32(2)+div(idx,Int32(64))]=(getlong(S,Int32(2048),tid+Int32(32))<<32)+UInt64(bitmap)
        end
        loc=scan32(loc,S,tid)
        if v2!=0;setlong!(S,Int32(2048),loc-Int32(1),v2);end
        if tid==1023
            start=initial
            if chunk>0
                start=vload(pointer(goffset,bid+Int32(1)))
                while start<0;start=vload(pointer(goffset,bid+Int32(1)));end
            end
            vstore(pointer(goffset,bid1+Int32(1)),start+loc);vstore(pointer(goffset,bid+Int32(1)),Int32(-1))
            if chunk==last;@inbounds compressed[1]=(UInt64(start+loc)<<32)+UInt64(0x43504cff)+UInt64(dim);end
            @inbounds S[4097]=reinterpret(UInt32,start)
            @inbounds S[4098]=reinterpret(UInt32,loc)
        end
        CUDA.sync_threads()
        @inbounds top=reinterpret(Int32,S[4098]);@inbounds start=reinterpret(Int32,S[4097])
        if tid<top;@inbounds compressed[start+tid+Int32(1)]=getlong(S,Int32(2048),tid);end
        chunk+=gdim
    end
    nothing
end
@inline function decompress_body!(S,compressed,decompressed,goffset)
    @inbounds header=compressed[1]
    dim=Int32(header&UInt64(31))+Int32(1);n=Int32(header>>32)
    tid=Int32(CUDA.threadIdx().x)-Int32(1);bid=Int32(CUDA.blockIdx().x)-Int32(1);gdim=Int32(CUDA.gridDim().x)
    lanex=tid&Int32(63);warpx=tid&Int32(0x3c0);bid1=bid+Int32(1)==gdim ? Int32(0) : bid+Int32(1)
    initial=Int32(1)+div(n+Int32(63),Int32(64));nru=(n-Int32(1))|Int32(63);last=div(n+Int32(1023),Int32(1024))-Int32(1);chunk=bid
    while chunk<=last
        idx=tid+chunk*Int32(1024);flag=Int32(0)
        if idx<=nru;@inbounds flag=Int32((compressed[Int32(2)+div(idx,Int32(64))]>>lanex)&UInt64(1));end
        CUDA.sync_threads();loc=scan32(flag,S,tid)
        if tid==1023
            start=initial
            if chunk>0
                start=vload(pointer(goffset,bid+Int32(1)))
                while start<0;start=vload(pointer(goffset,bid+Int32(1)));end
            end
            vstore(pointer(goffset,bid1+Int32(1)),start+loc);vstore(pointer(goffset,bid+Int32(1)),Int32(-1))
            @inbounds S[4097]=reinterpret(UInt32,start)
            @inbounds S[4098]=reinterpret(UInt32,loc)
        end
        CUDA.sync_threads()
        @inbounds top=reinterpret(Int32,S[4098]);@inbounds start=reinterpret(Int32,S[4097])
        if tid<top;@inbounds setlong!(S,Int32(2048),tid,compressed[start+tid+Int32(1)]);end
        CUDA.sync_threads();v2=flag!=0 ? getlong(S,Int32(2048),loc-Int32(1)) : UInt64(0)
        v2=scan64(v2,S,tid);setlong!(S,Int32(2048),tid,v2)
        CUDA.sync_threads();v1=UInt64(0)
        for i=Int32(63):-Int32(1):Int32(0)
            v1=(v1<<1)+((getlong(S,Int32(2048),warpx+i)>>lanex)&UInt64(1))
        end
        v1=scandim64(v1,S,tid,dim)
        if idx<n;@inbounds decompressed[idx+Int32(1)]=v1;end
        chunk+=gdim
    end
    nothing
end
