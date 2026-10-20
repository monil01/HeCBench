@inline packed(x::Int32,y::Int32)=UInt64(reinterpret(UInt32,x)) | (UInt64(reinterpret(UInt32,y))<<32)
@inline low(x::UInt64)=reinterpret(Int32,UInt32(x&0xffffffff))
@inline high(x::UInt64)=reinterpret(Int32,UInt32(x>>32))
@inline function edge_score(nxt::UInt64,cur::UInt64,avg::Float32)
    tag1=nxt>>48;tag0=cur>>48
    dx=Int32(nxt&0xffff)-Int32(cur&0xffff)
    dy=Int32((nxt>>16)&0xffff)-Int32((cur>>16)&0xffff)
    dd=abs(dx-dy)
    if tag1!=tag0 || dx==0 || dx>5000 || dy>5000 || dy<=0 || dd>500
        return -Int32(0x3fffffff)
    end
    logdd=dd<2 ? Int32(0) : dd<4 ? Int32(1) : dd<8 ? Int32(2) : dd<16 ? Int32(3) : dd<32 ? Int32(4) : dd<64 ? Int32(5) : dd<128 ? Int32(6) : dd<256 ? Int32(7) : Int32(8)
    width=Int32((nxt>>32)&0xffff)
    min(min(dx,dy),width)-unsafe_trunc(Int32,Float32(dd)*(0.01f0*avg))-(logdd>>1)
end
