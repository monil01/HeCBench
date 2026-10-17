@inline function rotation_less(a::Int32,b::Int32,sequence,n::Int32)
    a<0 && return false
    b<0 && return true
    for offset in Int32(0):n-Int32(1)
        @inbounds x=sequence[(a+offset)%n+Int32(1)]
        @inbounds y=sequence[(b+offset)%n+Int32(1)]
        x!=y && return x<y
    end
    false
end
