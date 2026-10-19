# Authored XORWOW state/Box-Muller/Marsaglia pipeline; initial state via declared native setup interop.
@inline function rng_word!(s)
    t=s[2]⊻(s[2]>>2)
    s[2]=s[3];s[3]=s[4];s[4]=s[5];s[5]=s[6]
    s[6]=(s[6]⊻(s[6]<<4))⊻(t⊻(t<<1));s[1]+=UInt32(362437)
    s[6]+s[1]
end
@inline rng_uniform!(s)=fma(Float32(rng_word!(s)),2.3283064f-10,2.3283064f-10/2f0)
@inline function rng_normal!(s)
    if s[7]==UInt32(1)
        s[7]=UInt32(0);return reinterpret(Float32,s[9])
    end
    x=rng_word!(s);y=rng_word!(s)
    u=fma(Float32(x),2.3283064f-10,2.3283064f-10/2f0)
    c=reinterpret(Float32,UInt32(0x30c90fdb));v=fma(Float32(y),c,c/2f0)
    radius=sqrt(-2f0*log(u));ss,cc=Base.FastMath.sincos_fast(v)
    s[9]=reinterpret(UInt32,cc*radius);s[7]=UInt32(1)
    ss*radius
end
@inline function rng_gamma!(s,shape,rate)
    d=shape-1f0/3f0;c=1f0/sqrt(9f0*d)
    flag=true;v=0f0
    while flag
        z=rng_normal!(s)
        if z> -1f0/c
            v=fma(c,z,1f0)^3f0;u=rng_uniform!(s)
            rhs=fma(d,log(v),fma(-d,v,fma(z,.5f0*z,d)))
            flag=log(u)>rhs
        end
    end
    d*v/rate
end
@inline function sample_index!(id,state,theta,y,n,a,b)
    base=17+id*12;s=MVector{9,UInt32}(undef)
    for k=1:9;s[k]=state[base+k-1];end
    shape=a+Float32(y[17+id]);rate=b+n[17+id]
    value=if shape<1f0
        v=rng_gamma!(s,shape+1f0,rate);v*rng_uniform!(s)^(1f0/shape)
    else
        rng_gamma!(s,shape,rate)
    end
    theta[17+id]=value
    for k=1:9;state[base+k-1]=s[k];end
    nothing
end
