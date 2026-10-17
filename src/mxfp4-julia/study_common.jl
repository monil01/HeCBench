using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
@inline function fp16_fp4_bits(x::Float16)
    bits=Int32(reinterpret(UInt16,x));exp=(bits>>10)&31;sign=bits>>15;last=(bits>>9)&1
    targetexp=exp-14;tailbits=min(Int32(16),Int32(9)+max(Int32(1)-targetexp,Int32(0)))
    halfway=Int32(1)<<(tailbits-Int32(1));tail=(bits&2047)&((Int32(1)<<tailbits)-Int32(1))
    if tail<halfway
        mant=targetexp>0 ? last : Int32(0);newexp=exp
    elseif tail>halfway
        mant=targetexp>0&&last==0 ? Int32(1) : Int32(0)
        newexp=exp+((targetexp<=0||last==1) ? Int32(1) : Int32(0))
    else
        mant=Int32(0);newexp=exp>13 ? exp+(last==1 ? Int32(1) : Int32(0)) : Int32(0)
    end
    mant+=(newexp>17&&mant==0) ? Int32(1) : Int32(0)
    newexp=newexp>=13 ? clamp(newexp,Int32(13),Int32(17)) : Int32(0)
    return reinterpret(Float16,UInt16((sign<<15)+(newexp<<10)+(mant<<9)))
end
@inline function scaled_fp4(x::Float16,maxval::Float16)
    rounded=(Int32(reinterpret(UInt16,maxval))+Int32(256))&Int32(0xfc00)
    exponent=((rounded>>10)&31)-17
    scale=Float16(exp2(Float32(exponent)))
    normalized=Float16(x/scale)
    return Float16(fp16_fp4_bits(normalized)*scale)
end
function mxfp4_study(factory;n=131072,repeats=10)
    n>0&&mod(n,128)==0||error("namedFP16study requires multiple128")
    host=Vector{Float16}(undef,n);expected=Vector{Float32}(undef,n)
    levels=Float32[0,.5,1,1.5,2,3,4,6]
    for i in 0:n-1
        group=div(i,32);scale=exp2(Float32(mod(group,9)-4))
        value=Float32(mod(17*i,193)-96)/16f0
        mod(i,32)==31&&(value=isodd(group) ? -6f0 : 6f0)
        host[i+1]=Float16(value*scale)
    end
    # Independent nearest representable magnitude, ties to even FP4 code.
    for group in 0:div(n,32)-1
        maxval=maximum(abs.(Float32.(host[group*32+1:group*32+32])))
        scale=exp2(floor(log2(maxval))-2f0)
        for lane in 0:31
            value=Float32(host[group*32+lane+1])/scale;a=abs(value);best=1;distance=abs(a-levels[1])
            for j in 2:8
                e=abs(a-levels[j])
                if e<distance||(e==distance&&iseven(j-1));best=j;distance=e;end
            end
            expected[group*32+lane+1]=Float32(Float16(copysign(levels[best],value)*scale))
        end
    end
    input=CuArray(host);output=CUDA.zeros(Float16,n);launch! = factory(input,output,n)
    OwnedStudy.run_trials!(launch!,()->fill!(output,Float16(0)),()->Float32.(Array(output)),expected,repeats;atol=0.,rtol=0.)
end
