using CUDA,KernelAbstractions
using Printf

# Julia port of atan2-cuda: polynomial approximations of atan2 at various
# precisions.  We reproduce the polynomials from the CUDA source verbatim
# (Sollya-generated).

# Constants match main.cu.  We inline these polynomials at the call sites
# in device kernels because CUDA.jl kernels don't easily support templated
# constexpr calls; the CPU reference uses the same functions.

# We use hex-float parsing via reinterpret to match float(-0xf.8eed2p-4) etc.
# In Julia, we can just write these as Float32 hex-decimal literals.

# ---- float polynomials (safe_atan2f) ----
@inline function approx_atan2f_P3(x::Float32)
    return x * (Float32(-0xf.8eed2p-4) + x * x * Float32(0x3.1238p-4))
end
@inline function approx_atan2f_P5(x::Float32)
    z = x * x
    return x * (Float32(-0xf.ecfc8p-4) + z * (Float32(0x4.9e79dp-4) + z * Float32(-0x1.44f924p-4)))
end
@inline function approx_atan2f_P7(x::Float32)
    z = x * x
    return x * (Float32(-0xf.fcc7ap-4) + z * (Float32(0x5.23886p-4) + z * (Float32(-0x2.571968p-4) + z * Float32(0x9.fb05p-8))))
end
@inline function approx_atan2f_P9(x::Float32)
    z = x * x
    return x * (Float32(-0xf.ff73ep-4) + z * (Float32(0x5.48ee1p-4) + z * (Float32(-0x2.e1efe8p-4) + z * (Float32(0x1.5cce54p-4) + z * Float32(-0x5.56245p-8)))))
end
@inline function approx_atan2f_P11(x::Float32)
    z = x * x
    return x * (Float32(-0xf.ffe82p-4) + z * (Float32(0x5.526c8p-4) + z * (Float32(-0x3.18bea8p-4) + z * (Float32(0x1.dce3bcp-4) + z * (Float32(-0xd.7a64ap-8) + z * Float32(0x3.000eap-8))))))
end
@inline function approx_atan2f_P13(x::Float32)
    z = x * x
    return x * (Float32(-0xf.fffbep-4) + z * (Float32(0x5.54adp-4) + z * (Float32(-0x3.2b4df8p-4) + z * (Float32(0x2.1df79p-4) + z * (Float32(-0x1.46081p-4) + z * (Float32(0x8.99028p-8) + z * Float32(-0x1.be0bc4p-8)))))))
end
@inline function approx_atan2f_P15(x::Float32)
    z = x * x
    c0 = Float32(-0xf.ffff4p-4)
    c1 = Float32(0x5.552f9p-4)
    c2 = Float32(-0x3.30f728p-4)
    c3 = Float32(0x2.39826p-4)
    c4 = Float32(-0x1.8a880cp-4)
    c5 = Float32(0xe.484d6p-8)
    c6 = Float32(-0x5.93d5p-8)
    c7 = Float32(0x1.0875dcp-8)
    tail = z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * c7)))))
    inner = Float32(0x5.552f9p-4 + Float64(tail))
    return x * (c0 + z * inner)
end

# helper implementations
@inline function unsafe_atan2f_impl(y::Float32, x::Float32, deg::Val{D}) where {D}
    pi4f = Float32(3.1415926535897932384626434 / 4)
    pi34f = Float32(3.1415926535897932384626434 * 3 / 4)
    absx = abs(x); absy = abs(y)
    r = (absx - absy) / (absx + absy)
    if x < 0f0
        r = -r
    end
    angle = x >= 0f0 ? pi4f : pi34f
    angle += @inline (D == 3 ? approx_atan2f_P3(r) :
                     D == 5 ? approx_atan2f_P5(r) :
                     D == 7 ? approx_atan2f_P7(r) :
                     D == 9 ? approx_atan2f_P9(r) :
                     D == 11 ? approx_atan2f_P11(r) :
                     D == 13 ? approx_atan2f_P13(r) :
                     approx_atan2f_P15(r))
    return y < 0f0 ? -angle : angle
end

@inline function safe_atan2f(y::Float32, x::Float32, ::Val{D}) where {D}
    xs = ((y == 0f0) & (x == 0f0)) ? 0.2f0 : x
    return unsafe_atan2f_impl(y, xs, Val(D))
end

# ---- int polynomials ----
@inline function approx_atan2i_P3(x::Float32); z = x*x; x * (-664694912f0 + z * 131209024f0); end
@inline function approx_atan2i_P5(x::Float32); z = x*x; x * (-680392064f0 + z * (197338400f0 + z * (-54233256f0))); end
@inline function approx_atan2i_P7(x::Float32); z = x*x; x * (-683027840f0 + z * (219543904f0 + z * (-99981040f0 + z * 26649684f0))); end
@inline function approx_atan2i_P9(x::Float32); z = x*x; x * (-683473920f0 + z * (225785056f0 + z * (-123151184f0 + z * (58210592f0 + z * (-14249276f0))))); end
@inline function approx_atan2i_P11(x::Float32); z = x*x; x * (-683549696f0 + z * (227369312f0 + z * (-132297008f0 + z * (79584144f0 + z * (-35987016f0 + z * 8010488f0))))); end
@inline function approx_atan2i_P13(x::Float32); z = x*x; x * (-683562624f0 + z * (227746080f0 + z * (-135400128f0 + z * (90460848f0 + z * (-54431464f0 + z * (22973256f0 + z * (-4657049f0))))))); end
@inline function approx_atan2i_P15(x::Float32); z = x*x; x * (-683562624f0 + z * (227746080f0 + z * (-135400128f0 + z * (90460848f0 + z * (-54431464f0 + z * (22973256f0 + z * (-4657049f0))))))); end

@inline function unsafe_atan2i_impl(y::Float32, x::Float32, ::Val{D}) where {D}
    maxint = Int64(typemax(Int32)) + 1
    pi4 = Int32(maxint ÷ 4)
    pi34 = Int32(3 * maxint ÷ 4)
    absx = abs(x); absy = abs(y)
    r = (absx - absy) / (absx + absy)
    if x < 0f0
        r = -r
    end
    angle = x >= 0f0 ? pi4 : pi34
    poly = D == 3 ? approx_atan2i_P3(r) :
           D == 5 ? approx_atan2i_P5(r) :
           D == 7 ? approx_atan2i_P7(r) :
           D == 9 ? approx_atan2i_P9(r) :
           D == 11 ? approx_atan2i_P11(r) :
           D == 13 ? approx_atan2i_P13(r) :
           approx_atan2i_P15(r)
    angle += unsafe_trunc(Int32, poly)
    return y < 0f0 ? -angle : angle
end

# ---- short polynomials ----
@inline function approx_atan2s_P3(x::Float32); z = x*x; x * (-10142.439453125f0 + z * 2002.0908203125f0); end
@inline function approx_atan2s_P5(x::Float32); z = x*x; x * (-10381.9609375f0 + z * (3011.1513671875f0 + z * (-827.538330078125f0))); end
@inline function approx_atan2s_P7(x::Float32); z = x*x; x * (-10422.177734375f0 + z * (3349.97412109375f0 + z * (-1525.589599609375f0 + z * 406.64190673828125f0))); end
@inline function approx_atan2s_P9(x::Float32); z = x*x; x * (-10428.984375f0 + z * (3445.20654296875f0 + z * (-1879.137939453125f0 + z * (888.22314453125f0 + z * (-217.42669677734375f0))))); end

@inline function unsafe_atan2s_impl(y::Float32, x::Float32, ::Val{D}) where {D}
    maxshort = Int32(typemax(Int16)) + Int32(1)
    pi4 = Int16(maxshort ÷ 4)
    pi34 = Int16(3 * maxshort ÷ 4)
    absx = abs(x); absy = abs(y)
    r = (absx - absy) / (absx + absy)
    if x < 0f0
        r = -r
    end
    angle = x >= 0f0 ? pi4 : pi34
    poly = D == 3 ? approx_atan2s_P3(r) :
           D == 5 ? approx_atan2s_P5(r) :
           D == 7 ? approx_atan2s_P7(r) :
           approx_atan2s_P9(r)
    angle += unsafe_trunc(Int16, poly)
    return y < 0f0 ? -angle : angle
end

# ---- kernels ----
@kernel function compute_f_kernel!(n::Int32, x, y, r)
    i=@index(Global,Linear)
    if i<=n
    @inbounds vy = y[i]
    @inbounds vx = x[i]
    val = safe_atan2f(vy, vx, Val(3)) +
          safe_atan2f(vy, vx, Val(5)) +
          safe_atan2f(vy, vx, Val(7)) +
          safe_atan2f(vy, vx, Val(9)) +
          safe_atan2f(vy, vx, Val(11)) +
          safe_atan2f(vy, vx, Val(13)) +
          safe_atan2f(vy, vx, Val(15))
    @inbounds r[i] = val
    end
end

@kernel function compute_i_kernel!(n::Int32, x, y, r)
    i=@index(Global,Linear)
    if i<=n
    @inbounds vy = y[i]
    @inbounds vx = x[i]
    val = unsafe_atan2i_impl(vy, vx, Val(3)) +
          unsafe_atan2i_impl(vy, vx, Val(5)) +
          unsafe_atan2i_impl(vy, vx, Val(7)) +
          unsafe_atan2i_impl(vy, vx, Val(9)) +
          unsafe_atan2i_impl(vy, vx, Val(11)) +
          unsafe_atan2i_impl(vy, vx, Val(13)) +
          unsafe_atan2i_impl(vy, vx, Val(15))
    @inbounds r[i] = val
    end
end

@kernel function compute_s_kernel!(n::Int32, x, y, r)
    i=@index(Global,Linear)
    if i<=n
    @inbounds vy = y[i]
    @inbounds vx = x[i]
    val = unsafe_atan2s_impl(vy, vx, Val(3)) +
          unsafe_atan2s_impl(vy, vx, Val(5)) +
          unsafe_atan2s_impl(vy, vx, Val(7)) +
          unsafe_atan2s_impl(vy, vx, Val(9))
    @inbounds r[i] = val
    end
end

# ---- CPU references ----
function reference_f(n, x, y)
    r = Vector{Float32}(undef, n)
    for i in 1:n
        vy = y[i]; vx = x[i]
        r[i] = safe_atan2f(vy, vx, Val(3)) +
               safe_atan2f(vy, vx, Val(5)) +
               safe_atan2f(vy, vx, Val(7)) +
               safe_atan2f(vy, vx, Val(9)) +
               safe_atan2f(vy, vx, Val(11)) +
               safe_atan2f(vy, vx, Val(13)) +
               safe_atan2f(vy, vx, Val(15))
    end
    return r
end
function reference_i(n, x, y)
    r = Vector{Int32}(undef, n)
    for i in 1:n
        vy = y[i]; vx = x[i]
        r[i] = unsafe_atan2i_impl(vy, vx, Val(3)) +
               unsafe_atan2i_impl(vy, vx, Val(5)) +
               unsafe_atan2i_impl(vy, vx, Val(7)) +
               unsafe_atan2i_impl(vy, vx, Val(9)) +
               unsafe_atan2i_impl(vy, vx, Val(11)) +
               unsafe_atan2i_impl(vy, vx, Val(13)) +
               unsafe_atan2i_impl(vy, vx, Val(15))
    end
    return r
end
function reference_s(n, x, y)
    r = Vector{Int16}(undef, n)
    for i in 1:n
        vy = y[i]; vx = x[i]
        r[i] = unsafe_atan2s_impl(vy, vx, Val(3)) +
               unsafe_atan2s_impl(vy, vx, Val(5)) +
               unsafe_atan2s_impl(vy, vx, Val(7)) +
               unsafe_atan2s_impl(vy, vx, Val(9))
    end
    return r
end

function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function launch(case,n,x,y,out)
 if case==1
  compute_f_kernel!(get_backend(x),256)(Int32(n),x,y,out;ndrange=n)
 elseif case==2
  compute_i_kernel!(get_backend(x),256)(Int32(n),x,y,out;ndrange=n)
 else
  compute_s_kernel!(get_backend(x),256)(Int32(n),x,y,out;ndrange=n)
 end
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 4096;repeat=length(args)>1 ? parse(Int,args[2]) : 10
 x=Float32[4f0+Float32(i%1024)/256f0 for i in 0:n-1];y=Float32[.25f0+Float32((i*37)%1024)/1024f0 for i in 0:n-1]
 expected=vcat(Float64.(reference_f(n,x,y)),Float64.(reference_i(n,x,y)),Float64.(reference_s(n,x,y)))
 dx=CuArray(x);dy=CuArray(y);outputs=(CUDA.zeros(Float32,n),CUDA.zeros(Int32,n),CUDA.zeros(Int16,n))
 for case in 1:3;launch(case,n,dx,dy,outputs[case]);end;CUDA.synchronize()
 for trial in 1:5
  result=Float64[];times=Float64[]
  for case in 1:3
   CUDA.synchronize();start=time_ns();out=outputs[case]
   for _ in 1:repeat;launch(case,n,dx,dy,out);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);append!(result,Float64.(Array(out)))
  end
  emit(trial,sum(times),result,expected,repeat;atol=1e-6,case_times=times)
 end
end
main(ARGS)
