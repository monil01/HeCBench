using CUDA,JACC
using Printf

const P_X = Int32(5)
const P_Y = Int32(1)
const MODULUS = UInt32(17)
const A_COEFF = Int32(2)

@inline function ext_euclidian_alg(a::Int32, b::UInt32)
    x = Int32(1)
    y = Int32(0)
    x1 = Int32(0)
    y1 = Int32(1)
    a1 = a
    b1 = Int32(b)

    while b1 != Int32(0)
        q = a1 ÷ b1
        s = x1
        t = x - q * x1
        x = s
        x1 = t
        s = y1
        t = y - q * y1
        y = s
        y1 = t
        s = b1
        t = a1 - q * b1
        a1 = s
        b1 = t
    end

    return a1, x, y
end

@inline function make_positive(a::Int32, m::UInt32)
    v = a
    mm = Int32(m)
    while v < Int32(0)
        v += mm
    end
    return Int32(v % mm)
end

@inline function find_inverse(a::Int32, m::UInt32)
    _, t, _ = ext_euclidian_alg(a, m)
    return make_positive(t, m)
end

@inline function point_addition(m::UInt32, x1::Int32, y1::Int32, x2::Int32, y2::Int32)
    temp = make_positive(x2 - x1, m)
    slope = make_positive((y2 - y1) * find_inverse(temp, m), m)
    x3 = make_positive(slope * slope - x1 - x2, m)
    y3 = make_positive(slope * (x1 - x3) - y1, m)
    return x3, y3
end

@inline function point_doubling(m::UInt32, a::Int32, x1::Int32, y1::Int32)
    slope = (Int32(3) * x1 * x1 + a) * find_inverse(Int32(2) * y1, m)
    x3 = make_positive(slope * slope - Int32(2) * x1, m)
    y3 = make_positive(slope * (x1 - x3) - y1, m)
    return x3, y3
end

@inline function first_set_bit(n::Int32)
    for i in Int32(31):-Int32(1):Int32(0)
        if ((Int32(1) << i) & n) != Int32(0)
            return i
        end
    end
    return Int32(0)
end

@inline function make_pk_fast(sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32)
    tx = px
    ty = py

    for i in (first_set_bit(sk) - Int32(1)):-Int32(1):Int32(0)
        tx, ty = point_doubling(m, a, tx, ty)
        if ((Int32(1) << i) & sk) != Int32(0)
            tx, ty = point_addition(m, tx, ty, px, py)
        end
    end

    return tx, ty
end

@inline function make_pk_slow(sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32)
    tx, ty = point_doubling(m, a, px, py)
    remaining = sk - Int32(2)

    while remaining > Int32(0)
        tx, ty = point_addition(m, tx, ty, px, py)
        remaining -= Int32(1)
    end

    return tx, ty
end

function slow_kernel!(i,pk_x, pk_y, sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32, num_pk::Int32)
    if i <= num_pk
        x, y = make_pk_slow(sk, px, py, m, a)
        @inbounds pk_x[i] = x
        @inbounds pk_y[i] = y
    end
end

function fast_kernel!(i,pk_x, pk_y, sk::Int32, px::Int32, py::Int32, m::UInt32, a::Int32, num_pk::Int32)
    if i <= num_pk
        x, y = make_pk_fast(sk, px, py, m, a)
        @inbounds pk_x[i] = x
        @inbounds pk_y[i] = y
    end
end

function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||error("numerical failure")
end
function oracle_add(p,q)
 p===nothing&&return q;q===nothing&&return p
 x,y=p;u,v=q
 x==u&&mod(y+v,17)==0&&return nothing
 slope=p==q ? mod((3x*x+2)*invmod(2y,17),17) : mod((v-y)*invmod(u-x,17),17)
 z=mod(slope*slope-x-u,17);return(z,mod(slope*(x-z)-y,17))
end
function launch(case,x,y,n)
 if case==1
  JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,slow_kernel!,x,y,Int32(18),P_X,P_Y,MODULUS,A_COEFF,Int32(n))
 else
  JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,fast_kernel!,x,y,Int32(18),P_X,P_Y,MODULUS,A_COEFF,Int32(n))
 end
end
function main(args)
 n=length(args)>0 ? parse(Int,args[1]) : 4096;repeat=length(args)>1 ? parse(Int,args[2]) : 10
 point=nothing;for _ in 1:18;point=oracle_add(point,(5,1));end
 expected=vcat(fill(Float64(point[1]),n),fill(Float64(point[2]),n));expected=vcat(expected,expected)
 x=CUDA.zeros(Int32,n);y=CUDA.zeros(Int32,n)
 for case in 1:2;launch(case,x,y,n);end;CUDA.synchronize()
 for trial in 1:5
  result=Float64[];times=Float64[]
  for case in 1:2
   CUDA.synchronize();start=time_ns();for _ in 1:repeat;launch(case,x,y,n);end;CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);append!(result,Float64.(Array(x)));append!(result,Float64.(Array(y)))
  end
  emit(trial,sum(times),result,expected,repeat;case_times=times)
 end
end
main(ARGS)
