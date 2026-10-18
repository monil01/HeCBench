const TAUSWORTHE_NUM_THREADS=Int32(256)
const TAUSWORTHE_TOTAL_NUM_THREADS=Int32(256)
const LOOKBACK_PATHS_PER_SIM=Int32(512)
const PI_F=Float32(pi)
@inline function taus_step(z::UInt32, s1::Int32, s2::Int32, s3::Int32, m::UInt32)
    b = ((z << s1) ⊻ z) >> s2
    nz = ((z & m) << s3) ⊻ b
    return nz
end

@inline function lcg_step(z::UInt32)
    return UInt32(1664525) * z + UInt32(1013904223)
end

@inline function uniform_taus(z1::UInt32, z2::UInt32, z3::UInt32, z4::UInt32)
    z1n = taus_step(z1, Int32(13), Int32(19), Int32(12), UInt32(4294967294))
    z2n = taus_step(z2, Int32(2), Int32(25), Int32(4), UInt32(4294967288))
    z3n = taus_step(z3, Int32(3), Int32(11), Int32(17), UInt32(4294967280))
    z4n = lcg_step(z4)
    v = 2.3283064f-10 * Float32((z1n ⊻ z2n ⊻ z3n) ⊻ z4n)
    return v, z1n, z2n, z3n, z4n
end

@inline function box_muller(u1::Float32, u2::Float32)
    z1 = sqrt(-2.0f0 * log(u1))
    a = 2.0f0 * PI_F * u2
    return z1 * sin(a), z1 * cos(a)
end

@inline function random_taus(z1::UInt32, z2::UInt32, z3::UInt32, z4::UInt32,
                             temporary::Float32, phase::UInt32)
    if (phase & UInt32(1)) != UInt32(0)
        return temporary, temporary, z1, z2, z3, z4
    end
    u1, z1, z2, z3, z4 = uniform_taus(z1, z2, z3, z4)
    u2, z1, z2, z3, z4 = uniform_taus(z1, z2, z3, z4)
    r1, r2 = box_muller(u1, u2)
    return r1, r2, z1, z2, z3, z4
end

@inline function lookback_sim(num_cycles::UInt32, vol0::Float32, eps0::Float32,
                              a0::Float32, a1::Float32, a2::Float32, s0::Float32,
                              mu::Float32, z1::UInt32, z2::UInt32, z3::UInt32,
                              z4::UInt32, path, tx)
    temporary = 0.0f0
    vol = vol0
    eps = eps0
    s = s0
    base = tx

    for ti in 1:Int(num_cycles)
        t=UInt32(ti-1)
        path[base] = s
        base += TAUSWORTHE_NUM_THREADS
        vterm=muladd(a1*vol,vol,a0)
        vol=sqrt(muladd(a2*eps,eps,vterm))
        r, temporary, z1, z2, z3, z4 = random_taus(z1, z2, z3, z4, temporary, t)
        eps = r * vol
        eps = max(min(eps, 1.0f0), -1.0f0)
        s *= exp(mu + eps)
    end

    total = 0.0f0
    for _ in 1:Int(num_cycles)
        base -= TAUSWORTHE_NUM_THREADS
        total += max(path[base] - s, 0.0f0)
    end
    return total, z1, z2, z3, z4
end

@inline function lookback_site!(address0,tx,num_cycles::UInt32,seed_values,means,variances,vol0,eps0,a0,a1,a2,s0,mu,path)
    address = address0 + Int32(1)

    z1 = seed_values[address]
    z2 = seed_values[address + TAUSWORTHE_TOTAL_NUM_THREADS]
    z3 = seed_values[address + Int32(2) * TAUSWORTHE_TOTAL_NUM_THREADS]
    z4 = seed_values[address + Int32(3) * TAUSWORTHE_TOTAL_NUM_THREADS]

    mean = 0.0f0
    variance = 0.0f0
    for i in UInt32(1):UInt32(LOOKBACK_PATHS_PER_SIM)
        res, z1, z2, z3, z4 = lookback_sim(
            num_cycles, vol0[address], eps0[address], a0[address], a1[address],
            a2[address], s0[address], mu[address], z1, z2, z3, z4, path,tx)
        delta = res - mean
        mean += delta / Float32(i)
        variance=muladd(delta,res-mean,variance)
    end

    means[address] = mean
    variances[address] = variance / Float32(LOOKBACK_PATHS_PER_SIM - Int32(1))
    return
    nothing
end

@inline function rng_probe!(id,draws,seeds,out)
 z1=seeds[id+1];z2=seeds[id+257];z3=seeds[id+513];z4=seeds[id+769];u1=0f0;u2=0f0
 for j in Int32(1):draws
  u,z1,z2,z3,z4=uniform_taus(z1,z2,z3,z4)
  if j==1;u1=u;elseif j==2;u2=u;end
 end
 @inbounds out[6id+1]=Float64(z1);out[6id+2]=Float64(z2);out[6id+3]=Float64(z3);out[6id+4]=Float64(z4);out[6id+5]=Float64(u1);out[6id+6]=Float64(u2)
 nothing
end
