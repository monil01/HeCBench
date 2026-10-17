function potential_device(a::Float64, b::Float64, x::Float64, y::Float64)
    return 2.0 * ((x / a / a)^2.0 + (y / b / b)^2.0) + 1.0 / a / a + 1.0 / b / b
end

function r8_uniform_01_device(seed::Int32)
    k = seed ÷ Int32(127773)
    seed = Int32(16807) * (seed - k * Int32(127773)) - k * Int32(2836)
    if seed < 0
        seed += Int32(2147483647)
    end
    return seed, Float64(seed) * 4.656612875e-10
end

function point(i,j,ni,nj,seed0,n_paths,a,b,h,rth)
    if i <= ni && j <= nj
        x = (Float64(nj - j) * (-a) + Float64(j - Int32(1)) * a) / Float64(nj - Int32(1))
        y = (Float64(ni - i) * (-b) + Float64(i - Int32(1)) * b) / Float64(ni - Int32(1))
        chk = (x / a)^2.0 + (y / b)^2.0

        if !(1.0 < chk)
            inside=Int32(1)
            w_exact = exp((x / a)^2.0 + (y / b)^2.0 - 1.0)
            wt = 0.0
            seed = seed0
            for _ in Int32(1):n_paths
                x1 = x
                x2 = y
                w = 1.0
                chk = 0.0
                while chk < 1.0
                    seed, ut = r8_uniform_01_device(seed)
                    if ut < 0.5
                        seed, us = r8_uniform_01_device(seed)
                        dx = us - 0.5 < 0.0 ? -rth : rth
                    else
                        dx = 0.0
                    end

                    seed, ut = r8_uniform_01_device(seed)
                    if ut < 0.5
                        seed, us = r8_uniform_01_device(seed)
                        dy = us - 0.5 < 0.0 ? -rth : rth
                    else
                        dy = 0.0
                    end

                    vs = potential_device(a, b, x1, x2)
                    x1 += dx
                    x2 += dy
                    vh = potential_device(a, b, x1, x2)
                    we = (1.0 - h * vs) * w
                    w = w - 0.5 * h * (vh * we + vs * w)
                    chk = (x1 / a)^2.0 + (x2 / b)^2.0
                end
                wt += w
            end
            wt /= Float64(n_paths)
            return Int32(1),(w_exact-wt)^2.0
        end
    end
 return Int32(0),0.0
end
const NI=Int32(17);const NJ=Int32(9);const PATHS=Int32(32)
function study(run!)
 seed=Int32(123456789);a=2.;b=1.;h=.001;rth=sqrt(2h);args=(NI,NJ,seed,PATHS,a,b,h,rth)
 expected=zeros(Float64,2)
 for j in Int32(1):NJ,i in Int32(1):NI
  n,e=point(i,j,args...);expected[1]+=n;expected[2]+=e
 end
 count=CUDA.zeros(Int32,1);err=CUDA.zeros(Float64,1);run!(args...,count,err);CUDA.synchronize()
 for trial in 1:5
  total=0.
  for k in 1:3
   fill!(count,0);fill!(err,0.);CUDA.synchronize();t=time_ns();run!(args...,count,err);CUDA.synchronize();total+=(time_ns()-t)*1e-9/3
  end
  got=Float64[Array(count)[1],Array(err)[1]];error=maximum(abs.(got.-expected));ok=all(isfinite,got)&&got[1]==expected[1]&&abs(got[2]-expected[2])<=1e-8+1e-8*abs(expected[2]);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":2,\"max_abs_error\":",error,",\"timing_region\":\"mean_stochastic_kernel_excluding_reset\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":2,\"byte_order\":\"little\"}}")
  ok||Base.error("Stochastic companion CPU oracle failed")
 end
end
