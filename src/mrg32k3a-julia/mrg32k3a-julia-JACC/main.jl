using CUDA,JACC
using Printf

const CURAND = CUDA.CURAND
const LIBCURAND = getfield(CURAND, :libcurand)

function create_generator(host::Bool)
    gen_ref = Ref{CURAND.curandGenerator_t}()
    rng = CURAND.CURAND_RNG_PSEUDO_MRG32K3A
    if host
        CURAND.curandCreateGeneratorHost(gen_ref, rng)
    else
        CURAND.curandCreateGenerator(gen_ref, rng)
    end
    return gen_ref[]
end

function configure_generator!(gen)
    CURAND.curandSetGeneratorOffset(gen, UInt64(0))
    CURAND.curandSetGeneratorOrdering(gen, CURAND.CURAND_ORDERING_PSEUDO_BEST)
    CURAND.curandSetPseudoRandomGeneratorSeed(gen, UInt64(1234))
    return gen
end

function run_on_device(n::Int)
    d_data = JACC.zeros(Float32,n)
    gen = configure_generator!(create_generator(false))
    CURAND.curandGenerateUniform(gen, d_data, Csize_t(n))
    h_data = Vector{Float32}(undef, n)
    copyto!(h_data, d_data)
    CUDA.unsafe_free!(d_data)
    CURAND.curandDestroyGenerator(gen)
    return h_data
end

function run_on_host(n::Int)
    h_data = Vector{Float32}(undef, n)
    gen = configure_generator!(create_generator(true))
    status = ccall((:curandGenerateUniform, LIBCURAND), Cint,
                   (CURAND.curandGenerator_t, Ptr{Float32}, Csize_t),
                   gen, pointer(h_data), Csize_t(n))
    status == Int(CURAND.CURAND_STATUS_SUCCESS) || error("curandGenerateUniform host failed: $status")
    CURAND.curandDestroyGenerator(gen)
    return h_data
end

function main(args)
    n=length(args)>=1 ? parse(Int,args[1]) : 4096
    repeat=length(args)>=2 ? parse(Int,args[2]) : 1
    run_on_host(n);run_on_device(n);CUDA.synchronize()
    for trial in 1:5
        start=time_ns();host=Float32[]
        for _ in 1:repeat
            host=run_on_host(n)
        end
        host_seconds=(time_ns()-start)*1e-9/repeat
        CUDA.synchronize();start=time_ns();result=Float32[]
        for _ in 1:repeat
            result=run_on_device(n)
        end
        CUDA.synchronize();seconds=(time_ns()-start)*1e-9/repeat
        error=maximum(abs.(result.-host));correct=all(isfinite,result)&&error<=1f-3
        artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin")
        open(artifact,"w") do io;write(io,result);end
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,
         ",\"checked_values\":",n,",\"max_abs_error\":",error,",\"timing_region\":\"vendor_rng_allocate_create_configure_generate_d2h_release_destroy\",\"jit_excluded\":true,\"internal_repeats\":",repeat,
         ",\"host_time_seconds\":",host_seconds,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"float32\",\"count\":",n,",\"byte_order\":\"little\"}}")
        correct||return 1
    end
    return 0
end
exit(main(ARGS))
