using CUDA
using Printf
# Load the preserved kernel and CPU verifier without executing the legacy CLI.
source = read(joinpath(@__DIR__, "main.jl"), String)
source = replace(source, "exit(main(ARGS))" => "")
include_string(Main, source, joinpath(@__DIR__, "main.jl"))
function study(args=ARGS)
    n = length(args)>=1 ? parse(Int,args[1]) : 1048576
    repeats = length(args)>=2 ? parse(Int,args[2]) : 100
    n>0 && n%256==0 && repeats>0 || error("invalid length/repeat")
    input=Int32.(0:n+Int(RADIUS)-1)
    expected=zeros(Int32,n)
    for i in 1:n, off in -Int(RADIUS):Int(RADIUS)
        j=i+off
        j>0 && (expected[i]+=input[j])
    end
    a=CuArray(input);b=CUDA.zeros(Int32,n)
    @cuda threads=256 blocks=cld(n,256) stencil_1d!(a,b,Int32(n))
    CUDA.synchronize()
    allok=true
    for trial in 1:5
        CUDA.fill!(b,Int32(0));CUDA.synchronize()
        t0=time_ns()
        for _ in 1:repeats
            @cuda threads=256 blocks=cld(n,256) stencil_1d!(a,b,Int32(n))
        end
        CUDA.synchronize()
        elapsed=(time_ns()-t0)*1e-9/repeats
        actual=Array(b)
        maxabs=maximum(abs.(Int64.(actual).-Int64.(expected)))
        ok=actual==expected;allok &= ok
        artifact = joinpath(get(ENV, "STUDY_OUTPUT_DIR", pwd()), "stencil1d_trial$(trial).bin"); write(artifact, actual)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0.0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),n,maxabs,repeats,artifact,n)
    end
    println(allok ? "PASS" : "FAIL")
    allok ? 0 : 1
end
exit(study())
