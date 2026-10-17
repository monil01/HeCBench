using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function stream_study(factory)
    op=isempty(ARGS) ? "copy" : ARGS[1];n=262144;repeats=10
    a=Float32[Float32(i%31-15)/32f0 for i in 0:n-1];b=Float32[Float32((7*i)%17-8)/16f0 for i in 0:n-1];c=Float32[Float32((11*i)%23-11)/16f0 for i in 0:n-1]
    expected=op=="copy" ? copy(a) : op=="mul" ? 0.4f0.*c : op=="add" ? a.+b : op=="triad" ? b.+0.4f0.*c : op=="nstream" ? copy(a) : zeros(Float32,256)
    if op=="nstream"
        for _ in 1:repeats
            expected .+= b.+0.4f0.*c
        end
    elseif op=="dot"
        # Float64 independent block sums: native block b covers all256 lanes and stride65536.
        for group in 0:255
            s=0.0
            for lane in 0:255, i in group*256+lane+1:65536:n
                s+=Float64(a[i])*Float64(b[i])
            end
            expected[group+1]=Float32(s)
        end
    end
    da=CuArray(a);db=CuArray(b);dc=CuArray(c);partial=CUDA.zeros(Float32,256)
    launch! = factory(op,da,db,dc,partial,n)
    reset! = ()->begin copyto!(da,a);copyto!(db,b);copyto!(dc,c);fill!(partial,0f0);end
    output=op=="copy"||op=="add" ? dc : op=="mul" ? db : op=="dot" ? partial : da
    OwnedStudy.run_trials!(launch!,reset!,()->Array(output),expected,repeats;atol=1e-4,rtol=0.)
end
