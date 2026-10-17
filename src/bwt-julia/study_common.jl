using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
include(joinpath(@__DIR__,"study_helpers.jl"))
function bwt_study(factory)
    n=1021;table_size=1024;repeats=3
    alphabet=UInt8[0x41,0x54,0x43,0x47];sequence=Vector{UInt8}(undef,n);state=UInt32(0x12345678)
    for i in 1:n
        state=state*UInt32(1664525)+UInt32(1013904223)
        sequence[i]=alphabet[Int((state>>28)&UInt32(3))+1]
    end
    # Independent CPU stdlib lexical sorting of explicitly materialized rotations.
    rotations=[String(vcat(sequence[i:end],sequence[1:i-1])) for i in 1:n]
    order=sortperm(rotations)
    @assert all(rotations[order[i]]!=rotations[order[i+1]] for i in 1:n-1)
    expected=vcat(Int32[sequence[mod(i-2,n)+1] for i in order],Int32.(order.-1),fill(Int32(-1),table_size-n))
    input=CuArray(sequence);table=CUDA.zeros(Int32,table_size);output=CUDA.zeros(UInt8,n)
    launch! = factory(input,table,output,Int32(n),Int32(table_size))
    reset!()=(fill!(table,Int32(-9));fill!(output,UInt8(0)))
    # Lossless byte->Int32 artifact widening, followed by full sorttable including padding.
    readback()=vcat(Int32.(Array(output)),Array(table))
    OwnedStudy.run_trials!(launch!,reset!,readback,expected,repeats;atol=0.,rtol=0.)
end
