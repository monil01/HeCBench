using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main())"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","..","_portable_study","OwnedStudy.jl"))
function study()
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 1024
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    n>=2 && n*n<=2^24 || error("invalid Float32 sequential fixture")
    host=Float32.(0:n*n-1); device=CuArray(host)
    expected=reference_matrix(n,repeats)
    launch!()=@cuda threads=256 blocks=cld(n÷2,256) rotate_matrix_kernel!(device,Int32(n))
    OwnedStudy.run_trials!(launch!,()->copyto!(device,host),()->Array(device),expected,repeats)
end
study()
