using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function bscan_study(launch!;args=ARGS)
    ns=[32,64,128,256,512,1024];inputs=[Int32[(i*37+11)%101-50 for i in 0:n-1] for n in ns]
    expected=Int32[]
    for a in inputs
        count=Int32(0)
        for x in a;push!(expected,count);count+=Int32(x>0);end
    end
    ins=CuArray.(inputs);outs=[CUDA.zeros(Int32,n) for n in ns];scratch=[CUDA.zeros(Int32,n) for n in ns]
    function launch()
        for k in eachindex(ns);launch!(outs[k],ins[k],scratch[k],Int32(ns[k]));end
    end
    OwnedStudy.run_trials!(launch,()->nothing,()->vcat(Array.(outs)...),expected,10;atol=0)
end
