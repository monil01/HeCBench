using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function fma_study(launch!;args=ARGS)
    na=1024;nc=2048;ch=64;ops=1024;r=20
    a=Float32[(i%17-8)/8 for i in 0:na*ch-1];b=Float32[(i%7-3)/4 for i in 0:ch-1]
    ins=Int32[(i*37)%na for i in 0:ops-1];outs=Int32[(i*19)%nc for i in 0:ops-1]
    expected=zeros(Float32,nc*ch)
    for op in 1:ops,k in 0:ch-1;expected[outs[op]*ch+k+1]=a[ins[op]*ch+k+1]*b[k+1];end
    da=CuArray(a);db=CuArray(b);di=CuArray(ins);do_=CuArray(outs);c1=CUDA.zeros(Float32,nc*ch);c2=similar(c1)
    function launch()
        CUDA.fill!(c1,0f0);CUDA.fill!(c2,0f0)
        launch!(da,db,c1,c2,di,do_,Int32(ops),Int32(ch),Int32(na),Int32(nc))
    end
    OwnedStudy.run_trials!(launch,()->nothing,()->vcat(Array(c1),Array(c2)),vcat(expected,expected),r;atol=1e-3)
end
