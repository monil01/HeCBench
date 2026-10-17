using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function daphne_study(launch!;args=ARGS)
    n=100000;repeat=20;cp=zeros(Float32,n*8);s=UInt64(20260722);expected=zeros(Float32,4n)
    for i in 1:n
        u=ntuple(_ -> begin;s=s*UInt64(6364136223846793005)+UInt64(1442695040888963407);Float32((s>>33)&UInt64(0x7fffffff))/Float32(0x7fffffff);end,4)
        base=(i-1)*8+1;cp[base]=(u[1]-.5f0)*2f0;cp[base+1]=(u[2]-.5f0)*2f0;cp[base+2]=i%5==0 ? -1f0 : u[3]*10f0+15f0;cp[base+4]=u[4]
        x,y,z,v=ref_project(cp[base],cp[base+1],cp[base+2]);expected[i]=x;expected[n+i]=y;expected[2n+i]=z;expected[3n+i]=Float32(v)
    end
    dcp=CuArray(cp);x=CUDA.zeros(Float32,n);y=similar(x);z=similar(x);v=CUDA.zeros(Int32,n)
    OwnedStudy.run_trials!(() -> launch!(dcp,x,y,z,v,Int32(n)),()->nothing,()->vcat(Array(x),Array(y),Array(z),Float32.(Array(v))),expected,repeat;atol=1e-3)
end
