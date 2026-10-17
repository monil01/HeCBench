using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function quant3_study(factory)
    m=4096;n=1024;repeats=10;height=div(m,32)*3
    vec=Float32[Float32(mod(r,17)-8)/8f0 for r in 0:m-1]
    scales=Float32[Float32(mod(c,7)+1)/8f0 for c in 0:n-1];zeropoints=Float32[Float32(mod(c,5)-2)/4f0 for c in 0:n-1]
    packed=zeros(UInt32,height*n);expected=zeros(Float32,n)
    for r in 0:m-1,c in 0:n-1
        q=UInt32(mod(17*r+13*c,8));bit=3*r;word=div(bit,32);shift=mod(bit,32)
        packed[word*n+c+1]|=q<<shift
        if shift>29;packed[(word+1)*n+c+1]|=q>>(32-shift);end
        expected[c+1]+=(scales[c+1]*Float32(q)-zeropoints[c+1])*vec[r+1]
    end
    dv=CuArray(vec);dm=CuArray(collect(reinterpret(Int32,packed)));ds=CuArray(scales);dz=CuArray(zeropoints);output=CUDA.zeros(Float32,n)
    operation! = factory(dv,dm,output,ds,dz)
    launch! = ()->begin;fill!(output,0f0);operation!();end
    OwnedStudy.run_trials!(launch!,()->fill!(output,0f0),()->Array(output),expected,repeats;atol=1e-3,rtol=0.)
end
