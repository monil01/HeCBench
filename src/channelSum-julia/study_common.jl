using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function channel_study(factory)
    nchw=length(ARGS)>0&&ARGS[1]=="nchw";n=4;c=128;hxw=1024;repeats=10
    host=zeros(Int32,n*c*hxw);sums=zeros(Int32,c);squares=zeros(Int32,c)
    for ni in 0:n-1,ci in 0:c-1,hw in 0:hxw-1
        value=Int32(mod(17*(ni*hxw+hw)+13*ci,19)-9)
        index=nchw ? (ni*c+ci)*hxw+hw+1 : (ni*hxw+hw)*c+ci+1
        host[index]=value;sums[ci+1]+=value;squares[ci+1]+=value*value
    end
    input=CuArray(host);sum=CUDA.zeros(Int32,c);sumsq=CUDA.zeros(Int32,c);launch! = factory(input,sum,sumsq,nchw)
    OwnedStudy.run_trials!(launch!,()->(fill!(sum,0);fill!(sumsq,0)),()->vcat(Array(sum),Array(sumsq)),vcat(sums,squares),repeats;atol=0.,rtol=0.)
end
