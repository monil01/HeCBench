using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function damage_study(factory;n=262144,repeats=10)
 n>0&&mod(n,256)==0||error("namedalignedstudy requires positive multiple256")
 m=div(n,256);input=Int32[mod(17*i+13*div(i,256),23)>=mod(div(i,256),11) ? Int32(mod(i,m)) : Int32(-1) for i in 0:n-1]
 counts=Int32[sum(input[g*256+1:g*256+256].!=-1) for g in 0:m-1]
 family=Int32[counts[g+1]+1+mod(13*g,64) for g in 0:m-1]
 expected=vcat(Float64.(counts),Float64[1.0-Float64(counts[g])/Float64(family[g]) for g in 1:m])
 di=CuArray(input);df=CuArray(family);out=CUDA.zeros(Int32,m);damage=CUDA.zeros(Float64,m);launch! = factory(di,df,out,damage,n,m)
 OwnedStudy.run_trials!(launch!,()->(fill!(out,0);fill!(damage,0.)),()->vcat(Float64.(Array(out)),Array(damage)),expected,repeats;atol=1e-6,rtol=0.)
end
