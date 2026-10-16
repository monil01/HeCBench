using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(Main,replace(source,"exit(main(ARGS))"=>""),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","..","_portable_study","OwnedStudy.jl"))
function study()
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 1048576
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    T=length(ARGS)>2 && ARGS[3]=="float32" ? Float32 : Float64
    lon=T[mod(17*i,360)-180 for i in 0:n-1];lat=T[mod(13*i,180)-90 for i in 0:n-1]
    a=CuArray(lon);b=CuArray(lat);x=CUDA.zeros(T,n);y=CUDA.zeros(T,n)
    refx,refy=host_transform(lon,lat)
    launch!()=@cuda threads=256 blocks=cld(n,256) transform_kernel!(a,b,x,y,Int32(n),T(90),T(45),DEGREE_TO_RADIAN_F64,EARTH_CIRCUMFERENCE_KM_PER_DEGREE_F64)
    reset!()=(fill!(x,zero(T));fill!(y,zero(T)))
    OwnedStudy.run_trials!(launch!,reset!,()->vcat(Array(x),Array(y)),vcat(refx,refy),repeats;atol=1e-3,rtol=0.)
end
study()
