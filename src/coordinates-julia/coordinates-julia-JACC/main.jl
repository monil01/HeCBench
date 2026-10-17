using CUDA, JACC
include(joinpath(@__DIR__,"..","..","_portable_study","OwnedStudy.jl"))
function coordinates_jacc!(i,lon,lat,x,y,n::Int32,ox,oy,rad,km)
    if i<=n
        mid=(lat[i]+oy)/typeof(oy)(2)
        x[i]=(ox-lon[i])*km*cos(mid*rad)
        y[i]=(oy-lat[i])*km
    end
end
function main()
    n=length(ARGS)>0 ? parse(Int,ARGS[1]) : 1048576
    repeats=length(ARGS)>1 ? parse(Int,ARGS[2]) : 10
    T=length(ARGS)>2 && ARGS[3]=="float32" ? Float32 : Float64
    n>0 || error("n must be positive")
    lon=T[mod(17*i,360)-180 for i in 0:n-1];lat=T[mod(13*i,180)-90 for i in 0:n-1]
    ox=T(90);oy=T(45);rad=pi/180.0;km=40000.0/360.0
    refx=T[(ox-lon[i])*km*cos(((lat[i]+oy)/T(2))*rad) for i in 1:n]
    refy=T[(oy-lat[i])*km for i in 1:n]
    a=CuArray(lon);b=CuArray(lat);x=CUDA.zeros(T,n);y=CUDA.zeros(T,n)
    spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
    launch!()=JACC.parallel_for(spec,n,coordinates_jacc!,a,b,x,y,Int32(n),ox,oy,rad,km)
    reset!()=(fill!(x,zero(T));fill!(y,zero(T)))
    OwnedStudy.run_trials!(launch!,reset!,()->vcat(Array(x),Array(y)),vcat(refx,refy),repeats;atol=1e-3,rtol=0.)
end
main()
