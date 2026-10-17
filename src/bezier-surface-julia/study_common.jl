using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
if !isdefined(Main,:XYZ)
    include(joinpath(@__DIR__,"study_types.jl"))
end
function bezier_study(factory)
    n=512;repeats=3;input=[XYZ(-0.508359f0,-0.280288f0,-0.160131f0),XYZ(-0.199345f0,-0.515802f0,-0.0100263f0),XYZ(0.133988f0,-0.515802f0,-0.0100263f0),XYZ(0.418629f0,-0.382526f0,-0.407321f0),XYZ(-0.532679f0,-0.182468f0,-0.0100263f0),XYZ(0.0505979f0,-0.195094f0,1.08832f0),XYZ(0.383931f0,-0.195094f0,1.08832f0),XYZ(0.467321f0,-0.182468f0,-0.0100263f0),XYZ(-0.532679f0,0.150865f0,-0.0100263f0),XYZ(0.0505979f0,0.13824f0,1.08832f0),XYZ(0.383931f0,0.13824f0,1.08832f0),XYZ(0.467321f0,0.150865f0,-0.0100263f0),XYZ(-0.483195f0,0.349859f0,-0.0568787f0),XYZ(-0.199345f0,0.484198f0,-0.0100263f0),XYZ(0.133988f0,0.484198f0,-0.0100263f0),XYZ(0.30311f0,0.389596f0,-0.0811675f0)]
    # Independent Float64 Bernstein polynomial oracle for the cubic sourcefixture.
    coeff=(1.,3.,3.,1.);reference=zeros(Float32,3*n*n)
    for i in 0:n-1,j in 0:n-1
        u=Float64(i)/(n-1);v=Float64(j)/(n-1);x=0.;y=0.;z=0.
        for ki in 0:3,kj in 0:3
            w=coeff[ki+1]*u^ki*(1-u)^(3-ki)*coeff[kj+1]*v^kj*(1-v)^(3-kj)
            a=input[ki*4+kj+1];x+=Float64(a.x)*w;y+=Float64(a.y)*w;z+=Float64(a.z)*w
        end
        k=3*(i*n+j);reference[k+1]=x;reference[k+2]=y;reference[k+3]=z
    end
    source_cpu=collect(reinterpret(Float32,bezier_cpu(input,3,3,n,n)))
    out=CuArray{XYZ}(undef,n*n);launch! = factory(CuArray(input),out,Int32(n))
    reset!()=fill!(out,XYZ(0f0,0f0,0f0))
    function readback()
        got=collect(reinterpret(Float32,Array(out)))
        l1=sum(abs(Float64(x)-Float64(y)) for (x,y) in zip(got,source_cpu))/sum(abs,Float64.(source_cpu))
        @assert isfinite(l1)&&l1<1e-6 "original source relativeL1 criterion"
        println("STUDY_ACCURACY relative_l1_source_cpu=",l1)
        got
    end
    OwnedStudy.run_trials!(launch!,reset!,readback,reference,repeats;atol=1e-5,rtol=0.)
end
