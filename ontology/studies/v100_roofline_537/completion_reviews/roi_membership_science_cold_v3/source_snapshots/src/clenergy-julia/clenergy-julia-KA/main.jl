using CUDA, KernelAbstractions
@kernel function clenergy_ka!(atoms,out,n)
    lane=@index(Global,Linear)
    x=(lane-1)%8;y=(lane-1)÷8
    addr=y*64+x+1;xx=Float32(x)*.1f0;yy=Float32(y)*.1f0;step=.1f0*8f0
    e1=0f0;e2=0f0;e3=0f0;e4=0f0;e5=0f0;e6=0f0;e7=0f0;e8=0f0
    @inbounds for i in 1:n
        a=atoms[i];dy=yy-a[2];dyz=dy*dy+a[3];dx1=xx-a[1];dx2=dx1+step;dx3=dx2+step;dx4=dx3+step;dx5=dx4+step;dx6=dx5+step;dx7=dx6+step;dx8=dx7+step
        e1+=a[4]*inv(sqrt(dx1*dx1+dyz));e2+=a[4]*inv(sqrt(dx2*dx2+dyz));e3+=a[4]*inv(sqrt(dx3*dx3+dyz));e4+=a[4]*inv(sqrt(dx4*dx4+dyz));e5+=a[4]*inv(sqrt(dx5*dx5+dyz));e6+=a[4]*inv(sqrt(dx6*dx6+dyz));e7+=a[4]*inv(sqrt(dx7*dx7+dyz));e8+=a[4]*inv(sqrt(dx8*dx8+dyz))
    end
    @inbounds begin out[addr]+=e1;out[addr+8]+=e2;out[addr+16]+=e3;out[addr+24]+=e4;out[addr+32]+=e5;out[addr+40]+=e6;out[addr+48]+=e7;out[addr+56]+=e8;end
    nothing
end
launch_ka!(a,o,n)=clenergy_ka!(get_backend(o),64)(a,o,n;ndrange=64)
include(joinpath(@__DIR__,"..","study_common.jl"))
clenergy_study(launch_ka!)
