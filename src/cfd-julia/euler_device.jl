# Faithful scalar/immutable-tuple form of the original five Euler CUDA entries.
# Float3 is unpacked into an immutable three-Float32 tuple; runtime inputs unchanged.
@inline function primitive(rho,mx,my,mz,energy)
    vx=mx/rho;vy=my/rho;vz=mz/rho
    vsq=muladd(vz,vz,muladd(vy,vy,vx*vx))
    pressure=(1.4f0-1f0)*muladd(-0.5f0*rho,vsq,energy)
    speed=sqrt(vsq);sound=sqrt(1.4f0*pressure/rho)
    de=energy+pressure
    fmx=(muladd(vx,mx,pressure),vx*my,vx*mz)
    fmy=(fmx[2],muladd(vy,my,pressure),vy*mz)
    fmz=(fmx[3],fmy[3],muladd(vz,mz,pressure))
    fde=(vx*de,vy*de,vz*de)
    return pressure,speed,sound,fmx,fmy,fmz,fde
end
@inline function cell_primitives(i,n,u)
    @inbounds return primitive(u[i],u[i+n],u[i+2n],u[i+3n],u[i+4n])
end
@inline function initialize_buffer_at(i,d,value,n)
    if i<=n;@inbounds d[i]=value;end
    return nothing
end
@inline function initialize_variables_at(i,n,u,ff)
    @inbounds for j in 0:4;u[i+j*n]=ff[j+1];end
    return nothing
end
@inline function compute_step_factor_at(i,n,u,areas,step)
    if i<=n
        p,s,c,_,_,_,_=cell_primitives(i,n,u)
        @inbounds step[i]=0.5f0/(sqrt(areas[i])*(s+c))
    end
    return nothing
end
@inline function compute_flux_at(i,n,neighbors,normals,u,ff,flux,ffde,ffmx,ffmy,ffmz)
    if i>n;return nothing;end
    @inbounds begin
        rho=u[i];mx=u[i+n];my=u[i+2n];mz=u[i+3n];energy=u[i+4n]
        pressure,speed,sound,fmx,fmy,fmz,fde=primitive(rho,mx,my,mz,energy)
        f0=0f0;f1=0f0;f2=0f0;f3=0f0;f4=0f0
        for face in 0:3
            nb=neighbors[i+face*n]
            nx=normals[i+face*n];ny=normals[i+(face+4)*n];nz=normals[i+(face+8)*n]
            nlen=sqrt(muladd(nz,nz,muladd(ny,ny,nx*nx)))
            if nb>=0
                j=Int(nb)+1
                rj=u[j];xj=u[j+n];yj=u[j+2n];zj=u[j+3n];ej=u[j+4n]
                pj,sj,cj,mxj,myj,mzj,dej=primitive(rj,xj,yj,zj,ej)
                factor=((-nlen*0.2f0)*0.5f0)*(((speed+sj)+sound)+cj)
                f0=muladd(factor,rho-rj,f0);f1=muladd(factor,mx-xj,f1);f2=muladd(factor,my-yj,f2);f3=muladd(factor,mz-zj,f3);f4=muladd(factor,energy-ej,f4)
                for d in 1:3
                    normal=d==1 ? nx : d==2 ? ny : nz
                    factor=0.5f0*normal
                    mi=d==1 ? mx : d==2 ? my : mz;mj=d==1 ? xj : d==2 ? yj : zj
                    f0=muladd(factor,mj+mi,f0)
                    f1=muladd(factor,mxj[d]+fmx[d],f1);f2=muladd(factor,myj[d]+fmy[d],f2);f3=muladd(factor,mzj[d]+fmz[d],f3);f4=muladd(factor,dej[d]+fde[d],f4)
                end
            elseif nb == -1
                f1=muladd(nx,pressure,f1);f2=muladd(ny,pressure,f2);f3=muladd(nz,pressure,f3)
            elseif nb == -2
                for d in 1:3
                    factor=0.5f0*(d==1 ? nx : d==2 ? ny : nz)
                    mi=d==1 ? mx : d==2 ? my : mz
                    f0=muladd(factor,ff[1+d]+mi,f0);f1=muladd(factor,ffmx[d]+fmx[d],f1);f2=muladd(factor,ffmy[d]+fmy[d],f2);f3=muladd(factor,ffmz[d]+fmz[d],f3);f4=muladd(factor,ffde[d]+fde[d],f4)
                end
            end
        end
        flux[i]=f0;flux[i+n]=f1;flux[i+2n]=f2;flux[i+3n]=f3;flux[i+4n]=f4
    end
    return nothing
end
@inline function time_step_at(i,stage,n,old,u,step,flux)
    if i<=n
        @inbounds begin
            factor=step[i]/Float32(4-stage)
            for k in 0:4;u[i+k*n]=muladd(factor,flux[i+k*n],old[i+k*n]);end
        end
    end
    return nothing
end
