# Complete source-equivalent tensor stages. No framework indexing/backend calls here.
@inline function adv_stage!(idx,vgeo,cg,D,I,u,out,r1,r2,cu,flux,tp,sp,E,::Val{S}) where S
    if S==1
        if idx<3*E*1024
            q=idx%1024;fe=idx÷1024;i=q%16;j=(q÷16)%8;c=q÷128;s=0.0
            @inbounds for a in Int32(0):Int32(7);s+=I[a*16+i+1]*u[fe*512+c*64+j*8+a+1];end
            @inbounds r1[idx+1]=s
        end
    elseif S==2
        if idx<3*E*2048
            q=idx%2048;fe=idx÷2048;i=q%16;j=(q÷16)%16;c=q÷256;s=0.0
            @inbounds for a in Int32(0):Int32(7);s+=I[a*16+j+1]*r1[fe*1024+c*128+a*16+i+1];end
            @inbounds r2[idx+1]=s
        end
    elseif S==3
        if idx<3*E*4096
            q=idx%4096;fe=idx÷4096;i=q%16;j=(q÷16)%16;k=q÷256;s=0.0
            @inbounds for a in Int32(0):Int32(7);s+=I[a*16+k+1]*r2[fe*2048+a*256+j*16+i+1];end
            @inbounds cu[idx+1]=s
        end
    elseif S==4
        if idx<3*E*4096
            q=idx%4096;fe=idx÷4096;e=fe%E;i=q%16;j=(q÷16)%16;k=q÷256;dr=0.0;ds=0.0;dt=0.0;base=fe*4096
            @inbounds for n in Int32(0):Int32(15)
                dr+=D[i*16+n+1]*cu[base+k*256+j*16+n+1];ds+=D[j*16+n+1]*cu[base+k*256+n*16+i+1];dt+=D[k*16+n+1]*cu[base+n*256+j*16+i+1]
            end
            @inbounds begin
                un=cu[e*4096+q+1];vn=cu[(E+e)*4096+q+1];wn=cu[(2*E+e)*4096+q+1];gid=e*4096*12+q+1;jw=cg[gid+5*4096]
                uh=jw*(un*cg[gid]+vn*cg[gid+4096]+wn*cg[gid+7*4096]);vh=jw*(un*cg[gid+2*4096]+vn*cg[gid+3*4096]+wn*cg[gid+8*4096]);wh=jw*(un*cg[gid+9*4096]+vn*cg[gid+10*4096]+wn*cg[gid+11*4096]);flux[idx+1]=uh*dr+vh*ds+wh*dt
            end
        end
    elseif S==5
        if idx<3*E*512
            q=idx%512;fe=idx÷512;i=q%8;j=(q÷8)%8;c=q÷64;s=0.0
            @inbounds for k in Int32(0):Int32(15);s+=I[c*16+k+1]*flux[fe*4096+k*256+j*16+i+1];end
            @inbounds tp[idx+1]=s
        end
    elseif S==6
        if idx<3*E*512
            q=idx%512;fe=idx÷512;i=q%8;j=(q÷8)%8;c=q÷64;s=0.0
            @inbounds for k in Int32(0):Int32(7);s+=I[j*16+k+1]*tp[fe*512+c*64+k*8+i+1];end
            @inbounds sp[idx+1]=s
        end
    else
        if idx<3*E*512
            q=idx%512;fe=idx÷512;e=fe%E;i=q%8;j=(q÷8)%8;c=q÷64;s=0.0
            @inbounds for k in Int32(0):Int32(7);s+=I[i*16+k+1]*sp[fe*512+c*64+j*8+k+1];end
            @inbounds out[idx+1]=vgeo[e*512*12+6*512+q+1]*s
        end
    end
    nothing
end
function adv_workspaces(E)
    CUDA.zeros(Float64,3*E*1024),CUDA.zeros(Float64,3*E*2048),CUDA.zeros(Float64,3*E*4096),CUDA.zeros(Float64,3*E*4096),CUDA.zeros(Float64,3*E*512),CUDA.zeros(Float64,3*E*512)
end
