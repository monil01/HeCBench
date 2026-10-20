# Source24activeGPUdefinitions/25semantic tags (real/complex BroadcastColumn).
# No native authored-kernel delegation. Originalinactive BroadcastRow not certified.
using Atomix
@fastmath function tsne_body!(::Val{tag}, tid, a, c) where tag
    N=c.N;K=c.K;B=c.B;H=c.H;F=c.F
    i=Int32(tid-1)
    if tag==1
        i<N*K && (a.idx[i+1]=Int32(a.rawidx[i+1]))
    elseif tag==2
        if i<N*K
            row=i÷K;col=i%K;d=a.dist[i+1]
            a.prob[i+1]=(col==0 && d==0f0) ? 0f0 : exp(-a.beta[row+1]*d)
        end
    elseif tag==3 || tag==4
        if i<N
            s=0f0
            for j=Int32(0):K-1
                v=a.prob[i*K+j+1]
                s += tag==3 ? v : (v==0f0 ? 0f0 : v*log(v))
            end
            if tag==3;a.rowsum[i+1]=s;else;a.entropy[i+1]=-s;end
        end
    elseif tag==5
        if i<N
            d=a.entropy[i+1]/a.rowsum[i+1]+log(a.rowsum[i+1])-log(c.perplexity)
            done=d<1f-4 && -d<1f-4
            if !done
                b=a.beta[i+1];lo=a.lower[i+1];hi=a.upper[i+1]
                if d>0f0
                    lo=b;b=(hi==floatmax(Float32)||hi== -floatmax(Float32)) ? b*2f0 : (b+hi)/2f0
                else
                    hi=b;b=(lo== -floatmax(Float32)||lo==floatmax(Float32)) ? b/2f0 : (b+lo)/2f0
                end
                a.beta[i+1]=b;a.lower[i+1]=lo;a.upper[i+1]=hi
            end
            a.found[i+1]=Int32(done)
        end
    elseif tag==6
        # Source BroadcastMatrixVector ignores axis; columnindex selects divisor.
        i<N*K && (a.prob[i+1] /= a.rowsum[i%K+1])
    elseif tag==7
        if i<N*K
            row=i÷K;j=a.idx[i+1];s=a.prob[i+1]
            for k=j*K:(j+1)*K-1
                s += a.idx[k+1]==row ? a.prob[k+1] : 0f0
            end
            a.sym[i+1]=s/(2f0*N)
        end
    elseif tag==8
        if i<N
            x=a.points[i+1];y=a.points[N+i+1]
            a.charges[4i+1]=1f0;a.charges[4i+2]=x;a.charges[4i+3]=y;a.charges[4i+4]=x*x+y*y
        end
    elseif tag==9
        if i<B*B
            row=i÷B;col=i%B
            a.boxlo[i+1]=col*c.width+c.mincoord;a.boxhi[i+1]=(col+1)*c.width+c.mincoord
            a.boxlo[B*B+i+1]=row*c.width+c.mincoord;a.boxhi[B*B+i+1]=(row+1)*c.width+c.mincoord
        end
    elseif tag==10
        if i<H*H
            row=i÷H;col=i%H;h=c.width/3f0
            y0=c.mincoord+h/2f0;x0=c.mincoord+h/2f0
            d1=y0-(y0+row*h);d2=x0-(x0+col*h);v=(1f0+d1*d1+d2*d2)^(-2f0)
            a.kernel[(H+row)*F+H+col+1]=v;a.kernel[(H-row)*F+H+col+1]=v
            a.kernel[(H+row)*F+H-col+1]=v;a.kernel[(H-row)*F+H-col+1]=v
        end
    elseif tag==11 || tag==12 || tag==18 || tag==19
        # Preserve source16x16threads and2D block grouping using flattened launch.
        gx=cld(F,Int32(16));lane=i%Int32(256);block=i÷Int32(256)
        col=(block%gx)*16+lane%16;row=(block÷gx)*16+lane÷16
        nr=tag==12 ? H+1 : F
        if col<F && row<nr
            theta_phase=(tag==11||tag==12 ? -2f0 : 2f0)*Float32(pi)*(Float32(col)/Float32(F))
            s=ComplexF32(0,0);offset=max(c.term,Int32(0))*F*F
            if tag==11
                for k=Int32(0):F-1
                    sn,co=sincos(theta_phase*Float32(k));v=c.term<0 ? a.kernel[row*F+k+1] : a.fin[offset+row*F+k+1]
                    s+=v*ComplexF32(co,sn)
                end
                a.scratch[offset+col*F+row+1]=s
            elseif tag==12
                for k=Int32(0):F-1
                    sn,co=sincos(theta_phase*Float32(k));s+=a.scratch[offset+row*F+k+1]*ComplexF32(co,sn)
                end
                if c.term<0;a.fkernel[col*(H+1)+row+1]=s
                else;a.fw[c.term*F*(H+1)+col*(H+1)+row+1]=s;end
            elseif tag==18
                for k=Int32(0):H
                    sn,co=sincos(theta_phase*Float32(k));s+=a.fw[c.term*F*(H+1)+row*(H+1)+k+1]*ComplexF32(co,sn)
                end
                for k=H+1:F-1
                    sn,co=sincos(theta_phase*Float32(k));s+=conj(a.fw[c.term*F*(H+1)+((F-row)%F)*(H+1)+(F-k)%F+1])*ComplexF32(co,sn)
                end
                a.scratch[offset+col*F+row+1]=s
            else
                real_sum=0f0
                for k=Int32(0):F-1
                    sn,co=sincos(theta_phase*Float32(k));real_sum+=real(a.scratch[offset+row*F+k+1]*ComplexF32(co,sn))
                end
                a.fout[offset+col*F+row+1]=real_sum
            end
        end
    elseif tag==13
        if i<N
            x=a.points[i+1];y=a.points[N+i+1];xi=clamp(trunc(Int32,(x-c.mincoord)/c.width),Int32(0),B-1);yi=clamp(trunc(Int32,(y-c.mincoord)/c.width),Int32(0),B-1);b=yi*B+xi
            a.boxidx[i+1]=b;a.xin[i+1]=(x-a.boxlo[b+1])/c.width;a.yin[i+1]=(y-a.boxlo[B*B+b+1])/c.width
        end
    elseif tag==14
        if i<3N
            pt=i%N;j=i÷N;v=1f0;x=c.axis==0 ? a.xin[pt+1] : a.yin[pt+1]
            for k=Int32(0):Int32(2)
                if j!=k;v*=x-a.spacing[k+1];end
            end
            if c.axis==0;a.xinterp[i+1]=v/a.den[j+1];else;a.yinterp[i+1]=v/a.den[j+1];end
        end
    elseif tag==15 || tag==21
        if i<36N
            term=i%4;pt=(i÷4)%N;j=((i÷4)÷N)%3;k=((i÷4)÷N)÷3;b=a.boxidx[pt+1]
            loc=((b%B)*3+k)*H+(b÷B)*3+j;v=a.xinterp[pt+k*N+1]*a.yinterp[pt+j*N+1]
            if tag==15
                v*=a.charges[pt*4+term+1];Atomix.@atomic a.w[loc*4+term+1] += v
            else
                v*=a.tilde[loc*4+term+1];Atomix.@atomic a.potential[pt*4+term+1] += v
            end
        end
    elseif tag==16 || tag==20
        if i<4H*H
            term=i÷(H*H);loc=i%(H*H);row=loc÷H;col=loc%H
            if tag==16;a.fin[term*F*F+row*F+col+1]=a.w[term+loc*4+1]
            else;a.tilde[term+4loc+1]=a.fout[term*F*F+(row+H)*F+col+H+1]/Float32(F*F);end
        end
    elseif tag==17
        i<4F*(H+1) && (a.fw[i+1]*=a.fkernel[i%(F*(H+1))+1])
    elseif tag==22
        if i<N
            x=a.points[i+1];y=a.points[N+i+1];p1=a.potential[4i+1];p2=a.potential[4i+2];p3=a.potential[4i+3];p4=a.potential[4i+4]
            a.norm[i+1]=(1f0+x*x+y*y)*p1-2f0*(x*p2+y*p3)+p4;a.rep[i+1]=x*p1-p2;a.rep[N+i+1]=y*p1-p3
        end
    elseif tag==23
        if i<N*K
            pt=i÷K;j=a.idx[i+1];dx=a.points[pt+1]-a.points[j+1];dy=a.points[N+pt+1]-a.points[N+j+1];v=a.sym[i+1]/(1f0+dx*dx+dy*dy)
            a.workspace[i+1]=v*dx;a.workspace[N*K+i+1]=v*dy
        end
    elseif tag==24
        if i<N
            x=0f0;y=0f0
            for j=i*K:(i+1)*K-1;x+=a.workspace[j+1];y+=a.workspace[N*K+j+1];end
            a.attr[i+1]=x;a.attr[N+i+1]=y
        end
    elseif tag==25
        if i<N
            for axis=Int32(0):Int32(1)
                j=i+axis*N+1;u=a.old[j];g=a.gains[j];d=c.exaggeration*a.attr[j]-a.rep[j]/c.normalization
                g=signbit(d)!=signbit(u) ? g+.2f0 : g*.8f0;g=max(g,.01f0);u=c.momentum*u-200f0*g*d
                a.points[j]+=u;a.attr[j]=0f0;a.rep[j]=0f0;a.old[j]=u;a.gains[j]=g
            end
        end
    end
    return nothing
end
