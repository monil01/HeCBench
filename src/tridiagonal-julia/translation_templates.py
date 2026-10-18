PCR=r'''
    thid=lid;blid=bid;delta=Int32(1);ao=Int32(0);bo=n+Int32(1);co=2bo;doff=3bo;xo=4bo
    @inbounds begin
    sh[ao+thid+1]=ad[thid+blid*n+1];sh[bo+thid+1]=bd[thid+blid*n+1];sh[co+thid+1]=cd[thid+blid*n+1];sh[doff+thid+1]=dd[thid+blid*n+1]
    BARRIER
    for j in Int32(0):iterations-Int32(1)
      i=thid
      if OPT==0 && i<delta
        t2=sh[co+i+1]/sh[bo+i+delta+1];bn=sh[bo+i+1]-sh[ao+i+delta+1]*t2;dn=sh[doff+i+1]-sh[doff+i+delta+1]*t2;an=0f0;cn=-sh[co+i+delta+1]*t2
      elseif OPT==0 && n-i-Int32(1)<delta
        t1=sh[ao+i+1]/sh[bo+i-delta+1];bn=sh[bo+i+1]-sh[co+i-delta+1]*t1;dn=sh[doff+i+1]-sh[doff+i-delta+1]*t1;an=-sh[ao+i-delta+1]*t1;cn=0f0
      else
        left=OPT==1 ? ((i-delta)&(n-Int32(1))) : i-delta;right=OPT==1 ? ((i+delta)&(n-Int32(1))) : i+delta
        t1=sh[ao+i+1]/sh[bo+left+1];t2=sh[co+i+1]/sh[bo+right+1]
        bn=sh[bo+i+1]-sh[co+left+1]*t1-sh[ao+right+1]*t2;dn=sh[doff+i+1]-sh[doff+left+1]*t1-sh[doff+right+1]*t2;an=-sh[ao+left+1]*t1;cn=-sh[co+right+1]*t2
      end
      BARRIER
      sh[bo+i+1]=bn;sh[doff+i+1]=dn;sh[ao+i+1]=an;sh[co+i+1]=cn;delta*=Int32(2)
      BARRIER
    end
    if thid<delta
      u=thid;v=thid+delta;t3=sh[bo+v+1]*sh[bo+u+1]-sh[co+u+1]*sh[ao+v+1]
      sh[xo+u+1]=(sh[bo+v+1]*sh[doff+u+1]-sh[co+u+1]*sh[doff+v+1])/t3
      sh[xo+v+1]=(sh[doff+v+1]*sh[bo+u+1]-sh[doff+u+1]*sh[ao+v+1])/t3
    end
    BARRIER
    xd[thid+blid*n+1]=sh[xo+thid+1]
    end
'''
CR=r'''
    thid=lid;blid=bid;stride=Int32(1);half=n÷Int32(2);active=half;ao=Int32(0);bo=n;co=2n;doff=3n;xo=4n
    @inbounds begin
    for off in (Int32(0),half)
      sh[ao+thid+off+1]=ad[thid+off+blid*n+1];sh[bo+thid+off+1]=bd[thid+off+blid*n+1];sh[co+thid+off+1]=cd[thid+off+blid*n+1];sh[doff+thid+off+1]=dd[thid+off+blid*n+1]
    end
    BARRIER
    for j in Int32(0):iterations-Int32(1)
      BARRIER
      stride<<=1;delta=stride>>1
      if thid<active
        i=stride*thid+stride-Int32(1)
        if OPT==0 && i==n-Int32(1)
          t1=sh[ao+i+1]/sh[bo+i-delta+1];sh[bo+i+1]=sh[bo+i+1]-sh[co+i-delta+1]*t1;sh[doff+i+1]=sh[doff+i+1]-sh[doff+i-delta+1]*t1;sh[ao+i+1]=-sh[ao+i-delta+1]*t1;sh[co+i+1]=0f0
        else
          right=OPT==1 ? ((i+delta)&(n-Int32(1))) : i+delta
          t1=sh[ao+i+1]/sh[bo+i-delta+1];t2=sh[co+i+1]/sh[bo+right+1]
          sh[bo+i+1]=sh[bo+i+1]-sh[co+i-delta+1]*t1-sh[ao+right+1]*t2;sh[doff+i+1]=sh[doff+i+1]-sh[doff+i-delta+1]*t1-sh[doff+right+1]*t2;sh[ao+i+1]=-sh[ao+i-delta+1]*t1;sh[co+i+1]=-sh[co+right+1]*t2
        end
      end
      active>>=1
    end
    if thid<Int32(2)
      u=stride-Int32(1);v=(stride<<1)-Int32(1);t3=sh[bo+v+1]*sh[bo+u+1]-sh[co+u+1]*sh[ao+v+1]
      sh[xo+u+1]=(sh[bo+v+1]*sh[doff+u+1]-sh[co+u+1]*sh[doff+v+1])/t3;sh[xo+v+1]=(sh[doff+v+1]*sh[bo+u+1]-sh[doff+u+1]*sh[ao+v+1])/t3
    end
    active=Int32(2)
    for j in Int32(0):iterations-Int32(1)
      delta=stride>>1
      BARRIER
      if thid<active
        i=stride*thid+(stride>>1)-Int32(1)
        if i==delta-Int32(1);sh[xo+i+1]=(sh[doff+i+1]-sh[co+i+1]*sh[xo+i+delta+1])/sh[bo+i+1]
        else;sh[xo+i+1]=(sh[doff+i+1]-sh[ao+i+1]*sh[xo+i-delta+1]-sh[co+i+1]*sh[xo+i+delta+1])/sh[bo+i+1];end
      end
      stride>>=1;active<<=1
    end
    BARRIER
    xd[thid+blid*n+1]=sh[xo+thid+1];xd[thid+half+blid*n+1]=sh[xo+thid+half+1]
    end
'''
SWEEP=r'''
    @inbounds begin
    if MODE==8
      i=globalid*Int32(4);i>=m&&return nothing
      step=m;base=i;idx=base
      c1=load4(cd,base);c2=load4(bd,base);fi=load4(dd,base);store4!(wd,i+m,-c1./c2);prev=fi./c2;store4!(xd,base,prev)
      for k in Int32(1):n-Int32(2)
        idx+=step;c1=load4(cd,idx);c2=load4(bd,idx);c3=load4(ad,idx);fi=load4(dd,idx);q=c3.*load4(wd,i+m*k).+c2;t=SVector{4,Float32}(1f0,1f0,1f0,1f0)./q;prev=(fi.-c3.*prev).*t;store4!(xd,idx,prev);store4!(wd,i+m*(k+Int32(1)),-c1.*t)
      end
      idx+=step;c2=load4(bd,idx);c3=load4(ad,idx);fi=load4(dd,idx);q=c3.*load4(wd,i+m*(n-Int32(1))).+c2;t=SVector{4,Float32}(1f0,1f0,1f0,1f0)./q;prev=(fi.-c3.*prev).*t;store4!(xd,idx,prev)
      for k in n-Int32(2):-Int32(1):Int32(0)
        idx-=step;prev=load4(xd,idx).+prev.*load4(wd,i+m*(k+Int32(1)));store4!(xd,idx,prev)
      end
    else
      i=globalid;i>=m&&return nothing
      reordered=MODE==5||MODE==7;step=reordered ? m : Int32(1);base=reordered ? i : i*n;idx=base
      priv=MVector{128,Float32}(undef);localmode=MODE==6||MODE==7
      c1=cd[base+1];c2=bd[base+1];fi=dd[base+1]
      if localmode;priv[2]=-c1/c2;else;wd[i+m+1]=-c1/c2;end
      prev=fi/c2;xd[base+1]=prev
      for k in Int32(1):n-Int32(2)
        idx+=step;c1=cd[idx+1];c2=bd[idx+1];c3=ad[idx+1];fi=dd[idx+1];wk=localmode ? priv[k+1] : wd[i+m*k+1];q=c3*wk+c2;t=1f0/q;prev=(fi-c3*prev)*t;xd[idx+1]=prev
        if localmode;priv[k+2]=-c1*t;else;wd[i+m*(k+Int32(1))+1]=-c1*t;end
      end
      idx+=step;c2=bd[idx+1];c3=ad[idx+1];fi=dd[idx+1];wk=localmode ? priv[n] : wd[i+m*(n-Int32(1))+1];q=c3*wk+c2;t=1f0/q;prev=(fi-c3*prev)*t;xd[idx+1]=prev
      for k in n-Int32(2):-Int32(1):Int32(0)
        idx-=step;wk=localmode ? priv[k+2] : wd[i+m*(k+Int32(1))+1];prev=xd[idx+1]+prev*wk;xd[idx+1]=prev
      end
    end
    end
'''
TRANS=r'''
    i0=bx*Int32(16)+tx;j0=by*Int32(16)+ty
    # Named complete16x16tiles make originalearly-returnguardsuniformlyfalse.
    (i0>=width||j0>=height)&&return nothing
    i1=by*Int32(16)+tx;j1=bx*Int32(16)+ty
    (i1>=height||j1>=width)&&return nothing
    @inbounds sh[ty*Int32(17)+tx+1]=src[i0+j0*width+1]
    BARRIER
    @inbounds dst[i1+j1*height+1]=sh[tx*Int32(17)+ty+1]
'''
