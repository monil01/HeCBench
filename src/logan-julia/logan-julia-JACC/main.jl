using CUDA
import JACC
JACC.@init_backend
function kernel!(ii,seed,q,t,res,oq,ot,off,ant,drop,direction,template)
 sh=JACC.shared(template)

    b=Int32((Int32(ii)-Int32(1))÷Int32(128)+Int32(1));tid=Int32((Int32(ii)-Int32(1))%Int32(128));base=(b-Int32(1))*off*Int32(3)
    q0=b==1 ? Int32(0) : oq[b-1];t0=b==1 ? Int32(0) : ot[b-1]
    cols=oq[b]-q0+Int32(1);rows=ot[b]-t0+Int32(1)
    if rows>1 && cols>1
      p1=base;p2=base+off;p3=base+Int32(2)*off
      a1=Int32(0);a2=Int32(1);a3=Int32(2);o1=Int32(0);o2=Int32(0);o3=Int32(0)
      if tid==0;ant[p2+1]=Int16(0);ant[p3+1]=Int16(-1);ant[p3+2]=Int16(-1);end
      JACC.sync_workgroup()
      mn=Int32(1);mx=Int32(2);d=Int32(1);best=Int32(0);lo=Int32(0);hi=Int32(0)
      while mn<mx
        d+=Int32(1);p1,p2,p3=p2,p3,p1;a1,a2,a3=a2,a3,a1;o1,o2,o3=o2,o3,mn-Int32(1)
        a3=mx+Int32(1)-o3
        if tid==0
          ant[p3+1]=Int16(-32767);ant[p3+mx-o3+1]=Int16(-32767)
          if -d>best-drop
            if o3==0;ant[p3+1]=Int16(-d);end
            if d-mx==0;ant[p3+mx-o3+1]=Int16(-d);end
          end
        end
        JACC.sync_workgroup()
        ii=Int32(0)
        while ii<mx
          c=tid+mn+ii
          if c<mx
            val=max(Int32(ant[p2+c-o2+1]),Int32(ant[p2+c-o2]))-Int32(1)
            sc=q[q0+c]==t[t0+c+rows-d] ? Int32(1) : Int32(-1)
            val=max(val,Int32(ant[p1+c-o1])+sc)
            ant[p3+tid+ii+2]=Int16(val<best-drop ? -32767 : val)
          end
          ii+=Int32(128)
        end
        JACC.sync_workgroup()
        ab=Int32(-32767);ii=Int32(0)
        while ii<a3
          sh[tid+1]=tid<a3-ii ? ant[p3+tid+ii+1] : Int16(-32767)
          JACC.sync_workgroup()
          stride=Int32(64)
          while stride>0
            if tid<stride;sh[tid+1]=max(sh[tid+1],sh[tid+stride+1]);end
            JACC.sync_workgroup()
            stride>>=Int32(1)
          end
          ab=max(ab,Int32(sh[1]));ii+=Int32(128)
          JACC.sync_workgroup()
        end
        best=max(best,ab)
        while mn-o3<a3 && ant[p3+mn-o3+1]==Int16(-32767) && mn-o2-1<a2 && ant[p2+mn-o2]==Int16(-32767)
          mn+=Int32(1)
        end
        while mx-o3>0 && ant[p3+mx-o3]==Int16(-32767) && ant[p2+mx-o2]==Int16(-32767)
          mx-=Int32(1)
        end
        mx+=Int32(1);lo=min(lo,mn-(d-mn));hi=max(hi,mx-Int32(2)-(d+Int32(2)-mx))
        mn=max(mn,d+Int32(2)-rows);mx=min(mx,cols)
        JACC.sync_workgroup()
      end
      c=a3+o3-Int32(2);rr=d-c;score=Int32(ant[p3+c-o3+1])
      if score==-32767
        if ant[p2+a2-1]!=Int16(-32767)
          c=a2+o2-Int32(2);rr=d-Int32(1)-c;score=Int32(ant[p2+a2-1])
        elseif a2>2 && ant[p2+a2-2]!=Int16(-32767)
          c=a2+o2-Int32(3);rr=d-Int32(1)-c;score=Int32(ant[p2+a2-2])
        end
      end
      if score==-32767
        jj=Int32(0)
        while jj<a1
          vv=Int32(ant[p1+jj+1])
          if vv>score;score=vv;c=jj+o1;rr=d-Int32(2)-c;end
          jj+=Int32(1)
        end
      end
      if tid==0
        sb=(b-Int32(1))*Int32(10)
        if score!=-32767
          if direction==1
            bd=seed[sb+8];seed[sb+6]=min(seed[sb+6],bd+lo);seed[sb+7]=max(seed[sb+7],bd+hi)
            seed[sb+1]-=rr;seed[sb+2]-=c
          else
            ed=seed[sb+9];seed[sb+7]=max(seed[sb+7],ed-lo);seed[sb+6]=min(seed[sb+6],ed-hi)
            seed[sb+3]+=rr;seed[sb+4]+=c
          end
        end
        seed[sb+5]=Int32(0) # capture canonicalization of source's unread/uninitialized copy member
        seed[sb+10]=Int32(0);res[b]=score
      end
    end

 return nothing
end
const TEMPLATE=JACC.zeros(Int16,128)
function launch(s,q,t,r,oq,ot,off,a,drop,dir,stream)
 CUDA.stream!(stream) do
  spec=JACC.launch_spec(;threads=128,blocks=length(r),shmem_size=256,sync=false)
  JACC.parallel_for(spec,128*length(r),kernel!,s,q,t,r,oq,ot,off,a,drop,dir,TEMPLATE)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->JACC.zeros(T,n),launch)
