"""Independent manufactured Navier operator and exact untouched-halo checks."""
from pathlib import Path
import json,array,struct,hashlib,sys

def read(p):
 a=array.array('d');a.frombytes(Path(p).read_bytes());return a

def main():
 result=Path(sys.argv[1]).resolve();d=json.loads(result.read_text());base=read(d['models']['cuda']['trials'][0]['output_artifact']['saved_path']);cases=[];pos=0
 for ni,nj,nk in [(13,11,18),(19,13,20)]:
  for mode in range(3):
   assert list(base[pos:pos+11])==[ni,nj,nk,mode,3,-2,ni-3,-2,nj-3,-2,nk+3];pos+=11;n=ni*nj*(nk+6);vec=[]
   for length in [3*n,n,n,4*n,n,3*n,ni,nj,864]:vec.append(base[pos:pos+length]);pos+=length
   u,mu,la,met,jac,init,sx,sy,cof=vec;assert cof==array.array('d',[q/1000 for q in range(864)])
   for iz in range(nk+6):
    for iy in range(nj):
     for ix in range(ni):
      q=ix+ni*(iy+nj*iz);x,y,z=ix-2,iy-2,iz-2
      uv=[x*x+2*x*y+3*x*z,2*y*y+4*x*y+5*y*z,3*z*z+6*x*z+7*y*z]if mode==0 else[1,2,3]if mode==1 else[((q*17+c*13)%97-48)/16 for c in range(3)]
      assert all(u[c*n+q]==uv[c] for c in range(3))
      if mode<2:assert mu[q]==2 and la[q]==3 and jac[q]==1 and [met[c*n+q]for c in range(4)]==[1,0,0,1]
   for q,v in enumerate(init):assert v==((q*7)%31-15)/32
   cases.append((ni,nj,nk,mode,n,init))
 prefix=pos;records=[];maxerr=0.;halo_count=0;physical_count=0
 for model,rec in d['models'].items():
  for tr in rec.get('trials',[]):
   p=Path(tr['output_artifact']['saved_path']);a=read(p);assert a[:prefix].tobytes()==base[:prefix].tobytes();pos=prefix;h=0;ph=0;err=0.
   for ni,nj,nk,mode,n,init in cases:
    out=a[pos:pos+3*n];pos+=3*n
    for iz in range(nk+6):
     for iy in range(nj):
      for ix in range(ni):
       x,y,z=ix-2,iy-2,iz-2;inside=0<=x<=ni-5 and 0<=y<=nj-5 and 1<=z<=nk
       for c in range(3):
        q=ix+ni*(iy+nj*iz)+c*n
        if not inside:assert struct.pack('d',out[q])==struct.pack('d',init[q]);h+=1
        elif mode<2 and 7<=z<=nk-6:
         expected=init[q]-([64,73,82][c]if mode==0 else 0);e=abs(out[q]-expected);assert e<=1e-8,(model,tr['trial'],ni,mode,x,y,z,c,out[q],expected);err=max(err,e);ph+=1
    # Boundary arrays use original synthetic q/1000 tables; full native
    # and Julia expression witnesses cover them, but are not independent SBP physics.
   assert pos==len(a);maxerr=max(maxerr,err);halo_count=h;physical_count=ph;records.append({'model':model,'trial':tr['trial'],'artifact':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exact_untouched_halo_values':h,'independent_interior_navier_values':ph,'max_interior_abs_error':err})
 out={'benchmark':'sw4ck','result':str(result),'input_prefix_count':prefix,'full_output_count':len(base)-prefix,'capture_count':len(records),'independent_method':'For homogeneous Cartesian metrics and lambda3/mu2, use L(u)=mu*Laplacian(u)+(lambda+mu)*grad(div(u)). Manufactured quadratic has div(u)=12x+13y+14z, Laplacian(u)=(2,4,6), hence L=(64,73,82). Constant displacement has L=0. Source operation is initial-L. This analytic witness shares no source finite-difference expressions.','independent_interior_values_per_capture':physical_count,'max_independent_interior_abs_error':maxerr,'exact_untouched_halo_values_per_capture':halo_count,'boundary_evidence_limit':'All boundary fields compared against nativeGPU/CUDAjl and scalar source-expression references, not an independently proven SBP discretization. Synthetic original q/1000 coefficient table is not certified as physical boundary data. No time integration, energy stability, convergence or original dataset norm claim.','captures':records,'pass':len(records)==20}
 (Path(__file__).parent/'independent_witness.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='captures'},indent=2))
if __name__=='__main__':main()
