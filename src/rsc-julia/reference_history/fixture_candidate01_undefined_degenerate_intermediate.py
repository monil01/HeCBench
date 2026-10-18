"""Bounded optical-flow fixtures; independent exact-rational first-order fits.

The source fits displacement = [D,-R;R,D] (position-center), then
classifies integer pixel residuals after separately truncating each term.
No probabilistic success/optimality assertion is made.
"""
from pathlib import Path
from fractions import Fraction as F
import ctypes, struct, json, hashlib
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',float(x)))[0]
libc=ctypes.CDLL(None)
pre=[4.]; gold=[]; descriptions=[]
guard=1515870810
for ci,(n,m,e,conv,cpu,blocks,threads) in enumerate([(257,131,3,.75,3,17,64),(65,67,1,.5,1,3,32),(96,97,2,0.,2,11,64),(128,129,3,1.,4,19,64)]):
    flows=[]
    for i in range(n):
        x=40+4*(i%16); y=40+4*((i//16)%17)
        u=(2*(x-32)-(y-32))//4;v=((x-32)+2*(y-32))//4
        if i%7==0:u+=20;v-=16
        if i%19==0:u-=12;v+=24
        flows.append((x,y,x+u,y+v))
    # Explicit repeated positions and zero/different displacement degeneracies.
    flows[-1]=flows[0];flows[-2]=(flows[0][0],flows[0][1],flows[0][2]+8,flows[0][3]-4)
    libc.srand(123)
    pairs=[libc.rand()%n for _ in range(2*m)]
    # Source RNG sequence retained, with a named explicit degeneracy control prefix.
    pairs[:6]=[0,n-1,0,n-2,1,1]
    params=[]; counts=[]; accepted=[]; max_int=0; invalid=0
    for model in range(m):
        a,b=[flows[k] for k in pairs[2*model:2*model+2]]
        x,y,hx,hy=a; xx,yy,hhx,hhy=b
        u,v=hx-x,hy-y;uu,vv=hhx-xx,hhy-yy
        dx,dy=x-xx,y-yy;du,dv=u-uu,v-vv
        velden=du*du+dv*dv;posden=dx*dx+dy*dy
        # Independent solved linear equations, not the source expanded polynomial.
        if not velden:
            q=[-2011.,0.,0.,0.];invalid+=1
        else:
            dn=dx*du+dy*dv;rn=dx*dv-dy*du
            cx=F(x)-F(dn*u+rn*v,velden)
            cy=F(y)-F(-rn*u+dn*v,velden)
            q=[f32(cx),f32(cy),0.,0.]
            if not posden:q[0]=-2011.;invalid+=1
            else:q[2:]=[f32(F(dn,posden)),f32(F(rn,posden))]
        # Actual native polynomial intermediates independently evaluated in unbounded
        # integers: every constituent and final numerator must fit signed Int32.
        intermediates=[u*(u-2*uu),uu*uu,v*v,vv*(2*v-vv),
          -uu*x,u*xx,-uu*xx,vv*y,-vv*yy,
          -vv*x,v*xx,-vv*xx,-uu*y,uu*yy,
          u*(-uu*x+u*xx-uu*xx+vv*y-vv*yy),
          v*(-vv*x+v*xx-vv*xx-uu*y+uu*yy),x*(vv*vv+uu*uu),
          uu*(v*x-v*xx-u*y+uu*y-u*yy),
          vv*(-u*x+u*xx-v*y+vv*y-v*yy),yy*(u*u+v*v),
          x*(x-2*xx),xx*xx,y*(y-2*yy),yy*yy,dn,rn]
        max_int=max(max_int,*map(abs,intermediates))
        assert all(-(2**31)<=z<2**31 for z in intermediates)
        assert q[0]!=-2011. or not velden or not posden
        params.extend(q)
        cnt=-1
        if q[0]!=-2011.:
            cnt=0
            for x,y,hx,hy in flows:
                rx=x+int(f32(f32(f32(x)-q[0])*q[2]))-int(f32(f32(f32(y)-q[1])*q[3]))-hx
                ry=y+int(f32(f32(f32(y)-q[1])*q[2]))+int(f32(f32(f32(x)-q[0])*q[3]))-hy
                cnt+=abs(rx)>=e or abs(ry)>=e
            if cnt<f32(f32(n)*f32(conv)):accepted.append((model,cnt))
        counts.append(cnt)
    # Cross-check exact-rational solved parameters with the unmodified original
    # fitting routine, independently of every GPU implementation.
    class Flow(ctypes.Structure):_fields_=[(z,ctypes.c_int) for z in ('x','y','vx','vy')]
    fit=ctypes.CDLL(str(p.parent.parent/'.porting-state/porting_logs/rsc-julia-shared/study_fit.so'))
    fit.rsc_fit.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.POINTER(Flow),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    par=(ctypes.c_float*(4*m))();fv=(Flow*n)(*[Flow(*a) for a in flows]);rp=(ctypes.c_int*(2*m))(*pairs);ident=ctypes.c_int(0)
    fit.rsc_fit(par,fv,n,rp,m,e,conv,ctypes.byref(ident),cpu)
    assert list(par)==params,[(j,x,y) for j,(x,y) in enumerate(zip(par,params)) if x!=y][:10]
    flat=[a for flow in flows for a in flow]
    header=[n,m,e,conv,cpu,blocks,threads]
    pre.extend(header+flat+pairs)
    # Canonical full accepted-ID/count set, NOT the raw concurrent append order.
    padded=[z for row in accepted for z in row]+[-1.]*(2*(m-len(accepted)))
    minimum=min((z[1] for z in accepted),default=n)
    vals=[len(accepted),minimum,1.]+params+padded+flat+params+[guard]*64
    gold.extend(vals)
    descriptions.append(dict(case=ci+1,n=n,models=m,error_threshold=e,convergence=conv,cpu_threads=cpu,blocks=blocks,threads=threads,accepted=len(accepted),minimum_outliers=minimum,minimizer_ids=[a for a,b in accepted if b==minimum],invalid_models=invalid,all_model_outlier_counts=counts,source_int_intermediate_max=max_int,source_int32_safe=True,artifact_values=len(vals),raw_record_values=4+4*m+2*m+4*n+4*m+64))
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]:
    (p/name).write_bytes(struct.pack('<'+str(len(a))+'d',*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=descriptions,input_prefix_values=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),independent_oracle='Exact Fraction linear-system first-order motion fits rounded once to Float32, separately truncated source Float32 pixel-residual convention, full candidate membership/counts/minimizer membership. Shared original C++ fit crosschecked on CPU before GPU trials.',random='libc srand123/rand()%N; first3pairs deliberately repeated-coordinate/displacement controls, remaining exact libc sequence',property_contract='Canonical sorted accepted ID/count pairs compared exactly; full raw append sequence and bestID retained durably, valid minimum membership checked rather than tied bestID identity. Numerical/raw-output identity claim false.',parameter_bits='Source integer intermediates checked Int32-safe, original fitted parameters equal independent exact-rational solution on every model; incomplete invalid parameters retained.',unverified='Probabilistic RANSAC success, global optimum, application accuracy, supplied5922flow dataset/default2000modelCLI, nonselected overflow/near-threshold states.'),indent=2)+'\n')
print(len(pre),len(gold),len(pre)+len(gold),[(x['accepted'],x['minimum_outliers'],x['invalid_models']) for x in descriptions])
