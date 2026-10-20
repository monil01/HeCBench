#!/usr/bin/env python3
"""Independent Float64 DG curl/linear flux/RK evaluator and conservative FP32 forward bounds."""
from pathlib import Path
import argparse,json,numpy as np,hashlib
U=2.0**-24
def gamma(n):return n*U/(1-n*U)
def load(p,name,dtype='<f4'):return np.fromfile(p/name,dtype=dtype)
def main():
 a=argparse.ArgumentParser();a.add_argument('--assets',type=Path,required=True);a.add_argument('--actual-trace',type=Path);a.add_argument('--output',type=Path,required=True);x=a.parse_args();x.output.mkdir(parents=True,exist_ok=True);models=[];audit=[]
 for r in [0,1]:
  pre=f'rank{r}_';meta=list(map(int,(x.assets/(pre+'meta.txt')).read_text().split()));rank,nr,K,N,Nf,F,B,nh=meta;assert(rank,nr,N,Nf,F,B)==(r,2,84,28,4,96)
  read=lambda name,dtype='<f4':load(x.assets,pre+name+'.bin',dtype).astype(np.float64)
  D=read('DrDsDt').reshape(B,B,4).transpose(2,1,0)[:3,:N,:N]
  G=read('vgeo').reshape(K,3,4)[:,:,:3].transpose(0,2,1)
  L=read('LIFT').reshape(Nf,N,4).transpose(1,0,2).reshape(N,F*Nf)
  Q=read('initial_Q').reshape(K,6,B);rhs=read('initial_rhs').reshape(K,6,B);res=read('initial_res').reshape(K,6,B);assert not rhs.any()and not res.any()
  surf=read('surfinfo').reshape(K,7,F*Nf);ids=surf[:,:2].astype(np.int64);assert np.array_equal(ids,surf[:,:2]);mapout=load(x.assets,pre+'parmapOUT.bin','<i4').astype(np.int64);assert np.all((mapout>=0)&(mapout<Q.size))
  # Manufactured linear coordinate fields: nodal derivative + actual metric must recover Cartesian identity.
  control=[]
  for c,name in enumerate(['x','y','z']):
   coord=read(name,'<f8').reshape(K,N);dr=np.einsum('rnm,km->krn',D,coord);grad=np.einsum('kar,krn->kan',G,dr);expected=np.zeros_like(grad);expected[:,c,:]=1
   db=gamma(170)*np.einsum('rnm,km->krn',abs(D),abs(coord))+1e-10*np.einsum('rnm,km->krn',abs(D),np.ones_like(coord))
   budget=np.einsum('kar,krn->kan',abs(G),db)+gamma(8)*np.einsum('kar,krn->kan',abs(G),abs(dr))+1e-9
   assert np.all(abs(grad-expected)<=budget),(r,name,float(np.max(abs(grad-expected)/budget)))
   control.append(dict(coordinate=name,max_abs_error=float(abs(grad-expected).max()),max_budget_ratio=float(np.max(abs(grad-expected)/budget))))
  models.append(dict(rank=r,K=K,N=N,B=B,Q=Q,E=np.zeros_like(Q),res=res,Er=np.zeros_like(Q),D=D,G=G,L=L,surf=surf,ids=ids,map=mapout,rka=read('rk4a_f64','<f8').astype(np.float32).astype(np.float64),rkb=read('rk4b_f64','<f8').astype(np.float32).astype(np.float64),control=control,previous_actual=Q.copy()))
 for stage in range(1,6):
  sends=[m['Q'].ravel()[m['map']]for m in models];senderr=[m['E'].ravel()[m['map']]for m in models];actualg=[]
  for m in models:
   r=m['rank']
   if x.actual_trace is not None:
    ag=load(x.actual_trace,f'rank{r}_stage{stage}_gathered.bin');assert np.array_equal(ag,m['previous_actual'].ravel()[m['map']]);actualg.append(ag)
  for m in models:
   r=m['rank'];K,N,B=m['K'],m['N'],m['B'];Q,E=m['Q'],m['E'];D,G,L,surf,ids=m['D'],m['G'],m['L'],m['surf'],m['ids'];recv=sends[1-r];recve=senderr[1-r];
   if x.actual_trace is not None:assert np.array_equal(load(x.actual_trace,f'rank{r}_stage{stage}_received.bin'),actualg[1-r])
   der=np.einsum('rnm,kfm->kfrn',D,Q[:,:,:N]);de=np.einsum('rnm,kfm->kfrn',abs(D),E[:,:,:N])+gamma(168)*np.einsum('rnm,kfm->kfrn',abs(D),abs(Q[:,:,:N])+E[:,:,:N]);phy=np.einsum('kar,kfrn->kfan',G,der);pe=np.einsum('kar,kfrn->kfan',abs(G),de)+gamma(12)*np.einsum('kar,kfrn->kfan',abs(G),abs(der)+de)
   vol=np.zeros_like(Q);ve=np.zeros_like(Q)
   for f0,sign in [(0,-1),(3,1)]:
    src=3-f0
    for c,j,k in [(0,1,2),(1,2,0),(2,0,1)]:vol[:,f0+c,:N]=sign*(phy[:,src+k,j,:]-phy[:,src+j,k,:]);ve[:,f0+c,:N]=pe[:,src+k,j,:]+pe[:,src+j,k,:]
   M,P=ids[:,0],ids[:,1];remote=P<0;safe=np.maximum(P,0);Fsc,bc=surf[:,2],surf[:,3];norm=surf[:,4:7].transpose(0,2,1);jumps=np.empty((K,112,6));je=np.empty_like(jumps)
   for f in range(6):
    qm=Q.ravel()[M+f*B];em=E.ravel()[M+f*B];qp=Q.ravel()[safe+f*B];ep=E.ravel()[safe+f*B];remoteids=6*(-1-P[remote])+f;qp[remote]=recv[remoteids];ep[remote]=recve[remoteids];coef=np.where(remote,1,bc)if f>=3 else np.ones_like(bc)
    jumps[:,:,f]=Fsc*(coef*qp-qm);je[:,:,f]=abs(Fsc)*(abs(coef)*ep+em)+gamma(3 if f>=3 else 2)*abs(Fsc)*(abs(coef)*(abs(qp)+ep)+abs(qm)+em)
   H,A=jumps[:,:,:3],jumps[:,:,3:];eH,eA=je[:,:,:3],je[:,:,3:];nh=np.sum(norm*H,axis=2);na=np.sum(norm*A,axis=2)
   fluxH=H-norm*nh[:,:,None]-np.cross(norm,A);fluxA=A-norm*na[:,:,None]+np.cross(norm,H)
   def cross_error(n,e):return np.stack([n[:,:,1]*e[:,:,2]+n[:,:,2]*e[:,:,1],n[:,:,2]*e[:,:,0]+n[:,:,0]*e[:,:,2],n[:,:,0]*e[:,:,1]+n[:,:,1]*e[:,:,0]],axis=2)
   an=abs(norm);eh=eH+an*np.sum(an*eH,axis=2)[:,:,None]+cross_error(an,eA);ea=eA+an*np.sum(an*eA,axis=2)[:,:,None]+cross_error(an,eH);flux=np.concatenate([fluxH,fluxA],axis=2).transpose(0,2,1);fe=np.concatenate([eh,ea],axis=2).transpose(0,2,1);fe+=(U/(1-U))*(abs(flux)+fe)+1e-13*(1+abs(flux))
   lift=np.einsum('nm,kfm->kfn',L,flux);le=np.einsum('nm,kfm->kfn',abs(L),fe)+gamma(224)*np.einsum('nm,kfm->kfn',abs(L),abs(flux)+fe)
   rhs=vol.copy();re=ve.copy();rhs[:,:,:N]+=lift;re[:,:,:N]+=le+(U/(1-U))*(abs(rhs[:,:,:N])+ve[:,:,:N]+le)
   fa,fb,dt=m['rka'][stage-1],m['rkb'][stage-1],float(np.float32(.0005));res=fa*m['res']+dt*rhs;er=abs(fa)*m['Er']+abs(dt)*re+gamma(4)*(abs(fa)*(abs(m['res'])+m['Er'])+abs(dt)*(abs(rhs)+re));new=Q+fb*res;en=E+abs(fb)*er+gamma(3)*(abs(Q)+E+abs(fb)*(abs(res)+er))
   for name,ref,budget in [('volume_rhs',vol,ve),('surface_rhs',rhs,re),('res',res,er),('Q',new,en)]:
    allow=budget+1e-10
    if x.actual_trace is not None:
     actual=load(x.actual_trace,f'rank{r}_stage{stage}_{name}.bin').reshape(K,6,B).astype(np.float64);err=abs(actual-ref);ok=bool(np.isfinite(actual).all()and np.all(err<=allow));row=dict(rank=r,stage=stage,field=name,checked_values=actual.size,passed=ok,max_abs_error=float(err.max()),max_normalized_budget_ratio=float(np.max(err/allow)),max_allowed_error=float(allow.max()));audit.append(row);assert ok,row
    ref.astype('<f8').tofile(x.output/f'rank{r}_stage{stage}_{name}_reference_f64.bin');allow.astype('<f8').tofile(x.output/f'rank{r}_stage{stage}_{name}_bound_f64.bin')
   m.update(Q=new,E=en,res=res,Er=er,previous_actual=load(x.actual_trace,f'rank{r}_stage{stage}_Q.bin').reshape(K,6,B).astype(np.float64)if x.actual_trace is not None else None)
 for m in models:
  r=m['rank'];np.concatenate([m['Q'].ravel(),m['res'].ravel()]).astype('<f8').tofile(x.output/f'rank{r}_final_Q_res_f64.bin');np.concatenate([m['E'].ravel(),m['Er'].ravel()]).astype('<f8').tofile(x.output/f'rank{r}_final_Q_res_bound_f64.bin')
 report=dict(independent_evaluator='Float64matrixcurl+linearMaxwellflux+lowstorageRK; no GPU/sourcekerneltranscription or nativeoutputs as oracle inputs',roundoff_budget='prederivedFP32gamma(2n),absoluteoperatorpropagation,Float32jumpandfluxcast/RKroundoff; inputexact, sourcecoefficientsFloat32, all5stages propagated. Float64arithmeticfloor1e-10; no postobservationalrelaxation',NumPy=np.__version__,checks=audit,manufactured_linear_gradient_controls={str(m['rank']):m['control']for m in models},all_passed=True if x.actual_trace is not None else None,actual_trace_observed=x.actual_trace is not None,checks_count=len(audit),full_original_CLI_physical_solution_certificate=False,timestep_policy='SourcecomputedCFLdt~0.000381 discarded byoriginalrun hardcoded.001, clampedtoFinalTime.0005; no stabilitycertificate',extra_GPU_runs=0)
 (x.output/'independent_oracle_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('INDEPENDENT_DG_AUDIT',len(audit),'passed maxratio',max((z['max_normalized_budget_ratio']for z in audit),default=None))
if __name__=='__main__':main()
