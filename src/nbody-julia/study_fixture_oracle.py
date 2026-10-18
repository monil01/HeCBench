from pathlib import Path
import struct,json,math,hashlib
p=Path(__file__).resolve().parent
F=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
U=2**-24
G=F(6.67259e-11);SOFT=F(1e-3);DT=F(.1)
def gamma(k):return k*U/(1-k*U)
def forces(P):
 A=[];terms=[]
 for i,pi in enumerate(P):
  contributions=[]
  for j,pj in enumerate(P):
   if i==j:continue
   d=[pj[k]-pi[k]for k in range(3)];r2=math.fsum(x*x for x in d)+SOFT;strength=G*pj[9]/(r2*math.sqrt(r2));contributions.append([x*strength for x in d])
  A.append([math.fsum(t[k]for t in contributions)for k in range(3)]);terms.append([math.fsum(abs(t[k])for t in contributions)for k in range(3)])
 return A,terms

def oracle(initial,steps):
 n=len(initial);P=[list(v)for v in initial];Ep=[[0.]*3 for _ in P];Ev=[[0.]*3 for _ in P];outputs=[];budgets=[];info=[];A0,T0=forces(P);replay=[v[:6]+a+[v[9]]for v,a in zip(P,A0)];force_budget=[(gamma(n+12)+64*U)*x for row in T0 for x in row];replay_budget=[]
 for i in range(n):replay_budget +=[0.]*6+force_budget[3*i:3*i+3]+[0.]
 momentum0=[math.fsum(v[9]*v[3+k]for v in P)for k in range(3)]
 for step in range(steps):
  A,T=forces(P);new=[];newEp=[];newEv=[];em=[];energy=[];Pmax=max(max(row)for row in Ep);delta=2*math.sqrt(3)*Pmax
  for i,pi in enumerate(P):
   # SpectralnormofJacobian d/(|d|²+soft)^(3/2) <=2/(|d|²+soft)^(3/2).
   # Alongboxpath ||delta(d)||2 <=2sqrt3 Pmax; soften lowerbound alwayspositive.
   input_bound=math.fsum(2*G*pj[9]*delta/((max(math.dist(pi[:3],pj[:3])-delta,0.)**2+SOFT)**1.5)for j,pj in enumerate(P)if j!=i)
   fa=[(gamma(n+12)+64*U)*T[i][k]+input_bound for k in range(3)]
   vel=[pi[3+k]+DT*A[i][k]for k in range(3)];pos=[pi[k]+DT*vel[k]for k in range(3)]
   ve=[Ev[i][k]+DT*fa[k]+gamma(2)*(abs(pi[3+k])+DT*abs(A[i][k])+Ev[i][k]+DT*fa[k])for k in range(3)]
   pe=[Ep[i][k]+DT*ve[k]+gamma(2)*(abs(pi[k])+DT*abs(vel[k])+Ep[i][k]+DT*ve[k])for k in range(3)]
   e=pi[9]*math.fsum(x*x for x in vel);eb=pi[9]*math.fsum(2*abs(vel[k])*ve[k]+ve[k]**2 for k in range(3))+gamma(6)*pi[9]*math.fsum((abs(vel[k])+ve[k])**2 for k in range(3))
   new.append(pos+vel+[0.,0.,0.,pi[9]]);newEp.append(pe);newEv.append(ve);energy.append(e);em.append(eb)
  P=new;Ep=newEp;Ev=newEv;total=math.fsum(energy);total_budget=math.fsum(em)+gamma(n)*math.fsum(x+b for x,b in zip(energy,em));energy[0]=total;em[0]=total_budget
  values=[v for particle in P for v in particle]+energy+[.5*total]
  bound=[b for pe,ve in zip(Ep,Ev)for b in pe+ve+[0.]*4]+em+[.5*total_budget]
  assert len(values)==len(bound)
  outputs+=values;budgets+=bound;info.append({'step':step+1,'kinetic_energy':.5*total,'max_position_budget':max(max(x)for x in Ep),'max_velocity_budget':max(max(x)for x in Ev),'kinetic_budget':.5*total_budget,'momentum_Float64_oracle_residual':[math.fsum(v[9]*v[3+k]for v in P)-momentum0[k]for k in range(3)]})
 outputs +=[v for particle in replay for v in particle];budgets+=replay_budget
 return outputs,budgets,info,A0,force_budget
raw=list(struct.unpack('<'+'f'*((p/'native_initial_f32.bin').stat().st_size//4),(p/'native_initial_f32.bin').read_bytes()));offset=0
ballistic=[[F(.3),F(.4),F(.5),F(.001),F(-.001),F(.0005),0.,0.,0.,2.]]
symmetric=[[.25,.5,.5,0.,0.,0.,0.,0.,0.,2.],[.75,.5,.5,0.,0.,0.,0.,0.,0.,2.]]
cases=[('ballistic1',1,3,ballistic),('symmetric_stationary2',2,4,symmetric)]
for n,steps in [(33,3),(257,5)]:
 initial=[raw[offset+10*i:offset+10*i+10]for i in range(n)];offset+=10*n;cases.append(('source_MT42_'+str(n),n,steps,initial))
fixture=[len(cases)];reference=[];budget=[];info=[]
for name,n,steps,P in cases:
 out,bnd,stepinfo,A0,AB=oracle(P,steps);fixture +=[n,steps,DT,SOFT,G]+[v for particle in P for v in particle];reference+=out;budget+=bnd
 assert all(v>=0 and math.isfinite(v)for v in bnd)
 info.append({'name':name,'n':n,'steps':steps,'step_metrics':stepinfo,'initial_force_max':max(abs(x)for row in A0 for x in row),'initial_force_max_bound':max(AB),'source_initialization':'ExactC++stdmt19937/distributionsnativeInitPos/Vel/Massdraworder'if n>2 else'analyticmanufacturedfixture'})
(p/'fixtures_f64.bin').write_bytes(struct.pack('<'+'d'*len(fixture),*fixture));(p/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(reference),*reference));(p/'error_budget_f64.bin').write_bytes(struct.pack('<'+'d'*len(budget),*budget))
derivation={'status':'PredeclaredconditionalengineeringFP32forwardmodel; notformalglobalcompiler/library/applicationproof','u':U,'gamma':'k*u/(1-k*u)','force_rounding':'(gamma(n+12)+64u)*sumabscomponentterms; normals/softening>=Float32(1e-3), no overflow/subnormals; conservativearithmetic/SFUmargin, onepureinitialforcereplay checksnontrivialtinyGoperators','source_rsqrt_accuracy':'NVIDIA CUDAguide12.9 documents2ULPrsqrtf; PTX12.0.1rsqrt.approx.f32normalaccuracy, using16umargininSFUchaincoveredby64u term','sources':['https://docs.nvidia.com/cuda/archive/12.9.0/pdf/CUDA_C_Programming_Guide.pdf','https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html'],'force_position_error':'delta=2sqrt3*maxpreviouspositionbudget; forceinputsum 2*G*mj*delta/(max(||dref||-delta,0)^2+soft)^(3/2); excludesi=jidenticallyzero selfterm','velocity':'Evnew=Evold+dtEforce+gamma2*(|vold|+dt|aref|+Evold+dtEforce)','position':'Epnew=Epold+dtEvnew+gamma2*(|pold|+dt|vnewref|+Epold+dtEvnew)','particle_energy':'m*sum(2|vref|Ev+Ev²)+gamma6*m*sum((|vref|+Ev)²)','serial_sum':'sumparticleenergybudgets+gamma(n)*sum(Eref+Ebudget), sameorderpreservednative/target; e[0]stores2KEtotal, KEhalf','exact_fields':['mass','postupdateacc0','initialforce-replayposition/velocity/mass'],'acceptance':'Allactualoutputswithincomponentwisecomputedbudgets, guards/exactzero fieldsbyteexact; primarydualrefsatol1e-10rtol5e-5 commonall4; no thresholdsadjustedfromGPUobservations','limitations':['ConditionalnormalFP32/SFUmodel+codegeninspection, no certifiedintervalcompilerproof','Finite3/4/3/5steps, sourcekick-driftEuler andsoftenedforce, no continuumerror/long-termorbitalaccuracy orenergyconservationclaim']}
(p/'error_model_derivation.json').write_text(json.dumps(derivation,indent=2)+'\n');(p/'independent_oracle.json').write_text(json.dumps({'method':'IndependentFloat64pairwisevectorforce/mathfsum, softenedJacobianforwardbudget/symplecticEuler/perparticle2KE+serialsumbudget, ballistic/pairedsymmetry/strictinitialforcereplay; everyparticle/energyfieldnotglobalscalarsalone','cases':info,'input_words':len(fixture),'oracle_words':len(reference),'budget_words':len(budget),'negative_controls':'N2zeroforce/zeroenergywouldpassoriginalglobalKEabs1e-3scalarcriterion butstrictforce/velocity/energycomponentbudgetsreject; notsourcealgorithmdefect.','error_model_sha256':hashlib.sha256((p/'error_model_derivation.json').read_bytes()).hexdigest()},indent=2)+'\n');print({'input_words':len(fixture),'output_words':len(reference),'budget_max':max(budget),'cases':[{'n':x['n'],'force':x['initial_force_max'],'force_bound':x['initial_force_max_bound']}for x in info]})
