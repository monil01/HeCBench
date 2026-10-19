from pathlib import Path
import re,struct,json,math,hashlib
p=Path(__file__).resolve().parent;r=p.parent.parent;h=(r/'src/rushlarsen-cuda/utils.h').read_text();u=(r/'src/rushlarsen-cuda/utils.cu').read_text();states=re.findall(r'^  STATE_(\w+),',h,re.M);params=re.findall(r'^  PARAM_(\w+),',h,re.M);sv={n:float(v)for n,v in re.findall(r'states\[n \* STATE_(\w+) \+ i\] = ([^;]+);',u)};pv={n:float(v)for n,v in re.findall(r'parameters\[n \* PARAM_(\w+) \+ i\] = ([^;]+);',u)};assert len(states)==19 and len(params)==53
cases=[];out=bytearray(struct.pack('<I',3));gate=bytearray(struct.pack('<I',3));Vlist=[-85.423,-60.,-40.0001,-40.,-39.9999,-15.02,-15.,-14.98,0.,20.,-80.]
def analytic(s,pa,dt):
 V=s[17];Css=s[14];exp=math.exp
 inf=[1/(1+exp(-26/7-V/7)),1/(1+exp(11/3+V/24)),1/(1+exp(-5/14-V/14)),1/(1+.00184221158116513*exp(-.110741971207087*V))**2,1/(1+15212.5932856544*exp(.134589502018843*V))**2]
 inf+=[inf[4],1/(1+.344153786865412*exp(-.133333333333333*V)),1/(1+exp(20/7+V/7)),.33+.67/(1+exp(5+V/7)),.4+.6/(1+400*Css*Css),1/(1+exp(4+V/5)),1/(1+exp(10/3-V/6))]
 ah=4.43126792958051e-7*exp(-.147058823529412*V)if V < -40 else 0.;bh=310000*exp(.3485*V)+2.7*exp(.079*V)if V < -40 else .77/(.13+.0497581410839387*exp(-.0900900900900901*V));aj=(37.78+V)*(-25428*exp(.2444*V)-6.948e-6*exp(-.04391*V))/(1+50262745825.954*exp(.311*V))if V < -40 else 0.;bj=.02424*exp(-.01052*V)/(1+.00396086833990426*exp(-.1378*V))if V < -40 else .6*exp(.057*V)/(1+.0407622039783662*exp(-.1*V))
 tau=[450/(1+exp(-9/2-V/10))*6/(1+13.5813245225782*exp(.0869565217391304*V)),3/(1+exp(-3-V/20))*1.12/(1+exp(-3+V/20)),80+1400/math.sqrt(1+exp(5/6-V/6))/(1+exp(-7/3+V/15)),1/(1+exp(-12-V/5))*(.1/(1+exp(7+V/5))+.1/(1+exp(-1/4+V/200))),1/(ah+bh),1/(aj+bj),(.25+1.4/(1+exp(-35/13-V/13)))*1.4/(1+exp(1+V/5))+1/(1+exp(5/2-V/20)),20+180/(1+exp(3+V/10))+200/(1+exp(13/10-V/10))+1102.5*exp(-(27+V)**2/225),31/(1+exp(5/2-V/10))+80/(1+exp(3+V/10))+562*exp(-(27+V)**2/240),2+80/(1+400*Css*Css),3+5/(1+exp(-4+V/5))+85*exp(-(45+V)**2/320),.8+9.5*exp(-(40+V)**2/1800)]
 y=[s[k]+(inf[k]-s[k])*(-math.expm1(-dt/tau[k]))for k in range(12)];par=dict(zip(params,pa));kcasr=par['max_sr']-(par['max_sr']-par['min_sr'])/(1+par['EC']**2/s[12]**2);lam=par['k4']+Css*par['k2_prime']*kcasr;y.append(s[15]if lam==0 else s[15]+(par['k4']/lam-s[15])*(-math.expm1(-dt*lam)));return y
for ci,(n,steps,np,t,dt)in enumerate([(33,1,11,9.9999,0.),(257,100,11,9.9999,2e-5),(100000,100,1,0.,2e-5)]):
 out+=struct.pack('<III2d',n,steps,np,t,dt);patterns=[];gold=[]
 for k in range(np):
  s=[sv[x]for x in states];pa=[pv[x]for x in params]
  if ci<2:
   s[17]=Vlist[k]
   for j in range(12):s[j]=min(1.,s[j]*(.9+.02*k))
   for j in [12,13,14,16,18]:s[j]*=.9+.02*k
   pa[params.index('stim_start')]=10.+(k%3)*2e-5;pa[params.index('stim_duration')]=.0007+(k%2)*.0002;pa[params.index('stim_amplitude')]=52.*(.9+.02*k)
   if k==10:pa[params.index('k2_prime')]=0.;pa[params.index('k4')]=0.
  out+=struct.pack('<72d',*(s+pa));patterns.append({'states':s,'parameters':pa});gold.append(analytic(s,pa,dt))
 gate+=struct.pack('<I',np)
 for j in range(13):
  for k in range(np):gate+=struct.pack('<d',gold[k][j])
 cases.append({'case':ci,'nodes':n,'steps':steps,'patterns':np,'t_start':t,'dt':dt,'scope':['manufactured_zero_step_identity_mixed_voltage','heterogeneous_stimulus_crossing_and_partial_tail','original_Makefile_run_100x100000_default_states_params'][ci],'pattern_values':patterns})
(p/'fixtures_compact.bin').write_bytes(out);(p/'independent_gates.bin').write_bytes(gate);total=sum(x['nodes']for x in cases);info={'cases':cases,'state_names':states,'parameter_names':params,'input_prefix_count':1+3*5+72*total,'output_count':1+3*5+110*total+3*256,'fixture_sha256':hashlib.sha256(out).hexdigest(),'source_CPU_oracle':'All19states afterfirst/finalstep computedperuniqueindependentinputpattern; GPUcomputesall100000nodes andchecksallactualvalues, noGPUfallback','independent_checks':'12affinegatefrozen-coefficientclosedformsusingstableexpm1 plusRprimereceptor affineformula, dt0exactidentity, finiteness/positiveconcentrations/gatebounds/sourceinputreadbacks/fullguards/readonlyparameters','dual_tolerances_preGPU':{'atol':1e-12,'rtol':1e-10},'gate_closed_form_atol_preGPU':2e-13,'error_rationale':'ConservativeFloat64combined1e-12+1e-10relative for100step/full474lineoperator withtranscendental/FMAimplementationdifferences, unchangedbeforeGPU; separateaffineanalyticgatebudget2e-13 and exactdt0/input/guards. Notclinicallycalibratedaccuracyorlonghorizonstabilitybound.','limitations':['SourceMakefile100stepsx100000nodes included, default1node1millionsteps/higherhorizonsnotcertified','Timeunitnotinferredfrommodelname; dt2e-5/stimperiod1000 asactualsource','Firststepcaptureisuntimedreplay aftersavingactualtimedfinalstate, nottimedintermediate','ScalarpatternsourceCPUoracle hascommonalgorithmprovenance; independentgate/derivativewitnesses donotvalidateall19ionicRHS/modelphysiology']};(p/'fixture_manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(info['input_prefix_count'],info['output_count'])
