#!/usr/bin/env python3
"""Compare complete declared mathematical witnesses to actual saved GPU pipeline outputs."""
import argparse,array,hashlib,json,math,mmap,struct
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('result');parser.add_argument('--model',default='cuda');parser.add_argument('--output',required=True);args=parser.parse_args();p=Path(__file__).resolve().parent;result=json.load(open(args.result));oracles={};f=(p/'independent_math_oracle.bin').open('rb')
while True:
 data=f.read(28)
 if not data:break
 c,kind,r,step,orb,M,ne=struct.unpack('<7i',data);a=array.array('d');a.frombytes(f.read(8*(2*M+ne)));oracles[(c,kind,r,step,orb)]={'M':M,'ne':ne,'moments':a[:2*M],'scientific':a[2*M:]}
f.close();states={};f=(p/'independent_math_oracle.bin.states.bin').open('rb')
while True:
 data=f.read(20)
 if not data:break
 c,kind,r,role,n=struct.unpack('<5i',data);a=array.array('d');a.frombytes(f.read(16*n));states[(c,kind,r,role)]=a
f.close();fixtures={}
for c,name in enumerate(['diffusive','localized','small_lattice','general_tail','spin_pairs']):
 f=(p/(name+'_fixture.bin')).open('rb');h=struct.unpack('<12i',f.read(48));n,maxn,ne,nt,M,R,v0,vac,msd,spin,ldos,nl=h;f.read(16+8*(ne+nt)+4*nl)
 def read(T,N):
  a=array.array(T);a.frombytes(f.read(N*a.itemsize));return a
 nn=read('i',n);neighbor=read('i',n*maxn);potential=read('d',n);hr=read('d',n*maxn);hi=read('d',n*maxn);dx=read('d',n*maxn);expected={}
 for r in range(R):
  re=read('d',n);im=read('d',n);psi=[complex(a,b)for a,b in zip(re,im)];norm=math.fsum(abs(z)**2 for z in psi);Jpsi=[]
  for i in range(n):
   Jpsi.append(sum((-1j*complex(hr[i*maxn+k],hi[i*maxn+k])*dx[i*maxn+k]*psi[neighbor[i*maxn+k]]for k in range(nn[i])),0j))
  currentnorm=math.fsum(abs(z)**2 for z in Jpsi);expected[r]={'state_norm':norm,'current_norm':currentnorm}
 fixtures[c]={'n':n,'expected':expected};f.close()
all_trials=[]
for trial in result['models'][args.model]['trials']:
 file=Path(trial['output_artifact']['saved_path']);fp=file.open('rb');mem=mmap.mmap(fp.fileno(),0,access=mmap.ACCESS_READ);values=memoryview(mem).cast('d');at=int(trial['input_prefix_values']);got_mom={};got_sci={};got_state={};release_count=0;work={}
 while at<len(values):
  code=int(values[at])
  if code in (1,2,3):
   _,c,kind,r,step,orb,n=map(int,values[at:at+7]);at+=7;key=(c,kind,r,step,orb)
   if code==1:got_mom[key]=array.array('d',values[at:at+2*n]);at+=2*n
   elif code==2:got_sci[key]=array.array('d',values[at:at+n]);at+=n
   else:role=int(values[at]);at+=1;got_state[(c,kind,r,role)]=array.array('d',values[at:at+2*n]);at+=2*n
  elif code==4:
   c=int(values[at+1]);identity=int(values[at+2]);n=int(values[at+3]);at+=4+n;release_count+=1
  elif code==5:
   c=int(values[at+1]);assert values[at+2]==14;work[c]={'all_launches':list(map(int,values[at+3:at+17])),'timed_launches':list(map(int,values[at+17:at+31]))};at+=31
  else:raise AssertionError(('unknownrecord',code,at))
 assert at==len(values);comparisons=[];max_abs=0.;max_budget=0.;bad=0;count=0
 def compare(key,label,actual,expected):
  global max_abs,max_budget,bad,count
  assert len(actual)==len(expected);fail=0;ma=0.;mb=0.
  for a,b in zip(actual,expected):
   error=abs(a-b);budget=1e-9+1e-7*abs(b);ma=max(ma,error);mb=max(mb,error/budget);fail+=not(math.isfinite(a)and error<=budget)
  max_abs=max(max_abs,ma);max_budget=max(max_budget,mb);bad+=fail;count+=len(actual);comparisons.append({'key':key,'kind':label,'values':len(actual),'failures':fail,'max_abs_error':ma,'max_normalized_budget':mb})
 for key,o in oracles.items():compare(key,'complex_moments',got_mom[key],o['moments']);compare(key,'scientific_values',got_sci[key],o['scientific'])
 for key,expected in states.items():compare(key,'complete_small_final_wavefunction',got_state[key],expected)
 norms=[]
 for (c,kind,r,role),a in got_state.items():
  if role==13:continue # derivative wavefunction is not norm-preserving propagation.
  n=len(a)//2;actual=math.fsum(a[i]**2+a[n+i]**2 for i in range(n));expected=fixtures[c]['expected'][r]['current_norm'if role==11 else'state_norm'];error=abs(actual-expected);budget=1e-10+1e-9*abs(expected);valid=error<=budget;bad+=not valid;norms.append({'case':c,'stage':kind,'random':r,'role':role,'actual_norm_squared':actual,'expected_norm_squared':expected,'absolute_error':error,'budget':budget,'pass':valid})
 all_trials.append({'trial':trial['trial'],'pass':bad==0,'independently_compared_values':count,'mathematical_failures':bad,'max_abs_error':max_abs,'max_normalized_budget':max_budget,'comparisons':comparisons,'final_norm_witnesses':norms,'complete_released_GPU_allocations':release_count,'actual_work':work,'full_artifact_sha256':hashlib.sha256(mem).hexdigest()});values.release();mem.close();fp.close()
receipt={'pass':bool(all_trials)and all(t['pass']for t in all_trials),'model':args.model,'run_id':result['run_id'],'checked_trials':len(all_trials),'oracle_sha256':hashlib.sha256((p/'independent_math_oracle.bin').read_bytes()).hexdigest(),'oracle_states_sha256':hashlib.sha256((p/'independent_math_oracle.bin.states.bin').read_bytes()).hexdigest(),'trials':all_trials,'scope':'AllinitialDOScomplexmoments5models plus allsmallcontrolscientificfunctions/finalwavefunctions; independentlargefinalpropagatornorms. Alllargecorrelationcurves additionallyneedfullGPUdualcomparison, not independentlyreplayedpointwise. No transportlimit/stochasticconvergenceclaim.'};Path(args.output).write_text(json.dumps(receipt,indent=2)+'\n');print({'pass':receipt['pass'],'trials':len(all_trials),'independent_values':sum(t['independently_compared_values']for t in all_trials),'failures':sum(t['mathematical_failures']for t in all_trials),'max_abs':max(t['max_abs_error']for t in all_trials)});raise SystemExit(0 if receipt['pass']else 1)
