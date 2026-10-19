#!/usr/bin/env python3
"""Deterministic connectivity-mapping fixtures and independent CPU dot/statistical oracle."""
from pathlib import Path
import array,struct,math,json,hashlib,subprocess,tempfile,sys
P=Path(__file__).resolve().parent;N=22283;R=4099;SIG=31;T=128;DB=(N+127)//128;SB=(R+127)//128;M=R*SIG;UC=SIG*N-SIG*(SIG+1)//2+SIG
f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
def arrayread(p,code):a=array.array(code);a.frombytes(p.read_bytes());return list(a)
def refs():
 a=[(i*73)%N-11141 for i in range(N)];b=[(i*97+13)%N-11141 for i in range(N)];return [[x+y for x,y in zip(a,b)],[-x for x in a]]
def query():
 q=[0]*N
 for k in range(SIG):q[(k*677+19)%N]=-1 if k%3==0 else 1
 return q
def score_oracle(random,ref,setsize,generations):
 sums=[0]*generations;indices=[];scaled=[]
 for j,v in enumerate(random):
  x=f32(1-float(v));sign=-1 if x>=.5 else 1
  if x>=.5:x=f32(x-.5)
  i=math.floor(f32(f32(x*N)*2));assert 0<=i<N;indices.append(i);sums[j%generations]+=sign*ref[i]
 # set sizes1/2 produce exact dyadic accumulation in Float32 under our checked bound.
 assert max(map(abs,sums))<2**24
 return [f32((v/setsize)/f32(UC))for v in sums],indices

def pvalue(k,n):
 # Independent two-tail formula; primary n is odd, so original exact50% branch defect impossible.
 return 2*min(max(.5,k)/n,max(.5,n-k)/n)

def primary(randoms):
 prefix=[];tail=[];meta=[];q=query()
 for group,(ref,random,setsize)in enumerate(zip(refs(),randoms,[2,1])):
  prefix +=[N,R,SIG,setsize,UC,DB,SB]
  parts=[(ref,1234567),(q,1234567),(random,-999),( [0]*DB,-12345),([0]*R,-999),([0]*SB,-12345)]
  for vals,guard in parts:prefix +=[guard]*16+vals+[guard]*16
  dotpartial=[sum(ref[i]*q[i]for i in range(b*T,min(N,(b+1)*T)))for b in range(DB)];dot=sum(dotpartial);avg=(dot/UC)/setsize;threshold=f32(avg)
  scores,indices=score_oracle(random,ref,setsize,R);countpartial=[sum(v>threshold for v in scores[b*T:min(R,(b+1)*T)])for b in range(SB)];above=sum(countpartial);pv=pvalue(above,R);enfp=pv*1309;significant=int(enfp<1)
  for vals,guard in [(dotpartial,-12345),(scores,-999),(countpartial,-12345)]:tail +=[guard]*16+vals+[guard]*16
  tail +=[dot,avg,threshold,above,pv,enfp,significant]
  meta.append({'group':group,'set_size':setsize,'dot':dot,'average_score':avg,'threshold_f32':threshold,'above':above,'p_value':pv,'ENFP':enfp,'significant':significant,'random_min':min(random),'random_max':max(random),'selected_index_min':min(indices),'selected_index_max':max(indices),'independent_score_bound':max(map(abs,scores))})
 return prefix,tail,meta
if __name__=='__main__':
 assert sys.byteorder=='little'
 with tempfile.TemporaryDirectory(prefix='cm-CPU-rng-')as tmp:
  exe=Path(tmp)/'rng';cmd=['g++','-O2','-std=c++17',str(P/'native_diagnostics/generate_random.cpp'),'-o',str(exe)];subprocess.run(cmd,check=True);subprocess.run([str(exe),str(R),str(P)],check=True)
 randoms=[arrayread(P/f'random{k}_f32.bin','f')for k in range(2)]
 for k,v in enumerate(refs()):(P/f'ref{k}_i32.bin').write_bytes(array.array('i',v).tobytes())
 (P/'query_i32.bin').write_bytes(array.array('i',query()).tobytes());prefix,tail,meta=primary(randoms);(P/'oracle_f64.bin').write_bytes(array.array('d',prefix+tail).tobytes())
 h=f'#pragma once\nconstexpr int N={N},R={R},SIG={SIG},M={M},T={T},DB={DB},SB={SB},UC={UC},PREFIX={len(prefix)},COUNT={len(prefix+tail)};\n';(P/'study_params.h').write_text(h)
 jl='\n'.join(f'const {k}={v}'for k,v in [('N',N),('R',R),('SIG',SIG),('M',M),('T',T),('DB',DB),('SB',SB),('UC',UC),('PREFIX',len(prefix)),('COUNT',len(prefix+tail))])+'\n';(P/'study_params.jl').write_text(jl)
 names=['ref0_i32.bin','ref1_i32.bin','query_i32.bin','random0_f32.bin','random1_f32.bin','oracle_f64.bin','study_params.h','study_params.jl'];files={name:{'bytes':(P/name).stat().st_size,'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest()}for name in names}
 plan={'benchmark':'cm','genes':N,'random_generations':R,'signature_genes':SIG,'reference_profiles':3,'drug_sets':2,'set_sizes':[2,1],'threads':128,'dot_blocks':DB,'score_count_blocks':SB,'input_prefix_count':len(prefix),'artifact_count':len(prefix+tail),'groups':meta,'random_generator':'Actual libstdc++ std::default_random_engine(123) and uniform_real_distribution<float>(0,1), same source recurrence/draw order; stored bits commonall4, no crosslanguage RNG claim.','precision':'Int32dot/count (bound within2^31), Float32random/index arithmetic/randomscores, Float64hostnormalization/statistics. Setsize1/2 exact dyadic terms, unboundedinteger CPUscore numerator; Float32divisionrounding.','timing_scope':'Resident two-drug-set source3kernel pipeline with D2H partialdot/hostreduce→GPUthreshold, D2H count/hostpvalueENFP; inputparsing/RNG/allocation/H2D/scoreD2H excluded all4.','native_integration_adapted':True,'native_reference_repaired':False,'exclude_untouched_native_aggregates':True,'source_statistic_limit':'Original pvaluehelper returns0 instead of1 at exacthalf. Primaryodd4099 cannotencountertie; actualcomponentwitness retained, arbitraryfullstatisticaldomain notcertified. Strict > sourcecountpredicate preserved.','files':files,'python':sys.version}
 (P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'prefix':len(prefix),'count':len(prefix+tail),'groups':meta}))
