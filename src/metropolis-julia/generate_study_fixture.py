from pathlib import Path
import struct,json
from independent_cpu import *
p=Path(__file__).resolve().parent;pre=[3];gold=[];cases=[];IG=-12345;UG=0x5a5a5a5a5a5a5a5a

def encode(a,bits):
 if bits==32:return list(a)
 out=[]
 for x in a:out.extend([x&0xffffffff,x>>32])
 return out
for ci,(h,B,ns) in enumerate([(.5,-.5,2),(.125,-.25,1),(0.,0.,3)]):
 s=[1 if ci==1 else (1 if (x*11+y*7+z*3)%7<4 else -1) for z in range(L) for y in range(L) for x in range(L)]
 H=[1 if ci==1 else (1 if (x*7+y*3+z)%5<2 else -1) for z in range(L) for y in range(L) for x in range(L)]
 st=[((i+1+ci*17)*0x9e3779b97f4a7c15)&mask for i in range(N//4)];inc=[2*i+1+ci*2 for i in range(N//4)];result,newstate,margin,flips=simulate(s,H,st,inc,h,B,ns);assert margin>1e-7
 if B==0:assert result==[-x for x in s]
 pre.extend([L,h,B,ns]);arrays=[s,H,st,inc]
 for a,b in zip(arrays,[32,32,64,64]):pre.extend(encode(a,b))
 for fields in [[result,H,newstate,inc],arrays]:
  for a,b in zip(fields,[32,32,64,64]):gold.extend(encode([IG if b==32 else UG]*8+a+[IG if b==32 else UG]*8,b))
 def energy(a):return -sum(a[idx(x,y,z)]*(a[idx(x+1,y,z)]+a[idx(x,y+1,z)]+a[idx(x,y,z+1)]+h*H[idx(x,y,z)]) for z in range(L) for y in range(L) for x in range(L))
 cases.append(dict(case=ci,L=L,h=h,B=B,sweeps=ns,min_uniform_probability_margin=margin,accepted_proposals=flips,initial_magnetization=sum(s),final_magnetization=sum(result),initial_Ising_energy=energy(s),final_Ising_energy=energy(result),zero_beta_analytical_flip_control=B==0))
for name,a in [('fixture.bin',pre),('oracle.bin',gold)]:(p/name).write_bytes(struct.pack('<'+'d'*len(a),*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(pre),oracle_values=len(gold),artifact_values=len(pre)+len(gold),oracle='Independent direct global Ising energy-flip/PCG oracle with source two-level blockcheckerboard schedule, no shared halo implementation reuse. Full states, actualinputecho andguards exact. Float32uniformrounding and modulo2^64 PCGstate, Float64probability with recordeddecisionmargins.',source_scope='Only full Metropolis spin-update component (the original timed loop). Excludes PRNGsetup/randomfieldinitialization, GPUenergyreduction, replicaexchange/temperatureadaptation, nativeCLIdefaultparameter certification.',acceptance='Exact fullspins/64bitPCGstate/inc, actualinputbytes andguards. UInt64 values losslessly encodedlow/highUInt32words inFloat64artifact; neverconvertedwholeUInt64toFloat64.'),indent=2)+'\n')
print(len(pre),len(gold))
