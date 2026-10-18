from pathlib import Path
import json,struct,math,hashlib
R=Path('/home/imo/HeCBench_julia');p=R/'src/dwconv1d-julia';f=[];oracle=[];cases=[]
for B,C,T,mode in[(8,1,16,0),(16,3,64,1),(8,1,768,2)]:
 w=[((i*17%41)-20)/256 for i in range(C*T)];k=[((i*29%67)-33)/128 for i in range(B*C*T)];g=[((i*19%53)-26)/64 for i in range(B*C*T)]
 if mode==0:w=[float(i%T==T-1)for i in range(C*T)];k=[float(i%T==3)for i in range(B*C*T)];g=[float(i%T==5)for i in range(B*C*T)]
 eps=.125;f +=[B,C,T,mode,eps]+w+k+g;x=[];gw=[];gk=[]
 for b in range(B):
  for c in range(C):
   for t in range(T):
    x.append(eps+math.fsum(w[c*T+T-1-t+u]*k[(b*C+c)*T+u]for u in range(t+1)))
    gw.append(math.fsum(g[(b*C+c)*T+v]*k[(b*C+c)*T+v+t-(T-1)]for v in range(T-1-t,T)))
    gk.append(math.fsum(g[(b*C+c)*T+v]*w[c*T+T-1-v+t]for v in range(t,T)))
 oracle+=x+gw+gk;cases.append({'B':B,'C':C,'T':T,'mode':mode,'oracle_outputs':3*B*C*T})
(p/'fixtures_f32.bin').write_bytes(struct.pack('<%sf'%len(f),*f));(p/'oracle_f64.bin').write_bytes(struct.pack('<%sd'%len(oracle),*oracle));(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent direct causal convolution and symbolic derivatives with math.fsum Float64; signed dyadic fixtures make exact binary multiplication; epsilon constant derivative zero','cases':cases,'oracle_outputs':len(oracle),'input_prefix_count':len(f),'gradW':'Native per-batch derivative; host batch sum checked outside kernel ROI, not timed vendor reduction','scope':'Kernel forward/backward on arbitrary supplied upstream derivative, not full tanh-loss/autograd/PyTorch pipeline'},indent=2)+'\n')
