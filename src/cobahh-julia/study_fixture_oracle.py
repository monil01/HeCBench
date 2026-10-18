from pathlib import Path
import re,math,struct,json
p=Path(__file__).resolve().parent;R=p.parent.parent
source=(R/'src/cobahh-cuda/neuron_update.h').read_text();f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];dt=f(.0001);clock=f(.01);L={1:30}
for i,ex in re.findall(r'const float\s+_lio_(\d+)\s*=\s*([^;]+);',source):
 ex=ex.strip();L[int(i)]=f(math.exp(f((-2000 if int(i)==23 else -100)*dt)))if 'expf' in ex else f(float(ex.rstrip('f')))
def timestep(t):return int(f(f(t+f(f(.001)*dt))/dt))
def evolve(state):
 h,m,n,ge,v,gi,last=state;Ah=L[2]*math.exp(L[3]*v);Bh=-4/(f(.001)+L[4]*math.exp(L[5]*v))-Ah;den=L[7]+L[8]*math.exp(L[9]*v);d2=L[13]+L[14]*math.exp(L[15]*v);Am=(L[6]+L[10]-f(.32)*v)/den;Bm=(L[11]-L[10]+f(.32)*v)/den+(L[12]+L[16]-f(.28)*v)/d2;d3=L[7]+L[18]*math.exp(L[5]*v);An=(L[17]+L[19]-f(.032)*v)/d3;Bn=(L[20]-L[19]+f(.032)*v)/d3-L[21]*math.exp(L[22]*v);Av=L[24]+L[25]*n**4+L[26]*h*m**3+L[27]*ge+L[28]*gi;Bv=L[29]+L[30]*n**4-(L[31]*h*m**3+L[32]*ge+L[32]*gi);upd=lambda x,A,B:f(-A/B+(x+A/B)*math.exp(dt*B));return [upd(h,Ah,Bh),upd(m,Am,Bm),upd(n,An,Bn),f(L[23]*ge),upd(v,Av,Bv),f(L[33]*gi),last]
cases=[(33,1,0),(257,1,1),(129,3,2)];fixture=[];oracle=[];flags=[];predicted=[]
for n,steps,mode in cases:
 arrays=[[]for k in range(7)]
 for i in range(n):
  last=f(clock-f([.0025,.003,.0035,-.001][i%4]))
  if mode==0:s=[f(.6),f(.05),f(.32),f(1e-9),f(-.065+(i%5)*.0005),f(1e-9),last]
  elif mode==1:s=[f(.35+(.3 if i%2 else -.3)),f(.45+(.4 if i%3 else -.4)),f(.55+(.5 if i%4 else -.5)),f(.15+(.1 if i%2 else -.1)),f(.65+(.6 if i%5 else -.6)),f(.25+(.2 if i%3 else -.2)),last]
  else:s=[f(.35),f(.45),f(.55),f(.05+((i%7)/128)),f(.05),f(.05+((i%7)/128)),last]
  for a,x in zip(arrays,s):a.append(x)
 fixture.extend([n,steps,mode,dt,clock]+[L[i]for i in range(1,34)]+sum(arrays,[]));states=[list(a)for a in zip(*arrays)]
 for st in range(steps):states=[evolve(s)for s in states]
 out=sum([list(a)for a in zip(*states)],[]);nr=[float(timestep(f(clock-x))>=L[1])for x in arrays[6]];oracle.extend(out+nr);flags.extend(nr);predicted.append({'mode':mode,'n':n,'iterations':steps,'m_min':min(s[1]for s in states),'m_max':max(s[1]for s in states),'all_finite':all(math.isfinite(x)for s in states for x in s)})
for name,data in [('fixtures_f32.bin',fixture),('oracle_f32.bin',oracle)]:
 with(p/name).open('wb')as io:io.write(struct.pack('<'+'f'*len(data),*data))
w={'method':'Independent affine frozen-coefficient ODE update A+B*x with equilibrium -A/B and Float64 exponential, rounded to actual Float32 storage each step; recomposes grouped rates rather than copying nested BA expressions','source_constants':{str(k):v for k,v in L.items()},'cases':predicted,'scientific_defect_witness':{'physiological_initial_state':{'v':f(-.065),'m':f(.05),'h':f(.6),'n':f(.32),'ge':f(1e-9),'gi':f(1e-9)},'predicted_source_m_after_one_step':predicted[0]['m_min'],'compiled_lio8':0,'comment_expression_lio8_as_float32':f(25.79034*(.001*.001*math.exp(.25*(-.063)/.001))),'compiled_lio18':0,'comment_expression_lio18_as_float32':f(20.085537*(.001*.001*math.exp(.2*(-.063)/.001))),'contract':'For dm/dt=alpha(1-m)-beta*m with nonnegative rates, frozen-voltage exponential update preserves gate probability [0,1]. Source zero coefficients make effective alpha_m negative.','scientific_context_source':'https://brian.readthedocs.io/en/stable/examples-misc_COBAHH.html','context_limit':'Official example establishes related mathematical intent; exact generated-code provenance not claimed. Actual native GPU output still to be checked.'},'limits':['Compiled source coefficient realization checked, not faithful calibrated Hodgkin-Huxley physics','Time held fixed across repeated updates as source; no network spikes/synaptic/refractory/reset simulation','Full original initializer leaves index0 uninitialized; named fixtures fully initialize every neuron']};(p/'independent_oracle.json').write_text(json.dumps(w,indent=2)+'\n')
