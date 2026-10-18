from pathlib import Path
import numpy as np,json,hashlib,math
B=Path(__file__).resolve().parent;G=16;SENTINEL=12345.5
specs=[(1,1,17,3,0),(5,3,33,2,0),(7,2,512,64,0),(3,2,65,3,1),(5,3,31,2,2)]

def schedule(s,l):
 groups=[];ls=le=rs=0
 while True:
  if le==0:ls=0;le=1;rs=0
  else:
   ls+=1;le+=1;rs-=2
   if le>l or rs<0:rs+=(ls+1)*2;ls=0;le=1
   while rs>=s and le<=l:ls+=1;le+=1;rs-=2
   if le>l or rs<0:break
  groups.append((ls,le,rs,min(rs+2,s)))
 visited=[(j,t)for a,b,c,d in groups for j in range(a,b)for t in range(c,d)]
 assert sorted(visited)==[(j,t)for j in range(l)for t in range(s)]
 for j in range(l):assert[t for k,t in visited if k==j]==list(range(s))
 return groups

def lcg(n):
 i=np.arange(n,dtype=np.uint64);seed=i^n;bits=((26757677*seed+1)&0xffffffff)&0x7fffffff
 return bits.astype(np.float32)/np.float32(2147483648)

def golden(c,b,th,ti,s,l,h,batch):
 ne=h*batch;hc=c.astype(np.float64).reshape(l,s+1,batch,h).copy();hh=np.full_like(hc,SENTINEL);ii=np.full((l+1,s,batch,h),SENTINEL);gg=np.full((l,s,batch,4,h),SENTINEL);bs=b.astype(np.float64).reshape(l,8,h);ths=th.astype(np.float64).reshape(l,batch,4,h);tis=ti.astype(np.float64).reshape(s,batch,4,h)
 # Straight layer/time order is independent of the original diagonal state machine.
 for layer in range(l):
  for t in range(s):
   g=tis[t]+ths[layer]+(bs[layer,:4]+bs[layer,4:])[None,:,:];gg[layer,t]=g
   a=1/(1+np.exp(-g[:,0]));f=1/(1+np.exp(-g[:,1]));v=np.tanh(g[:,2]);o=1/(1+np.exp(-g[:,3]));cell=f*hc[layer,t]+a*v;hc[layer,t+1]=cell;hidden=o*np.tanh(cell);hh[layer,t+1]=hidden;ii[layer+1,t]=hidden
 return [hh.ravel(),ii.ravel(),hc.ravel(),b.astype(float),th.astype(float),ti.astype(float),gg.ravel()]

def main():
 inputs=[];oracle=[];cases=[]
 for s,l,h,batch,mode in specs:
  ne=h*batch;sizes=[(s+1)*l*ne,s*(l+1)*ne,(s+1)*l*ne,l*h*8,4*l*ne,4*s*ne,4*s*l*ne]
  c,b,th,ti=[lcg(sizes[k])for k in [2,3,4,5]]
  if mode==1:c=(2*c-1).astype(np.float32);b=(8*b-4).astype(np.float32);th=(8*th-4).astype(np.float32);ti=(8*ti-4).astype(np.float32)
  if mode==2:
   c=((np.arange(len(c))%9)-4).astype(np.float32)/8;b.fill(0);th.fill(0);ti.fill(0)
  off=len(inputs);offsets=[]
  for v in [c,b,th,ti]:offsets.append(len(inputs));inputs.extend(v.tolist())
  go=len(oracle);gold=golden(c,b,th,ti,s,l,h,batch)
  for v in gold:oracle.extend(np.r_[np.full(G,SENTINEL),v,np.full(G,SENTINEL)].tolist())
  cases.append(dict(seq=s,layers=l,hidden=h,batch=batch,mode=mode,sizes=sizes,input_offsets=offsets,oracle_offset=go,groups=schedule(s,l)))
 (B/'inputs_f32.bin').write_bytes(np.array(inputs,dtype='<f4').tobytes());(B/'oracle_f64.bin').write_bytes(np.array(oracle,dtype='<f8').tobytes())
 files={n:dict(bytes=(B/n).stat().st_size,sha256=hashlib.sha256((B/n).read_bytes()).hexdigest())for n in ['inputs_f32.bin','oracle_f64.bin']}
 # Actual historical CPU initializer comparison, not an invented failed GPU trial.
 count=257;i=np.arange(count,dtype=np.uint64);old=((26757677*(i^count)+1)&0xffffffff).astype(np.float32)/np.float32(2147483648);new=lcg(count);bad=np.flatnonzero(old.view(np.uint32)!=new.view(np.uint32));assert len(bad)>0
 witness=dict(size=count,mismatches=len(bad),first=dict(index=int(bad[0]),historical=float(old[bad[0]]),native_contract=float(new[bad[0]])),cause='HistoricalUInt32multiplywrapomitsmod2^31; sourceGPU/C++ref bothmod2^31',scope='CPUexecutedinitializercounterexample only; historicalGPUtranslationattemptcountunknown')
 (B/'historical_initializer_witness.json').write_text(json.dumps(witness,indent=2)+'\n')
 (B/'fixture_provenance.json').write_text(json.dumps(dict(cases=cases,files=files,oracle='IndependentFloat64straightlayer/time LSTMpointwisealgebra; sigmoid/tanhFloat64, fullgates/states; nativewavefrontscheduleindependentlycheckeduniquecompleteandtimeordered',source_initializer='Exactintegeruint32multiply/addwrapthenmod2^31; Float32convert/divide commonmaterializedfixtures; actualnativeinitGPU diagnostics forsource-modecases outsideROI',initial_undefined_scope='h/i/lgoriginalnativeuninitializedbutneverreadbyelementwise; studyinitializesuntouchedslots+guards to12345.5outsideROI andchecksasdiagnosticstate, notsourceundefinednumericoutput',atol=1e-5,rtol=2e-6,independent_atol=2e-5,independent_rtol=2e-6,source_default='MakefileS100L8H512B64R10; namedlargestS7L2H512B64 preservesper-kernelgeometry/workn32768, notfull100x8pipeline',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
