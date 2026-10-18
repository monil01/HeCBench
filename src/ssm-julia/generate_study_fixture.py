from pathlib import Path
import random,struct,json,math,hashlib
p=Path(__file__).resolve().parent;rng=random.Random(20261018);f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
prefix=[3.];gold=[];cases=[]
for batch,dim,L in [(1,1,1),(2,33,33),(1,257,1024)]:
 sizes=[batch*dim*L,batch*dim*L,dim*8,batch*8*L,batch*8*L,dim,dim,batch*dim*L,batch*dim*8]
 arrays=[]
 for i,n in enumerate(sizes):
  lo,hi=((-1.,-.0625) if i==2 else ((-4.,4.) if i==1 else ((-2.,2.) if i==7 else (-.25,.25))))
  arrays.append([f32(rng.uniform(lo,hi)) for _ in range(n)])
 u,delta,A,B,C,D,db,z,h0=arrays
 for b in range(batch):
  for t in range(L):u[(b*dim)*L+t]=0.
 for n in range(8):A[n]=0.
 if dim>1:
  for b in range(batch):
   for t in range(L):u[(b*dim+1)*L+t]=0.
 delta[-1]=24.
 prefix.extend([float(batch),float(dim),float(L)])
 for x in arrays:prefix.extend([float(len(x))]);prefix.extend(x)
 case={'batch':batch,'dim':dim,'seqlen':L,'dstate':8,'array_sizes':sizes,'chains':[]}
 for chain in [1,3]:
  h=h0.copy();ys=[]
  for step in range(chain):
   y=[0.]*(batch*dim*L)
   for b in range(batch):
    for d in range(dim):
     base=(b*dim+d)*L;sb=(b*dim+d)*8;ab=d*8;bc=b*8*L
     for t in range(L):
      dt=delta[base+t]+db[d];dt=dt if dt>20 else math.log1p(math.exp(dt));uv=u[base+t];v=D[d]*uv
      for n in range(8):
       j=sb+n;h[j]=math.exp(dt*A[ab+n])*h[j]+dt*B[bc+n*L+t]*uv;v+=C[bc+n*L+t]*h[j]
      zz=z[base+t];y[base+t]=v*zz/(1+math.exp(-zz))
   ys.extend(y+[-12345.]*17)
  gold.extend(ys+h+[-12345.]*17);case['chains'].append({'calls':chain,'output_values':len(ys)+len(h)+17})
 cases.append(case)
raw=struct.pack('<'+str(len(prefix))+'f',*prefix);graw=struct.pack('<'+str(len(gold))+'d',*gold);(p/'fixture.bin').write_bytes(raw);(p/'expected_outputs.bin').write_bytes(graw)
meta={'seed':20261018,'cases':cases,'prefix_count':len(prefix),'output_count':len(gold),'fixture_sha256':hashlib.sha256(raw).hexdigest(),'gold_sha256':hashlib.sha256(graw).hexdigest(),'oracle':'IndependentFloat64 gatedselectivelinearrecurrence with stablelog1pexp, fullalltimeoutputs forallchaincalls plusfinalhiddenstates. Channel0 u=0,A=0 =>unchangedh; channel1 u=0,A<0 homogeneousdecay. Nonzeroinitialstates; all8statecomponents.','scope':'Onlysource serialbase; VLLMCUBparallelscan/defaultCLIunverified. Gatingtrue,softplustrue,dstate8,finiteinputs.'};(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:meta[k] for k in ['prefix_count','output_count']})
