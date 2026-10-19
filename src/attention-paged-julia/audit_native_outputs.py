"""CPU-only full-record audit: exact actual inputs and independent dense attention.
The key cache is unpacked into a logical dense tensor before any dot products.
"""
from pathlib import Path
import array,ctypes,hashlib,json,math,struct,sys
P=Path(__file__).resolve().parent

def load(path):
 a=array.array('d');a.frombytes(Path(path).read_bytes());return a
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
libm=ctypes.CDLL('libm.so.6');libm.fmaf.argtypes=[ctypes.c_float]*3;libm.fmaf.restype=ctypes.c_float

def random32(state):
 state^=(state<<13)&0xffffffff;state^=state>>17;state^=(state<<5)&0xffffffff
 return state&0xffffffff

def audit(path):
 wire=load(path);fixture=load(P/'fixture.bin');prefix=[3.];fo=1;cases=[];rngchecked=0;normalerror=0.
 for ci in range(3):
  head=list(fixture[fo:fo+8]);fo+=8;S,H,D,L,B,style=map(int,head[:6]);scale,kvscale=head[6:];M=(L+15)//16;Q=S*H*D;KV=B*H*D*16
  lens=list(map(int,fixture[fo:fo+S]));fo+=S;tables=list(map(int,fixture[fo:fo+S*M]));fo+=S*M
  baseline=[];actual=[]
  for destination in [baseline,actual]:
   for n in [Q,KV,KV,H]:destination.append(list(fixture[fo:fo+n]));fo+=n
  for values,sc in zip(baseline[:3],[scale,kvscale,kvscale]):
   low=f32(-sc);high=f32(sc);width=f32(high-low)
   for i,observed in enumerate(values):
    state=random32(1234^i);u=(state>>8)*2**-24;want=float(libm.fmaf(width,u,low));assert observed==want,(ci,i,observed,want);rngchecked+=1
  for i,observed in enumerate(baseline[3]):
   r=random32(1234^i);u1=(r>>8)*2**-24;r=random32(r);u2=(r>>8)*2**-24;theta=f32(f32(6.28318530718)*u2);want=math.sqrt(-2.*math.log(u1))*math.cos(theta);e=abs(observed-want);normalerror=max(normalerror,e);assert e<=3e-6
  def guard(v):return [-37.]*32+list(v)+[-37.]*32
  initial=[guard([0.]*n)for n in [Q,KV,KV,H,Q]]+[guard(tables),guard(lens)]
  prefix+=head
  for v in initial:prefix+=v
  for v in baseline:prefix+=guard(v)
  attention=[guard(v)for v in actual]+[guard([0.]*Q),guard(tables),guard(lens)]
  for v in attention:prefix+=v
  cases.append((head,lens,tables,baseline,actual,initial,attention))
 assert fo==len(fixture)and len(prefix)==1016993
 assert list(wire[:len(prefix)])==prefix,'actualinputprefixmismatch'
 pos=len(prefix);full=[];allmath=0.;convex=0.;outputs=0;normalization=0.;constantresidual=0.
 for ci,(head,lens,tables,baseline,actual,initial,attention)in enumerate(cases):
  S,H,D,L,B,style=map(int,head[:6]);M=(L+15)//16;Q=S*H*D;KV=B*H*D*16;q,key,value,alibi=actual
  # Independently decode the serialized 5D packed key coordinates.
  dense=[0.]*KV
  for index,x in enumerate(key):
   component=index%4;rest=index//4;token=rest%16;rest//=16;group=rest%(D//4);rest//=D//4;h=rest%H;page=rest//H
   dense[((page*H+h)*D+group*4+component)*16+token]=x
  actualout=list(wire[pos:pos+Q+64]);pos+=Q+64;assert actualout[:32]==[-37.]*32 and actualout[-32:]==[-37.]*32
  full.append(actualout)
  for sequence in range(S):
   N=lens[sequence]
   for h in range(H):
    logits=[];densev=[]
    for t in range(N):
     page=tables[sequence*M+t//16];token=t%16;assert 0<=page<B
     dot=sum(q[(sequence*H+h)*D+d]*dense[((page*H+h)*D+d)*16+token]for d in range(D));logits.append(dot*head[6]+alibi[h]*(t-N+1))
    largest=max(logits);exp=[math.exp(x-largest)for x in logits];den=sum(exp)+1e-6;weights=[x/den for x in exp];total=math.fsum(weights);assert 0<total<1.;normalization=max(normalization,1-total)
    for d in range(D):
     vv=[value[((tables[sequence*M+t//16]*H+h)*D+d)*16+t%16]for t in range(N)];want=sum(p*v for p,v in zip(weights,vv));observed=actualout[32+(sequence*H+h)*D+d];e=abs(observed-want);allmath=max(allmath,e);outputs+=1;assert math.isfinite(observed)and e<=2e-5+2e-5*abs(want),(ci,sequence,h,d,observed,want)
     residual=max(0.,min(vv)*total-observed,observed-max(vv)*total);convex=max(convex,residual);assert residual<=2e-5
     if style==2:assert max(vv)==min(vv);constantresidual=max(constantresidual,abs(observed-vv[0]));assert abs(observed-vv[0])<=2e-7
 for ci,(head,lens,tables,baseline,actual,initial,attention)in enumerate(cases):
  for i,expected in enumerate(attention):
   n=len(expected);after=list(wire[pos:pos+n]);pos+=n
   assert after==(full[ci]if i==4 else expected),'readonly/output/guardmismatch'
 assert pos==len(wire)==1357921
 return {'all_pass':True,'artifact_words':len(wire),'exact_actual_input_prefix_words':len(prefix),'independent_source_uniform_values':rngchecked,'normal_BoxMuller_max_abs_error':normalerror,'independent_dense_attention_output_values':outputs,'max_abs_vs_dense_Float64_attention':allmath,'maximum_softmax_mass_deficit':normalization,'convex_hull_residual':convex,'constantV_output_residual':constantresidual,'scope':'FP32 dense paged single-query attention, equalQ/KVheads, ALiBi+source1e-6normalizer, block16; noBF16/sparse/GQA/partition/globalLLMqualityclaim.'}
if __name__=='__main__':
 path=Path(sys.argv[1]);out=audit(path);out['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();print(json.dumps(out,indent=2))
