from pathlib import Path
import struct,json,ctypes,math
p=Path(__file__).resolve().parent;lib=ctypes.CDLL('libm.so.6');lib.fmaf.argtypes=[ctypes.c_float]*3;lib.fmaf.restype=ctypes.c_float;lib.fma.argtypes=[ctypes.c_double]*3;lib.fma.restype=ctypes.c_double
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
prefix=[192];gold=[];cases=[];cache={}
for bits in [32,64]:
 R=f32 if bits==32 else float;F=lib.fmaf if bits==32 else lib.fma
 for N in [256,512]:
  for rep in [0,1,2]:
   for op in range(1,5):
    for lanes in [1,2,4,8]:
     v1=R(10 if op in [1,3] else .5 if op==2 else 1);v2=R(.9899 if op==3 else .125 if op==4 else 0);updates=([240,120,60,30] if op in [1,3] else [200,100,50,25] if op==2 else [160,80,40,20])[[1,2,4,8].index(lanes)];xs=[R(3.25+(i%15)*.25) for i in range(N)];out=[]
     for x in xs:
      key=(bits,op,lanes,rep,x)
      if key in cache:out.append(cache[key]);continue
      if op==2:s=[R(R(x-x)+R(f32(.999)))];s +=[R(s[0]-R(f32(.0001*i))) for i in range(1,8)]
      else:
       s=[x,R(R(f32(10))-x)];s +=[R(R(f32(v))-s[k]) for v,k in [(9,0),(9,1),(8,0),(8,1),(7,0),(7,1)]]
      for _ in range(rep*updates):
       for k in range(lanes):
        z=s[k]
        if op==1:s[k]=R(v1-z)
        elif op==2:s[k]=R(R(z*z)*v1)
        elif op==3:s[k]=R(F(-z,v2,v1))
        else:s[k]=R(R(F(-v2,z,v1))*z)
      if lanes==1:y=s[0]
      elif lanes==2:y=R(s[0]+s[1])
      elif lanes==4:y=R(R(s[0]+s[1])+R(s[2]+s[3]))
      else:y=R(R(R(s[0]+s[1])+R(s[2]+s[3]))+R(R(s[4]+s[5])+R(s[6]+s[7])))
      assert math.isfinite(y);cache[key]=y;out.append(y)
     prefix +=[bits,N,op,lanes,rep,v1,v2]+xs;gold +=[-12345]*8+out+[-12345]*8+[-12345]*8+xs+[-12345]*8
     cases.append(dict(bits=bits,N=N,op=op,lanes=lanes,repeat=rep,updates_per_accumulator_per_repeat=updates,v1=v1,v2=v2,atol=1e-4 if bits==32 else 1e-11,rtol=1e-5 if bits==32 else 1e-12,min_output=min(out),max_output=max(out),exact=rep==0 or op in [1,2]))
for n,a in [('fixture.bin',prefix),('oracle.bin',gold)]:(p/n).write_bytes(struct.pack('<'+'d'*len(a),*a))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),oracle='Independent scalar recurrence, explicit IEEEFP32/64 round aftereach sourceoperation, libm fmaf/fma for source fused multiply-subtract. CorrectF32 literal promotion eveninFP64kernels. Sourcegrouped finalreduction. Separate native compiler maycontract finalreduction acrosslast update; dtype-specific fixed engineering budgets declaredbeforeGPU.',budgets={'float32':[1e-4,1e-5],'float64':[1e-11,1e-12]},argument_scope='Finite runtime controls Mulv.5,MulMAddv1=1,v2=.125; original Add10/MAdd10,.9899. Allselectedinitiallanes positive3.25..6.75firsttwo, remainingpositive. OriginalMul1.01overflow/chaoticMulMAdd3.75,.355unverified, nopeakFLOPsclaim.'),indent=2)+'\n')
print(len(prefix),len(gold))
