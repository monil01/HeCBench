from pathlib import Path
import math,struct,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];N=512;T=257
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
cases=[];allout=[]
for case in range(3):
 weights=[f32(float(x))for x in(R/'src/clink-cuda'/('weight_1.hpp'if case!=1 else'weight_2.hpp')).read_text().split()]if case<2 else [f32(((i*7)%19-9)/256)for i in range(146)]
 xs=[f32(.001)if case<2 else f32((((g%8)*11+t*7)%37-18)/32)for g in range(N)for t in range(T)]
 outputs=[];cache={}
 for g in range(N):
  key=g%8 if case==2 else 0
  if key not in cache:
   h=[0.]*5;c=[0.]*5;ys=[]
   for x in xs[g*T:(g+1)*T]:
    gates=[]
    for gate in range(4):
     vals=[]
     for j in range(5):
      a=weights[gate*5+j]*x+math.fsum(h[i]*weights[20+gate*25+j*5+i]for i in range(5))+weights[120+gate*5+j]
      vals.append(math.tanh(a)if gate==3 else 1/(1+math.exp(-a)))
     gates.append(vals)
    c=[c[j]*gates[1][j]+gates[3][j]*gates[0][j]for j in range(5)]
    h=[math.tanh(c[j])*gates[2][j]for j in range(5)]
    ys.append(weights[145]+math.fsum(h[j]*weights[140+j]for j in range(5)))
   cache[key]=ys
  outputs.extend(cache[key])
 cases.append(xs+weights);allout.extend(outputs)
flat=sum(cases,[])
(P/'study_inputs_f32.bin').write_bytes(struct.pack('<%df'%len(flat),*flat));(P/'study_expected_f64.bin').write_bytes(struct.pack('<%dd'%len(allout),*allout))
meta={'N':N,'T':T,'cases':3,'inputs_per_case':N*T+146,'output_count':len(allout),'source_scope':'Source8192x20000 reducedto512x257 withoutkernelbodychange. Suppliedweightfilesidenticalconstant. Cases0/1 retainactualsourceinput/weights,case2 separatelynamedvariedsigneddyadicdata,8distinctsequencesrepeated64times.','oracle':'Float64scalarLSTM withmath.fsum, gates/state updatedafterallfourgatevectors computed; outputsalltimesteps. NotFP64GPUcompute.','sha256':{name:hashlib.sha256((P/name).read_bytes()).hexdigest()for name in['study_inputs_f32.bin','study_expected_f64.bin']}}
(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(meta)
