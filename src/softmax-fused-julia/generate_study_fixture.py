from pathlib import Path
import struct,random,math,json,hashlib
p=Path(__file__).resolve().parent;r=random.Random(20261018);f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def quant(x,kind):
 if kind==0:return struct.unpack('<e',struct.pack('<e',x))[0]
 bits=struct.unpack('<I',struct.pack('<f',x))[0];bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;return struct.unpack('<f',struct.pack('<I',bits))[0]
frames=[];prefix=[8.];gold=[];oldmaskmax=0.;cases=[]
for Q,C in [(256,1),(32,31),(32,128),(16,1024)]:
 batch,heads=2,3;rows=batch*heads*Q;n=rows*C;raw=[f32(r.uniform(-20,20)) for _ in range(n)];mask=[]
 for row in range(rows):
  for j in range(C):mask.append(1 if row%7==0 or (row%7==2 and j%2==0) or (row%7>2 and j<row%C) else 0)
 for kind in [0,1]:
  x=[quant(v,kind) for v in raw];scale=f32(1/math.sqrt(C));prefix.extend([float(kind),float(batch),float(heads),float(Q),float(C),scale,float(n)]);prefix.extend(x);prefix.extend(map(float,mask));case={'kind':['Float16','BFloat16'][kind],'batch':batch,'heads':heads,'query':Q,'key':C,'scale':scale,'elements':n}
  for pad in [0,1]:
   out=[];allmasked=0;mass=0.;quantmax=0.
   for row in range(rows):
    bi=row//(heads*Q);q=row%Q;mr=q if pad==1 else bi*Q+q;vals=[float(x[row*C+j])*scale if mask[mr*C+j]!=1 else -10000. for j in range(C)];mx=max(vals)
    if mx==-10000.:probs=[0.]*C;allmasked+=1
    else:
     e=[math.exp(v-mx) for v in vals];den=math.fsum(e);probs=[v/den for v in e]
    rounded=[quant(v,kind) for v in probs];quantmax=max(quantmax,max(abs(v-qv) for v,qv in zip(probs,rounded)));mass=max(mass,abs(math.fsum(rounded)-(0 if mx==-10000 else 1)));out.extend(rounded)
    if pad==0:
     ov=[float(x[row*C+j])*scale if mask[row*C+j]!=1 else -10000. for j in range(C)];om=max(ov);oe=[math.exp(v-om) for v in ov];od=math.fsum(oe);old=[0. if om==-10000 else v/od for v in oe];oldmaskmax=max(oldmaskmax,max(abs(a-b) for a,b in zip(probs,old)))
   gold.extend(out+[-8.]*17);case.setdefault('paths',[]).append({'pad':pad,'allmasked_rows':allmasked,'quantized_mass_max_abs_error':mass,'storage_quantization_max_abs_error':quantmax})
  cases.append(case)
raw=struct.pack('<'+str(len(prefix))+'f',*prefix);graw=struct.pack('<'+str(len(gold))+'d',*gold);(p/'fixture.bin').write_bytes(raw);(p/'expected_outputs.bin').write_bytes(graw);meta={'seed':20261018,'cases':cases,'prefix_count':len(prefix),'output_count':len(gold),'fixture_sha256':hashlib.sha256(raw).hexdigest(),'gold_sha256':hashlib.sha256(graw).hexdigest(),'historical_wrong_BERT_mask_CPU_witness_max_probability_error':oldmaskmax,'oracle':'IndependentFloat64scaledmaskednormalization, originalfinite-10000sentinel/allmaskedzero, nearest-evenFloat16/BFloat16outputquantization. Maskbroadcast BERT(batch,query),GPT(query).','domain':'FiniteactualFP16/BF16inputs, unmaskedscaledvalues>-10000; key1/31scalar and128/1024divisible4vectornativepath; allnativequerydivisibilityconditionsretained. Otherlengths/vectorpartialtail/NaNInf/unboundedsentineldomain unverified.'};(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n');print({k:meta[k] for k in ['prefix_count','output_count','historical_wrong_BERT_mask_CPU_witness_max_probability_error']})
