from pathlib import Path
import struct,math,json,collections
p=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
u32=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
f32bits=lambda x:struct.unpack('<f',struct.pack('<I',x))[0]
def encode(x,t):
 if t==1:return struct.unpack('<H',struct.pack('<e',x))[0]
 if t==2:
  b=u32(x);return ((b+0x7fff+((b>>16)&1))>>16)&65535
 return u32(x)
def decode(b,t):
 if t==1:return float(struct.unpack('<e',struct.pack('<H',b))[0])
 return f32bits(b<<16) if t==2 else f32bits(b)
def clamp(x):return min(127,max(-128,x))
NT=17;scale=f32(.1);AZP=54;prefix=[];expected=[];descriptors=[]
for t in (1,2,3):
 for H in (513,5137):
  master=[]
  for token in range(NT):
   row=[f32(((j*13+token*17)%1025-512)/8) for j in range(H)]
   row[0]=127.;row[-1]=-128.
   for j,x in enumerate([.05,-.05,.15,.25,.35,-.15],1):row[j]=f32(x)
   master+=row
  bits=[encode(x,t) for x in master];values=[decode(b,t) for b in bits]
  assert all(min(values[i*H:(i+1)*H])==-128 and max(values[i*H:(i+1)*H])==127 for i in range(NT))
  prefix += [t,H,NT,u32(scale),AZP]+bits
  for path in range(4):
   codes=[];scales=[-12345.]*NT;zps=[-123456]*NT;clipped=0;err2=0.;maxerr=0.
   for token in range(NT):
    row=values[token*H:(token+1)*H];absmax=max(map(abs,row));tmp=f32(127/absmax)
    if path<2:s=scale;z=AZP if path==1 else 0
    elif path==2:s=f32(max(map(abs,row))/127);z=0;scales[token]=s
    else:
     s=f32(f32(max(row)-min(row))/255);z=round(f32(-128.-f32(min(row)/s)));scales[token]=s;zps[token]=z
    for x in row:
     quant=round(f32(x*tmp)) if path==2 else round(f32(x/s))+z
     clipped+=quant< -128 or quant>127;code=clamp(quant);codes.append(code);error=abs((code-z)*s-x);err2+=error*error;maxerr=max(maxerr,error)
   out=codes+[-99]*17;so=scales+[-12345.]*17;zo=zps+[-123456]*17;expected+=out+so+zo
   descriptors.append({'type':['FP16','BF16','FP32'][t-1],'storage_bits':16 if t!=3 else 32,'hidden':H,'tokens':NT,'threads_per_block':min(H,1024),'path':['static_symmetric','static_asymmetric','dynamic_symmetric','dynamic_asymmetric'][path],'encoding_clipped_values':clipped,'reconstruction_max_abs_error':maxerr,'reconstruction_RMSE':math.sqrt(err2/(NT*H)),'Int8_code_histogram':dict(collections.Counter(codes)),'input_roundtie_prefix_values':values[1:7],'dynamic_scales':scales if path>=2 else None,'dynamic_zero_points':zps if path==3 else None})
(p/'fixture_bits.bin').write_bytes(struct.pack('<'+'I'*len(prefix),*prefix));(p/'expected_full_outputs.bin').write_bytes(struct.pack('<'+'d'*len(expected),*expected))
(p/'fixture_definition.json').write_text(json.dumps({'tokens':NT,'hidden_sizes':[513,5137],'input_prefix_uint32_words':len(prefix),'public_values':len(expected),'case_order':'FP16, BF16, FP32; eachhidden513then5137; each4staticSym/staticAsym/dynamicSym/dynamicAsym','typed_storage':'RawUInt16half/bfloat16 orUInt32float bits copiedexactlytoactualnative/JuliaGPUtypedarrays; Float64artifact widensrawbits/codes/scales/AZPlosslesslyonly.','scientific_contract':'Allrowsmixedsign/nondegenerate,min-128,max127. Originalnativeallnegativemaxinitialhazardandzero/constantdivisiondomain excludedwithseparatewitness, notclaimedunsupportedbyframework.','oracle':'IndependentFloat32operationsequence, nearest-evenintegerrounding/clamp; trueper-rowextrema, actualdtypequantizedinputs. Dynamiccodecorrectnessseparatefromreconstructionerror.','guards':{'Int8_output':[-99,17],'Float32_scale':[-12345,17],'Int32_azp':[-123456,17]},'descriptors':descriptors},indent=2)+'\n')
print(len(prefix),len(expected))
