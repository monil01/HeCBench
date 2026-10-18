from pathlib import Path
import struct,json,hashlib
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;state=R/'.porting-state/portable_julia/streamCreateCopyDestroy_reference_review';state.mkdir(exist_ok=True)
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];bits=lambda x:struct.unpack('<I',struct.pack('<f',x))[0]
values=[f32(f32(1.618)+f32(i))for i in range(4096)];payload=b''.join(struct.pack('<I',bits(v))for v in values);(p/'oracle_u32.bin').write_bytes(payload);guard=struct.pack('<I',bits(-777.))*64;cases=[];prefix=struct.pack('<I',8)
for mode in [0,1]:
 for test in [0,5,10,15]:
  group=test//4;streams=[1,2,4,8][test%4];buffers=[1,100,1000,5000][group];it=256//(streams*(1<<(group+1)));cases.append(dict(test=test,mode=mode,streams=streams,buffers=buffers,iterations=it));prefix+=struct.pack('<5I',test,mode,streams,buffers,it)+payload
suffix=[]
for c in cases:
 suffix.extend([guard+payload+guard]*c['buffers']);it=c['iterations'];s=c['streams'];suffix.append(struct.pack('<4I',it*s*c['buffers'],it*s if c['mode']else 0,it*s if c['mode']else 0,it*s if c['mode']else it))
golden=state/'golden_u32.bin'
with golden.open('wb')as f:
 f.write(prefix)
 for part in suffix:f.write(part)
info={'cases':cases,'input_prefix_count':len(prefix)//4,'output_count':golden.stat().st_size//4,'golden_path':str(golden.relative_to(R)),'golden_sha256':hashlib.sha256(golden.read_bytes()).hexdigest(),'oracle_sha256':hashlib.sha256(payload).hexdigest(),'source_binary32_input':'round32(round32(1.618)+round32(i)), all i0..4095','independent_contract':'Every actual source host rawIEEE32 payload, every complete destination+guards, copy/create/destroy/sync counts exact','no_authored_device_kernels':True};(p/'independent_oracle.json').write_text(json.dumps(info,indent=2)+'\n');print(info['input_prefix_count'],info['output_count'])
