from pathlib import Path
import struct,hashlib,json
p=Path(__file__).resolve().parent
x=[(128*(i%2)+((i*37+d*19+(i//2)*d*3)%31)-15)/128 for i in range(512) for d in range(8)]
b=struct.pack('<%df'%len(x),*x);(p/'input_f32.bin').write_bytes(b)
print(json.dumps(dict(count=len(x),sha256=hashlib.sha256(b).hexdigest(),dtype='little Float32',shape=[512,8])))
