"""Generate source-range Float32 inputs and independent Float64 log2 values."""
from pathlib import Path
import struct,math
B=Path(__file__).parent
vals=[float(i) for i in range(1,8194)]+[1.25,1.5,1.75,9999999.,10000000.]
for k in range(24):
 u=struct.unpack('<I',struct.pack('<f',float(1<<k)))[0]
 for du in [-1,0,1]:
  x=struct.unpack('<f',struct.pack('<I',u+du))[0]
  if 1<=x<=1e7:vals.append(x)
(B/'study_input_f32.bin').write_bytes(b''.join(struct.pack('<f',x) for x in vals))
(B/'oracle_f64.bin').write_bytes(b''.join(struct.pack('<d',math.log2(x)) for x in vals))
