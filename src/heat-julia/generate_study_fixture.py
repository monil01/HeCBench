from pathlib import Path
import math,struct,json,hashlib
P=Path(__file__).resolve().parent
cases=[(63,17),(127,18),(255,19)]
raw=bytearray()
for n,steps in cases:
 dx=1000.0/(n+1)
 for j in range(n):
  for i in range(n):
   v=math.sin(math.pi*(dx*(i+1))/1000.0)*math.sin(math.pi*(dx*(j+1))/1000.0)
   raw.extend(struct.pack('<d',v))
f=P/'study_initial.bin';f.write_bytes(raw)
metadata={'cases':cases,'alpha':0.1,'length':1000.0,'total_time':0.5,'values':85123,'dtype':'Float64littleendian','sha256':hashlib.sha256(raw).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Source analytic sine initial data evaluated once on host and shared byteidentically. Native GPUinitializer/setup excluded from solve timing; all original physical constants retained.'}
(P/'study_initial_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
