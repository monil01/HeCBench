"""Reproduce the predeclared exact Int32 fixtures; verify existing bytes before reuse."""
from pathlib import Path
import hashlib,struct
rows=[[i%17 for i in range(1024)], [((i*73+19)%201)-100 for i in range(1024)], [17 if i==0 else -31 if i==511 else 47 if i==1023 else 0 for i in range(1024)]]
data=struct.pack('<3072i',*(x for row in rows for x in row));path=Path(__file__).with_name('fixture.bin')
if path.exists():
 if path.read_bytes()!=data:raise RuntimeError('Existing fixture differs; preserve and investigate it before regenerating.')
else:path.write_bytes(data)
print(hashlib.sha256(data).hexdigest())
