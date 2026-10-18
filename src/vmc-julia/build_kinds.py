from pathlib import Path
import json
p=Path(__file__).resolve().parent;N=57344;k=bytearray()
def add(n,kind=0):k.extend(bytes([2])*64);k.extend(bytes([kind])*n);k.extend(bytes([2])*64)
def capture(stats):
 for i in range(7):add(N,1 if i==6 else 0)
 if stats:
  k.extend(bytes([2])*64)
  for j in range(4):k.extend(bytes([2 if j==3 else 0])*N)
  k.extend(bytes([2])*64)
 add(N,2)
for ci in range(2):
 capture(False);capture(True)
 for sample in range(2):
  capture(True);k.extend(bytes([2])*64);k.extend(bytes([0,0,0,2]));k.extend(bytes([2])*64);add(224,2)
d=json.loads((p/'fixture_description.json').read_text());assert len(k)==d['oracle_count'];(p/'kinds.bin').write_bytes(k);print(len(k))
