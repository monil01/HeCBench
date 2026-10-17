#!/usr/bin/env python3
"""Source CSV Float32 fixture; independent count-weighted Float64 recurrence.
All10000 GPU queries/neighbors and1000updates remain unchanged. Only the CPU
oracle groups the21 identical source rows with exact multiplicities.
"""
from pathlib import Path
import csv,struct,collections,math,json,hashlib
P=Path(__file__).resolve().parent
rows=[tuple(struct.unpack('<3f',struct.pack('<3f',*map(float,r))))for r in csv.reader(open(P.parent/'meanshift-cuda/dataset/data.csv'))]
assert len(rows)==10000;counter=collections.Counter(rows);keys=list(counter);positions=[list(x)for x in keys];weights=[counter[k]for k in keys];original=positions;source_index={v:i for i,v in enumerate(keys)}
for iteration in range(1000):
 new=[]
 for a in positions:
  total=0.;num=[0.,0.,0.]
  for j,v in enumerate(positions):
   distance=sum((a[k]-v[k])**2 for k in range(3))
   if distance<=60:
    weight=math.exp(-distance/32)*weights[j];total+=weight
    for k in range(3):num[k]+=weight*v[k]
  assert total>=1;new.append([v/total for v in num])
 positions=new
expected=[v for row in rows for v in positions[source_index[row]]];input_values=[v for row in rows for v in row]+[0.]*144
raw=struct.pack('<30144f',*input_values);(P/'study_input_f32.bin').write_bytes(raw);(P/'study_expected_f64.bin').write_bytes(struct.pack('<30000d',*expected))
centroids=[positions[source_index[rows[i]]]for i in range(3)];assert all(sum((centroids[i][j]-100*i)**2 for j in range(3))<=10 for i in range(3))
meta={'source_CSV_sha256':hashlib.sha256((P.parent/'meanshift-cuda/dataset/data.csv').read_bytes()).hexdigest(),'source_rows':10000,'unique_rows':len(keys),'unique_multiplicities':weights,'iterations':1000,'logical_N':10000,'allocated_rows':10048,'defined_padding_values':144,'input_prefix_count':30144,'cpu_recurrence':'Float64 weighted unique rows; same all-point update mathematically, no GPU grouping shortcut','expected_centroids':centroids,'files':{q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in P.glob('study_*.bin')}};(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
