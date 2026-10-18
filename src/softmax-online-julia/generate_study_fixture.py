"""Finite exact-offset fixtures and independent Float64 normalized exponentials."""
from pathlib import Path
import random,math,struct,json,hashlib
p=Path(__file__).resolve().parent;r=random.Random(20261018);f=[5.];gold=[];records=[];sizes=[(17,1),(19,31),(33,33),(17,257),(3,50257)]
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
for case,(n,c) in enumerate(sizes):
 rows=[]
 for row in range(n):
  a=[f32(r.randrange(-16,17)/16) for _ in range(c)]
  if row==0:a=[0.]*c
  elif row==1:
   for index in sorted({0,c//2,c-1}):a[index]=40.
  elif row==2:a=[x+1000. for x in rows[1]]
  elif row==3:a=[x-1000. for x in rows[1]]
  elif row==4:a=[-80.]*c;a[-1]=80.
  rows.append([f32(x) for x in a])
 for dst in (2,3):
  if dst<n:
   assert all(rows[dst][j]-rows[1][j]==(1000 if dst==2 else -1000) for j in range(c))
 data=[v for row in rows for v in row]+[-12345.]*17;f.extend([float(n),float(c),float(len(data))]);f.extend(data);out=[];maxsum=0.
 for row in rows:
  m=max(row);v=[math.exp(x-m) for x in row];den=math.fsum(v);vals=[x/den for x in v];maxsum=max(maxsum,abs(math.fsum(vals)-1));out.extend(vals)
 for w in (32,64,128,256,512,1024):gold.extend(out+[-12345.]*17)
 records.append({'case':case,'rows':n,'cols':c,'threads':[32,64,128,256,512,1024],'max_input':max(data[:-17]),'min_input':min(data[:-17]),'largest_probability':max(out),'Float64_max_row_sum_error':maxsum,'shift_pairs_zero_based':[[1,d] for d in (2,3) if d<n],'uniform_row0':True,'pointmass_lastcolumn_row4':n>4,'input_values':len(data)})
fb=struct.pack('<%df'%len(f),*f);gb=struct.pack('<%dd'%len(gold),*gold);(p/'fixture.bin').write_bytes(fb);(p/'expected_outputs.bin').write_bytes(gb)
(p/'fixture_description.json').write_text(json.dumps({'seed':20261018,'cases':records,'selected_native_kernel':4,'unverified_native_kernel_modes':[1,2,3],'path_order':[f'case{case}_threads{w}' for case in range(5) for w in (32,64,128,256,512,1024)],'prefix_count':len(f),'output_count':len(gold),'fixture_sha256':hashlib.sha256(fb).hexdigest(),'gold_sha256':hashlib.sha256(gb).hexdigest(),'oracle':'IndependentFloat64 max-shifted exponential normalization withmath.fsum. Fullprobabilities,nonnnegative/unitmass, exactoffsetrowpairs; allfiniteinputs.','scope':'Namedkernel4block-online,6blocksizesand5boundary/fullvocabularywidthshapes, source512? actual50257columnsretainedlastcase3rows. Other3nativekernels/default8192rows/NaN-Infinputdomain/fullmodelaccuracyunverified. Exactoffsetrowsarecontrolledpairedinvariancecasesnotindependentdata.'},indent=2)+'\n')
