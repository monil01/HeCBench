from pathlib import Path
import struct,json
B=Path(__file__).resolve().parent
cases=[];payload=[];oracle=[]
for n,mode in [(1,0),(17,1),(33,2),(257,3),(1024,4),(2048,5)]:
 start=len(payload);o=len(oracle)
 for i in range(n):
  count=0
  for j in range(n):
   nonzero=mode==1 or (mode==2 and (i*7+j*11)%19==0) or (mode==3 and i%7!=0 and (j==i or j==(i*5)%n)) or (mode==4 and i in [0,n//2,n-1]) or (mode==5 and (i*17+j*13)%101<3)
   value=((-1 if (i+j)%2 else 1)*((i*3+j*5)%31+1)/8) if nonzero else (-0.0 if (i+j)%2 else 0.0)
   payload.append(struct.unpack('<I',struct.pack('<f',value))[0]);count+=value!=0
  oracle.append(count)
 cases.append(dict(n=n,mode=mode,input_offset=start,oracle_offset=o,total=sum(oracle[o:])))
(B/'input_bits_u32.bin').write_bytes(struct.pack('<%dI'%len(payload),*payload));(B/'oracle_counts_u32.bin').write_bytes(struct.pack('<%dI'%len(oracle),*oracle))
(B/'cases.json').write_text(json.dumps(dict(cases=cases,scope='Square row-major dense is column-major transpose; source COLUMN returns logical row counts. Exact signed dyadics and signed zeros, nonuniform sparse masks, empty and hot rows. No rectangular or full original size claim.'),indent=2)+'\n')
(B/'study_cases.h').write_text('struct Case{int n,mode,io,oo,total;};\nstatic const Case cases[] = {'+','.join('{%d,%d,%d,%d,%d}'%(c['n'],c['mode'],c['input_offset'],c['oracle_offset'],c['total'])for c in cases)+'};\n')
