from pathlib import Path
import json,struct,math,hashlib
B=Path(__file__).resolve().parent;inputs=[];gold=[];cases=[]
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
for typ in ['f32','f64']:
 cast=f32 if typ=='f32' else float
 for m,n in [(128,32),(256,64),(256,256)]:
  for fixture in range(3):
   if fixture==0:A=[1.]*(2*m);C=[0.]*(2*n);sa=[1.]*m;sb=[1.]*n
   elif fixture==2:A=[0.]*(2*m);C=[0.]*(2*n);sa=[1.]*m;sb=[1.]*n
   else:
    A=[cast(((i*7+d*13)%61-30)/16) for d in range(2) for i in range(m)];C=[cast(((j*11+d*17)%67-33)/16) for d in range(2) for j in range(n)];sa=[cast(.25+(i%7)/8) for i in range(m)];sb=[cast(.5+(j%5)/8) for j in range(n)]
   blocks=[]
   for by in range(n//32):
    for bx in range(m//128):
     terms=[]
     for i in range(bx*128,(bx+1)*128):
      for j in range(by*32,(by+1)*32):terms.append(math.exp(-math.fsum((A[i+d*m]-C[j+d*n])**2 for d in range(2))/(sa[i]+sb[j])))
     blocks.append(math.fsum(terms))
   total=math.fsum(blocks);cases.append(dict(dtype=typ,m=m,n=n,fixture=fixture,input_offset=len(inputs),input_count=3*(m+n),oracle_offset=len(gold),nblocks=len(blocks),absolute_tolerance=.001 if typ=='f32' else 1e-9,relative_tolerance=1e-5 if typ=='f32' else 1e-12,exact_coincident=fixture==2));inputs.extend(A+C+sa+sb);gold.extend(blocks+[total])
(B/'inputs_f64.bin').write_bytes(struct.pack('<'+'d'*len(inputs),*inputs));(B/'oracle_f64.bin').write_bytes(struct.pack('<'+'d'*len(gold),*gold));prefix=sum(c['input_count']+128+(c['nblocks']+32)+33 for c in cases);out=sum(c['nblocks']+32+33 for c in cases)
(B/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=prefix,complete_capture_count=prefix+out,guards_per_array_side=16,internal_repeats=10,inputs='2DSoA A/B andpositiveperpointscale arrays. Sourceconstant A1/B0/scale1, signeddyadicvariedpoints/scales, allcoincidentzero closedform. TypedFloat32/64, captureFloat64losslessly widensF32.',oracle='PythonFloat64 math.fsum pairwise squaredEuclideandistance Gaussianaffinities withdirectper-source-tile sums andfinalsum. No shared Julia/nativekernel/reduction implementation; coincidentblocks4096,totalm*n exactly.',timing='Sum18kernelpair synchronizedhostwindows/R10, distance thenfinalreduction eachrepeat; input/setup/JIT/readback/oracle/captureoutside. 5checkedtrials mean2-5.',scope='Finite unnormalizedsum exp(-squared2Ddistance/(scaleA+scaleB)), sourcecalledBhattacharyacost; notnormalizedprobability/Bhattacharyyadistanceorregistrationquality. Fulltilem128multiples/n32multiples preventoriginalunguardedsharedloadOOB; original16384squareCLIunmeasured.'),indent=2)+'\n')
