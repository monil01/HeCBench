from pathlib import Path
import struct,json,hashlib
p=Path(__file__).resolve().parent
f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
A=[-.166667,.5,-.5,.166667,.5,-1,0,.666667,-.5,.5,.5,.166667,.166667,0,0,0]
dA=[0,-.5,1,-.5,0,1.5,-2,0,0,-1.5,1,.5,0,.5,0,0]
d2A=[0,0,-1,1,0,0,3,-2,0,0,-3,1,0,0,1,0]
cases=[(0,0,0,0.,0.,0.),(1,0,1,.25,.5,.75),(0,1,1,.875,.125,.625),(1,1,0,.5,.25,.125)]
basis=[]
for ix,iy,iz,tx,ty,tz in cases:
 for table in [A,dA,d2A]:
  for t in [tx,ty,tz]:
   for row in range(4):
    a,b,c,d=[f(v) for v in table[row*4:row*4+4]];basis.append(f(f(f(f(f(a*t)+b)*t)+c)*t+d))
 basis.extend([float(ix),float(iy),float(iz),5.,7.,9.])
blob=struct.pack('<'+'f'*len(basis),*basis);(p/'study_basis.bin').write_bytes(blob)
(p/'study_basis_metadata.json').write_text(json.dumps({'cases':cases,'layout':'perwalker9basisvectors(a,b,c,da,db,dc,d2a,d2b,d2c),3cellindices,3inversegrids','float_count':len(basis),'sha256':hashlib.sha256(blob).hexdigest(),'coefficient_note':'SourceAf six-decimal1/6 approximation retained, derivative tables sourceideal; ideal-basis comparison separatelyquantifiesapproximation.'},indent=2)+'\n')
