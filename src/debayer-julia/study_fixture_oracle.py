from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent
cases=[(32,10,0,0),(35,7,1,1),(65,11,2,2),(64,10,3,3),(64,15,0,4)];fixture=[];oracle=[];reviews=[]
# Sparse integer convolution filters, rather than GPU shared-memory expressions.
filters=[({(0,0):4,(-1,0):2,(1,0):2,(0,-1):2,(0,1):2,(-2,0):-1,(2,0):-1,(0,-2):-1,(0,2):-1},8),({(-1,0):8,(1,0):8,(0,0):10,(0,-2):1,(0,2):1,(-1,-1):-2,(1,-1):-2,(-1,1):-2,(1,1):-2,(-2,0):-2,(2,0):-2},16),({(0,-1):8,(0,1):8,(0,0):10,(-2,0):1,(2,0):1,(-1,-1):-2,(1,-1):-2,(-1,1):-2,(1,1):-2,(0,-2):-2,(0,2):-2},16),({(0,0):12,(-2,0):-3,(2,0):-3,(0,-2):-3,(0,2):-3,(-1,-1):4,(1,-1):4,(-1,1):4,(1,1):4},16)]

def reflect(i,n):
 i=abs(i)
 return 2*n-i-2 if i>=n else i

def truncdiv(a,d):return (abs(a)//d)*(-1 if a<0 else 1)
for w,h,pat,mode in cases:
 n=w*h;data=[]
 for r in range(h):
  for c in range(w):
   x=(7*c+13*r)%256 if mode==0 else 255*((r+c)%2)if mode==1 else ((c*73+r*151+(c*r*17))^((c+11)*(r+7)))%256 if mode==2 else 123 if mode==3 else 255*int(c==w//2 and r==h//2);data.append(x)
 fixture +=[w,h,pat,mode,5]+data;out=[0]*(4*n);rx,ry=[(0,0),(1,0),(0,1),(1,1)][pat];raw_min=0;raw_max=0;clip=0
 for br in range((h+4)//5):
  for bc in range((w+31)//32):
   assert all(0<=reflect(bc*32+ac-2,w)<w and 0<=reflect(br*5+ar-2,h)<h for ac in range(36)for ar in range(9))
 for r in range(h):
  for c in range(w):
   responses=[]
   for taps,den in filters:
    numer=sum(weight*data[reflect(r+dy,h)*w+reflect(c+dx,w)]for(dx,dy),weight in taps.items());responses.append(truncdiv(numer,den))
   center=data[r*w+c];R1,R2,R3,R4=responses;isr=(r%2==ry and c%2==rx);isb=(r%2!=ry and c%2!=rx);rr=R4 if isb else center if isr else R2 if r%2==ry else R3;bb=R4 if isr else center if isb else R3 if r%2==ry else R2;gg=R1 if isr or isb else center
   for ch,x in enumerate([rr,gg,bb]):out[r*4*w+c*3+ch]=max(0,min(255,x));raw_min=min(raw_min,x);raw_max=max(raw_max,x);clip+=int(x<0 or x>255)
 oracle +=out;reviews.append({'width':w,'height':h,'pattern':pat,'mode':mode,'all_apron_reflections_in_bounds':True,'raw_response_min':raw_min,'raw_response_max':raw_max,'saturated_channels':clip,'constant_state':mode==3,'row_padding_zero':True})
(p/'fixtures_u32.bin').write_bytes(struct.pack('<'+'I'*len(fixture),*fixture));(p/'oracle_u8.bin').write_bytes(bytes(oracle));(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent sparse integer convolution taps at reflected global imagecoordinates; C-style truncation-toward-zero after exactinteger numerator, saturatingclamp, phasecolor lookup and3-byteRGB/4wrowpitch','cases':reviews,'native_default_channels':3,'limits':['Sourceinteger MHC filter, not floating originalpaperquality proof','One-fold borderreflection must bevalid forentire paddedapron; allselectedcases checked','Width33 with2tiles has apronx65 reflectto-1 (sourcedomain limitation, GPUunmeasured), excluded fromnamedfixtures','No sensor calibration, photographicquality/PSNR,optional4channelbuild orallsmallshapeCLI claim']},indent=2)+'\n');print({'prefix':len(fixture),'output_bytes':len(oracle),'artifact_u32_count':len(fixture)+len(oracle)+20})
