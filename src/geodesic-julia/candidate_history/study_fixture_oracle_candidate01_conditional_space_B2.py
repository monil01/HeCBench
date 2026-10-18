from pathlib import Path
import math,struct,json
p=Path(__file__).resolve().parent
A=6378137.;B=6356752.31424518;F=1-B/A
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def inverse(lat1,lon1,lat2,lon2):
 if lat1==lat2 and lon1==lon2:return 0.,0
 u1=math.atan((1-F)*math.tan(math.radians(lat1)));u2=math.atan((1-F)*math.tan(math.radians(lat2)));s1,c1=math.sin(u1),math.cos(u1);s2,c2=math.sin(u2),math.cos(u2);L=math.radians((lon2-lon1+180)%360-180);lam=L
 for it in range(100):
  sl,cl=math.sin(lam),math.cos(lam);ss=math.hypot(c2*sl,c1*s2-s1*c2*cl);cs=s1*s2+c1*c2*cl;sig=math.atan2(ss,cs);sa=c1*c2*sl/ss;ca=1-sa*sa;c2m=0. if ca<1e-30 else cs-2*s1*s2/ca;C=F/16*ca*(4+F*(4-3*ca));old=lam;lam=L+(1-C)*F*sa*(sig+C*ss*(c2m+C*cs*(-1+2*c2m*c2m)))
  if abs(lam-old)<1e-14:break
 assert it<99
 usq=ca*(A*A-B*B)/(B*B);aa=1+usq/16384*(4096+usq*(-768+usq*(320-175*usq)));bb=usq/1024*(256+usq*(-128+usq*(74-47*usq)));delta=bb*ss*(c2m+bb/4*(cs*(-1+2*c2m*c2m)-bb/6*c2m*(-3+4*ss*ss)*(-3+4*c2m*c2m)))
 return B*aa*(sig-delta),it+1

def meridian(a,b):
 # Independent composite Simpson integration of ellipsoidal meridional metric, not inverse iteration.
 lo,hi=sorted([math.radians(a),math.radians(b)]);e2=1-B*B/(A*A);N=4096;h=(hi-lo)/N
 def metric(phi):return A*(1-e2)/(1-e2*math.sin(phi)**2)**1.5
 return h/3*math.fsum((1 if i in[0,N]else4 if i%2 else2)*metric(lo+i*h)for i in range(N+1))
controls=[(0,12,0,12),(45,12,45,12),(-33.5,12,-33.5,12),(0,0,0,.05),(0,0,0,45),(20,12,30,12),(-50,40,40,40),(40.7128,-74.006,51.5074,-.1278),(-33.8688,151.2093,1.3521,103.8198),(10,179,10,-179)]
f=[];o=[];cases=[]
for n in[33,257]:
 points=controls.copy()
 for j in range(n-len(points)):
  lat1=(j*17%121)-60;lon1=(j*29%241)-120;lat2=(j*13%111)-55;lon2=lon1+(j*7%121)-60;points.append((lat1,lon1,lat2,lon2))
 points=[tuple(map(f32,r))for r in points];values=[];iterations=[];checks=[]
 for j,r in enumerate(points):
  v,it=inverse(*r);values.append(v);iterations.append(it)
  if r[0]==r[2]and r[1]==r[3]:checks.append({'index':j,'kind':'coincident','exact_oracle':0.})
  elif r[0]==r[2]==0:
   exact=A*abs(math.radians((r[3]-r[1]+180)%360-180));assert abs(v-exact)<1e-6;checks.append({'index':j,'kind':'equatorial_exact','oracle':exact,'Vincenty_difference':v-exact})
  elif r[1]==r[3]:
   exact=meridian(r[0],r[2]);assert abs(v-exact)<.001;checks.append({'index':j,'kind':'meridional_metric_quadrature','oracle':exact,'Vincenty_difference':v-exact})
 f +=[n,10]+[v for r in points for v in r];o+=values;cases.append({'n':n,'internal_repeats':10,'oracle_max_iterations':max(iterations),'all_independent_inverse_iterations':iterations,'controls':checks})
(p/'fixtures_f32.bin').write_bytes(struct.pack('<%sf'%len(f),*f));(p/'oracle_f64.bin').write_bytes(struct.pack('<%sd'%len(o),*o));(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent Float64 Vincenty inverse with standard u² A/B series and1e-14rad termination, vsnative compactHelmert-form series/Float32/EPS5e-6; independent meridional metric Simpson4096 quadrature andequatorial A*deltaLongitude +coincidence identities','primary_reference':'https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf','primary_reference_sections':'Vincenty1975 inverse equations13–19 and A/B3–6; notesinversecanfailnearantipodes','semimajor':A,'semiminor':B,'cases':cases,'input_prefix_count':len(f),'oracle_values':len(o),'acceptance':{'atol_meters':50.,'rtol':0.,'coincident_requires_exact_zero':True,'before_first_GPU':True,'justification':'NativeEPS5e-6*semimajor=31.890685m; predeclared50mengineeringbudgetallows sourcecoarsestoppingplusFloat32geometry/trig/seriesrounding on regularnonantipodalfixtures. Notrigorousglobalbound orsurveyaccuracyclaim; actualerrors/short-equatorbiasreported.','near_antipodal_scope':'Unmeasured; nativeunboundediteration andJulia100cap notcertified.'}},indent=2)+'\n')
