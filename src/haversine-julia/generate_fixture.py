from pathlib import Path
import json,struct,math,hashlib
B=Path(__file__).resolve().parent
def geo(a,b,c,d):
 a,b,c,d=map(math.radians,[a,b,c,d]);u=(math.cos(a)*math.cos(b),math.cos(a)*math.sin(b),math.sin(a));v=(math.cos(c)*math.cos(d),math.cos(c)*math.sin(d),math.sin(c));cross=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]);return 6371*math.atan2(math.hypot(*cross),math.fsum(x*y for x,y in zip(u,v)))
if __name__ == '__main__':
 normal=[(0.,0.,0.,0.),(0.,0.,0.,90.),(0.,0.,90.,0.),(90.,0.,90.,90.),(10.,179.999,10.,-179.999),(20.,30.,20.+1e-7,30.),(-40.,10.,-40.,10.+1e-7)]
 cities=[(19.076,72.8777),(-37.8136,144.9631),(42.3765,-71.2356),(55.7558,37.6173),(55.8642,-4.2518),(31.6295,-7.9811)]
 normal +=[(a,b,c,d) for a,b in cities for c,d in cities]
 i=0
 while len(normal)<4099:
  a=((i*197)%17801)/100-89;b=((i*317)%36001)/100-180;c=((i*503+19)%17801)/100-89;d=((i*419+7)%36001)/100-180;i+=1
  if geo(a,b,c,d)<.95*math.pi*6371:normal.append((a,b,c,d))
 stress=[]
 for lat in [-89.,-60.,-45.,-30.,-15.,0.,15.,30.,45.,60.,89.]:
  for lon in [0.,11.25,90.]:
   anti=lon-180 if lon>=0 else lon+180
   for eps in [0.,1e-8,-1e-8]:stress.append((lat,lon,-lat+eps,anti))
 for key,points in [('normal',normal),('stress',stress)]:
  raw=b''.join(struct.pack('<4d',*p) for p in points);(B/('input_'+key+'_f64.bin')).write_bytes(raw);(B/('oracle_'+key+'_f64.bin')).write_bytes(b''.join(struct.pack('<d',geo(*p)) for p in points))
 (B/'fixture_description.json').write_text(json.dumps({'normal_n':len(normal),'stress_n':len(stress),'radius_km':6371,'oracle':'IndependentFloat64unit3Dvectors/crossnorm/dotatan2centralangle,notduplicatedhaversineasinformula','normal_budget_km':1e-6,'stress_budget_km':1e-3,'budget_scope':'Predeclarednormal1mm andnear-antipodal1mabsolutecomputationalbudgets forsourcefixed-radiusspheremodel; notphysicalaccuracyforrealEarth ellipsoid.','stress_rationale':'Nearh=1, arcssqrtconditionamplifiesO(eps64)hroundofftoO(sqrt(eps64))angle, ~.0002km perulp;1mabsolute budget allowsseveralulpandtestsfiniteoutput. Noafter-failuretolerancechanges.','normal_scope':'4099city/syntheticpairs; randomangles<.95pi; self/quartercircle/poles/dateline/nearcoincident included.','stress_scope':'99exact/near-antipodal pairs, latitude±89/lon0,11.25,90; shiftedlat±1e-8deg.'},indent=2)+'\n')
