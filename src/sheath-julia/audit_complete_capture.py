#!/usr/bin/env python3
"""CPU replay: rehash/parse all physical stages, independent physics, and exact primary joins."""
from pathlib import Path
import numpy as np,json,sys,hashlib,re,struct,ctypes,math
r=Path(__file__).resolve().parent;result=Path(sys.argv[1]).resolve();model=sys.argv[2];lib=ctypes.CDLL(sys.argv[3]);lib.witness_deposition.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_double];lib.witness_field.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int];lib.witness_motion.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_double,ctypes.c_int]
x=json.loads(result.read_text());m=x['models'][model];assert m['complete']and m['timing_valid'];log=(result.parent/(model+'_run1.log')).read_text();paths={int(a):Path(b).resolve()for a,b in re.findall(r'SHEATH_RAW trial=(\d+) path=(\S+)',log)};P=np.dtype({'names':['x','v','alive'],'formats':['<f8','<f8','u1'],'offsets':[0,8,16],'itemsize':24});K=1.38065e-23;EV=11604.52;Q=1.602176565e-19;ME=9.10938215e-31;AMU=1.660538921e-27;phiscale=K*(3*EV)/Q;escale=phiscale/1e-4;rhoscale=Q*1e16;phiunits=np.fromfile(r/'data/case0_phi_normalization_v2.bin',dtype='<f8');v2='_neutralphi_inputbound_acceptance_v2' in x['configuration_id'];ptr=lambda a:ctypes.c_void_p(a.ctypes.data);reports=[]
for row in m['trials']:
 primary=Path(row['output_artifact']['saved_path']);data=primary.read_bytes();assert hashlib.sha256(data).hexdigest()==row['output_artifact']['sha256'];actual=np.frombuffer(data,dtype='<f8');rawpath=paths[row['trial']];rawhash=hashlib.sha256();prefix=[];outputs=[];field_counts=[0]*3;snaps=[0]*3;prev=[None]*3;lastguards=[None]*3;lastparts=[None]*3;currentfield=[None]*3;raw_guard=0;operator_checks=0
 with rawpath.open('rb')as f:
  def read(n):
   b=f.read(n);assert len(b)==n;rawhash.update(b);return b
  while True:
   b=f.read(16)
   if not b:break
   assert len(b)==16;rawhash.update(b);ci,stage,step,nbuf=struct.unpack('<iiii',b);n,steps=[(500000,1),(4099,25),(257,1000)][ci]
   if nbuf==-1:
    assert stage==1;ni=np.frombuffer(read(404),dtype='<f4');ne=np.frombuffer(read(404),dtype='<f4');rho=np.frombuffer(read(808),dtype='<f8');phi=np.frombuffer(read(808),dtype='<f8');ef=np.frombuffer(read(808),dtype='<f8');assert lib.witness_field(ptr(ni),ptr(ne),ptr(rho),ptr(phi),ptr(ef),1)==0;currentfield[ci]=(ni,ne,rho,phi,ef);arr=np.column_stack((ni.astype(np.float64)/1e16,ne.astype(np.float64)/1e16,rho/rhoscale,phi/(phiunits if ci==0 and v2 else phiscale),ef/escale,np.ones(101)));outputs.append(('field',ci,arr.ravel()));field_counts[ci]+=1;operator_checks+=303;continue
   assert nbuf==5 and stage in(0,2,3);buf=[];guards=[]
   for size in [n*24,n*24,404,404,808]:
    (count,)=struct.unpack('<Q',read(8));assert count==size+512;b=read(count);assert b[:256]==b[-256:]==bytes([0xa5])*256;raw_guard+=512;guards.append(np.frombuffer(b[:256]+b[-256:],dtype='u1').astype(np.float64));buf.append(b[256:-256])
   parts=[np.frombuffer(buf[0],dtype=P),np.frombuffer(buf[1],dtype=P)];assert all(np.all((a['alive']==0)|(a['alive']==1))for a in parts)
   if stage==0:
    assert prev[ci]is None;fix=np.fromfile(r/'data'/f'case{ci}.bin',dtype='<f8',offset=12).reshape(2,n,2);prefix.append(np.array([ci,n,steps],dtype='<f8'))
    for s,a in enumerate(parts):assert np.array_equal(a['x'],fix[s,:,0])and np.array_equal(a['v'],fix[s,:,1])and np.all(a['alive']==1);prefix.append(np.column_stack((a['x'],a['v'],a['alive'])).ravel())
   else:
    assert prev[ci]is not None;ni,ne,rho,phi,ef=currentfield[ci];assert np.frombuffer(buf[4],dtype='<f8').tobytes()==ef.tobytes();rawden=[np.frombuffer(buf[2],dtype='<f4'),np.frombuffer(buf[3],dtype='<f4')];weight=1e16*(100*1e-4)/n
    for s,a in enumerate(parts):
     before=prev[ci][s];assert lib.witness_deposition(ptr(before),n,ptr(rawden[s]),ptr(ne if s else ni),weight)==0;qm=(-Q/ME)if s else Q/(16*AMU);assert lib.witness_motion(ptr(before),ptr(a),n,ptr(ef),qm,int(stage==2))==0;operator_checks+=3*n+202
   prev[ci]=parts;lastparts[ci]=parts;lastguards[ci]=guards;snaps[ci]+=1
 # Nativeprimary groups allfieldrows foronecase, thenfinalparticlesandguards, thennextcase.
 rebuilt_prefix=np.concatenate(prefix);assert rebuilt_prefix.size==3026145;out=[]
 for ci,(n,steps)in enumerate([(500000,1),(4099,25),(257,1000)]):
  assert field_counts[ci]==steps+1 and snaps[ci]==steps+2
  out.extend(a for kind,c,a in outputs if c==ci)
  for s,a in enumerate(lastparts[ci]):
   mass=ME if s else 16*AMU;temp=3. if s else 1.;vth=math.sqrt(2*K*temp*EV/mass);out.append(np.column_stack((a['x']/(100*1e-4),a['v']/vth,a['alive'])).ravel())
  out.extend(lastguards[ci])
 rebuilt=np.concatenate([rebuilt_prefix,*out]);assert rebuilt.size==actual.size==6683535
 if not np.array_equal(rebuilt.view('<u8'),actual.view('<u8')):
  different=np.flatnonzero(rebuilt.view('<u8')!=actual.view('<u8'));print('join mismatch',different[:8],rebuilt[different[:8]],actual[different[:8]]);raise AssertionError('fullprimary bit join')
 reports.append({'trial':row['trial'],'primary_sha256':hashlib.sha256(data).hexdigest(),'physical_path':str(rawpath),'physical_sha256':rawhash.hexdigest(),'physical_snapshots_per_case':snaps,'field_frames_per_case':field_counts,'actual_guard_bytes_checked_over_snapshots':raw_guard,'unique_allocation_guard_bytes_per_trial':7680,'independent_local_component_checks':operator_checks,'full_primary_actual_raw_join':True})
lib.witness_report();out={'benchmark':'sheath','model':model,'result':str(result),'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'all5_full_physical_scientific_replay_and_primary_bitjoins':True,'physical_allocations_per_trial':15,'raw_output_identity_claim':False,'scope':'CompleteactualAoSparticletrajectories+unscaledrawCPUfieldtraces+GPUdensities/fieldinputs+actualguards; primarynumericunitsqualifiedbycontract, deadAoSpaddingnotsemantic. No fullhot500k×1000claim.','trials':reports};(r/(model+'_complete_capture_audit.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'model':model,'all5_full_raw_and_primary_pass':True,'trials':len(reports),'guard_bytes_rechecked':sum(q['actual_guard_bytes_checked_over_snapshots']for q in reports)}))
