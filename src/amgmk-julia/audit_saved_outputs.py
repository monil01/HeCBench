"""Independent complete prefix/host-output/raw-state/property audit; bounded NumPy memory."""
from pathlib import Path
import json,hashlib,platform
import numpy as np
RROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;F=RROOT/'.porting-state/portable_julia/runs/20261019T172742019982Z_amgmk/result.json';run=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());p=plan['parameters'];N,E,R=p['n'],p['e'],p['repeats'];shape=(p['nz'],p['ny'],p['nx']);bindings={};records=[];prefix_cache=set();raw_hashes=set()
def digest(f):
 h=hashlib.sha256()
 with f.open('rb')as s:
  for b in iter(lambda:s.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def asset(name,dtype,count):
 f=P/name;assert f.stat().st_size==count*np.dtype(dtype).itemsize;assert digest(f)==plan['files'][name]['sha256'];return np.fromfile(f,dtype=dtype)
I=asset('csr_i_i32.bin','<i4',N+1);J=asset('csr_j_i32.bin','<i4',E);V=asset('csr_values_f64.bin','<f8',E);rhs=asset('rhs_f64.bin','<f8',N);w=asset('weights_f64.bin','<f8',N);u0=asset('relax_u0_f64.bin','<f8',N);mx=asset('matvec_x_f64.bin','<f8',N);ax=asset('axpy_x_f64.bin','<f8',N);ay0=asset('axpy_y0_f64.bin','<f8',N);em=asset('matvec_expected_f64.bin','<f8',N);ea=asset('axpy_expected_f64.bin','<f8',N)
pad=lambda a,g=91.25:np.pad(a,(16,16),constant_values=g)
def neighbor_sum(a):
 z=np.zeros_like(a);z[:,:,1:]+=a[:,:,:-1];z[:,:,:-1]+=a[:,:,1:];z[:,1:,:]+=a[:,:-1,:];z[:,:-1,:]+=a[:,1:,:];z[1:,:,:]+=a[:-1,:,:];z[:-1,:,:]+=a[1:,:,:];return z
deg=neighbor_sum(np.ones(shape));assert np.array_equal(rhs,(6-deg).ravel());assert np.array_equal(em,(6*mx.reshape(shape)-neighbor_sum(mx.reshape(shape))).ravel());assert np.array_equal(ea,ay0+.5*R*ax)
qactual=float(np.max(neighbor_sum(w.reshape(shape)).ravel()/(6*w)));assert qactual<=p['q_bound'];assert abs(min(w)-p['wmin'])<1e-18;assert np.max(np.abs(u0-1)/w)<=p['M0_upper'];delta=2.0**-50+2.0**-52;rounding=float(delta/min(w)/(1-p['q_bound'])+1e-10);assert abs(rounding-p['weighted_roundoff_budget'])<1e-15;bound=p['q_bound']**R*p['M0_upper']+rounding;assert abs(bound-p['weighted_error_bound'])<1e-15;assert bound<.5*p['M0_upper']
for model,m in run['models'].items():
 assert m['status']=='native_check_pass'and m['provenance_valid']and len(m['trials'])==5
 for name,sha in m['adapter_file_hashes'].items():
  f=Path(name);assert digest(f)==sha;bindings[str(f.resolve().relative_to(RROOT))]=sha
 for trial in m['trials']:
  a=trial['output_artifact'];f=Path(a['saved_path']);assert digest(f)==a['sha256'];x=np.memmap(f,mode='r',dtype='<f8');assert len(x)==p['artifact_count'];assert np.isfinite(x).all();prefix_sha=hashlib.sha256(memoryview(x[:p['input_prefix_count']])).hexdigest();reused=prefix_sha in prefix_cache
  if not reused:
   offset=0
   def consume(expected):
    global offset
    n=len(expected);got=x[offset:offset+n];assert np.array_equal(got,expected),(model,offset);offset+=n
   consume(np.array([p['nx'],p['ny'],p['nz'],N,E,R,.5,p['q_bound'],p['wmin'],p['M0_upper'],p['weighted_roundoff_budget'],bound,p['evaluation_margin'],N,p['nx'],p['ny'],p['nz'],R]))
   for z in [I,J,V,pad(I,-123456789),pad(J,-123456789),pad(V),pad(rhs),pad(u0),pad(mx),pad(ax),pad(np.zeros(N)),pad(ay0),w,I,J,V,pad(I,-123456789),pad(J,-123456789),pad(V),pad(rhs),pad(mx),pad(ax)]:consume(z)
   assert offset==p['input_prefix_count'];prefix_cache.add(prefix_sha)
  offset=p['input_prefix_count'];assert np.array_equal(x[offset:offset+N+32],pad(em));offset+=N+32;assert np.array_equal(x[offset:offset+N+32],pad(ea));offset+=N+32;assert np.all(x[offset:offset+32]==91.25);offset+=32;assert np.array_equal(x[offset:],np.ones(8))
  rawpath=Path(trial['raw_artifact']['path']);assert rawpath.stat().st_size==(N+32)*8;rawsha=digest(rawpath);raw=np.fromfile(rawpath,dtype='<f8');assert np.all(raw[:16]==91.25)and np.all(raw[-16:]==91.25);u=raw[16:-16];assert np.isfinite(u).all()and np.all((u>=0)&(u<=2));err=np.abs(u-1);weighted=float(np.max(err/w));unweighted=float(np.max(err));res=(6*u.reshape(shape)-neighbor_sum(u.reshape(shape))-(6-deg)).ravel();wr=float(np.max(np.abs(res)/w));assert weighted<=bound and unweighted<=p['E0_max']+1e-10 and wr<=p['weighted_residual_bound'];assert abs(weighted-trial['relax_weighted_error'])<1e-10;assert abs(wr-trial['relax_residual_weighted_max'])<1e-8;raw_hashes.add(rawsha)
  eqpath=Path(trial['equilibrium_artifact']['path']);eqsha=digest(eqpath);eq=np.fromfile(eqpath,dtype='<f8');assert len(eq)==N+32 and np.array_equal(eq,pad(np.ones(N)))
  records.append(dict(model=model,trial=trial['trial'],score_artifact=dict(path=str(f.relative_to(RROOT)),sha256=a['sha256'],complete_values_checked=len(x)),actual_prefix_sha256=prefix_sha,input_prefix_calculation_reused_after_full_rehash=reused,raw_GPU_state=dict(path=str(rawpath.relative_to(RROOT)),sha256=rawsha,full_values_checked=N,guard_values_checked=32),equilibrium=dict(path=str(eqpath.relative_to(RROOT)),sha256=eqsha,full_values_checked=N),independent_properties=dict(weighted_error=weighted,weighted_bound=bound,unweighted_error=unweighted,weighted_residual=wr,weighted_residual_bound=p['weighted_residual_bound']),CPU_matvec_Axpy_full_exact=True));del x
assert len(records)==20
(P/'saved_output_audit.json').write_text(json.dumps(dict(result=dict(path=str(F.relative_to(RROOT)),sha256=digest(F)),all20_pass=True,records=records,actual_source_bindings=bindings,unique_raw_GPU_state_sha256_count=len(raw_hashes),numerical_agreement_kind='property_equivalence_not_identical_raw_values',raw_output_identity_claim=False,independent_operator='NumPy coordinate7pointstencil, noCSR forresidual/mathoracle; fullprefixasset/sourceCSR/hostoutputs exact',roundoff_bound_recomputed=dict(FP64_local_delta=delta,q_actual=qactual,weighted_roundoff_budget=rounding,weighted_error_bound=bound,no_op_rejected=bool(bound<.5*p['M0_upper'])),scope='Allrawstatesretainedandvalidated; scoreartifactidentity doesnotimplyrawstateidentity; originalglobalnonatomicraceformalvalidityunproven.',environment=dict(python=platform.python_version(),numpy=np.__version__),raw_GPU_captures_not_to_commit=True),indent=2)+'\n');print('AMGMK_FULL20_RAW_PROPERTY_PASS',len(records),len(bindings),len(raw_hashes))
