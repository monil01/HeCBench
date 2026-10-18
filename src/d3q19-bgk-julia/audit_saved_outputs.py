"""CPU-only full cavity fields, exact metadata/guards, actual input bytes and invariants."""
from pathlib import Path
import struct,json,hashlib,sys,math
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();gold=struct.unpack('<221197d',(p/'oracle.bin').read_bytes());meta=json.loads((p/'fixture_description.json').read_text());nl=4913;pop=93347
fields=[];off=0
for name,count in [('f_current',pop),('f_previous',pop),('u_star',3*nl),('r_initial_immutable',nl),('boundary_flags',nl),('boundary_dirs',nl),('boundary_direction_bits',nl)]:fields.append((name,off,count));off+=count+16
assert off==len(gold);records=[]
for arg in sys.argv[1:]:
 run=Path(arg);result=json.loads((run/'result.json').read_text())
 for model,md in result['models'].items():
  for t in md.get('trials',[]):
   file=Path(t['output_artifact']['saved_path']);a=file.read_bytes();assert a[:len(raw)]==raw;got=struct.unpack('<221197d',a[len(raw):]);assert all(math.isfinite(v) for v in got);metrics=[]
   for name,off,count in fields:
    assert got[off:off+8]==gold[off:off+8] and got[off+count+8:off+count+16]==gold[off+count+8:off+count+16]
    vals=got[off+8:off+count+8];refs=gold[off+8:off+count+8]
    if name in ['r_initial_immutable','boundary_flags','boundary_dirs','boundary_direction_bits']:assert vals==refs
    absmax=max(abs(x-y) for x,y in zip(vals,refs));normalized=max(abs(x-y)/(1e-7+1e-6*abs(y)) for x,y in zip(vals,refs));assert normalized<=1
    nonzero=[abs(x-y)/abs(y) for x,y in zip(vals,refs) if abs(y)>=1e-8];nearzero=[abs(x-y) for x,y in zip(vals,refs) if abs(y)<1e-8]
    metrics.append({'field':name,'count':count,'max_abs_error':absmax,'max_acceptance_fraction':normalized,'max_relative_error_where_oracle_abs_ge_1e-8':max(nonzero,default=0),'max_abs_error_where_oracle_abs_lt_1e-8':max(nearzero,default=0),'guard_values_exact':16,'metadata_or_immutable_payload_exact':name in ['r_initial_immutable','boundary_flags','boundary_dirs','boundary_direction_bits']})
   current=got[8:pop+8];assert min(current)>0
   flags=gold[fields[4][1]+8:fields[4][1]+8+nl];mass=math.fsum(current[q*nl+i] for i,f in enumerate(flags) if f<4 for q in range(19));active=sum(f<4 for f in flags);drift=abs(mass-active)/active;assert drift<1e-6
   for name,sha in md.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,(model,name)
   records.append({'model':model,'trial':t['trial'],'path':str(file),'sha256':hashlib.sha256(a).hexdigest(),'field_metrics':metrics,'active_mass':mass,'mass_relative_drift':drift,'population_min':min(current),'all_declared_dependency_hashes_current':True})
print(json.dumps({'configuration':'moving_wall_cavity17_20steps','conditional_native_initialized_metadata':True,'atol':1e-7,'rtol':1e-6,'prefix_bytes':len(raw),'all_guards_exact':True,'guard_count_per_trial':112,'metadata_and_r_exact':True,'source_native_undefined_setup_not_executed':True,'oracle_description':'Independent generic vector-formD3Q19 CPU withFloat32velocities, Float64standard equilibrium; optimizedsourceF32equilibriumsubexpressions canrounddifferently. Fullbuffers/fields checked underpredeclared tolerance; notcontinuum/gridconvergencevalidation.','relative_error_reporting':'Rawtrial e/(abs(ref)+1e-300) may be enormous around exactzero; preserved unchanged. Fieldreport separatesabsoluteerrornearzero and clearlylabelledrelativeerrorabove1e-8. Normalizedhybridtolerancefraction includesALLvalues.','firststep_analytic_movingwall_witness':{'cell':'mid-x/mid-y,z1','initial_rho':1,'expected_streamed_x_momentum':meta['lid_Float32_widened']/3,'wrong_lid_sign_changes_momentum_by':2*meta['lid_Float32_widened']/3,'explanation':'Pairedincomingdiagonalcorrections±lid/6 yieldnetxmomentum lid/3; signflip not hiddenbyzero/equilibriumfixture'},'rows':records},indent=2))
