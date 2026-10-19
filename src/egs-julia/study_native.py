"""Five checked native EGS trials through unchanged kernels and guarded source preparation."""
from pathlib import Path
import argparse,ctypes as C,json,time,hashlib,struct,subprocess,sys
from assemble_artifact import assemble
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();HERE=Path(__file__).resolve().parent;lib=C.CDLL(str(a.library.resolve()));lib.egs_create.argtypes=[C.c_char_p];lib.egs_create.restype=C.c_int;lib.egs_name.argtypes=[C.c_int];lib.egs_name.restype=C.c_char_p;lib.egs_encoding.argtypes=[C.c_int];lib.egs_encoding.restype=C.c_char_p;lib.egs_bytes.argtypes=[C.c_int];lib.egs_bytes.restype=C.c_size_t;lib.egs_copy.argtypes=[C.c_int,C.c_void_p];lib.egs_scalar.argtypes=[C.c_int];lib.egs_scalar.restype=C.c_double;lib.egs_native_sim.argtypes=[C.c_int]
n=lib.egs_create(str(a.data.resolve()).encode());desc=[dict(index=i,name=lib.egs_name(i).decode(),encoding=lib.egs_encoding(i).decode(),bytes=lib.egs_bytes(i),mutable=bool(lib.egs_mutable(i)))for i in range(n)];scalars=[lib.egs_scalar(i)for i in range(22)]
subprocess.run(['g++','-O3','-std=c++17',str(HERE/'check_mt_trajectory.cpp'),'-o','mtcheck'],check=True)
# Full source production/drain warmup outside allfive reportedtrials.
def counts():
 i=next(d['index']for d in desc if d['name']=='step_counts');raw=(C.c_ubyte*72)();lib.egs_copy(i,raw);return struct.unpack('<9Q',bytes(raw))
for phase in range(8):
 lib.egs_native_sim(phase);lib.egs_native_sum();lib.egs_sync()
 if not sum(counts()[:8]):break
else:raise AssertionError('Warmup failedtodrain')
for trial in range(1,6):
 lib.egs_reset();D=Path(f'capture_trial{trial}');D.mkdir();manifest={'role':'native_checked_trial','scalars':scalars,'allocations':desc,'captures':[]};times=[]
 def capture(label,full):
  records={}
  for d in desc:
   if not full and not d['mutable']:continue
   raw=(C.c_ubyte*d['bytes'])();lib.egs_copy(d['index'],raw);b=bytes(raw);fn=f"{label}_{d['name']}.bin";(D/fn).write_bytes(b);records[d['name']]={'path':fn,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
  assert lib.egs_guards()==1 and lib.egs_readonly()==1;manifest['captures'].append({'label':label,'records':records,'guards_pass':True,'all_readonly_unchanged':True})
 capture('initial',True)
 for phase in range(8):
  lib.egs_sync();t=time.perf_counter();lib.egs_native_sim(phase);lib.egs_sync();t1=time.perf_counter();lib.egs_native_sum();lib.egs_sync();t2=time.perf_counter();times.extend([t1-t,t2-t1]);capture(f'phase{phase}',False)
  if not sum(counts()[:8]):break
 else:raise AssertionError('Sourceactivityfailedtodrain')
 (D/'capture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 checkpath=D/'independent_check.json';subprocess.run([sys.executable,str(HERE/'check_full_physics.py'),'--capture-dir',str(D),'--output',str(checkpath)],stdout=subprocess.DEVNULL,check=True);review=json.loads(checkpath.read_text())
 with(D/'independent_mt_check.json').open('w')as f:subprocess.run(['./mtcheck',str(D)]+[f'phase{i}'for i in range(phase+1)],stdout=f,check=True)
 artifact=f'output_trial{trial}.bin';encoding=assemble(D,artifact)
 row={'trial':trial,'time_seconds':sum(times),'correct':review['all_pass'],'checked_values':review['complete_pixel_values_independently_CPU_accumulated'],'max_abs_error':review['all_pixels_accumulation_max_absolute_error'],'max_rel_error':None,'timing_region':'sum_source_simulation_and_detector_accumulation_contextcompleted_segments_excludes_captures','jit_excluded':True,'internal_repeats':1,'output_artifact':{'path':artifact,'dtype':'float64','count':encoding['count'],'byte_order':'little'},'input_prefix_count':encoding['input_prefix_count'],'segment_seconds':times,'actual_histories':review['history'][-1]['started_cumulative'],'inner_iterations_by_phase':[x['inner_iterations']for x in review['history']],'independent_squared_direction_norm_max_error':review['max_particle_direction_squared_norm_error'],'metric_scope':'max_abs_error isfullimageCPUaccumulationerror only; noindependentfulltransporttrajectoryerror computed.'};print('STUDY_TRIAL '+json.dumps(row,allow_nan=False),flush=True)
lib.egs_destroy()
