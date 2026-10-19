"""Diagnostic full-buffer capture through owned C ABI; no five-trial/target success claim."""
from pathlib import Path
import argparse,ctypes as C,json,time,hashlib,struct
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
lib=C.CDLL(str(a.library.resolve()));lib.egs_create.argtypes=[C.c_char_p];lib.egs_create.restype=C.c_int;lib.egs_name.argtypes=[C.c_int];lib.egs_name.restype=C.c_char_p;lib.egs_encoding.argtypes=[C.c_int];lib.egs_encoding.restype=C.c_char_p;lib.egs_bytes.argtypes=[C.c_int];lib.egs_bytes.restype=C.c_size_t;lib.egs_copy.argtypes=[C.c_int,C.c_void_p];lib.egs_scalar.argtypes=[C.c_int];lib.egs_scalar.restype=C.c_double;lib.egs_native_sim.argtypes=[C.c_int]
n=lib.egs_create(str(a.data.resolve()).encode());desc=[dict(index=i,name=lib.egs_name(i).decode(),encoding=lib.egs_encoding(i).decode(),bytes=lib.egs_bytes(i),mutable=bool(lib.egs_mutable(i)))for i in range(n)];scalars=[lib.egs_scalar(i)for i in range(22)];report={'role':'native_readiness_diagnostic_not_benchmark_trials','library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'scalars':scalars,'allocations':desc,'captures':[],'timings_not_primary':[]}
def capture(label,full):
 records={}
 for d in desc:
  if not full and not d['mutable']:continue
  raw=(C.c_ubyte*d['bytes'])();lib.egs_copy(d['index'],raw);b=bytes(raw);fn=f"{label}_{d['name']}.bin";(a.output/fn).write_bytes(b);records[d['name']]={'path':fn,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 assert lib.egs_guards()==1 and lib.egs_readonly()==1
 report['captures'].append({'label':label,'records':records,'guards_pass':True,'all_readonly_unchanged':True});return records
capture('initial',True)
for phase in range(8):
 lib.egs_sync();t=time.perf_counter();lib.egs_native_sim(phase);lib.egs_sync();t1=time.perf_counter();lib.egs_native_sum();lib.egs_sync();t2=time.perf_counter();report['timings_not_primary'].append({'phase':phase,'simulation_seconds':t1-t,'sum_seconds':t2-t1});r=capture(f'phase{phase}',False);counts=struct.unpack('<9Q',(a.output/r['step_counts']['path']).read_bytes());print('PHYSICS_PHASE',phase,counts,flush=True)
 if not sum(counts[:8]):break
else:raise AssertionError('Originalsourceactivity failedtodrain within8namedbatches')
lib.egs_destroy();(a.output/'capture_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'phases':phase+1,'complete_all_buffer_capture':True,'allocations':n,'output':str(a.output)}))
