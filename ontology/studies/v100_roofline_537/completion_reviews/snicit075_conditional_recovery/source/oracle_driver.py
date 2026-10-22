"""Fresh CPU-oracle runtime provenance; original audit source executes unchanged, outside ROI."""
from pathlib import Path
import sys,os,json,hashlib,runpy,platform
out=Path(sys.argv[1]).resolve();audit=Path('/home/imo/HeCBench_julia/src/snicit-julia/audit_model.py');pins=json.loads((Path(__file__).parent/'numpy_bindings.json').read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in pins['files'].items():
 if sha(p)!=h:raise RuntimeError('Pinned NumPy/CPUdependency changed: '+p)
import numpy as np
if np.__version__!='2.5.4' or not Path(np.__file__).resolve().is_relative_to('/tmp/pasc-julia-plot-tools/numpy'):raise RuntimeError('Unexpected CPU-oracle NumPy identity')
receipt={'status':'CPU_oracle_started','numpy_version':np.__version__,'numpy_module_file':str(Path(np.__file__).resolve()),'python_version':sys.version,'python_executable':sys.executable,'platform':platform.platform(),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),'PYTHONPATH':os.environ.get('PYTHONPATH'),'audit_source_path':str(audit),'audit_source_sha256':sha(audit),'pinned_dependency_manifest_sha256':sha(Path(__file__).parent/'numpy_bindings.json'),'primary_historical_loaded_version_identity':'Not inferred from this new process'}
try:
 sys.argv=[str(audit),str(out)];runpy.run_path(str(audit),run_name='__main__');receipt['status']='CPU_oracle_completed'
except BaseException as error:receipt.update(status='CPU_oracle_failed',error=repr(error));raise
finally:
 paths=set()
 for line in Path('/proc/self/maps').read_text().splitlines():
  part=line.split(maxsplit=5)
  if len(part)==6 and part[5].startswith('/')and'.so'in part[5]:paths.add(part[5])
 receipt['actual_loaded_shared_libraries']=[{'path':p,'sha256':sha(p),'bytes':Path(p).stat().st_size,'prebound':str(Path(p).resolve())in pins['files']}for p in sorted(paths)if Path(p).is_file()]
 mismatches=[z for z in receipt['actual_loaded_shared_libraries']if z['prebound']and z['sha256']!=pins['files'][str(Path(z['path']).resolve())]]
 receipt['prebound_loaded_dependency_mismatches']=mismatches
 if mismatches:receipt['status']='CPU_oracle_dependency_identity_failed'
 (out/'oracle_runtime_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print('SNICIT_ORACLE_RUNTIME',json.dumps({'status':receipt['status'],'numpy_version':np.__version__,'audit_sha256':receipt['audit_source_sha256']}),flush=True)

if receipt.get("prebound_loaded_dependency_mismatches"):raise RuntimeError("Loaded CPUdependency differs from prebound bytes")
