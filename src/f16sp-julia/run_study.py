from pathlib import Path
import json,subprocess,sys
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
if not(R/'.porting-state/portable_julia/builds/f16sp/libstudy_vendor.so').exists():subprocess.run([sys.executable,str(p/'build_vendor.py')],check=True)
sys.path.insert(0,str(R/'tools'));from verify_portable_julia import verify
c=json.load(open(p/'study_config.json'))
for key in list(c):
 if key.startswith('reuse_')or key in('allow_cross_configuration_reuse','cross_configuration_reuse_reason'):c.pop(key)
raise SystemExit(verify(c))
