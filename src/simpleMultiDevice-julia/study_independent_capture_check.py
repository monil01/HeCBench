"""Full exact bytes against pre-GPU independent IEEE oracle; no GPU/summary substitution."""
from pathlib import Path
import json,hashlib,sys
R=Path('/home/imo/HeCBench_julia');p=Path(__file__).resolve().parent;rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());g=int(r['models']['cuda_jl']['trials'][0]['active_devices']);info=json.loads((p/'independent_oracle.json').read_text())['configurations'][str(g)];gold=R/info['golden_path'];assert hashlib.sha256(gold.read_bytes()).hexdigest()==info['golden_sha256'];checks=[]
for model,data in r['models'].items():
 for t in data.get('trials',[]):
  a=Path(t['output_artifact']['saved_path']);assert a.stat().st_size==info['output_count']*8;h=hashlib.sha256();nbytes=0
  with a.open('rb')as actual,gold.open('rb')as expected:
   while True:
    x=actual.read(1048576);y=expected.read(1048576);assert x==y,(model,t['trial'],nbytes)
    if not x:break
    h.update(x);nbytes+=len(x)
  checks.append({'model':model,'trial':t['trial'],'all_input_raw_IEEE32_words_exact':True,'all_host_and_device_partialvectors_host_float32_folds_CPU_float64_reldiff_guards_exact':True,'bytes_compared':nbytes,'sha256':h.hexdigest()})
print(json.dumps({'result':str(rp),'active_devices':g,'count':info['output_count'],'prefix':info['input_prefix_count'],'independent_golden_sha256':info['golden_sha256'],'all_pass':True,'checks':checks,'scope':'RawIEEEbitoperatorcontract; sourceFloat32vsreal/CPUdoubleerror separateinindependent_oracle.json, max_abs_error0inrowsmeansrawencodingexactnotuniversalexactrealaddition.'},indent=2))
