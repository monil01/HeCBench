from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());meta=json.loads((p/'independent_oracle.json').read_text());gold=R/meta['golden_path'];assert hashlib.sha256(gold.read_bytes()).hexdigest()==meta['golden_sha256'];checks=[]
for model,data in r['models'].items():
 for t in data.get('trials',[]):
  a=Path(t['output_artifact']['saved_path']);assert a.stat().st_size==meta['output_count']*4;h=hashlib.sha256();offset=0
  with a.open('rb')as actual,gold.open('rb')as expected:
   while True:
    x=actual.read(1048576);y=expected.read(1048576);assert x==y,(model,t['trial'],offset)
    if not x:break
    h.update(x);offset+=len(x)
  checks.append(dict(model=model,trial=t['trial'],bytes_compared=offset,sha256=h.hexdigest(),all_actual_host_input_bits_destinations_guards_runtime_counts_exact=True))
print(json.dumps(dict(result=str(rp),all_pass=True,count=meta['output_count'],input_prefix_count=meta['input_prefix_count'],authored_device_kernels=0,independent_golden_sha256=meta['golden_sha256'],checks=checks),indent=2))
