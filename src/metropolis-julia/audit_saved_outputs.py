"""Read-only saved full-output audit. No GPU commands or timing changes."""
from pathlib import Path
import argparse,json,hashlib,struct
ap=argparse.ArgumentParser();ap.add_argument('result');a=ap.parse_args();p=Path(__file__).resolve().parent;d=json.loads(Path(a.result).read_text());prefix=(p/'fixture.bin').read_bytes();gold=(p/'oracle.bin').read_bytes();gv=struct.unpack('<'+'d'*(len(gold)//8),gold);rows=[]
for model,m in d['models'].items():
 for t in m['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert raw[:len(prefix)]==prefix;v=struct.unpack('<'+'d'*(len(raw[len(prefix):])//8),raw[len(prefix):]);assert v==gv;assert t['correct'];rows.append(dict(model=model,trial=t['trial'],path=str(f),sha256=hashlib.sha256(raw).hexdigest(),exact_numeric_oracle=True,exact_input_prefix_bytes=True,guard_and_input_fields_in_oracle=True))
assert len(rows)==20
out=dict(result=str(Path(a.result).resolve()),result_sha256=hashlib.sha256(Path(a.result).read_bytes()).hexdigest(),all_pass=True,rows=rows,artifact_values=len(prefix)//8+len(gv),all_artifact_sha256_count=len({r['sha256'] for r in rows}),no_GPU_rerun=True)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
