#!/usr/bin/env python3
"""Read-only complete capture/input/guard/source audit; no GPU execution."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def guarded(p):return bytes([165])*64+p.read_bytes()+bytes([165])*64
def audit(result):
 d=json.loads(result.read_text());expected=b''.join(guarded(P/n) for n in ['anchors_u16.bin','control_u8.bin','oracle_return_i32.bin','oracle_max_i32.bin','oracle_parent_i32.bin']);checks=[];bindings=[]
 for model in ['cuda','cuda_jl','ka','jacc']:
  v=d['models'][model];assert len(v['trials'])==5 and v['complete'] and v['status']=='native_check_pass'
  for t in v['trials']:
   a=t['output_artifact'];p=Path(a['saved_path']);p=p if p.exists() else ROOT/p
   h=hashlib.sha256();offset=0
   with p.open('rb') as f:
    while b:=f.read(1048576):
     assert b==expected[offset:offset+len(b)],(model,t['trial'],offset);h.update(b);offset+=len(b)
   assert offset==len(expected)==a['count'];assert h.hexdigest()==a['sha256'];checks.append({'model':model,'trial':t['trial'],'full_bytes_checked':offset,'sha256':h.hexdigest(),'inputs_payload_and_all_guards_exact':True})
  for path,h in v['adapter_file_hashes'].items():
   p=Path(path);assert p.is_file() and sha(p)==h,('source_changed',path);bindings.append({'model':model,'path':path,'sha256':h})
 report={'benchmark':'minimap2','result':str(result.relative_to(ROOT)),'result_sha256':sha(result),'all20_full_byte_checks_passed':True,'files_checked':len(checks),'bytes_checked':sum(x['full_bytes_checked'] for x in checks),'exact_input_prefix_bytes_each':d['input_prefix_count'],'independent_DP_and_guards_compared':True,'source_asset_bindings_rehashed':len(bindings),'checks':checks,'bindings':bindings,'scope':'All complete timed-trial outputs/finaltrackers, actual initial device inputs and guards. No claim overwritten timed intermediate snapshots. Sourceexpectedselected385248fields independentcontrol; fulloriginalCLIcontrolseparate.'};(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['checks','bindings']},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('result',type=Path);a=ap.parse_args();audit(a.result.resolve())
