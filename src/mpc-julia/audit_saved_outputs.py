#!/usr/bin/env python3
"""Read-only complete input/bitstream/decoded/offset/guard audit, both strata."""
import argparse,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent.parent
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def guarded(x):return bytes([165])*64+x+bytes([165])*64
def audit(results):
 reports=[]
 for result in results:
  result=result.resolve();d=json.loads(result.read_text());cfg=json.loads((result.parent/'config.json').read_text());set=cfg['models']['cuda']['command'][-1];rows=[list(map(int,x.split())) for x in (P/f'cases_{set}.txt').read_text().splitlines()];pre=bytearray();out=bytearray();lossless_layout=[]
  for index,n,dim,cap,used,consumed in rows:
   q=P/'fixtures'/set/f'case{index}';input=(q/'input_u64.bin').read_bytes();encoded=(q/'encoded_u64.bin').read_bytes();decoder=(q/'decoder_input_u64.bin').read_bytes();decoded=(q/'decoded_u64.bin').read_bytes();co=(q/'offset_comp_i32.bin').read_bytes();de=(q/'offset_dec_i32.bin').read_bytes();assert input==decoded;ip=len(pre)+64;dp=len(out)+len(encoded)+128+64;lossless_layout.append((ip,dp,n*8));pre.extend(guarded(input));pre.extend(guarded(decoder));out.extend(guarded(encoded));out.extend(guarded(decoded));out.extend(guarded(co));out.extend(guarded(de))
  assert len(pre)==cfg['input_prefix_count'];expected=pre+out;checks=[];bindings=[]
  for model in ['cuda','cuda_jl','ka','jacc']:
   v=d['models'][model];assert v['complete'] and v['status']=='native_check_pass' and len(v['trials'])==5
   for t in v['trials']:
    a=t['output_artifact'];p=Path(a['saved_path']);actual=p.read_bytes();assert len(actual)==len(expected)==a['count'];h=hashlib.sha256(actual).hexdigest();assert h==a['sha256'];assert actual==expected
    for ip,dp,nbytes in lossless_layout:assert actual[ip:ip+nbytes]==actual[len(pre)+dp:len(pre)+dp+nbytes]
    checks.append({'model':model,'trial':t['trial'],'bytes_checked':len(actual),'sha256':h,'full_actual_decoded_equals_actual_input':True,'exact_inputs_bitstreams_offsets_guards':True})
   for name,h in v['adapter_file_hashes'].items():
    p=Path(name);assert p.is_file() and digest(p)==h,('source_changed',name);bindings.append({'model':model,'path':name,'sha256':h})
  reports.append({'configuration_id':d['configuration_id'],'native_reference_repaired':cfg['native_reference_repaired'],'result':str(result.relative_to(ROOT)),'result_sha256':digest(result),'files':len(checks),'bytes_checked':sum(x['bytes_checked'] for x in checks),'actual_input_roundtrip_words_per_capture':sum(r[1] for r in rows),'source_asset_bindings_rehashed':len(bindings),'all_full_checks_passed':True,'checks':checks,'bindings':bindings})
 report={'benchmark':'mpc','all40captures_passed':True,'total_files':sum(r['files'] for r in reports),'total_bytes_checked':sum(r['bytes_checked'] for r in reports),'configurations':reports,'fidelity':'Raw IEEE64bits preserved; nofloatingnonfinitecanonicalization, no checksumonly tests. Completeencodedcapacity/actualdecoderinputs/finaloffsetstates/guards compared. Partialsecondarycapacitycorrection separate.'};(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='configurations'},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('results',nargs='+',type=Path);audit(ap.parse_args().results)
