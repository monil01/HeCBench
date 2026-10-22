from pathlib import Path
import json,gzip,hashlib,re,copy,fcntl,difflib
W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';O=Path(__file__).resolve().parent;P=S/'profiles/tsne/ka/batch054-tsne-checkers-complete';h=lambda b:hashlib.sha256(b).hexdigest()
def info(p):return {'path':str(p),'sha256':h(p.read_bytes()),'bytes':p.stat().st_size}
def patch_exact(text,udiff,reverse=False):
 lines=text.splitlines(keepends=True);out=[];pos=0;ds=udiff.splitlines(keepends=True);i=0
 while i<len(ds):
  if not ds[i].startswith('@@ '):i+=1;continue
  mat=re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@',ds[i]);assert mat;start=int(mat[2 if reverse else 1])-1;assert start>=pos;out+=lines[pos:start];pos=start;i+=1
  while i<len(ds)and not ds[i].startswith('@@ '):
   z=ds[i];i+=1
   if z.startswith(('--- ','+++ ')):continue
   kind=z[:1];body=z[1:]
   if reverse:kind='+'if kind=='-'else'-'if kind=='+'else kind
   if kind in [' ','-']:assert lines[pos]==body,('hunk mismatch',pos);pos+=1
   if kind in [' ','+']:out.append(body)
  
 out+=lines[pos:];return ''.join(out)
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lk:
 fcntl.flock(lk,fcntl.LOCK_SH)
 manifest=json.loads((P/'manifest.json').read_bytes());proofbytes=gzip.decompress((P/'source_proof.json.gz').read_bytes());proof=json.loads(proofbytes);plan=json.loads(gzip.decompress((P/'plan.json.gz').read_bytes()));ledger=json.loads((S/'benchmarks/tsne.json').read_bytes());ct=ledger['accepted_named_contract'];assert ledger['models']['ka']['latest_profile_attempt']['profile_id']==str(P.relative_to(W));assert ct['configuration_id']==plan['configuration_id']and ct['result_sha256']==plan['reference_result_sha256']and ct['raw_config_sha256']==plan['raw_config_sha256']
 e=next(x for x in manifest['files']if x['original_name']=='source_proof.json');assert h(proofbytes)==e['uncompressed_sha256']and h((P/'source_proof.json.gz').read_bytes())==e['compressed_sha256']
 old=plan['scratch_cwd'];new='/tmp/hecbench-roofline-v100-20261020/batch069/tsne/ka/scratch';rows=[]
 for q in proof['patches']:
  source=Path(q['source']);original=source.read_bytes();assert h(original)==q['source_sha256'];retained=q['text'].encode();assert h(retained)==q['copy_sha256']==plan['execution_bindings'][q['copy']];assert not Path(q['copy']).exists(),'Do not overwrite/reuse historical scratch'
  assert patch_exact(original.decode(),q['diff']).encode()==retained;assert patch_exact(q['text'],q['diff'],True).encode()==original
  recovered=O/'recovered_054'/Path(q['copy']).name;recovered.parent.mkdir(exist_ok=True);recovered.write_bytes(retained)
  before=q['text'];assert old in before;after=before.replace(old,new);assert after.replace(new,old)==before;fresh=O/'fresh_069'/Path(q['copy']).name;fresh.parent.mkdir(exist_ok=True);fresh.write_bytes(after.encode());future=q['copy'].replace(old,new)
  rows.append({'canonical_source':info(source),'retained_capture_source':q['copy'],'retained_capture_sha256':q['copy_sha256'],'recovered_exact_bytes':info(recovered),'original_to054_full_forward_inverse_diff_verified':True,'captured_diff':q['diff'],'fresh_source':info(fresh),'future_source':future,'future_sha256':h(after.encode()),'reviewed_replacement':{'before':old,'after':new,'exact_occurrences':before.count(old)},'054to069_full_forward_inverse_bytes_verified':True,'fresh_diff':''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=q['copy'],tofile=future))})
 assert len(rows)==3
 (O/'tsne_exact_source_recovery_proof.json').write_text(json.dumps({'status':'PRIVATE_PENDING_ROOT_SOURCE_ADMISSION','canonical_selected_contract':ct,'frozen_source_proof':info(P/'source_proof.json.gz'),'frozen_manifest':info(P/'manifest.json'),'old_scratch':old,'future_scratch':new,'sources':rows,'all_three_complete_retained_texts_match_original_capture_SHA':True,'historical_scratch_recreated_or_reused':False,'source_fabrication_or_hash_fallback':False,'scope':'Exact retained bytes plus full forward/inverse canonical→054 proof and literal054→069 path-only proof; mathematics/checker/codec/control/timer unchanged. No GPU/build/import.'},indent=2)+'\n')
print('TSNE all3 exact retainedSHA and canonical054/fresh069 full forward/inverse source bytes verified')
