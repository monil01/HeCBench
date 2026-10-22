from pathlib import Path
import json,gzip,hashlib,fcntl,datetime
O=Path(__file__).resolve().parent;W=Path('/home/imo/HeCBench_julia');S=W/'ontology/studies/v100_roofline_537';h=lambda b:hashlib.sha256(b).hexdigest();p=json.load(open(O/'proposed_stage_patches.json'));sel=json.load(open(O/'selection_authority_review.json'))
with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as l:
 fcntl.flock(l,fcntl.LOCK_SH);assert (S/'roi_reviews/INDEX.json').read_bytes()==(O/'selection_parent_INDEX.json').read_bytes();assert h(Path(sel['legacy']['path']).read_bytes())==sel['legacy']['sha256']
 for x in p['sources']:assert h(Path(x['path']).read_bytes())==x['sha256']
 for r in p['selected_capture_bindings']+p['withheld_models']:
  for k in ['ledger','manifest','plan','derivation','provenance']:assert h(Path(r[k]['path']).read_bytes())==r[k]['sha256']
 original=json.loads(gzip.decompress((O/'selected_review_records.json.gz').read_bytes()));cache={}
 for r in original:
  if r['parent']not in cache:cache[r['parent']]=json.loads(gzip.decompress(Path(r['parent']).read_bytes()))
  assert cache[r['parent']]['records'][r['index']]==r['record']
 (O/'final_locked_revalidation.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'INDEX_and_legacy_stable':True,'target_records_exact_equal_at_original_indices':len(original),'all_selected_capture_ledger_manifest_plan_derived_provenance_hashes_unchanged':len(p['selected_capture_bindings'])+len(p['withheld_models']),'all_source_hashes_unchanged':len(p['sources']),'metadata_lock':'LOCK_SH','target_execution':False},indent=2)+'\n')
s=json.load(open(O/'summary.json'));s['withheld_models']=[{k:v for k,v in x.items()if k in ['benchmark','model','reason']}for x in s['withheld_models']];s['withheld_unresolved_placeholders']=s.pop('withheld_vendor_placeholders');s['admitted_names']=sorted({x['record_key']['benchmark']for x in p['patches']});(O/'summary.json').write_text(json.dumps(s,indent=2)+'\n')
files={str(x.relative_to(O)):{'sha256':h(x.read_bytes()),'bytes':x.stat().st_size}for x in O.rglob('*')if x.is_file()and x.name!='DELIVERY_MANIFEST.json'};(O/'DELIVERY_MANIFEST.json').write_text(json.dumps({'status':'PRIVATE_PENDING_ROOT_REVIEW','publication_stratum':'Source-only proposal; actualphase/fullROI/causal unpromoted','files':files},indent=2)+'\n');print(json.dumps(s,indent=2))
