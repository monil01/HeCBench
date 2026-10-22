"""Private metadata enrichment only; do not import/run target programs."""
import json,gzip,hashlib
from pathlib import Path
O=Path(__file__).parent;p=O/'proposed_stage_patches.json';d=json.loads(p.read_text());reason={("mpc","jacc"):'Existing parent has operation placeholder only despite authored generated-wrapper extension; exact sourcecallsite candidate unavailable, no newstage introduced. Not frameworkunsupported or noSMwork.',("mtf","cuda"):'Existing native operation/library placeholder lacks exact emitted kernel-family source anchor; full corrected hybrid transform remains numerically accepted, no API-to-counter family inference.'}
for r in d['withheld_models']:r['reason']=reason.get((r['benchmark'],r['model']),r['reason'])
d['withheld_stages']=[]
for r in d['patches']:
 for sp in r['stage_patches']:
  sp['updates']['ROI_membership_proof'].update(sample_in_primary_timed_invocation=None,retained_trial=None,whole_elapsed_ROI_claim=False,complete_ROI=False,causal_bottleneck='unknown')
 parent=json.loads(gzip.decompress(Path(r['parent_review']['path']).read_bytes()));old=parent['records'][r['record_index']];changed={s['stage_index']for s in r['stage_patches']}
 for i,s in enumerate(old['stages']):
  if i not in changed:d['withheld_stages'].append({'record_key':r['record_key'],'stage_index':i,'stage_id':s['stage_id'],'expected_original_stage_canonical_sha256':hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'reason':'Unresolved vendor/operation placeholder or no exact source callsite; original stage untouched, no absence-of-SM-work inference.'})
p.write_text(json.dumps(d,indent=2)+'\n')
