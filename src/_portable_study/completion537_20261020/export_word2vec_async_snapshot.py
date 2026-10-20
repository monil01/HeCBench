#!/usr/bin/env python3
"""Export compact additional workflow evidence without raw research or old edits."""
import datetime,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[3]
P=R/'2026-PASC-Hecbench-Julia-paper';S=R/'src/word2vec-julia/async_workflow'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
obs=load(S/'paper_observations.json');effort=load(S/'effort_summary.json');protocol=load(S/'protocol.json')
result=R/'.porting-state/portable_julia/runs/20261020T155223860367Z_word2vec/result.json';d=load(result)
assert sha(result)==obs['final_result_sha256']
assert d['reference_agreement']and all(d['variant_verified'][m]for m in['ka','jacc'])
assert all(len(x['trials'])==5 and all(t['correct']for t in x['trials'])for x in d['models'].values())
assert obs['numerical_agreement_claim']is False and obs['raw_output_identity_claim']is False
out={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'benchmark':'word2vec','configuration_id':d['configuration_id'],'role':'workflow_property_secondary','frozen_primary_component_replaced':False,'named_numerical_identity_claim':False,'raw_embedding_identity_claim':False,'result':str(result.relative_to(R)),'result_sha256':sha(result),'scientific_observations':obs,'effort':effort,'protocol':protocol,'per_model_implementation_traits':load(S/'implementation_traits.json'),'source_support':{str(f.relative_to(R)):sha(f)for f in[S/'paper_observations.json',S/'effort_summary.json',S/'protocol.json',S/'full_saved_property_audit.json',S/'source_binding_audit.json',S/'implementation_traits.json']},'scope':'Additional bounded source-asynchronous workflow, generated named corpus rather than text8; shared CPU host controller and Julia request queue are explicit. Full raw models differ and are independently checked. No second benchmark name is counted; old numerical CBOW component and its original effort remain separate. Primary comparison artifact is a static-input/property certificate; complete model/trace files are checked by the mandatory saved audit.'}
receipt=R/'src/_portable_study/completion537_20261020/word2vec_workflow_commits.json'
if receipt.exists():out['source_commits']=load(receipt)
(P/'data/word2vec_async_workflow.json').write_text(json.dumps(out,indent=2)+'\n');print('Exported additional property workflow:',len(json.dumps(out)),'bytes')
