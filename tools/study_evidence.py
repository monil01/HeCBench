"""Apply hash-bound classification supplements without changing immutable runs."""
import copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'numerical_evidence_strength','native_reference_repaired','native_behavior_known_scientific_defect','reference_provenance','exclude_from_scientific_fidelity_aggregates','exclude_untouched_native_aggregates','exclude_from_aggregate','source_scope_adjudication','input_contract_metadata_erratum','reported_metric_definition_erratum'}
def effective_config(result_path,config):
    result_path=Path(result_path);raw_result=result_path.read_bytes();result=json.loads(raw_result)
    sidecar=ROOT/'src'/(result['benchmark']+'-julia')/'study_adjudications.json'
    if not sidecar.exists():return config
    raw_sidecar=sidecar.read_bytes();document=json.loads(raw_sidecar)
    if not isinstance(document.get('run_adjudications'),list):raise ValueError('Classification supplement requires a run_adjudications list: '+str(sidecar))
    entries=document['run_adjudications']
    entries=[e for e in entries if e['run_id']==result['run_id']]
    if not entries:return config
    if len(entries)!=1:raise ValueError('Ambiguous run classification supplement: '+str(sidecar))
    e=entries[0]
    if not e.get('classification_only') or set(e['updates'])-ALLOWED:raise ValueError('Supplement attempts to change execution or acceptance: '+str(sidecar))
    if e['result_sha256']!=hashlib.sha256(raw_result).hexdigest() or e['config_sha256']!=hashlib.sha256(result_path.with_name('config.json').read_bytes()).hexdigest():raise ValueError('Classification supplement hash mismatch: '+str(sidecar))
    c=copy.deepcopy(config);c.update(e['updates']);c['configuration_classification_adjudication']={'path':str(sidecar.relative_to(ROOT)),'sha256':hashlib.sha256(raw_sidecar).hexdigest(),'run_id':result['run_id'],'reason':e['reason'],'recorded_at':e.get('recorded_at'),'updated_fields':sorted(e['updates']),'immutable_execution_and_trials_unchanged':True}
    return c
