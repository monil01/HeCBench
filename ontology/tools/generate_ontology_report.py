#!/usr/bin/env python3
"""Build the offline ontology explorer from catalogue and reviewed Julia records."""
import argparse
import base64
import gzip
import datetime
import hashlib
import json
from pathlib import Path

import yaml
from study_report_metadata import context

ROOT = Path(__file__).resolve().parents[2]
MODELS = {'cuda_jl': 'CUDA.jl', 'ka': 'KernelAbstractions', 'jacc': 'JACC'}


def get(obj, path, default=None):
    for key in path.split('.'):
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def flatten(obj, prefix='', out=None):
    """Retain distinct scalar values, including false/zero; array paths use []."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        if not obj and prefix:
            out.setdefault(prefix, [])
        for key, value in obj.items():
            flatten(value, f'{prefix}.{key}' if prefix else key, out)
    elif isinstance(obj, list):
        if not obj:
            out.setdefault(prefix, [])
        for value in obj:
            flatten(value, prefix + '[]' if isinstance(value, (list, dict)) else prefix, out)
    elif prefix:
        values = out.setdefault(prefix, [])
        if obj is not None:
            # bool and integer zero are distinct ontology observations.
            if not any(type(v) is type(obj) and v == obj for v in values):
                values.append(obj)
    return out


# group, stable key, readable label, path in each row's scoped source object
COLUMNS = [
 ('Identity','id','Benchmark','id'), ('Identity','model','Implementation','model'),
 ('Identity','status','Outcome','record.status'),
 ('Scientific intent','family','Scientific family','domain.scientific_intent.primary_family'),
 ('Scientific intent','application','Application domain','domain.scientific_intent.application_domain'),
 ('Scientific intent','task','Scientific task','domain.scientific_intent.scientific_task'),
 ('Scientific intent','method','Mathematical method','domain.scientific_intent.mathematical_method'),
 ('Scientific intent','accuracy','Accuracy requirements','domain.scientific_intent.accuracy_requirements'),
 ('Scientific intent','use_case','Community use case','domain.scientific_intent.community_use_case'),
 ('Algorithm and computation','type','Benchmark type','base.taxonomy.benchmark_type'),
 ('Algorithm and computation','domain','Catalogue domains','base.taxonomy.scientific_domains'),
 ('Algorithm and computation','motif','Computational motifs','base.taxonomy.computational_motifs'),
 ('Algorithm and computation','algorithm','Actual algorithm','record.implementation_ontology_normalized.algorithm'),
 ('Algorithm and computation','patterns','Parallel patterns (catalogue)','base.parallelism.parallel_patterns'),
 ('Inputs and numerics','size','Input size regime','domain.scientific_intent.input_size_regime'),
 ('Inputs and numerics','input_shape','Input shape (catalogue)','base.taxonomy.input_data.shape'),
 ('Inputs and numerics','input_kind','Input kind (catalogue)','base.taxonomy.input_data.synthetic_or_real'),
 ('Inputs and numerics','storage','Storage precision','record.implementation_ontology_normalized.storage_precision'),
 ('Inputs and numerics','compute','Compute precision','record.implementation_ontology_normalized.compute_precision'),
 ('Inputs and numerics','timed_precision','Timed compute precision','record.implementation_ontology_normalized.timed_compute_precision'),
 ('Data and memory','structures','Data structures (catalogue)','base.taxonomy.data_structures'),
 ('Data and memory','layout','Actual layout','record.implementation_ontology_normalized.layout'),
 ('Data and memory','memory_access','Memory hierarchy / access','record.implementation_ontology_normalized.memory_hierarchy_and_access'),
 ('Data and memory','memory_model','Memory model (catalogue)','base.language_features.memory_model'),
 ('Data and memory','reuse','Data reuse (catalogue)','base.workload.data_reuse'),
 ('Parallel execution','mapping','Work-item mapping','record.implementation_ontology_normalized.work_item_mapping'),
 ('Parallel execution','geometry','Launch geometry','record.implementation_ontology_normalized.launch_geometry'),
 ('Parallel execution','sync','Synchronization','record.implementation_ontology_normalized.synchronization'),
 ('Parallel execution','atomics','Atomics / reductions','record.implementation_ontology_normalized.atomics_and_reductions'),
 ('Parallel execution','kernel_count','Authored kernel count','classes.authored_kernel_count'),
 ('Resource characteristics','resources','Resource policy','record.implementation_ontology_normalized.resource_policy'),
 ('Resource characteristics','registers','Registers','record.implementation_ontology_normalized.resource_policy.registers'),
 ('Resource characteristics','shared_bytes','Dynamic shared bytes','record.implementation_ontology_normalized.resource_policy.dynamic_shared_bytes'),
 ('Resource characteristics','spills','Spills','record.implementation_ontology_normalized.resource_policy.spills'),
 ('Resource characteristics','bias','Resource bias (inferred catalogue)','base.resource_model.resource_bias'),
 ('Resource characteristics','intensity','Arithmetic intensity (catalogue)','base.resource_model.arithmetic_intensity'),
 ('Resource characteristics','resource_confidence','Resource evidence (catalogue)','base.resource_model.confidence'),
 ('Implementation and portability','class','Reviewed implementation class','classes.controlled_class'),
 ('Implementation and portability','declared_class','Recorded implementation class','record.implementation_ontology_normalized.implementation_class'),
 ('Implementation and portability','composition','Execution composition','classes.execution_composition'),
 ('Implementation and portability','device_api','Device API dependence','classes.device_api_dependence'),
 ('Implementation and portability','host_role','Host role','classes.host_role'),
 ('Implementation and portability','library','Library / backend boundary','record.implementation_ontology_normalized.library_backend_boundary'),
 ('Implementation and portability','host','Host control boundary','record.implementation_ontology_normalized.host_control_boundary'),
 ('Implementation and portability','support','Support boundary','record.implementation_ontology_normalized.support_boundary'),
 ('Implementation and portability','mapping_fidelity','Native mapping fidelity','record.implementation_ontology_normalized.native_to_model_mapping'),
 ('Verification','agreement','Agreement kind','derived.agreement'),
 ('Verification','contract','Verification contract','derived.contract'),
 ('Verification','fidelity','Fidelity scope','record.implementation_ontology_normalized.fidelity_status_and_scope'),
 ('Verification','corrected','Corrected native reference','derived.corrected'),
 ('Verification','known_defect','Known native defect','derived.known_defect'),
 ('Verification','revised','Revised acceptance contract','derived.revised'),
 ('Verification','trials_correct','Recorded trial verdicts','derived.trials_correct'),
 ('Performance','configs','Configuration IDs','derived.configs'),
 ('Performance','timing_scope','Timed region','derived.timing_scope'),
 ('Performance','times','Configuration mean seconds (last four)','derived.times'),
 ('Performance','trials','All recorded trial seconds','derived.trials'),
 ('Performance','hardware_stratum','Hardware stratum','derived.hardware_stratum'),
 ('Performance','profiling','Profiling status','record.implementation_ontology.profiling_status'),
 ('V100 roofline observations','v100_status','V100 profiling status','roofline.profiling_status'),
 ('V100 roofline observations','v100_config','V100 profiled configurations','roofline.configuration_ids'),
 ('V100 roofline observations','v100_device','V100 device UUID','roofline.device_uuid'),
 ('V100 roofline observations','v100_kernels','V100 recorded kernel observations','roofline.kernel_observation_count'),
 ('V100 roofline observations','v100_loaded','V100 hash-verified kernel observations','roofline.loaded_kernel_observation_count'),
 ('V100 roofline observations','v100_coverage','V100 ROI coverage','roofline.roi_coverage'),
 ('V100 roofline observations','v100_regimes','Sampled kernel geometric regimes','roofline.geometric_regimes'),
 ('V100 roofline observations','v100_predicted_limit','Sampled compute/memory limit prediction','roofline.predicted_resource_limits'),
 ('V100 roofline observations','v100_causal','V100 causal bottleneck','roofline.causal_bottleneck'),
 ('V100 roofline observations','v100_precision','Sampled execution precision classes','roofline.precision_classes'),
 ('V100 roofline observations','v100_link','V100 profile evidence link status','roofline.evidence_link_status'),
 ('V100 roofline observations','v100_source','V100 profile evidence paths','roofline.evidence_paths'),
 ('Native CUDA roofline baseline','v100_native_status','Native CUDA profiling status','roofline_native.profiling_status'),
 ('Native CUDA roofline baseline','v100_native_kernels','Native CUDA recorded kernel observations','roofline_native.kernel_observation_count'),
 ('Native CUDA roofline baseline','v100_native_coverage','Native CUDA ROI coverage','roofline_native.roi_coverage'),
 ('Native CUDA roofline baseline','v100_native_regimes','Native CUDA sampled geometric regimes','roofline_native.geometric_regimes'),
 ('Native CUDA roofline baseline','v100_native_predicted_limit','Native CUDA compute/memory limit prediction','roofline_native.predicted_resource_limits'),
 ('Native CUDA roofline baseline','v100_native_causal','Native CUDA causal bottleneck','roofline_native.causal_bottleneck'),
 ('Native CUDA roofline baseline','v100_native_link','Native CUDA evidence link status','roofline_native.evidence_link_status'),
 ('Effort and failures','attempts','Candidate attempts','record.effort.candidate_attempts'),
 ('Effort and failures','compiler_checks','Compiler checks','record.effort.compiler_checks'),
 ('Effort and failures','compiler_repairs','Compiler feedback repairs','record.effort.compiler_feedback_repairs'),
 ('Effort and failures','runtime_repairs','Runtime repairs','record.effort.runtime_repairs'),
 ('Effort and failures','numerical_repairs','Numerical repairs','record.effort.numerical_repairs'),
 ('Effort and failures','failure_codes','Observed event codes','derived.failure_codes'),
 ('Effort and failures','failure_count','Observed failure count','record.failure_taxonomy.observed_failure_event_count'),
 ('Effort and failures','history','Failure history completeness','record.failure_taxonomy.history_completeness'),
 ('Effort and failures','tokens_status','Token accounting status','record.effort.translation_tokens.status'),
 ('Effort and failures','tokens','Recorded total translation tokens','record.effort.translation_tokens.total_tokens'),
 ('Effort and failures','token_windows','Observed token windows (overlap possible)','derived.token_windows'),
 ('Effort and failures','source_proxy','Estimated saved-source tokens (proxy)','record.effort.translation_tokens.estimated_saved_source_tokens'),
 ('Environment and provenance','julia','Julia release','record.environment.julia_release'),
 ('Environment and provenance','cuda_version','CUDA.jl version','record.environment.cuda_jl'),
 ('Environment and provenance','ka_version','KA version','record.environment.kernel_abstractions'),
 ('Environment and provenance','jacc_version','JACC version','record.environment.jacc'),
 ('Environment and provenance','gpu','GPU','record.environment.gpu'),
 ('Environment and provenance','review','Implementation review','record.implementation_review_status'),
 ('Environment and provenance','intent_review','Scientific intent review','domain.scientific_intent.review_status'),
 ('Environment and provenance','source_commit','Source commit','record.source_commit'),
 ('Environment and provenance','source_hash','Configuration source hashes','derived.source_hash'),
 ('Environment and provenance','source_evidence','Implementation source evidence','record.implementation_ontology.source_evidence'),
 ('Relationships and discovery','related','Related benchmarks','base.search.related_benchmarks'),
 ('Relationships and discovery','clusters','Cluster membership','base.relationships.cluster_membership'),
 ('Relationships and discovery','neighbors','Nearest neighbors','base.relationships.nearest_neighbors'),
 ('Relationships and discovery','keywords','Keywords','base.search.keywords'),
]


def distinct(values):
    result = []
    for v in values:
        if v is not None and v not in result:
            result.append(v)
    return result


def load_roofline_registry(root, population):
    """Read configuration-scoped profile links; never mutate inferred catalogue traits.

    Only hash-verified linked observations contribute sampled geometric regimes.
    Native CUDA is a baseline property, not a fourth Julia implementation row.
    """
    folder = root/'ontology/studies/v100_roofline_537'
    summaries, inputs, linked_inputs = {}, [], {}
    stats = {m:{'status_counts':{}, 'registry_slots':0, 'hash_verified_kernel_observations':0,
                'slots_with_hash_verified_observations':0, 'evidence_link_issues':0,
                'complete_ROI_slots':0} for m in ['cuda', *MODELS]}
    protocol_path = folder/'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if population == 537 and protocol_path.exists() else {}
    if population == 537 and protocol_path.exists(): inputs.append(protocol_path)
    for file in sorted((folder/'benchmarks').glob('*.json')) if population == 537 else []:
        registry_raw = file.read_bytes()
        registry = json.loads(registry_raw); inputs.append(file)
        bid = registry.get('benchmark', file.stem)
        config = get(registry, 'accepted_named_contract.configuration_id')
        for model in stats:
            entry = registry.get('models', {}).get(model, {})
            status = entry.get('profiling_status', 'not_recorded')
            ref = entry.get('kernel_observations', {})
            refs = ref if isinstance(ref, list) else [ref] if isinstance(ref, dict) and ref else []
            # Future compact records may also be inline in the registry.
            observations = entry.get('kernel_records', [])
            observations = list(observations) if isinstance(observations, list) else []
            link_status, paths = [], []
            for link in refs:
                name = link.get('path'); expected = link.get('sha256')
                if not name:
                    link_status.append('missing_path'); continue
                path = Path(name); path = path if path.is_absolute() else root/path
                paths.append(name)
                if not path.is_file():
                    link_status.append('missing_linked_evidence'); continue
                raw = path.read_bytes(); actual = hashlib.sha256(raw).hexdigest()
                if not expected:
                    link_status.append('missing_expected_hash'); continue
                if actual != expected:
                    link_status.append('hash_mismatch'); continue
                compression = link.get('compression') or ('gzip' if path.suffix == '.gz' else None)
                if compression not in [None, 'gzip']:
                    link_status.append('unsupported_compression'); continue
                try:
                    decoded = gzip.decompress(raw) if compression == 'gzip' else raw
                except (OSError, EOFError):
                    link_status.append('invalid_gzip_evidence'); continue
                decoded_hash = hashlib.sha256(decoded).hexdigest()
                decoded_expected = link.get('uncompressed_sha256')
                if compression == 'gzip' and not decoded_expected:
                    link_status.append('missing_uncompressed_hash'); continue
                if decoded_expected and decoded_hash != decoded_expected:
                    link_status.append('uncompressed_hash_mismatch'); continue
                try: report = json.loads(decoded)
                except (ValueError, UnicodeDecodeError):
                    link_status.append('invalid_linked_json'); continue
                records = report.get('records', report.get('kernel_records', [])) if isinstance(report, dict) else report
                if not isinstance(records, list):
                    link_status.append('invalid_kernel_records'); continue
                observations.extend(records); linked_inputs[str(path)] = {'sha256':actual, 'uncompressed_sha256':decoded_hash, 'compression':compression}
                link_status.append('hash_verified')
            if entry.get('kernel_records'): link_status.append('registry_inline')
            if not link_status: link_status = ['not_linked']
            regimes, precisions, observed_configs = [], [], []
            for obs in observations:
                if not isinstance(obs, dict): continue
                precisions.append(obs.get('precision_class'))
                observed_configs.append(get(obs, 'scope_context.configuration_id'))
                for point in obs.get('scalar_precision_projections', {}).values():
                    if isinstance(point, dict) and (point.get('executed_scalar_flops') or 0) > 0:
                        regimes.append(point.get('roofline_regime'))
            ids = distinct(observed_configs + [entry.get('profile_plan_configuration_id') or config])
            causal = entry.get('observed_bottleneck') or entry.get('causal_bottleneck') or 'unknown'
            recorded = entry.get('sampled_kernel_count')
            if recorded is None and observations: recorded = len(observations)
            summary = {'profiling_status':status, 'configuration_ids':ids,
                       'device_uuid':get(entry,'device.uuid') or get(protocol,'gpu.uuid'),
                       'kernel_observation_count':recorded,
                       'loaded_kernel_observation_count':len(observations) if observations else (0 if refs else None),
                       'roi_coverage':entry.get('roi_coverage','unknown'),
                       'geometric_regimes':distinct(regimes), 'precision_classes':distinct(precisions),
                       'predicted_resource_limits':distinct([{'memory_side':'memory_bound_roofline_prediction',
                           'compute_side':'compute_bound_roofline_prediction','near_ridge':'ridge_transition'}.get(x,'unclassified') for x in regimes]),
                       'causal_bottleneck':causal, 'evidence_link_status':distinct(link_status),
                       'evidence_paths':paths + entry.get('profile_ids', []),
                       'registry_path':str(file.relative_to(root)),
                       'registry_sha256':hashlib.sha256(registry_raw).hexdigest(),
                       'raw_model_record':entry, 'kernel_observations':observations,
                       'geometric_regime_scope':'Sampled per-kernel precision projections only; not whole-ROI bottleneck'}
            summaries[(bid,model)] = summary
            st = stats[model]; st['registry_slots'] += 1
            st['status_counts'][status] = st['status_counts'].get(status,0)+1
            st['hash_verified_kernel_observations'] += len(observations)
            st['slots_with_hash_verified_observations'] += bool(observations)
            st['evidence_link_issues'] += any(v not in ['not_linked','hash_verified','registry_inline'] for v in link_status)
            st['complete_ROI_slots'] += summary['roi_coverage'] in ['complete_device_ROI','complete_workflow','complete_ROI']
    return summaries, {'models':stats, 'registry_names':len({b for b,m in summaries}),
                       'population':537 if population == 537 else None,
                       'native_baseline_is_not_a_Julia_implementation_row':True,
                       'source':'ontology/studies/v100_roofline_537',
                       'linked_evidence_sha256':linked_inputs}, inputs


def build(root, population=447):
    ont = root/'ontology'
    study = ont/f'studies/julia_abstractions_{population}'
    roofline, roofline_summary, roofline_inputs = load_roofline_registry(root, population)
    normalized = root/'2026-PASC-Hecbench-Julia-paper/research/revision_experiments/classes/normalized_records.json'
    class_rows = json.loads(normalized.read_text()) if normalized.exists() else []
    classes = {(r['benchmark'], r['model']): r for r in class_rows}
    bases, data = {}, []
    mapping = {
        'backends':'catalogue.available_backends','models':'catalogue.programming_models',
        'suite':'catalogue.benchmark_suite_category','type':'taxonomy.benchmark_type',
        'domain':'taxonomy.scientific_domains','algo':'taxonomy.algorithm_families',
        'motif':'taxonomy.computational_motifs','layout':'taxonomy.data_layout',
        'precision':'taxonomy.precision_model','validation':'taxonomy.output_data.validation',
        'synth':'taxonomy.input_data.synthetic_or_real','kcount':'parallelism.kernel_count',
        'multi':'parallelism.multi_kernel','pattern':'parallelism.parallel_patterns',
        'sync':'parallelism.synchronization','atomics':'parallelism.atomics',
        'reductions':'parallelism.reductions','access':'parallelism.memory_access_patterns',
        'bias':'resource_model.resource_bias','ai':'resource_model.arithmetic_intensity',
        'div':'resource_model.branch_divergence','comm':'resource_model.communication_intensity',
        'rconf':'resource_model.confidence','code':'language_features.code_structure',
        'mem':'language_features.memory_model','depkind':'dependencies.dependency_kinds',
        'libs':'dependencies.libraries','completeness':'quality.completeness',
        'confidence':'quality.confidence','reviewer':'quality.reviewed_by','updated':'quality.last_updated'}
    scalars = {'validation','synth','kcount','multi','atomics','reductions','bias','ai','div','comm','rconf','completeness','confidence','reviewer','updated'}
    for file in sorted((ont/'benchmarks').glob('*.yaml')):
        b = yaml.safe_load(file.read_text()); bid = b['id']; bases[bid] = b
        row = {'id':bid, 'summary':b.get('summary','')}
        for key,path in mapping.items():
            row[key] = get(b,path, None if key in scalars else [])
        data.append(row)
    domains = {p.stem:json.loads(p.read_text()) for p in (study/'domains').glob('*.json')}
    inputs = sorted((study/'implementations').glob('*/*.json'))
    workflow_inputs=[]
    rows, counts = [], {m:{} for m in MODELS}
    for file in inputs:
        rec = json.loads(file.read_text()); bid = rec['benchmark_id']; model = rec['implementation_model']
        workflow=study/'workflow_studies'/bid/(model+'.json')
        if workflow.exists():
            rec['additional_workflow_studies']=json.loads(workflow.read_text());workflow_inputs.append(workflow)
        if model not in MODELS: continue
        cfg = rec.get('configurations',[]); dom = domains.get(bid,{})
        counts[model][rec['status']] = counts[model].get(rec['status'],0)+1
        derived = {name:distinct([c.get(field) for c in cfg]) for name,field in [
            ('agreement','numerical_agreement_kind'),('contract','verification_contract'),
            ('corrected','native_reference_repaired'),('known_defect','native_behavior_known_scientific_defect'),
            ('revised','verification_contract_revised'),('configs','configuration_id'),
            ('times','mean_last_four_seconds'),('hardware_stratum','hardware_stratum'),('source_hash','source_hash')]}
        derived['trials_correct'] = [v for c in cfg for v in c.get('trials_correct',[])]
        derived['trials'] = [v for c in cfg for v in c.get('trials_seconds',[])]
        derived['timing_scope'] = get(rec,'implementation_ontology_normalized.timing_scope') or get(rec,'shared_benchmark_traits.timing_scope')
        derived['failure_codes'] = distinct([e.get('taxonomy_code') for e in get(rec,'failure_taxonomy.events',[])])
        derived['token_windows'] = get(rec,'effort.translation_tokens.observed_activity_windows',[])
        sources = {'id':bid,'model':model,'record':rec,'base':bases.get(bid,{}),'domain':dom,'classes':classes.get((bid,model),{}),'derived':derived,'roofline':roofline.get((bid,model),{}),'roofline_native':roofline.get((bid,'cuda'),{})}
        props = flatten({k:v for k,v in sources.items() if k not in ['base','domain']})
        values = {key:get(sources,path) for _,key,_,path in COLUMNS}
        rows.append({'id':bid,'model':model,'values':values,'properties':props,'configurations':cfg,'record_path':str(file.relative_to(root))})
    covered = {r['id'] for r in rows}
    for bid in bases.keys()-covered:
        sources={'id':bid,'model':'catalogue','base':bases[bid]}
        rows.append({'id':bid,'model':'catalogue','values':{key:get(sources,path) for _,key,_,path in COLUMNS},'properties':{},'configurations':[],'record_path':f'ontology/benchmarks/{bid}.yaml'})
    base_props = {bid:flatten({'base':b,'domain':domains.get(bid,{})}) for bid,b in bases.items()}
    payload={'rows':sorted(rows,key=lambda r:(r['id'],list(MODELS).index(r['model']) if r['model'] in MODELS else 3)),
             'base_properties':base_props,'columns':[{'group':g,'key':k,'label':l,'path':p} for g,k,l,p in COLUMNS],
             'study_path':str(study.relative_to(root)),'status_counts':counts,'assessed_names':len(domains),'class_source':str(normalized.relative_to(root)) if normalized.exists() else None,
             'roofline_summary':roofline_summary, 'population_context':context(root,population),'source_files':len(inputs),'additional_workflow_source_files':len(workflow_inputs),'source_digest':hashlib.sha256(b''.join(p.read_bytes() for p in inputs+workflow_inputs+roofline_inputs)).hexdigest()}
    payload['roofline_source_digest'] = hashlib.sha256(json.dumps(roofline_summary.get('linked_evidence_sha256',{}),sort_keys=True).encode()).hexdigest()
    meta={'total':len(data),'status':{'recorded':len(data)},'generated':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return data,meta,payload


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path);parser.add_argument('--population',type=int,choices=[447,537],default=447)
    args=parser.parse_args();root=args.root.resolve();data,meta,payload=build(root,args.population)
    assets=Path(__file__).parent/'report';template=(assets/'template.html').read_text();labels=context(root,args.population)
    template=template.replace('__STUDY_SUMMARY__',labels['summary']).replace('__STUDY_POPULATION__',str(args.population))
    for marker,value in [('__BENCHMARK_DATA__',data),('__REPORT_META__',meta)]:
        template=template.replace(marker,json.dumps(value,separators=(',',':'),default=str).replace('<','\\u003c'))
    # Native browser gzip decoding keeps the report offline and avoids a large HTML download.
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),default=str).encode(),mtime=0)).decode()
    unpack="JSON.parse(await new Response(new Blob([Uint8Array.from(atob('"+packed+"'), c => c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).text())"
    template=template.replace('__STUDY_DATA__',unpack)
    explorer_js = (assets/'explorer.js').read_text()
    if args.population == 537:
        template = template.replace('<h2 id="explorer">', '<h2>V100 kernel observations</h2><p class="secnote">Native CUDA remains the baseline; the Julia implementation denominator is unchanged. Profiles are configuration/device-specific samples. Geometric roofline position and causal bottlenecks are separate; inferred catalogue resource bias is preserved.</p><div id="rooflineSummary"></div><h2 id="explorer">', 1)
        explorer_js += """
const RF=STUDY.roofline_summary;
const RM={cuda:'Native CUDA',cuda_jl:'CUDA.jl',ka:'KernelAbstractions',jacc:'JACC'};
const rs=Object.values(RF.models||{});
const statuses=[...new Set(rs.flatMap(s=>Object.keys(s.status_counts||{})))].sort();
$('rooflineSummary').innerHTML='<table class="study-table"><thead><tr><th>V100 profile assessment</th>'+Object.values(RM).map(m=>'<th>'+esc(m)+'</th>').join('')+'</tr></thead><tbody>'+statuses.map(k=>'<tr><td>'+esc(k.replaceAll('_',' '))+'</td>'+Object.keys(RM).map(m=>'<td class="study-count">'+(RF.models[m]?.status_counts[k]??0)+'</td>').join('')+'</tr>').join('')+[
 ['Linked kernel observations (hash verified)','hash_verified_kernel_observations'],
 ['Slots with linked observations','slots_with_hash_verified_observations'],
 ['Explicit complete ROI coverage','complete_ROI_slots'],
 ['Linked evidence issues','evidence_link_issues']
].map(([label,key])=>'<tr><td>'+esc(label)+'</td>'+Object.keys(RM).map(m=>'<td class="study-count">'+(RF.models[m]?.[key]??0)+'</td>').join('')+'</tr>').join('')+'</tbody></table><p class="cap">'+RF.registry_names+' benchmark registry names; status counts are per model, not extra Julia implementations. Completed profiling processes can still have partial or unassigned ROI coverage. No observations is not a measured zero. Filter V100 columns or search raw roofline fields below for exact configurations, device, source links and kernel records.</p>';
"""
    template=template.replace('__EXPLORER_JS__',explorer_js).replace('__EXPLORER_CSS__',(assets/'explorer.css').read_text())
    template=template.replace('<script>', '<script>\n(async () => {')
    template=template.replace('</script>', "\n})().catch(error => { document.getElementById('count').textContent = 'Could not load the report. Use a current Chrome, Firefox, Edge or Safari browser with gzip decompression support.'; console.error(error); });\n</script>")
    out=args.output or root/'ontology/reports/ontology_report.html';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(template)
    print(json.dumps({'output':str(out),'catalogue_benchmarks':len(data),'assessed_julia_names':payload['assessed_names'],'implementation_records':payload['source_files'],'explorer_rows':len(payload['rows']),'columns':len(COLUMNS),'bytes':out.stat().st_size}))


if __name__=='__main__': main()
