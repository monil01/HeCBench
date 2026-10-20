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


def build(root, population=447):
    ont = root/'ontology'
    study = ont/f'studies/julia_abstractions_{population}'
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
        sources = {'id':bid,'model':model,'record':rec,'base':bases.get(bid,{}),'domain':dom,'classes':classes.get((bid,model),{}),'derived':derived}
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
             'population_context':context(root,population),'source_files':len(inputs),'additional_workflow_source_files':len(workflow_inputs),'source_digest':hashlib.sha256(b''.join(p.read_bytes() for p in inputs+workflow_inputs)).hexdigest()}
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
    template=template.replace('__EXPLORER_JS__',(assets/'explorer.js').read_text()).replace('__EXPLORER_CSS__',(assets/'explorer.css').read_text())
    template=template.replace('<script>', '<script>\n(async () => {')
    template=template.replace('</script>', "\n})().catch(error => { document.getElementById('count').textContent = 'Could not load the report. Use a current Chrome, Firefox, Edge or Safari browser with gzip decompression support.'; console.error(error); });\n</script>")
    out=args.output or root/'ontology/reports/ontology_report.html';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(template)
    print(json.dumps({'output':str(out),'catalogue_benchmarks':len(data),'assessed_julia_names':payload['assessed_names'],'implementation_records':payload['source_files'],'explorer_rows':len(payload['rows']),'columns':len(COLUMNS),'bytes':out.stat().st_size}))


if __name__=='__main__': main()
